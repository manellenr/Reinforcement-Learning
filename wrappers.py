import os
import tempfile
import numpy as np
import imageio
import cv2
import warnings
from collections import defaultdict

class Wrapper:
    def __init__(self, env):
        self.env = env
    
    def step(self, action, **kwargs):
        return self.env.step(action, **kwargs)
    
    def reset(self):
        return self.env.reset()
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)
    
    def close(self):
        return self.env.close()

class ActionGather(Wrapper):
    ACTION_GATHER = 4
    
    def __init__(self, env, maze):
        Wrapper.__init__(self, env)
        self.golds = maze.golds
    
    def reset(self):
        obs = self.env.reset()
        obs = self.transform_obs(obs, 0.)
        y,x = self.obs_to_yx(obs)
        self.last_obs = obs
        self.last_gold = self.golds[y,x] 
        self.last_info = {}
        self.last_done = False
        return obs
    
    def step(self, action,  **kwargs):
        if action == ActionGather.ACTION_GATHER:
            obs = self.last_obs
            gold = self.last_gold
            info = self.last_info
            done = True
        else:
            obs, gold, done, info = self.env.step(action,  **kwargs)
            obs = self.transform_obs(obs, gold)
            self.last_obs = obs
            self.last_gold = gold
            self.last_info = info
            self.last_done = done
            gold = 0.
        return obs, gold, done, info
    
    def transform_obs(self, obs, gold):
        obs = list(obs) + [gold != 0]
        assert len(obs) == 7
        return obs
    
    def obs_to_yx(self, obs):
        y,x,top,left,right,bot,_ = obs
        return y,x

class VideoWriter(Wrapper):
 
    def __init__(self, env, file_prefix, fps, overwrite_every_episode=True):
  
        Wrapper.__init__(self, env)
        self.file_prefix = file_prefix
        fd, self.temp_filename = tempfile.mkstemp('.mp4', 'tmp', '/'.join(self.file_prefix.split('/')[:-1]))
        os.close(fd)
        self.video_writer = None
        self.counter = 0
        self.score = 0
        self.frame_size = None
        self.fps = fps
        self.overwrite_every_episode = overwrite_every_episode
        self.first_reset = True

    def _process_frame(self, render_infos=None):
        frame = self.env.render(render_infos=render_infos)
        frame = np.asarray(frame, dtype=np.uint8)
        if min(frame.shape[0], frame.shape[1]) < 30:
            f = 20
        else:
            f = 4
        frame = cv2.resize(frame, (frame.shape[1]*f, frame.shape[0]*f), interpolation=cv2.INTER_NEAREST) 
        if self.frame_size is None: 
            if frame.shape[0] % 16 or frame.shape[1] % 16:
                self.frame_size = (frame.shape[0] + 16 - (frame.shape[0] % 16), frame.shape[1] + 16 - (frame.shape[1] % 16)) + tuple(frame.shape[2:])
                self.frame_pos = (int((16 - frame.shape[0] % 16)//2), int((16 - frame.shape[1] % 16)//2))
            else:
                self.frame_size =  tuple(frame.shape[:])
                self.frame_pos = (0,0)
        f_out = np.zeros(self.frame_size, dtype=np.uint8)
        f_out[self.frame_pos[0]:self.frame_pos[0]+frame.shape[0], self.frame_pos[1]:self.frame_pos[1]+frame.shape[1]] = frame
        return f_out

    def step(self, action, render_infos=None, **kwargs):
        obs, gold, done, info = self.env.step(action, **kwargs)
        self.cur_step += 1
        self.score += gold
        self.video_writer.append_data(self._process_frame(render_infos=render_infos))
        return obs, gold, done, info
    
    def _save_current(self):
        if self.video_writer is not None:
            self.video_writer.close()
            filename = f'{self.file_prefix}.mp4'
            os.replace(self.temp_filename, filename)
            self.counter += 1

    def reset(self):
        self.score = 0
        self.cur_step = 0
        obs = self.env.reset()
        if self.overwrite_every_episode:
            self.ckpt_video()
        else:
            if self.first_reset:
                self.first_reset = False
                self.video_writer = imageio.get_writer(self.temp_filename, mode='I', fps=self.fps)
            self.video_writer.append_data(self._process_frame())
        return obs
    
    def ckpt_video(self):
        self._save_current()
        self.video_writer = imageio.get_writer(self.temp_filename, mode='I', fps=self.fps)
        self.video_writer.append_data(self._process_frame())
    
    def close(self):
        self._save_current()
        self.env.close()
    
    def __del__(self):
        self.video_writer.close()
        if os.path.exists(self.temp_filename):
            os.remove(self.temp_filename)

class Monitor(Wrapper):
 
    def __init__(self, env):
        Wrapper.__init__(self, env)
        self.tot_golds = 0.
        self.tot_steps = 0
        self.ep_lengths = []

    def reset(self):
        self.ep_gold = 0.
        self.ep_length = 0
        return self.env.reset()
    
    def step(self, action, **kwargs):
        obs, gold, done, info = self.env.step(action, **kwargs)
        self.ep_gold += gold
        self.ep_length += 1
        self.tot_golds += gold
        self.tot_steps += 1
        if done:
            self.ep_lengths.append(self.ep_length)
        info.update({
            "monitor.ep_gold": self.ep_gold,
            "monitor.ep_length": self.ep_length,
            "monitor.ep_lengths": self.ep_lengths,
            "monitor.tot_golds": self.tot_golds,
            "monitor.tot_steps": self.tot_steps,
        })
        return obs, gold, done, info

class StochasticGolds(Wrapper):
 
    def __init__(self, env, maze, std):
        Wrapper.__init__(self, env)
        self.std = std
        poss = np.argwhere(maze.golds != 0)
        poss = [tuple(pos) for pos in poss] 
        MAX_SEED = 2**16
        meta_rng = np.random.RandomState(self._hash_maze(maze) % MAX_SEED) 
        self.rngs = dict(zip(poss, [np.random.RandomState(meta_rng.randint(0, MAX_SEED)) for _ in range(len(poss))]))

    def _hash_maze(self, maze):
        t = tuple()
        for arr in [maze.layout, maze.golds, maze.dones, maze.starts]:
            t += tuple(arr.flatten())
        h = hash(t)
        return h

    def _from_obs(self, obs):
        y,x,top,left,right,bot,gold_info = obs
        return y,x
    
    def step(self, action, **kwargs):
        obs, gold, done, info = self.env.step(action, **kwargs)
        if gold != 0:
            y,x = self._from_obs(obs)
            gold = self.rngs[(y,x)].normal(loc=gold, scale=self.std)
        return obs, gold, done, info
