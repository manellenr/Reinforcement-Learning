import numpy as np
from tqdm import tqdm

import torch.optim as optim

from solution import Wallace
from maze import create_maze

TRAINING_ITER = 500
MAX_STEPS = 128

def gather_training_data(wallace):
    observations = []
    actions = []
    rewards = []
    dones = []


    env = create_maze(video_prefix="./videos/video_%d_train"%iter, overwrite_every_episode=False, fps=4, save_video=False)
    obs = env.reset()
    done = False
    gold_prev_step = 0
    step = 0

    for step in range(MAX_STEPS):
        if done:
            env.close()
            env = create_maze(video_prefix="./videos/video_%d_train"%iter, overwrite_every_episode=False, fps=4, save_video=False)
            obs = env.reset()
            gold_prev_step = 0.
            done = False
        action = wallace.act(obs, gold_prev_step, done)
        obs, gold_prev_step, done, info = env.step(action, render_infos=wallace.get_custom_render_infos())

        observations.append(obs)
        actions.append(action)
        rewards.append(gold_prev_step)
        dones.append(done)

    return observations, actions, rewards, dones

if __name__ == "__main__":
    exp_idx = 0

    wallace = Wallace()
    optimizer = optim.Adam(wallace.parameters(), lr=wallace.learning_rate, eps=1e-5)

    for iter in tqdm(range(TRAINING_ITER)):
        frac = 1.0 - (iter - 1.0) / TRAINING_ITER
        lrnow = frac * wallace.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

        observations, actions, rewards, dones = gather_training_data(wallace)

        wallace.learn(observations, actions, rewards, dones, optimizer= optimizer, global_step=iter*MAX_STEPS)
    
    wallace.save_model(f"./models/wallace_model.h5")
    #env.close()

    
    print("Training complete.")