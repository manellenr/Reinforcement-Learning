import enum
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Cell(enum.IntEnum):
    EMPTY = 0
    WALL = 1

class Action(enum.IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    GATHER = 4

# cleanrl ppo
class Wallace(nn.Module):

    def __init__(self):
        super(Wallace, self).__init__()
        self.sequence = [Action.UP, Action.RIGHT, Action.RIGHT, Action.GATHER]
        self.actions = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT, Action.GATHER]
        self.idx = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.learning_rate = 1e-3
        self.batch_size = 32  
        self.minibatch_size = 32 
        self.update_epochs = 4
        self.gamma = 0.99  
        self.gae_lambda = 0.95 
        self.clip_coef = 0.2  
        self.norm_adv = True 
        self.clip_vloss = True  
        self.target_kl = None 
        self.ent_coef = 0.01 
        self.vf_coef = 0.5 
        self.max_grad_norm = 0.5  

        self.observation_space = [7]  
        self.action_space = [5] 

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(self.observation_space).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(self.observation_space).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.action_space[0]), std=0.01),
        )
        self.to(self.device)

        self.writer = SummaryWriter(f"runs/wallace_{int(time.time())}")
    
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def learn(self, observations, actions, rewards, dones, optimizer=None, global_step=0):
        start_time = time.time()
        observations = torch.tensor(observations, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        num_steps = len(rewards)

        with torch.no_grad():
            action_probs = self.actor(observations)
            dist = Categorical(logits=action_probs)
            log_probs = dist.log_prob(actions)
            values = self.get_value(observations).squeeze()

            next_value = self.get_value(observations[-1]).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.get_action_and_value(observations[mb_inds], actions.long()[mb_inds])
                logratio = newlogprob - log_probs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                    v_clipped = values[mb_inds] + torch.clamp(
                        newvalue - values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        self.writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
        #print("SPS:", int(global_step / (time.time() - start_time)))
        self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        self.writer.add_scalar("charts/Reward", rewards.detach().cpu().mean(), global_step)

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)
        print(f"Model loaded from {path}")

    def act(self, obs, gold_received_at_previous_step, done):
        y,x,top,left,right,bottom,has_gold = obs
        gold = gold_received_at_previous_step
        if done:
            self.idx = 0
            return None
        else:
            if has_gold:
                action = Action.GATHER
            else:
                logits = self.actor(torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
                probs = Categorical(logits=logits)
                action = probs.sample().item()

            self.idx += 1
            if action is None:
                raise RuntimeError("Wallace should never return None when done is False")
            return action

    def get_custom_render_infos(self):
        return None
