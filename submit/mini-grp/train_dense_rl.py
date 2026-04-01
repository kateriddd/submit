"""
HW3 Part 1: Train a dense MLP policy with PPO on privileged state.

Usage:
    python hw3/train_dense_rl.py \
        experiment.name=hw3_dense_ppo_seed0 \
        r_seed=0 \
        sim.task_set=libero_spatial \
        sim.eval_tasks=[9] \
        training.total_env_steps=200000 \
        training.rollout_length=512 \
        training.ppo_epochs=10 \
        training.minibatch_size=64

Usage debugging:
    python hw3/train_dense_rl.py \
        experiment.name=hw3_dense_ppo_seed0 \
        r_seed=0 \
        sim.task_set=libero_spatial \
        "sim.eval_tasks=[9]" \
        training.total_env_steps=10000 \
        training.rollout_length=256 \
        training.ppo_epochs=10 \
        training.minibatch_size=64
"""
import os
import sys
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

# patched to skip OpenGL initialization, avoid error
def _patch_mujoco_gl():
    try:
        import mujoco.gl_context as gl_ctx
        class _NoOpGLContext:                           # Replace GLContext with a no-op class
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def make_current(self): pass
            def free(self): pass
        
        gl_ctx.GLContext = _NoOpGLContext
        print("[PATCH] MuJoCo GL context bypassed")
    except:
        pass  

_patch_mujoco_gl()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hw3.libero_env_fast import FastLIBEROEnv


# ----------------------------------------------------------------------------
# Policy and value networks
# ----------------------------------------------------------------------------

class DensePolicy(nn.Module):
    """
    MLP policy that maps privileged state observations to action distributions.
    Outputs a Gaussian distribution (mean + log_std) over the 7-DoF action space.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        # TODO: Build the policy network layers and output heads.
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #network layerrs
        self.nl = nn.Sequential(
        nn.Linear(obs_dim, hidden_dim), nn.SiLU(),
        *[module for _ in range(n_layers - 1) 
            for module in (nn.Linear(hidden_dim, hidden_dim), nn.SiLU())]   
        )
        #output heads 
        self.out_mean = nn.Linear(hidden_dim, action_dim)   # mean of 7 action dof
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))   #make it trainable

        nn.init.orthogonal_(self.out_mean.weight, gain=0.1)   # gain of 0.1 more stable than 1

    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: (B, obs_dim) float tensor
        Returns:
            dist: torch.distributions.Normal over actions
        """
        # TODO: Return a Normal distribution over actions given obs.
        actions = self.nl(obs)
        mean = self.out_mean(actions)
        std = self.log_std.clamp(-4, 0.5).exp()  # clamp for numerical stability
        return Normal(mean,std)


    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample an action and return (action, log_prob, entropy)."""
        dist = self.forward(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()

        log_prob = dist.log_prob(action).sum(-1)
        action = action.clamp(-1.0, 1.0) 
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy


class DenseValueFunction(nn.Module):
    """MLP value function V(s) for PPO critic."""
    def __init__(self, obs_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        # TODO: Build the value network layers.
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #network layerrs
        self.net = nn.Sequential(
        nn.Linear(obs_dim, hidden_dim), nn.SiLU(),
        *[module for _ in range(n_layers - 1) 
            for module in (nn.Linear(hidden_dim, hidden_dim), nn.SiLU())]  
        )

        self.value_head = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0) #gain of 1.0 to get more signal


    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns scalar value estimate of shape (B,)."""
        return self.value_head(self.net(obs)).squeeze(-1)


# ---------------------------------------------------------------------------
# PPO rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores a fixed-length on-policy rollout for PPO updates."""

    def __init__(self, rollout_length: int, obs_dim: int, action_dim: int, device: torch.device, store_images=False, image_shape: list = [64, 64, 3]):
        self.rollout_length = rollout_length
        self.device = device
        self.obs = torch.zeros(rollout_length, obs_dim, device=device)
        self.actions = torch.zeros(rollout_length, action_dim, device=device)
        self.log_probs = torch.zeros(rollout_length, device=device)
        self.rewards = torch.zeros(rollout_length, device=device)
        self.values = torch.zeros(rollout_length, device=device)
        self.dones = torch.zeros(rollout_length, device=device)
        self.ptr = 0

        #part 2 modif
        self.store_images = store_images    #added for part2 to be able to store images
        if self.store_images:
            self.images = torch.zeros((rollout_length, *image_shape), dtype=torch.uint8, device=device)
        else:
            self.images = None

    def add(self, obs, action, log_prob, reward, value, done, image = None):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        #part 2 update with images
        if self.store_images and image is not None:
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image.copy())
            self.images[self.ptr] = image.to(self.device, non_blocking=True).byte()

        self.ptr += 1

    def full(self):
        return self.ptr >= self.rollout_length

    def reset(self):
        self.ptr = 0

    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float, gae_lambda: float):
        """
        Compute discounted returns and GAE advantages.

        Args:
            last_value: bootstrap value V(s_T) of shape ()
            gamma: discount factor
            gae_lambda: GAE lambda
        Returns:
            returns: (rollout_length,) tensor
            advantages: (rollout_length,) tensor
        """
        # TODO: Compute GAE advantages and discounted returns.
        adv = 0.0
        advantages = torch.zeros_like(self.rewards)

        #backwards (bootstrap)
        for i in range(self.rollout_length -1, -1, -1):
            if i == self.rollout_length -1 :
                next_v = last_value            
            else:
                next_v = self.values[i+1]     
            
            not_done = 1.0 - self.dones[i]      #flags if at the episode ended at i, 1 if yes else 0
            delta = self.rewards[i] + (gamma * next_v * not_done) - self.values[i]      #TD error
            adv = delta + (gamma * gae_lambda * not_done * adv)
            advantages[i] = adv
        
        returns = advantages + self.values      # returns = advantage + baseline
        return returns, advantages, {
            "adv_mean_before_norm": advantages.mean().item(),
            "adv_std_before_norm": advantages.std().item(),
            "returns_mean": returns.mean().item(),
            "returns_std": returns.std().item(),
        }


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------
def ppo_update(policy, value_fn, v_optimizer, p_optimizer, buffer, returns, advantages, cfg, gpu_images=None):
    device = policy.device
    # 1. Pre-process non-image data 
    b_actions = buffer.actions.to(device)
    b_logprobs = buffer.log_probs.to(device)
    b_returns = returns.to(device)
    b_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    b_advantages = b_advantages.to(device)
    b_state_obs = buffer.obs.to(device) 

    # 2. Prepare inputs for the transformer
    if cfg.model.type == 'transformer':
        if gpu_images is not None:
            all_imgs = gpu_images           #already handled in the function
        else:
            imgs = buffer.images if isinstance(buffer.images, torch.Tensor) else torch.from_numpy(buffer.images)
            all_imgs = imgs.to(device).float() / 255.0
            if all_imgs.shape[-1] == 3:
                all_imgs = all_imgs.permute(0, 3, 1, 2)

        g_txt = policy.goal_text_ids 
        g_img = policy.goal_img

    # 3. Training Loop
    policy_l, entropy_l, value_l, total_l = [], [], [], []
    for j in range(cfg.training.ppo_epochs):
        idx = torch.randperm(buffer.rollout_length)
        
        for i in range(0, buffer.rollout_length, cfg.training.minibatch_size):
            mb_idx = idx[i:i + cfg.training.minibatch_size]
        
            if cfg.model.type == 'transformer':
                curr_imgs = all_imgs[mb_idx] 
                b_sz = curr_imgs.shape[0]
                g_txt_batch = g_txt.expand(b_sz, -1)                            # match batch size
                
                g_img_batch = g_img if g_img.ndim == 4 else g_img.unsqueeze(0)  # Arrange dim to get to [64, C, H, W]
                g_img_batch = g_img_batch.expand(b_sz, -1, -1, -1)

                a_mean, _ = policy.model(curr_imgs, g_txt_batch, g_img_batch)   # Forward policy pass
                dist = torch.distributions.Normal(a_mean, policy.log_std.exp())
                
                v = value_fn(b_state_obs[mb_idx])                               # Forward value pass
            
            else: 
                dist = policy.forward(b_state_obs[mb_idx])
                v = value_fn(b_state_obs[mb_idx])

            # 4. PPO loss
            new_log_prob = dist.log_prob(b_actions[mb_idx]).sum(-1)
            ratio = (new_log_prob - b_logprobs[mb_idx]).exp()
            
            surr1 = ratio * b_advantages[mb_idx]
            surr2 = torch.clamp(ratio, 1 - cfg.training.clip_eps, 1 + cfg.training.clip_eps) * b_advantages[mb_idx]
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(v, b_returns[mb_idx])
            entropy_loss = -dist.entropy().mean()
      
            if not cfg.model.policy.freeze:             # Update the Policy if not frozen
                p_optimizer.zero_grad()
                p_loss = policy_loss + (cfg.training.entropy_coeff * entropy_loss)
                p_loss.backward() 
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.training.max_grad_norm)
                p_optimizer.step()

            v_optimizer.zero_grad()
            v_loss = value_loss * cfg.training.value_coeff
            v_loss.backward()
            nn.utils.clip_grad_norm_(value_fn.parameters(), cfg.training.max_grad_norm)
            v_optimizer.step()

            policy_l.append(p_loss.item())
            value_l.append(v_loss.item())
            entropy_l.append(-entropy_loss.item())
            total_l.append((p_loss + v_loss).item())

    mean_losses = {
        "total_loss": np.mean(total_l),
        "policy_loss": np.mean(policy_l),
        "value_loss": np.mean(value_l),
        "entropy_loss": np.mean(entropy_l)
    }
    print(f"[PPO update epoch {j}] "
      f"policy={mean_losses['policy_loss']:.4f} "
      f"value={mean_losses['value_loss']:.4f} "
      f"entropy={mean_losses['entropy_loss']:.4f} "
      f"total={mean_losses['total_loss']:.4f}"
      )
    return mean_losses


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="dense_ppo", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.r_seed)
    np.random.seed(cfg.r_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Logging ---
    wandb.init(
        project=cfg.experiment.project,
        name=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # --- Environment ---
    task_id = int(cfg.sim.eval_tasks[0])
    env = FastLIBEROEnv(
        task_id=task_id,
        max_episode_steps=cfg.sim.episode_length,
        cfg=cfg,
    )

    obs_dim = cfg.policy.obs_dim
    action_dim = cfg.policy.action_dim

    # --- Models ---
    policy = DensePolicy(obs_dim, action_dim, cfg.policy.hidden_dim, cfg.policy.n_layers).to(device)
    value_fn = DenseValueFunction(obs_dim, cfg.policy.hidden_dim, cfg.policy.n_layers).to(device)

    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_fn.parameters()),
        lr=cfg.training.learning_rate,
    )

    buffer = RolloutBuffer(cfg.training.rollout_length, obs_dim, action_dim, device, store_images=False)

    # --- Rollout state ---
    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    episode_return = 0.0
    episode_steps = 0
    total_steps = 0
    episode_returns = []
    episode_successes = []
    episode_r_reach = 0.0
    episode_r_grasp = 0.0      
    episode_r_place = 0.0   

    # --- Main loop ---
    while total_steps < cfg.training.total_env_steps:
        buffer.reset()

        # Collect one rollout
        with torch.no_grad():
            for _ in range(cfg.training.rollout_length):
                action, log_prob, _ = policy.get_action(obs_tensor.unsqueeze(0))
                value = value_fn(obs_tensor.unsqueeze(0))
                action_np = action.squeeze(0).cpu().numpy()

                next_obs, reward, done, truncated, info = env.step(action_np)
                episode_return += reward
                episode_r_reach += info.get('r_reach', 0.0)
                episode_r_grasp += info.get('r_grasp', 0.0)
                episode_r_place += info.get('r_place', 0.0)
                episode_steps += 1
                total_steps += 1

                buffer.add(
                    obs_tensor,
                    action.squeeze(0),
                    log_prob.squeeze(0),
                    torch.tensor(reward, device=device),
                    value.squeeze(0),
                    torch.tensor(float(done or truncated), device=device),
                )

                if done or truncated:
                    episode_returns.append(episode_return)
                    episode_successes.append(float(info.get("success_placed", 0.0)))
                    wandb.log({
                        "total_reward": episode_return, 
                        "reward/r_reach":    episode_r_reach,
                        "reward/r_grasp":    episode_r_grasp,
                        "reward/r_place":    episode_r_place,
                        "reward/r_success":  info.get('r_success', 0.0),
                        "reward/is_grasping": info.get('is_grasping', 0.0),
                    }, step=total_steps)
                    # reset all episode trackers
                    episode_return = 0.0
                    episode_steps = 0
                    episode_r_reach = 0.0
                    episode_r_grasp = 0.0
                    episode_r_place = 0.0
                    obs, _ = env.reset()
                else:
                    obs = next_obs
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

                if buffer.full():
                    break

            # Bootstrap last value
            last_value = value_fn(obs_tensor.unsqueeze(0)).squeeze(0).detach()  #added detach to avoid gradient flow into gae computation

        returns, advantages, gae_stats = buffer.compute_returns_and_advantages(last_value, cfg.training.gamma, cfg.training.gae_lambda)
        
        update_info = ppo_update(policy, value_fn, optimizer, None , buffer, returns, advantages, cfg)  # PPO update
     
        with torch.no_grad():      # Log log_std diagnostics for debugging and optimization
            log_dict_std = {
                "train/log_std_mean": policy.log_std.mean().item(),
                "train/log_std_min": policy.log_std.min().item(),
                "train/log_std_max": policy.log_std.max().item(),
                "train/std_mean": policy.log_std.exp().mean().item(),
                "train/entropy_from_std": policy.log_std.sum().item() + 0.5 * policy.action_dim * (1 + torch.log(torch.tensor(2 * torch.pi))).item(),
            }
        wandb.log(log_dict_std, step=total_steps)

        if total_steps % cfg.log_interval < cfg.training.rollout_length:
            log_dict = {
                "train/total_steps": total_steps,
                **{f"train/{k}": v for k, v in update_info.items()},
                **{f"train/{k}": v for k, v in gae_stats.items()},    
                **log_dict_std, 
            }
            if episode_returns:
                log_dict["train/episode_return"] = np.mean(episode_returns[-10:])
                log_dict["train/success_rate"] = np.mean(episode_successes[-10:])
            wandb.log(log_dict, step=total_steps)

            print(f"[{total_steps}/{cfg.training.total_env_steps}] "
                  f"return={log_dict.get('train/episode_return', float('nan')):.3f} "
                  f"success={log_dict.get('train/success_rate', float('nan')):.3f} "
                  f"policy_loss={update_info['policy_loss']:.4f} "
                  f"value_loss={update_info['value_loss']:.4f} "
                  f"entropy={update_info['entropy_loss']:.4f} "
                  f"total={update_info['total_loss']:.4f}"
                  )

        # Checkpoint
        if total_steps % cfg.save_interval < cfg.training.rollout_length:
            ckpt = {
                "policy": policy.state_dict(),
                "value_fn": value_fn.state_dict(),
                "optimizer": optimizer.state_dict(),
                "total_steps": total_steps,
                "cfg": OmegaConf.to_container(cfg),
            }
            torch.save(ckpt, f"dense_ppo_{total_steps}.pth")

    # Final save
    torch.save({"policy": policy.state_dict(), "cfg": OmegaConf.to_container(cfg)},
               "dense_ppo_final_seed{cfg.r_seed}.pth")
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
