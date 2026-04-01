"""
HW3 Part 2: Fine-tune a transformer policy from HW1 with PPO or GRPO.

Usage (PPO):
    python hw3/train_transformer_rl.py \
        experiment.name=hw3_transformer_ppo_seed0 \
        r_seed=0 \
        init_checkpoint=/path/to/hw1/miniGRP.pth \
        rl.algorithm=ppo \
        sim.task_set=libero_spatial \
        sim.eval_tasks=[9]

Usage (GRPO with ground-truth resets):
    python hw3/train_transformer_rl.py \
        experiment.name=hw3_transformer_grpo_seed0 \
        r_seed=0 \
        init_checkpoint=/path/to/hw1/miniGRP.pth \
        rl.algorithm=grpo \
        sim.task_set=libero_spatial \
        sim.eval_tasks=[9]
"""

from math import log
import sys
import os
import h5py
from tensorflow.python.ops.gen_tpu_ops import is_tpu_embedding_initialized
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../mini-grp'))
from dreamerV3 import DreamerV3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import matplotlib
matplotlib.use('Agg') # needed for headless OSMesa environments, for the image plot
import matplotlib.pyplot as plt
from hw3.libero_env_fast import FastLIBEROEnv
from hw3.train_dense_rl import RolloutBuffer, ppo_update
os.environ['LIBERO_DATASET_PATH'] = '/teamspace/studios/this_studio/robot_learning_2026/LIBERO/libero/datasets' # downloaded datasets
sys.path.insert(0, '/teamspace/studios/this_studio/robot_learning_2026/LIBERO') #LIBERO code repo
import dill
# ---------------------------------------------------------------------------
# Separate value network (used with transformer policy)
# ---------------------------------------------------------------------------

class ValueFunction(nn.Module):
    """
    Separate MLP value network V(s).
    Keep this separate from the transformer policy as required by hw3.md.
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(obs)).squeeze(-1)


# ---------------------------------------------------------------------------
# Transformer policy wrapper
# ---------------------------------------------------------------------------

class TransformerPolicyWrapper(nn.Module):
    """
    Wraps the HW1 GRP transformer model to provide a gym-style action interface.

    The transformer policy expects a history of observations and actions;
    this wrapper maintains the required context window internally.
    """
    def __init__(self, checkpoint_path: str, device: torch.device, cfg: DictConfig):
        # TODO: Load the HW1 transformer checkpoint and reconstruct the model.
        super().__init__()

        self.device = device
        self.cfg = cfg
        model = torch.load(checkpoint_path, map_location=device, pickle_module=dill)    #whole model saved previously
        self.model = model.to(device)
        self.model.train() 
        self.log_std = nn.Parameter(torch.full((self.model._cfg.action_dim,), -0.5).to(device)) # Trainable log_std for exploration

    def reset_context(self):
        self._context = []

    def get_action(self, obs: np.ndarray, goal_txt_ids: torch.Tensor, goal_img: torch.Tensor, deterministic: bool = False):
        """
        Query the transformer for an action given the current observation.

        Args:
            obs: (obs_dim,) numpy array, (3, 64, 64) raw image from env.step
            deterministic: if True return mean, else sample
        Returns:
            action: (action_dim,) numpy array
            log_prob: scalar tensor
            entropy: scalar tensor
        """
        # TODO: Run a forward pass through the transformer and return (action, log_prob, entropy).
        # Observation preprocessing (Obs)
        if isinstance(obs, np.ndarray):
            obs_contiguous = obs.copy()
            img_in = torch.from_numpy(obs_contiguous).float().to(self.device)
        else:
            img_in = obs.float().to(self.device)
                                                
        if img_in.ndim == 3: img_in = img_in.unsqueeze(0)                           # Force shape to [1, 3, 64, 64], obs (image) has dim [64,64,3]
                   
        if img_in.shape[-1] == 3: img_in = img_in.permute(0, 3, 1, 2)               # Ensure obs is a torch tensor and move to device
            
        if img_in.max() > 1.0: img_in = img_in / 255.0                              # Scale to [0, 1] if needed                            
            
        # Goal image preprocessing
        if not torch.is_tensor(goal_img):
            goal_img = torch.from_numpy(goal_img).float().to(self.device)
        else:
            goal_img = goal_img.float().to(self.device)

        if goal_img.ndim == 3: goal_img = goal_img.unsqueeze(0)
        
        if goal_img.shape[-1] == 3: goal_img = goal_img.permute(0, 3, 1, 2)         # If goal is [1, 64, 64, 3], permute to [1, 3, 64, 64] to match Obs
            
        if goal_img.max() > 1.0: goal_img = goal_img / 255.0
        
        # Policy forward pass
        a_mean, _ = self.model(img_in, goal_txt_ids, goal_img)                     
        std = self.log_std.exp()
        dist = torch.distributions.Normal(a_mean, std)
        
        action = a_mean if deterministic else dist.rsample()                        # Sample action
 
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        # Convert from Transformer latent to 7D Sim Action
        decoded_action = self.model.decode_action(action)
        sim_action = decoded_action.detach().cpu().numpy().reshape(-1)
        if sim_action.shape[0] > 7: sim_action = sim_action[:7]            
            
        return sim_action, action.detach(), log_prob, entropy
    
    #commented bc was creating confusion for the training of self.log_std
    # def parameters(self):
    #     return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


# ---------------------------------------------------------------------------
# GRPO helpers
# ---------------------------------------------------------------------------

def collect_grpo_group(env: FastLIBEROEnv,
                       policy: TransformerPolicyWrapper,
                       init_state,
                       group_size: int,
                       max_steps: int,
                       device: torch.device):
    """
    Reset to the same initial state and collect `group_size` trajectories.

    Returns a list of trajectory dicts, each containing:
        obs, state_obs, actions, log_probs, rewards, dones, total_return
    """
    # TODO: Collect group_size trajectories all starting from init_state.
    trajectories = []
    for i in range(group_size):
        #reset to same initial state for all trajectories
        obs, info = env.reset(options={"init_state": init_state})
        obs_buf, state_buf, act_buf, logp_buf, rew_buf, done_buf  = [], [], [], [], [], []
        total_return = 0.0
        r_reach_buf, r_grasp_buf, r_place_buf = 0.0, 0.0, 0.0

        for t in range(max_steps):            
            with torch.no_grad():                       #Get samples with policy, 
                action_np, raw_action, logp, _ = policy.get_action(obs, policy.goal_text_ids, policy.goal_img)

            next_obs, r, d, truncated, info = env.step(action_np) 
            obs_buf.append(np.ascontiguousarray(obs))
            state_buf.append(info['state_obs'].copy()) 
            act_buf.append(raw_action.squeeze(0).cpu().numpy())
            logp_buf.append(logp.squeeze(0).cpu().numpy())
            rew_buf.append(r)
            done_buf.append(d)   
            total_return += r
            r_reach_buf += info.get('r_reach', 0.0)
            r_grasp_buf += info.get('r_grasp', 0.0)
            r_place_buf += info.get('r_place', 0.0)
            
            obs = next_obs
        
            if d or truncated:      #episode done
                break
        
        traj_dict = {
            "obs": np.array(obs_buf),
            "state_obs": np.array(state_buf),
            "actions": np.array(act_buf),
            "log_probs": np.array(logp_buf),
            "rewards": np.array(rew_buf),
            "dones": np.array(done_buf),
            "total_return": float(total_return),
            "r_reach": r_reach_buf,   
            "r_grasp": r_grasp_buf,
            "r_place": r_place_buf,
            "success": float(info.get("success_placed", 0.0))
        }
        trajectories.append(traj_dict)

    return trajectories


def grpo_update(policy: TransformerPolicyWrapper,
                value_fn: ValueFunction,
                policy_optimizer: torch.optim.Optimizer,
                trajectories_per_group: list,
                cfg: DictConfig,
                device: torch.device):
    """
    GRPO update: compute group-relative advantages and update policy.

    Args:
        trajectories_per_group: list of lists; each inner list is a group of
            trajectory dicts collected from the same initial state.
    Returns:
        dict with "policy_loss", "mean_return"
    """
    # TODO: Compute group-relative advantages and apply a clipped surrogate loss.
    all_o = []
    all_img = []
    all_a = []
    all_logp = []
    all_ad = []
    all_ret = []

    for g in trajectories_per_group:
        ret = np.array([traj["total_return"] for traj in g])
        all_ret.extend(ret.tolist())

        #Normalize like with PPO
        g_mean = ret.mean()
        g_std = ret.std() +1e-8
        adv_per_traj = (ret-g_mean)/g_std

        for traj, adv in zip(g, adv_per_traj):  #eval each traj advantage
            T = len(traj["rewards"])
            all_o.append(traj["obs"])
            all_img.append(traj["obs"])
            all_a.append(traj["actions"])
            all_logp.append(traj["log_probs"])
            all_ad.append(np.full(T, adv))  #broadcast advantage to all timesteps

    #turn the np arraysback to tensor
    b_img = torch.tensor(np.concatenate(all_img, axis=0), dtype=torch.float32, device=device)  # (N, obs_dim)
    b_imgs = b_img / 255.0 
    if b_imgs.shape[-1] == 3: b_imgs = b_imgs.permute(0, 3, 1, 2)

    b_actions  = torch.tensor(np.concatenate(all_a, axis=0), dtype=torch.float32, device=device)  # (N, act_dim)
    b_logp_old = torch.tensor(np.concatenate(all_logp, axis=0), dtype=torch.float32, device=device)  # (N,)
    b_advantages = torch.tensor(np.concatenate(all_ad, axis=0), dtype=torch.float32, device=device) 

    N = b_imgs.shape[0] 
    policy_losses, entropy_losses,  kl_losses = [],[],[]

    for epoch in range(cfg.training.grpo_epochs):
        index = torch.randperm(N, device=device)

        for i in range(0, N, cfg.training.minibatch_size):
            mb_idx =index[i: i + cfg.training.minibatch_size]
            if cfg.model.type == 'transformer':
                curr_imgs = b_imgs[mb_idx]
                b_size = curr_imgs.shape[0]
                g_txt_batch = policy.goal_text_ids.expand(b_size, -1)
                g_img_batch = policy.goal_img

                if g_img_batch.ndim == 3:
                    g_img_batch = g_img_batch.unsqueeze(0)
                g_img_batch = g_img_batch.expand(b_size, -1, -1, -1)
 
                a_mean, _ = policy.model(curr_imgs, g_txt_batch, g_img_batch)
                dist = torch.distributions.Normal(a_mean, policy.log_std.exp())
            
            else:
                # Dense policy (unchanged)
                b_obs = torch.tensor(np.concatenate(all_o, axis=0), dtype=torch.float32, device=device)
                dist = policy.forward(b_obs[mb_idx])
            
            new_log_prob = dist.log_prob(b_actions[mb_idx]).sum(-1) #actions dim treated as independednt, prob of action = sum of dim in action

            #all same as ppo for the clip
            ratio = (new_log_prob - b_logp_old[mb_idx]).exp() 
            surr1 = ratio * b_advantages[mb_idx]
            surr2 = torch.clamp(ratio, 1 - cfg.training.clip_eps, 1 + cfg.training.clip_eps) * b_advantages[mb_idx]

            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -dist.entropy().mean()
            kl = (b_logp_old[mb_idx] - new_log_prob).mean()
            loss = policy_loss + cfg.training.entropy_coeff * entropy_loss + cfg.training.kl_coeff * kl

            if not cfg.model.policy.freeze:
                policy_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.training.max_grad_norm)
                policy_optimizer.step()
 
            policy_losses.append(policy_loss.item())
            entropy_losses.append(entropy_loss.item())
            kl_losses.append(kl.item())

    print(
        f"[GRPO update] "
        f"policy={np.mean(policy_losses):.4f}"
        f"mean_return={np.mean(all_ret):.4f}",
        f"entropy_loss={np.mean(entropy_losses):.4f}"
        f"kl={np.mean(kl_losses):.4f}"
        )
 
    return {
        "policy_loss": float(np.mean(policy_losses)),
        "mean_return": float(np.mean(all_ret)),
        "entropy_loss": float(np.mean(entropy_losses)),
        "kl":           float(np.mean(kl_losses)),
    }


# ---------------------------------------------------------------------------
# GRPO with world model (Part 2d)
# ---------------------------------------------------------------------------

def grpo_worldmodel_update(policy: TransformerPolicyWrapper,
                            world_model,
                            current_obs: np.ndarray,
                            group_size: int,
                            horizon: int,
                            policy_optimizer: torch.optim.Optimizer,
                            cfg: DictConfig,
                            device: torch.device):
    """
    GRPO using the HW2 world model to generate imagined trajectories.

    Args:
        world_model: trained HW2 world model (SimpleWorldModel or DreamerV3)
        current_obs: (obs_dim,) current real observation used as rollout start
        group_size: number of imagined trajectories per state
        horizon: number of imagination steps
    Returns:
        dict with "policy_loss", "mean_imagined_return"
    """
    # TODO: Roll out imagined trajectories using the world model and apply GRPO.
    # get first observation, source of imagination
    if isinstance(current_obs, np.ndarray):
        obs_np = np.ascontiguousarray(current_obs)
    else:
        obs_np = current_obs.cpu().numpy()
    
     # preprocess images for dreamer
    obs_processed = world_model.preprocess_state(obs_np)     
    obs_tensor = torch.tensor(obs_processed, dtype=torch.float32, device=device)
    obs_seq = obs_tensor.unsqueeze(0).unsqueeze(0)  

     # encode real obs into initial latent state 
    with torch.no_grad():
        init_statee = world_model.encode_sequence(obs_seq,prev_actions=torch.zeros(1, 1, world_model.action_dim, device=device))
    init_state = {k: v.squeeze(1) if v.dim() == 3 else v for k, v in init_statee.items()}

    # get goal image, process the format, want (1, C, H, W)
    goal_img = policy.goal_img  
    if not torch.is_tensor(goal_img): goal_img = torch.tensor(goal_img, dtype=torch.float32, device=device)
    goal_img = goal_img.float().to(device)

    if goal_img.ndim == 3: goal_img = goal_img.unsqueeze(0)

    if goal_img.shape[-1] == 3: goal_img = goal_img.permute(0, 3, 1, 2)

    if goal_img.max() > 1.0: goal_img = goal_img / 255.0        
    
    imagined_traj = []
    for _ in range(group_size):
        state = {k: v.clone() for k, v in init_state.items()}
        o_b, a_b, log_b, r_b = [],[],[],[]
        total_ret = 0.0

        # Decode initial latent to image 
        with torch.no_grad():
            feat = torch.cat([state['h'], state['z']], dim=-1)      # (1, latent_dim)
            dec_img = world_model.deco_proj(feat).view(1, world_model.deco_seed_chan, 4, 4)
            for blk in world_model.deco_blocks:
                dec_img = blk(dec_img)
            current_img = dec_img 

        for t in range(horizon):                                    #sample horizon actions from the policy, collection no training
            policy_img = (current_img + 1.0) / 2.0      
 
            with torch.no_grad():
                g_txt = policy.goal_text_ids.expand(1, -1)
                g_img = goal_img.expand(1, -1, -1, -1)
                a_mean, _ = policy.model(policy_img, g_txt, g_img)
                dist = torch.distributions.Normal(a_mean, policy.log_std.exp())
                action = dist.sample()                              # (1, act_dim)
                logp = dist.log_prob(action).sum(-1)                # (1,)

            action_decoded = world_model.decode_action(action)      # (1, act_dim)#decode action

            #  RSSM prior step (imagination)
            with torch.no_grad():
                next_state, prior_logits, _ = world_model.rssm_step(state, action_decoded, embed=None)
                feat_next = torch.cat([next_state['h'], next_state['z']], dim=-1)
                reward_pred = world_model.r_head(feat_next).squeeze(-1)  # (1,)
                r = float(reward_pred.item())
          
                dec_img = world_model.deco_proj(feat_next).view(1, world_model.deco_seed_chan, 4, 4)
                for blk in world_model.deco_blocks:
                    dec_img = blk(dec_img)
             
                H_pol = cfg.model.image_shape[0]
                W_pol = cfg.model.image_shape[1]
                if dec_img.shape[-2:] != (H_pol, W_pol):            # Resize if needed to match policy input (64×64)
                    dec_img = torch.nn.functional.interpolate(dec_img, size=(H_pol, W_pol), mode='bilinear', align_corners=False)
                current_img = dec_img  
        
            # Store the decoded image rescaled to [0,255] uint8
            img_np = ((current_img.squeeze(0).cpu().numpy() + 1.0) / 2.0 * 255.0).astype(np.uint8)
            img_np = img_np.transpose(1, 2, 0)            
            o_b.append(img_np)
            a_b.append(action.squeeze(0).detach().cpu().numpy())
            log_b.append(logp.squeeze(0).detach().cpu().numpy())
            r_b.append(r)
            total_ret += r
            state = next_state

        imagined_traj.append({
            "obs": np.array(o_b, dtype=np.uint8),
            "actions": np.array(a_b, dtype=np.float32),
            "log_probs": np.array(log_b, dtype=np.float32),
            "rewards": np.array(r_b, dtype=np.float32),
            "total_return": float(total_ret),
        })
    # group-relative advantages 
    returns = np.array([t["total_return"] for t in imagined_traj], dtype=np.float32)
    g_mean = returns.mean()
    g_std = returns.std() + 1e-8        
    adv_per_traj = (returns - g_mean) / g_std
    all_img, all_ad, all_a, all_logp   = [], [], [], []

    for traj, adv in zip(imagined_traj, adv_per_traj):
        T = len(traj["rewards"])
        all_img.append(traj["obs"])
        all_a.append(traj["actions"])
        all_logp.append(traj["log_probs"])
        all_ad.append(np.full(T, adv, dtype=np.float32))
 
    b_imgs = torch.tensor(np.concatenate(all_img, axis=0), dtype=torch.float32, device=device)
    b_imgs = b_imgs / 255.0
    if b_imgs.shape[-1] == 3: b_imgs = b_imgs.permute(0, 3, 1, 2)                     # (N, 3, H, W)
 
    b_actions = torch.tensor(np.concatenate(all_a, axis=0), dtype=torch.float32, device=device)
    b_logp_old = torch.tensor(np.concatenate(all_logp, axis=0), dtype=torch.float32, device=device)
    b_advantages = torch.tensor(np.concatenate(all_ad, axis=0), dtype=torch.float32, device=device)
 
    N = b_imgs.shape[0]
    policy_losses,entropy_losses,kl_losses  = [],[],[]

    # minibatch update
    for epoch in range(cfg.training.grpo_epochs):
        index = torch.randperm(N, device=device)
 
        for i in range(0, N, cfg.training.minibatch_size):
            mb_idx = index[i : i + cfg.training.minibatch_size]
 
            curr_imgs = b_imgs[mb_idx]
            b_sz = curr_imgs.shape[0]
            g_txt_batch = policy.goal_text_ids.expand(b_sz, -1)
            g_img_batch = goal_img.expand(b_sz, -1, -1, -1)
 
            a_mean, _ = policy.model(curr_imgs, g_txt_batch, g_img_batch)
            dist = torch.distributions.Normal(a_mean, policy.log_std.exp())
            new_log_prob = dist.log_prob(b_actions[mb_idx]).sum(-1)
 
            ratio = (new_log_prob - b_logp_old[mb_idx]).exp()
            surr1 = ratio * b_advantages[mb_idx]
            surr2 = torch.clamp( ratio, 1 - cfg.training.clip_eps, 1 + cfg.training.clip_eps) * b_advantages[mb_idx]
 
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -dist.entropy().mean()
            kl = (b_logp_old[mb_idx] - new_log_prob).mean()
            loss = ( policy_loss + cfg.training.entropy_coeff * entropy_loss+ cfg.training.kl_coeff * kl )
 
            if not cfg.model.policy.freeze:
                policy_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.training.max_grad_norm)
                policy_optimizer.step()
 
            policy_losses.append(policy_loss.item())
            entropy_losses.append(entropy_loss.item())
            kl_losses.append(kl.item())
 
    print(
        f"[GRPO-WM] "
        f"policy={np.mean(policy_losses):.4f}  "
        f"imagined_return={returns.mean():.4f}  "
        f"return_std={returns.std():.4f}  "
        f"entropy={np.mean(entropy_losses):.4f}  "
        f"kl={np.mean(kl_losses):.4f}"
    )
 
    return {
        "policy_loss":          float(np.mean(policy_losses)),
        "mean_imagined_return": float(returns.mean()),
        "imagined_return_std":  float(returns.std()),
        "entropy_loss":         float(np.mean(entropy_losses)),
        "kl":                   float(np.mean(kl_losses)),
    }


def create_eval_alignment_plot(images, rewards, values, title="Evaluation: Episode Reward vs. Value Prediction"):
    """
    Helper function to plot the allignment plot required in part 2.
    """
    fig, (ax_img, ax_graph) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [1, 3]})
    indices = np.linspace(0, len(images) - 1, 10, dtype=int)
    sampled_images = [images[i] for i in indices]
    ax_img.imshow(np.concatenate(sampled_images, axis=1))
    ax_img.axis('off')
    
    # Reward vs Value (Bottom) (value is 0 is no value func (grpo))
    ax_graph.plot(rewards, color='blue', alpha=0.3, label='Step Reward (Immediate)')
    ax_graph.plot(values, color='red', linewidth=2, label='Critic Value Prediction ($V_s$)')
    ax_graph.set_title(title)
    ax_graph.set_xlabel("Timesteps")
    ax_graph.set_ylabel("Value / Reward")
    ax_graph.grid(True, linestyle='--', alpha=0.6)
    ax_graph.legend()
    
    plt.tight_layout()
    return fig

def prepare_gpu_images(buffer, device):
    """
    Moves images to GPU and prepares them for the Transformer.
    Goal is to increase speed.
    """
    imgs = buffer.images
    if not isinstance(imgs, torch.Tensor): imgs = torch.from_numpy(imgs)
    # 2. Move to GPU, cast to float, and Normalize
    images = imgs.to(device).float() / 255.0
    # 3. Shape check: (B, H, W, C) -> (B, C, H, W)
    if images.shape[-1] == 3: images = images.permute(0, 3, 1, 2)
    return images

def load_manual_goal(path):
    with h5py.File(path, "r") as f:
        # Pull the very last frame of the first demo
        return f["data/demo_0/obs/agentview_rgb"][-1]
# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
#manual path to task 0 or 9
#manual_path = "/teamspace/studios/this_studio/robot_learning_2026/LIBERO/libero/datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
manual_path = "/teamspace/studios/this_studio/robot_learning_2026/LIBERO/libero/datasets/libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5"
@hydra.main(config_path="conf", config_name="transformer_rl", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.r_seed)
    np.random.seed(cfg.r_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project=cfg.experiment.project, name=cfg.experiment.name, config=OmegaConf.to_container(cfg, resolve=True))
    task_id = cfg.sim.eval_tasks[0]
    #set ot returnn image obs for the hw1 transformer
    env = FastLIBEROEnv(task_id=task_id, max_episode_steps=cfg.sim.episode_length, cfg=cfg, output_image_obs=True)

    obs_dim = env.obs_dim
    action_dim = env._action_dim

    # Load transformer policy from HW1 checkpoint
    policy = TransformerPolicyWrapper(cfg.init_checkpoint, device, cfg)
    # Separate value function (required by hw3.md)
    value_fn = ValueFunction(obs_dim, cfg.value.hidden_dim, cfg.value.n_layers).to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.training.learning_rate)
    value_optimizer = torch.optim.Adam(value_fn.parameters(), lr=cfg.value.learning_rate)
    
    algorithm = cfg.rl.algorithm.lower()
    if algorithm == "ppo":
        # ------------------------------------------------------------------
        # PPO loop (reuses the buffer + update from Part 1)
        # ------------------------------------------------------------------
        buffer = RolloutBuffer(cfg.training.rollout_length, obs_dim, action_dim, device, store_images = cfg.output_image_obs, image_shape = cfg.model.image_shape)
        
        if not cfg.model.policy.freeze:
            policy.model.train()
        else:                                       # eval mode for frozen
            policy.model.eval()
            for param in policy.model.parameters():
                param.requires_grad = False
            policy.log_std.requires_grad = False

        obs, info = env.reset()
        state_vector = info['state_obs']
        obs_t = torch.tensor(state_vector, dtype=torch.float32, device=device)

        policy.reset_context()
        #hardcoded if buggy
        #goal_text_ids = policy.model.encode_text_goal("pick up the black bowl between the plate and the ramekin and place it on the plate")
        #goal_text_ids = policy.model.encode_text_goal("pick up the black bowl on the wooden cabinet and place it on the plate")
        total_steps = 0
        episode_returns, episode_successes = [], []
        ep_ret = 0.0
        #preprocess goals for transformer, one task same
        goal_text_ids = policy.model.encode_text_goal(env.goal_description)
        print(f"DEBUG: Goal Text: {env.goal_description}")
       
        env_goal_image = load_manual_goal(manual_path)
        goal_img = policy.model.preprocess_goal_image(env_goal_image)
        policy.goal_text_ids = goal_text_ids
        policy.goal_img = goal_img

        episode_r_reach, episode_r_grasp, episode_r_place = 0.0, 0.0, 0.0
        total_steps = 0
        current_ep_images, current_ep_rewards, current_ep_values = [], [], []
       
        #rollout
        while total_steps < cfg.training.total_env_steps:
            buffer.reset()
            policy.eval()   #dont train during rollout
            with torch.no_grad():
                for _ in range(cfg.training.rollout_length):
                    action_np, raw_action, log_prob, entropy_ = policy.get_action(obs,goal_text_ids, goal_img)
                    value = value_fn(obs_t.unsqueeze(0))
                    next_obs, reward, done, truncated, info = env.step(action_np)
                    ep_ret += reward
                    episode_r_reach += info.get('r_reach', 0.0)
                    episode_r_grasp += info.get('r_grasp', 0.0)
                    episode_r_place += info.get('r_place', 0.0)
                    total_steps += 1

                    if total_steps % 50 == 0:
                        print(f"Obs Range: [{obs.min():.3f}, {obs.max():.3f}]")
                    buffer.add(
                        obs = obs_t, # Pose for the critic
                        action = raw_action,
                        log_prob = log_prob,
                        reward = torch.tensor(reward, device=device),
                        value = value.squeeze(0),
                        done = torch.tensor(float(done or truncated), device=device),
                        image = obs #raw obs
                    )
                    current_ep_images.append(obs)
                    current_ep_rewards.append(reward)
                    current_ep_values.append(value.item())

                    if done or truncated:
                        episode_returns.append(ep_ret)
                        episode_successes.append(float(info.get("success_placed", 0.0)))
                        wandb.log({
                            "reward/total": ep_ret,
                            "reward/r_reach": episode_r_reach,
                            "reward/r_grasp": episode_r_grasp,
                            "reward/r_place": episode_r_place,
                            "reward/success": info.get("success_placed", 0.0),
                        }, step=total_steps)
                        if total_steps % cfg.log_interval < cfg.training.rollout_length:
                            fig = create_eval_alignment_plot(current_ep_images, current_ep_rewards, current_ep_values)
                            wandb.log({"analysis/value_alignment_episode": wandb.Image(fig)}, step=total_steps)
                            plt.close(fig)
                        
                        # clear lists for next episode
                        current_ep_images, current_ep_rewards, current_ep_values = [], [], []
                        ep_ret, episode_r_reach, episode_r_grasp, episode_r_place = 0.0, 0.0, 0.0, 0.0
                        obs, info = env.reset()
                        state_vector = info['state_obs']
                        policy.reset_context()
                    else:
                        obs = next_obs
                        state_vector = info['state_obs']
                    obs_t = torch.tensor(state_vector, dtype=torch.float32, device=device)
                        
                    if buffer.full():
                        break

                last_value = value_fn(obs_t.unsqueeze(0)).squeeze(0).detach()
            returns, advantages, gae_stats = buffer.compute_returns_and_advantages(last_value, cfg.training.gamma, cfg.training.gae_lambda)
            gpu_images = prepare_gpu_images(buffer, device)

            if not cfg.model.policy.freeze:
                policy.train()  #back to training

            update_info = ppo_update(policy, value_fn, value_optimizer, policy_optimizer, buffer, returns, advantages, cfg, gpu_images=gpu_images)
            if total_steps % cfg.log_interval < cfg.training.rollout_length:
                with torch.no_grad():
                    std_vals = policy.log_std.exp()
                    act_dim = policy.log_std.shape[0]
                    entropy_approx = policy.log_std.sum().item() + 0.5 * act_dim * (1 + np.log(2 * np.pi))
                log_dict = {
                    "train/total_steps": total_steps, 
                    "train/std_mean": std_vals.mean().item(),
                    "train/entropy_from_std": entropy_approx,
                    **{f"train/{k}": v for k, v in update_info.items()},
                    **{f"train/{k}": v for k, v in gae_stats.items()}
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
                    "p_optimizer": policy_optimizer.state_dict(), 
                    "v_optimizer": value_optimizer.state_dict(), 
                    "total_steps": total_steps,
                    "cfg": OmegaConf.to_container(cfg),
                }
                torch.save(ckpt, f"transformer_ppo_seed1500{total_steps}.pth")

    elif algorithm == "grpo":
        # ------------------------------------------------------------------
        # GRPO loop with ground-truth resets (Part 2c)
        # ------------------------------------------------------------------
        total_steps, update_count = 0, 0
        all_returns, episode_successes = [], []
        ep_ret, episode_r_reach, episode_r_grasp, episode_r_place = 0.0, 0.0, 0.0, 0.0
        current_ep_images, current_ep_rewards, current_ep_values = [], [], []
        max_step = cfg.sim.episode_length

        goal_text_ids = policy.model.encode_text_goal(env.goal_description)
        env_goal_image = load_manual_goal(manual_path)
        goal_img = policy.model.preprocess_goal_image(env_goal_image)
        policy.goal_img = goal_img
        policy.goal_text_ids = goal_text_ids
        policy.model.goal_img = policy.goal_img
        
        if cfg.grpo.wm:     #world model loop first
            from sim_eval import eval_libero_fast
            wm_checkpoint = cfg.grpo.wm_checkpoint
            dreamer_cfg = OmegaConf.load("/teamspace/studios/this_studio/robot_learning_2026/hw3/conf/dreamerconfig.yaml")
            world_model = DreamerV3(
                obs_shape=(3, 64, 64),
                action_dim=7,
                stoch_dim=32,
                discrete_dim=32,
                deter_dim=512,
                hidden_dim=512,
                cfg= dreamer_cfg
            ).to(device)

            state_dict = torch.load(wm_checkpoint, map_location=device)
            world_model.load_state_dict(state_dict)
            world_model.eval()
            for param in world_model.parameters():
                param.requires_grad = False
            print(f"[WM] DreamerV3 loaded from {wm_checkpoint}")
            horizon = cfg.grpo.horizon
            #main loop
            while total_steps < cfg.training.total_env_steps:
                # One real env step to get current observation as imagination seed
                obs, info = env.reset()
                policy.eval()
                if not cfg.model.policy.freeze: policy.train()
                update_info = grpo_worldmodel_update(policy, world_model, obs, cfg.grpo.group_size, horizon, policy_optimizer, cfg, device)
                update_count += 1

                total_steps  += cfg.grpo.group_size * horizon
                all_returns.append(update_info["mean_imagined_return"])
                #sim eval loop to compare real performance vs imagination
                if update_count % 40 == 0:
                    print(f"--- Running Real Sim Eval at Iteration {update_count} ---")
                    policy.eval()
                    
                    with torch.no_grad():
                        eval_results = eval_libero_fast(
                            model = policy.model,
                            device=device,
                            cfg=cfg,
                            iter_= update_count,
                            wandb=wandb,
                            render=True
                        )
                    policy.train()

                with torch.no_grad():
                    std_vals = policy.log_std.exp()
                    act_dim = policy.log_std.shape[0]
                    entropy_approx = (policy.log_std.sum().item()+ 0.5 * act_dim * (1 + np.log(2 * np.pi)))
    
                log_dict = {
                    "train/total_steps":         total_steps,
                    "train/update":              update_count,
                    "train/std_mean":            std_vals.mean().item(),
                    "train/entropy_from_std":    entropy_approx,
                    "train/wm_horizon":          horizon,
                    **{f"train/{k}": v for k, v in update_info.items()},
                    "train/episode_return":      np.mean(all_returns[-50:]) if all_returns else 0.0,
                }
                # Check if 'eval_results' was created in this loop iteration
                if 'eval_results' in locals():
                    log_dict["eval/real_success_rate"] = eval_results.get("success_rate", 0.0)
                    del eval_results    # Clean up 
                    
                wandb.log(log_dict, step=total_steps)
                print(
                    f"[GRPO-WM {total_steps}/{cfg.training.total_env_steps}] "
                    f"imagined_return={update_info['mean_imagined_return']:.3f}  "
                    f"return_std={update_info['imagined_return_std']:.4f}  "
                    f"policy_loss={update_info['policy_loss']:.4f}  "
                    f"entropy={update_info['entropy_loss']:.4f}  "
                    f"kl={update_info['kl']:.4f}"
                )
                # checkpoints
                if total_steps % cfg.save_interval < cfg.grpo.group_size * horizon:
                    torch.save({
                        "policy":      policy.state_dict(),
                        "p_optimizer": policy_optimizer.state_dict(),
                        "total_steps": total_steps,
                        "cfg":         OmegaConf.to_container(cfg),
                    }, f"transformer_grpo_wm_{total_steps}.pth")

        else:
            while total_steps < cfg.training.total_env_steps:
                # Collect groups: reset to different initial states
                # TODO: Reset env, capture init_state, call collect_grpo_group() num_groups times.
                #Collect groups, no policy training
                trajectories_per_group = []
                policy.eval()

                for g in range(cfg.grpo.num_groups):
                    #1. reset env to get valid sim init state
                    env.reset()
                    #2. capture init_state
                    init_state = env.env.sim.get_state().flatten()
                    #3. collect groups 
                    trajectories = collect_grpo_group(env, policy, init_state, cfg.grpo.group_size, max_step, device=device)
                    trajectories_per_group.append(trajectories)
                
                for group in trajectories_per_group:
                    for traj in group:
                        T = len(traj["rewards"])
                        total_steps += T
                        
                        for r in traj["rewards"]:
                            ep_ret += r
                            current_ep_rewards.append(float(r))
                        # Store images for alignment plot
                        for img in traj["obs"]:
                            current_ep_images.append(img)
                        current_ep_values.extend([0.0] * T) #No critic but pass in 0 so that the plot function dont crash
                        last_done = bool(traj["dones"][-1]) if len(traj["dones"]) > 0 else False
                        episode_ended = last_done or (T == max_step)

                        if episode_ended:
                            all_returns.append(traj["total_return"])
                            episode_successes.append(float(traj.get("success",0.0)))
                            wandb.log({
                                    "reward/total":   traj["total_return"],
                                    "reward/r_reach": traj.get("r_reach", 0.0),
                                    "reward/r_grasp": traj.get("r_grasp", 0.0),
                                    "reward/r_place": traj.get("r_place", 0.0),
                                    "reward/success": traj.get("success",  0.0),
                                }, step=total_steps)
                            
                            if total_steps % cfg.log_interval < cfg.grpo.group_size * max_step:
                                fig = create_eval_alignment_plot(current_ep_images, current_ep_rewards, current_ep_values)
                                wandb.log(
                                    {"analysis/value_alignment_episode": wandb.Image(fig)},
                                    step=total_steps,
                                )
                                plt.close(fig)
                                
                            ep_ret, episode_r_reach, episode_r_grasp, episode_r_place = 0.0, 0.0, 0.0, 0.0
                            current_ep_images, current_ep_rewards, current_ep_values = [], [], []
                #policy update section
                if not cfg.model.policy.freeze: policy.train()
                update_info = grpo_update(policy, value_fn, policy_optimizer, trajectories_per_group, cfg, device)
                update_count += 1
                policy.eval()

                if total_steps % cfg.log_interval < cfg.grpo.num_groups * cfg.grpo.group_size * max_step:
                    with torch.no_grad():
                        std_vals = policy.log_std.exp()
                        act_dim = policy.log_std.shape[0]
                        entropy_approx = (policy.log_std.sum().item()+ 0.5 * act_dim * (1 + np.log(2 * np.pi)))

                    log = {
                        "train/total_steps": total_steps,
                        "train/update": update_count,
                        "train/std_mean": std_vals.mean().item(),
                        "train/entropy_from_std": entropy_approx,
                        **{f"train/{k}": v for k, v in update_info.items()},
                        "train/episode_return": np.mean(all_returns[-50:]) if all_returns else 0.0,
                        "train/success_rate": np.mean(episode_successes[-50:]) if episode_successes else 0.0,
                        "train/grpo_return_std": float(np.std([t["total_return"] for g in trajectories_per_group for t in g]))
                        }

                    wandb.log(log, step=total_steps)
                    print(
                            f"[GRPO {total_steps}/{cfg.training.total_env_steps}] "
                            f"return={log['train/episode_return']:.3f}  "
                            f"success={log['train/success_rate']:.3f}  "
                            f"policy_loss={update_info['policy_loss']:.4f}  "
                            f"entropy={update_info['entropy_loss']:.4f}  "
                            f"kl={update_info['kl']:.4f}  "
                            f"return_std={log['train/grpo_return_std']:.4f}"
                        )
                if total_steps % cfg.save_interval < cfg.grpo.num_groups * cfg.grpo.group_size * max_step:
                        ckpt = {
                            "policy": policy.state_dict(),
                            "value_fn": value_fn.state_dict(),   # kept for compat
                            "p_optimizer": policy_optimizer.state_dict(),
                            "total_steps": total_steps,
                            "cfg": OmegaConf.to_container(cfg),
                        }
                        torch.save(ckpt, f"transformer_grpo_{total_steps}.pth")

    else:
        raise ValueError(f"Unknown rl.algorithm: {algorithm}. Choose 'ppo' or 'grpo'.")
    # Save final checkpoint
    torch.save({
        "policy": {k: v for k, v in policy.model.state_dict().items()},
        "value_fn": value_fn.state_dict(),
        "cfg": OmegaConf.to_container(cfg),
    }, "transformer_rl_grpo_wm_task9_seed0.pth")

    env.close()
    wandb.finish()

if __name__ == "__main__":
    main()