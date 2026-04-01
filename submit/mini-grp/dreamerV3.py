from pandas.core.internals import blocks
from pandas.errors import PossibleDataLossError
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Independent
import numpy as np
from torch.onnx.symbolic_opset17 import quantized_layer_norm

def symlog(x):
    """
    Symmetric log transformation.
    Squashes large values while preserving sign and small values.
    y = sign(x) * ln(|x| + 1)
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


class GRPBase(nn.Module):
    """Base class for GRP models"""
    def __init__(self, cfg):
        super(GRPBase, self).__init__()
        self._cfg = cfg

    def encode_text_goal(self, goal, tokenizer=None, text_model=None):
        import numpy as _np
        import torch as _torch
        if self._cfg.dataset.encode_with_t5:
            if tokenizer is None or text_model is None:
                raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
            # TODO:    
            #get tokens from goal
            tokens = tokenizer(goal, return_tensors="pt", padding="max_length", truncation=True,max_length=self._cfg.max_block_size)
            #move to right device
            m_device = next(text_model.parameters()).device
            tokens = {key: value.to(m_device) for key, value in tokens.items()}
            #encode with t5
            with _torch.no_grad():
                encoder_out = text_model.encoder(**tokens)
            
            hidden = encoder_out.last_hidden_state
            embedding = hidden.mean(dim=1)

            return embedding.to(self._cfg.device)
        else:
            pad = " " * self._cfg.max_block_size
            goal_ = goal[:self._cfg.max_block_size] + pad[len(goal):self._cfg.max_block_size]
            try:
                stoi = {c: i for i, c in enumerate(self._cfg.dataset.chars_list)}
                ids = [stoi.get(c, 0) for c in goal_]
            except Exception:
                ids = [0] * self._cfg.max_block_size
            return _torch.tensor(_np.expand_dims(_np.array(ids, dtype=_np.int64), axis=0), dtype=_torch.long, device=self._cfg.device)

    def process_text_embedding_for_buffer(self, goal, tokenizer=None, text_model=None):
        """
        Process text goal embedding for storing in the circular buffer.
        Returns a numpy array of shape (max_block_size, n_embd) without batch dimension.
        """
        import numpy as _np
        if tokenizer is None or text_model is None:
            raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
        
        goal_ = _np.zeros((self._cfg.max_block_size, self._cfg.n_embd), dtype=_np.float32)
        input_ids = tokenizer(goal, return_tensors="pt").input_ids
        goal_t = text_model.encoder(input_ids).last_hidden_state.detach().cpu().numpy()
        goal_[:len(goal_t[0]), :] = goal_t[0][:self._cfg.max_block_size]
        return goal_


    def resize_image(self, image):
        """Resize image to match model input size"""
        import cv2
        import numpy as _np
        img = _np.array(image, dtype=_np.float32)
        img = cv2.resize(img, (self._cfg.image_shape[0], self._cfg.image_shape[1]))
        return img

    def normalize_state(self, image):
        """Normalize image to [-1, 1] range"""
        enc = ((image / 255.0) * 2.0) - 1.0
        return enc
    
    def preprocess_state(self, image):
        """Preprocess observation image"""
        img = self.resize_image(image)
        img = self.normalize_state(img)
        return img

    def preprocess_goal_image(self, image):
        """Preprocess goal image"""
        return self.preprocess_state(image)

    def decode_action(self, action_tensor):
        """Decode normalized actions to original action space"""
        import torch as _torch
        action_mean = _torch.tensor(np.repeat([self._cfg.dataset.action_mean], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                   dtype=action_tensor.dtype, device=action_tensor.device)
        action_std = _torch.tensor(np.repeat([self._cfg.dataset.action_std], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                  dtype=action_tensor.dtype, device=action_tensor.device)
        return (action_tensor * (action_std)) + action_mean

    def encode_action(self, action_float):
        """Encode actions to normalized space [-1, 1]"""
        import torch as _torch
        ## If the action_float has length greater than action_dim then use stacking otherwise just use normal standardiaztion vectors
        if action_float.shape[1] == len(self._cfg.dataset.action_mean):
            action_mean = _torch.tensor(self._cfg.dataset.action_mean, dtype=action_float.dtype, device=action_float.device)
            action_std = _torch.tensor(self._cfg.dataset.action_std, dtype=action_float.dtype, device=action_float.device)
            return (action_float - action_mean) / (action_std)  

        action_mean = _torch.tensor(np.repeat([self._cfg.dataset.action_mean], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                   dtype=action_float.dtype, device=action_float.device)
        action_std = _torch.tensor(np.repeat([self._cfg.dataset.action_std], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                  dtype=action_float.dtype, device=action_float.device)
        return (action_float - action_mean) / (action_std)
    
    def decode_pose(self, pose_tensor):
        """
        Docstring for decode_pose
        
        :param self: Description
        :param pose_tensor: Description
        self._decode_state = lambda sinN: (sinN * state_std) + state_mean  # Undo mapping to [-1, 1]
        """
        import torch as _torch
        pose_mean = _torch.tensor(self._cfg.dataset.pose_mean, dtype=pose_tensor.dtype, device=pose_tensor.device)
        pose_std = _torch.tensor(self._cfg.dataset.pose_std, dtype=pose_tensor.dtype, device=pose_tensor.device)
        return (pose_tensor * (pose_std)) + pose_mean
    
    def encode_pose(self, pose_float):
        """
        Docstring for encode_pose
        
        :param self: Description
        :param pose_float: Description
        self._encode_pose = lambda pf:   (pf - pose_mean)/(pose_std) # encoder: take a float, output an integer
        """
        import torch as _torch
        pose_mean = _torch.tensor(self._cfg.dataset.pose_mean, dtype=pose_float.dtype, device=pose_float.device)
        pose_std = _torch.tensor(self._cfg.dataset.pose_std, dtype=pose_float.dtype, device=pose_float.device)
        return (pose_float - pose_mean) / (pose_std)

class DreamerV3(GRPBase):
    def __init__(self, 
                 obs_shape=(3, 128, 128),  # Updated default to match your error
                 action_dim=6, 
                 stoch_dim=32, 
                 discrete_dim=32, 
                 deter_dim=512, 
                 hidden_dim=512, cfg=None):
        # TODO: Part 3.1 - Initialize DreamerV3 architecture
        ## Define encoder, RSSM components (GRU, prior/posterior nets), and decoder heads
        super().__init__(cfg)
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.stoch_dim = stoch_dim
        self.discrete_dim = discrete_dim
        self.deter_dim = deter_dim
        self.hidden_dim = hidden_dim
        self.cfg = cfg
        c,h,w = obs_shape
        #hyper params from paper : encoding channels: 128,192,256,256, kernel=5 with maxpool(2)
        enco_chan = [128, 192, 256, 256]       
        k = 5

        #encoder: CNN Image (C, H, W)
        self.enco_blocks = nn.ModuleList()
        c_i = c
        for c_o in enco_chan:
            self.enco_blocks.append(nn.Sequential(
                nn.Conv2d(c_i, c_o, k, padding=k//2),
                nn.GroupNorm(8,c_o),
                nn.GELU(),
                nn.MaxPool2d(2)
            ))
            c_i = c_o

        #compute the encoder output size with a dummy input
        with torch.no_grad():
            x = torch.zeros(1, c, h, w)
            for block in self.enco_blocks:
                x = block(x)
            conv_out_dim = x.view(1,-1).shape[-1]

        self.emb_proj = nn.Linear(conv_out_dim, hidden_dim) 

        #RSSM
        #gru: Deterministic state (h)
        self.rssm_gru = nn.GRUCell(stoch_dim*discrete_dim + action_dim, deter_dim) #why
        self.rssm_norm = nn.LayerNorm(deter_dim) 

        #prior/posterior nets: Stochastic state (z), Discrete latent variables (categorical distribution)
        # Prior: p(z_t | h_t), NO observations
        self.rssm_prior = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, stoch_dim*discrete_dim)
        )

        # Posterior: q(z_t | h_t, embed_t), Observations
        self.rssm_posterior = nn.Sequential(
            nn.Linear(deter_dim+hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, stoch_dim*discrete_dim)
        )

        #decoder heads: reconstruct obs, CNN for images and MLP for vectors, like in paper
        #then project latent to space 
        latent_dim = deter_dim+stoch_dim*discrete_dim
        deco_chan = [256, 256, 192, 128]  
        self.deco_seed_chan = 256
        self.deco_proj = nn.Linear(latent_dim, self.deco_seed_chan *4 *4)
        self.deco_blocks = nn.ModuleList()
        d_in = deco_chan[0]
        dec_out_chan = deco_chan[1:] + [c]  
        for i, d_out in enumerate(dec_out_chan):
            if (i == len(dec_out_chan)-1):
                self.deco_blocks.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(d_in, d_out, k, padding=k//2)
                ))
            else:
                self.deco_blocks.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(d_in, d_out, k, padding=k //2),
                    nn.GroupNorm(8, d_out),
                    nn.GELU()
                    ))
            d_in = d_out
                
        #reward and continue pred head. 2 layer mlp layernormed and GELU like in paper
        self.r_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,1)    
        )
        
        self.c_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,1)   
        )

    def get_initial_state(self, batch_size, device):
        return {
            'h': torch.zeros(batch_size, self.deter_dim, device=device),
            'z': torch.zeros(batch_size, self.stoch_dim * self.discrete_dim, device=device),
            'z_probs': torch.zeros(batch_size, self.stoch_dim, self.discrete_dim, device=device)
        }

    def sample_stochastic(self, logits, training=True):
        # TODO: Part 3.1 - Implement stochastic sampling
        ## Sample from discrete categorical distribution using logits
        B = logits.shape[0]
        logits_2d = logits.view(B, self.stoch_dim, self.discrete_dim)
        z_probs = torch.softmax(logits_2d, dim=-1)

        if training:
            idx = torch.distributions.Categorical(probs=z_probs).sample()
            z_h = F.one_hot(idx, self.discrete_dim).float()
            z_st = z_h - z_probs.detach() + z_probs
        else:
            idx = torch.argmax(z_probs, dim=-1)
            z_st = F.one_hot(idx, self.discrete_dim).float()

        return z_st.view(B,-1), z_probs

    # def rssm_step(self, prev_state, action, embed=None):
    #     # TODO: Part 3.1 - Implement RSSM step
    #     ## Update deterministic state (h) with GRU, compute prior and posterior distributions
    #     h = self.rssm_gru(torch.cat([prev_state['z'], action], dim=-1), prev_state['h'])
    #     h = self.rssm_norm(h)
    #     prior = self.rssm_prior(h)
    #     z_prior, z_p_prob = self.sample_stochastic(prior, self.training)

    #     if embed is not None:
    #         post = self.rssm_posterior(torch.cat([h, embed], dim=-1))
    #         z, z_prob = self.sample_stochastic(post, self.training)
    #     else:
    #         post = None
    #         z, z_prob = z_prior, z_p_prob

    #     return {'h': h, 'z': z, 'z_probs': z_prob}, prior, post
    def rssm_step(self, prev_state, action, embed=None):
    # 1. Force everything to 2D for the GRU Cell
    # prev_state['z'] is often (B, 1, Z) -> we want (B, Z)
        z = prev_state['z']
        h_gru = prev_state['h']
        
        if z.dim() == 3:
            z = z.reshape(z.shape[0], -1)
        if h_gru.dim() == 3:
            h_gru = h_gru.reshape(h_gru.shape[0], -1)
        
        # action is often (B, A) but might be (B, 1, A) -> we want (B, A)
        if action.dim() == 3:
            action = action.reshape(action.shape[0], -1)

        # 2. Concatenate and Step the GRU (Line 307)
        # Both are now guaranteed 2D, so concat works on dim=-1
        h = self.rssm_gru(torch.cat([z, action], dim=-1), h_gru)
        h = self.rssm_norm(h)
        
        # 3. Compute Prior/Posterior (Same as your current code)
        prior = self.rssm_prior(h)
        z_prior, z_p_prob = self.sample_stochastic(prior, self.training)

        if embed is not None:
            if embed.dim() == 3:
                embed = embed.reshape(embed.shape[0], -1)
            post = self.rssm_posterior(torch.cat([h, embed], dim=-1))
            z, z_prob = self.sample_stochastic(post, self.training)
        else:
            post = None
            z, z_prob = z_prior, z_p_prob

        return {'h': h, 'z': z, 'z_probs': z_prob}, prior, post
    def forward(self, observations, prev_actions=None, prev_state=None,
                mask_=True, pose=None, last_action=None,
                text_goal=None, goal_image=None):
        # TODO: Part 3.2 - Implement DreamerV3 forward pass
        ## Encode images, unroll RSSM, and compute reconstructions and heads
        B, T, C, H, W = observations.shape
        device = observations.device

        state = prev_state if prev_state is not None else self.get_initial_state(B, device)
        if prev_actions is None:
            prev_actions = torch.zeros(B,T, self.action_dim, device=device)

        #encode images
        obs_flat = observations.view(B*T, C, H, W)
        x = obs_flat
        for blk in self.enco_blocks:
            x = blk(x)
        embeds = self.emb_proj(x.view(B*T, -1)).view(B,T,self.hidden_dim)
        recons, rewards, continues, prior_l, post_l, s_h, s_z  = [], [], [], [], [], [], []

        #unroll
        for t in range(T):
            s,p,q = self.rssm_step(state, prev_actions[:,t], embed = embeds[:,t])
            prior_l.append(p)
            post_l.append(q)
            s_h.append(s['h'])
            s_z.append(s['z'])
            feat = torch.cat([s['h'], s['z']], dim=-1)

            #decode
            x = self.deco_proj(feat).view(B, self.deco_seed_chan, 4, 4)
            for blk in self.deco_blocks:
                x = blk(x)
            if x.shape[-2:] != (H,W):
                x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            recons.append(x)
            rewards.append(self.r_head(feat))
            continues.append(self.c_head(feat))

        stack = lambda lst: torch.stack(lst, dim=1)

        return {
            'reconstructions': stack(recons),
            'rewards': stack(rewards),
            'continues': stack(continues),
            'priors_logits': stack(prior_l),
            'posts_logits': stack(post_l),
            'states_h': stack(s_h),
            'states_z': stack(s_z)
        }

    def preprocess_state(self, image):
        """Preprocess observation image"""
        img = self.resize_image(image)
        img = self.normalize_state(img)
        ## Change numpy array from channel-last to channel-first
        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        # img = img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        return img
    
    def compute_loss(self, output, images, rewards, dones, device):
        """
        Compute the total loss for DreamerV3 model training.
        
        Args:
            output: Dictionary containing model outputs (reconstructions, rewards, continues, priors_logits, posts_logits)
            images: Ground truth images tensor
            rewards: Ground truth rewards tensor
            dones: Ground truth done flags tensor
            device: Device to perform computations on
            pred_coeff: Coefficient for prediction losses (reconstruction + reward + continue)
            dyn_coeff: Coefficient for dynamics loss
            rep_coeff: Coefficient for representation loss
        
        Returns:
            Dictionary containing:
                - total_loss: Combined weighted loss
                - recon_loss: Reconstruction loss
                - reward_loss: Reward prediction loss
                - continue_loss: Continue prediction loss
                - dyn_loss: Dynamics loss (KL divergence)
                - rep_loss: Representation loss (KL divergence)
        """
        # TODO: Part 3.2 - Implement DreamerV3 loss computation
        ## Compute reconstruction, reward, KL divergence losses and combine them
        #recon loss:

        B,T = output['reconstructions'].shape[:2]
        #recon loss:
        recon_loss = F.mse_loss(symlog(output['reconstructions']), symlog(images))

        #reward loss: MSE like in paper?
        out_rewards = output["rewards"].squeeze(-1)
        reward_loss = F.mse_loss(out_rewards, symlog(rewards.view(B,T).to(device)))
        
        #continue loss
        c_pred = output['continues'].squeeze(-1)
        c_true = (1.0 - dones.view(B,T).float()).to(device)
        continue_loss = F.binary_cross_entropy_with_logits(c_pred, c_true)

        #kl divergence loss
        prior = output['priors_logits'].view(B*T, self.stoch_dim, self.discrete_dim)
        post = output['posts_logits'].view(B*T, self.stoch_dim, self.discrete_dim)
        log_post = F.log_softmax(post, dim=-1)
        log_prior = F.log_softmax(prior, dim=-1)
        post_prob = log_post.exp()

        # dyn: train prior toward sg(posterior) 
        kl_dyn = (post_prob.detach() * (post_prob.detach().log() - log_prior)).sum(-1)
        kl_dyn = kl_dyn.clamp(min=0.1).mean()

        # rep: train posterior toward sg(prior)  
        kl_rep = (post_prob * (log_post - log_prior.detach())).sum(-1)
        kl_rep = kl_rep.clamp(min=0.1).mean()

        #total loss
        total_loss = (self.cfg.loss_coeffs.pred_coeff * (recon_loss + reward_loss + continue_loss) + self.cfg.loss_coeffs.dyn_coeff * kl_dyn + self.cfg.loss_coeffs.rep_coeff * kl_rep)

        loss = {"total_loss": total_loss,
                "recon_loss": recon_loss.detach(),
                "reward_loss": reward_loss.detach(),
                "continue_loss": continue_loss.detach(),
                "dyn_loss": kl_dyn.detach(),
                "rep_loss": kl_rep.detach(),
                'pose_loss': torch.tensor(0.0, device=device)
                }

        return loss
        
    def encode_sequence(self, observations, prev_actions=None, prev_state=None):
        """
        Helper function to encode in planning, encode and unroll without decoding.
        Encode observations into latent state without decoding. Used for planning.
        """
        B, T, C, H, W = observations.shape
        device = observations.device

        state = prev_state if prev_state is not None else self.get_initial_state(B, device)
        if prev_actions is None:
            prev_actions = torch.zeros(B, T, self.action_dim, device=device)

        # Encode all frames
        obs_flat = observations.view(B * T, C, H, W)
        x = obs_flat
        for blk in self.enco_blocks:
            x = blk(x)
        embeds = self.emb_proj(x.view(B * T, -1)).view(B, T, self.hidden_dim)

        # Unroll RSSM only — no decoding
        for t in range(T):
            state, _, _ = self.rssm_step(state, prev_actions[:, t], embed=embeds[:, t])

        return state 

