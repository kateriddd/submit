import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

#changed get patches fast
def get_patches_fast(images, cfg):
    # images shape is confirmed as (B, H, W, C) -> (1, 64, 64, 3)
    patch_size = cfg.patch_size  # 8
    
    # We check the LAST dimension for channels
    channels_total = images.shape[-1]
    hs = cfg.policy.obs_stacking # Should be 1 if not stacking
    
    # Calculate base channels (usually 3 for RGB)
    c = channels_total // hs

    # The pattern must match (B, H, W, C)
    # We split H into (h*p1) and W into (w*p2)
    # Then we move the patches into the sequence dimension
    patches = rearrange(images, 'b (h p1) (w p2) (c hs) -> b (h w hs) (p1 p2 c)', 
                        p1=patch_size, p2=patch_size, hs=hs, c=c)
    
    return patches

def calc_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

def build_mask(sample, readout_index, device):
    """Build an attention mask for a single batch, handling:
      - Causal masking (no future attention)
      - Missing modalities (tokens not present)
      - Readout token (can attend to all, not attended by others)diagonal, obs can see prev obs.
      """
    B = len(sample)
    T = readout_index + 1
    mask = torch.ones(B, T, T, dtype=torch.bool)
    for i in range(B):
        n_t = sample[i]
        # attend to prev observations
        causal = torch.triu(torch.ones(n_t, n_t, device=device), diagonal=1).bool()
        mask[i, :n_t, :n_t] = causal
        # readout token, can attend to all real tokens
        mask[i, readout_index, :n_t] = False
        # no one can attend to readout
        mask[i, :readout_index, n_t] = True
    return mask


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        # TODO: 
        ## Provide the block masking logic for the attention head
        # Octo style attention mask

        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        if mask is not None:
            wei = wei.masked_fill(mask, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd=n_embd, dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        with torch.profiler.record_function("Self-Attention"):
            out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd=n_embd, dropout=dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        x = x + self.sa(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class GRP(nn.Module):
    def __init__(self, cfg, mlp_ratio=4):
        super(GRP, self).__init__()
        self._cfg = cfg
        chars = cfg.dataset.chars_list
        cfg.vocab_size = len(chars)

        ## Provide the logic for the GRP network
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)

        if self._cfg.dataset.encode_with_t5:
            self.t5_proj = nn.Linear(512, self._cfg.n_embd)

        patch_dim = cfg.patch_size * cfg.patch_size * 3
        self.patch_embed = nn.Linear(patch_dim, cfg.n_embd)

        # 4) Transformer encoder blocks
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.n_embd))
        img_h = cfg.image_shape[0]
        img_w = cfg.image_shape[1]

        num_goal_patches = (img_h // cfg.patch_size) * (img_w // cfg.patch_size)
        num_obs_patches = num_goal_patches * cfg.policy.obs_stacking
        max_goal_tokens = cfg.max_block_size

        max_seq_len = max_goal_tokens + num_goal_patches + num_obs_patches + 1  # +1 for CLS
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, cfg.n_embd))

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cfg.n_embd,
                nhead=cfg.n_head,
                dim_feedforward=cfg.n_embd * mlp_ratio,
                dropout=cfg.dropout,
                activation='gelu'
            )
            for _ in range(cfg.n_blocks)
        ])
        # 5) Classification MLPk
        if self._cfg.continous:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(cfg.n_embd),
                nn.Linear(cfg.n_embd, cfg.action_dim)
            )
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(cfg.n_embd),
                nn.Linear(cfg.n_embd, cfg.action_dim * cfg.n_bins)
            )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, images, goals_txt, goal_imgs, targets=None, pose=None, mask_=False):
        # --- NEW SAFETY CHECK FOR NCHW vs NHWC ---
        # If images are (B, 3, 64, 64), permute them to (B, 64, 64, 3)
        if images.shape[1] == 3 and images.shape[-1] != 3:
            images = images.permute(0, 2, 3, 1)
            
        # Do the same for goal_imgs if they are present
        if goal_imgs is not None and goal_imgs.shape[1] == 3 and goal_imgs.shape[-1] != 3:
            goal_imgs = goal_imgs.permute(0, 2, 3, 1)
        # -----------------------------------------

        if not hasattr(self, "_fwd_count"):
            self._fwd_count = 0
            self._fwd_count += 1
            # print("FORWARD CALL", self._fwd_count)
        B, H, W, C = images.shape
        if self._cfg.dataset.encode_with_t5:
            goals_e = goals_txt
            B, T_t, E = goals_txt.shape 
            # goals_e = self.t5_proj(goals_e)  # now [B, T, n_embd]
        else:
            # goals_e = self.token_embedding_table(goals_txt)
            # B, E = goals_txt.shape
            # T = self._cfg.max_block_size
            goals_e = self.token_embedding_table(goals_txt)
            B, T_t, E = goals_e.shape

        # TODO:
        ## Provide the logic to produce the output and loss for the GRP
        # Map patch to the hidden dim
        obs_patches = get_patches_fast(images, self._cfg)   #obs
        g_patches = get_patches_fast(goal_imgs, self._cfg)  #goalim
        obs_patches = self.patch_embed(obs_patches)
        g_patches = self.patch_embed(g_patches)

        # Adding classification, goal_img and goals_e tokens
        cls_tokens = self.cls_token.expand(B, 1, self._cfg.n_embd)
        task_tokens = torch.cat([goals_e, g_patches], dim=1)    #text + image goals
        tokens = torch.cat([task_tokens, obs_patches, cls_tokens], dim=1)

        # positional embedding
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]

        # blocked masks
        T = tokens.shape[1]
        if mask_:
            seq = [task_tokens.shape[1] + obs_patches.shape[1] for _ in range(B)]
            readout_index = T - 1
            mask = build_mask(seq, readout_index, self._cfg.device)
        else:
            mask = None

        # Transformer Blocks
        for block in self.blocks:
            tokens = block(tokens, src_mask=mask)

        # Get the classification token only
        N = self._cfg.policy.action_stacking
        readout = tokens[:, :N, :]

        # Compute output and loss
        out = self.mlp_head(readout)
        if targets is not None:
            if (self._cfg.continous):
                loss = F.mse_loss(out, targets)
            else:
                targets = targets.long()
                Bt, Nt, At = targets.shape
                out = out.view(Bt, Nt, At, self._cfg.n_bins)
                logits = out.reshape(Bt*Nt*At, self._cfg.n_bins)
                targets = targets.reshape(Bt*Nt*At).long()
                loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return (out, loss)

    def resize_image(self, image):
        """
        Docstring for resize_image
        :param self: Description
        :param image: Description
        self._resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  # resize state
        """
        import cv2
        import numpy as _np
        img = _np.array(image, dtype=_np.float32)
        img = cv2.resize(img, (self._cfg.image_shape[0], self._cfg.image_shape[1]))
        return img

    def normalize_state(self, image):
        """
        Docstring for preprocess_state
        :param self: Description
        :param image: Description
        self._encode_state = lambda af:   ((af/(255.0)*2.0)-1.0) # encoder: take a float, output an integer
        self._resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  # resize state
        """
        # img = _np.array(image, dtype=_np.float32)
        # img = cv2.resize(img, (self._cfg.image_shape[0], self._cfg.image_shape[1]))
        enc = ((image / 255.0) * 2.0) - 1.0
        # t = _torch.tensor(enc, dtype=_torch.float32, device=self._cfg.device)
        return enc

    def preprocess_state(self, image):
        # 1. Resize (input: H, W, 3 -> output: 64, 64, 3)
        img = self.resize_image(image) 
        # 2. Normalize
        img = self.normalize_state(img)
        # 3. Ensure it's a tensor and add batch dim
        import torch
        img_tensor = torch.from_numpy(img).float().to(self._cfg.device)
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0) # [1, 64, 64, 3]
        return img_tensor

    def preprocess_goal_image(self, image):
        return self.preprocess_state(image)

    def encode_text_goal(self, goal, tokenizer=None, text_model=None):
        import numpy as _np
        import torch as _torch
        if self._cfg.dataset.encode_with_t5:
            if tokenizer is None or text_model is None:
                raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
            encoded = tokenizer(
                goal,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self._cfg.max_block_size
            )
            input_ids = encoded["input_ids"].to(self._cfg.device)
            attention_mask = encoded["attention_mask"].to(self._cfg.device)
            with _torch.no_grad():
                encoder_outputs = text_model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_emb = encoder_outputs.last_hidden_state  # [1, seq_len, n_embd]
            # pad/truncate to max_block_size
            B, T, E = text_emb.shape
            if T < self._cfg.max_block_size:
                pad = _torch.zeros((B, self._cfg.max_block_size - T, E), device=text_emb.device)
                text_emb = _torch.cat([text_emb, pad], dim=1)
            else:
                text_emb = text_emb[:, :self._cfg.max_block_size, :]
            return text_emb

        else:
            pad = " " * self._cfg.max_block_size
            goal_ = goal[:self._cfg.max_block_size] + pad[len(goal):self._cfg.max_block_size]
            try:
                stoi = {c: i for i, c in enumerate(self._cfg.dataset.chars_list)}
                ids = [stoi.get(c, 0) for c in goal_]
            except Exception:
                ids = [0] * self._cfg.max_block_size
            return _torch.tensor(_np.expand_dims(_np.array(ids, dtype=_np.int64), axis=0), dtype=_torch.long,
                                 device=self._cfg.device)

    def process_text_embedding_for_buffer(self, goal, tokenizer=None, text_model=None):
        """
        Process text goal embedding for storing in the circular buffer.
        Returns a numpy array of shape (max_block_size, n_embd) without batch dimension.
        """
        import numpy as _np
        if tokenizer is None or text_model is None:
            raise ValueError("tokenizer and text_model must be provided when using T5 encoding")

        device = next(text_model.parameters()).device
        # allocate buffer directly on GPU
        goal_ = torch.zeros(self._cfg.max_block_size, self._cfg.n_embd, device=device)
        tokens = tokenizer(goal, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
    
        with torch.no_grad():
            goal_t = text_model.encoder(**tokens).last_hidden_state
            # keep only up to max_block_size and embedding dim
            goal_t = goal_t[:, :self._cfg.max_block_size, :self._cfg.n_embd]
            # take first batch element
            goal_[:goal_t.shape[1], :] = goal_t[0]
    
        return goal_

    def decode_action(self, action_tensor):
        """
        Docstring for decode_action
        :param self: Description
        :param action_tensor: Description
        self._decode_action = lambda binN: (binN * action_std) + action_mean  # Undo mapping to [-1, 1]
        """
        import torch as _torch
        ## The action tensor is of shape (batch_size, action_dim * action_stacking) so we need to repeat the mean and std per action stacking
        action_mean = _torch.tensor(np.repeat(self._cfg.dataset.action_mean, self._cfg.policy.action_stacking),
                                    dtype=action_tensor.dtype, device=action_tensor.device)
        action_std = _torch.tensor(np.repeat(self._cfg.dataset.action_std, self._cfg.policy.action_stacking),
                                   dtype=action_tensor.dtype, device=action_tensor.device)
        return (action_tensor * action_std) + action_mean

    def encode_action(self, action_float):
        """
        Docstring for encode_action
        :param self: Description
        :param action_float: Description
        self._encode_action = lambda af:   (af - action_mean)/(action_std) # encoder: take a float, output an integer
        """
        import torch as _torch
        action_mean = _torch.tensor(self._cfg.dataset.action_mean, dtype=action_float.dtype, device=action_float.device)
        action_std = _torch.tensor(self._cfg.dataset.action_std, dtype=action_float.dtype, device=action_float.device)
        return (action_float - action_mean) / action_std


@torch.no_grad()
def estimate_loss(model, dataset):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(model._cfg.eval_iters)
        for k in range(model._cfg.eval_iters):
            X, x_pose, x_goal, x_goal_img, Y = dataset.get_batch_grp(split, model._cfg, model._cfg.batch_size)
            logits, loss = model(X, x_goal, x_goal_img, Y, pose=x_pose)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
