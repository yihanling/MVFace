from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mvface.geometry import triangulate_dlt_batch
from mvface.units import MM_PER_METRE


class ProjectiveAttention(nn.Module):
    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(
        self,
        query_3d: torch.Tensor,
        feat_maps: torch.Tensor,
        proj: torch.Tensor, # P matrices
        image_hw: tuple[int, int], # original image size the P matrix projetc into
    ):
        B, N, C, Hf, Wf = feat_maps.shape
        Q = query_3d.shape[1]
        H_img, W_img = image_hw

        q = query_3d.unsqueeze(1).expand(B, N, Q, 3) # replicate queries across all views
        
        # project 3D query point onto 2D image of each view with batching
        ones = torch.ones(B, N, Q, 1, device=q.device, dtype=q.dtype)
        hom = torch.cat([q, ones], dim=-1)
        uvw = torch.einsum('bnij,bnqj->bnqi', proj, hom)
        z = uvw[..., 2:3]
        z_safe = z.abs().clamp(min=1e-6) * torch.where(z < 0, -1.0, 1.0)
        uv = uvw[..., :2] / z_safe

        # normalize uv (pixels) to a grid_sample grid in [-1, 1].
        gx = 2 * (uv[..., 0] + 0.5) / W_img - 1
        gy = 2 * (uv[..., 1] + 0.5) / H_img - 1
        grid = torch.stack([gx, gy], dim=-1)

        # sample features at projected, normalized query point on 2D
        fm = feat_maps.reshape(B * N, C, Hf, Wf)
        g = grid.reshape(B * N, 1, Q, 2)
        sampled = F.grid_sample(fm, g, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.reshape(B, N, C, Q).permute(0, 1, 3, 2)

        # feature fusion across views
        feat = sampled.mean(dim=1)

        return feat, sampled, uv

class QueryUpdate(nn.Module):
    """Refine query in a standard pre-norm transformer block  (self-attention over landmarks and FFN)"""

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 ffn_dim: int = 1024, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        ) # position-wise feed-forward network

    def forward(self, query_feat: torch.Tensor) -> torch.Tensor:
        x = self.norm1(query_feat) # pre-norm
        attn, _ = self.self_attn(x, x, x) # attention
        query_feat = query_feat + attn
        x = self.norm2(query_feat)
        query_feat = query_feat + self.ffn(x) # FFN
        return query_feat

class OffsetConfidenceHead(nn.Module):
    """From each per-view sampled feature it predicts:
    
      offset (B,N,Q,2)
        a pixel nudge added to that query's projected point, so
      conf   (B,N,Q)
        a non-negative weightfor that view in the confidence-weighted DLT 
    """

    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.offset = nn.Linear(d_model, 2)
        self.conf = nn.Linear(d_model, 1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, sampled: torch.Tensor):
        offset = self.offset(sampled)
        conf = F.softplus(self.conf(sampled)).squeeze(-1) # softplus for smoothness and non-negative
        return offset, conf



class DecoderLayer(nn.Module):
    """one refinement iteration of the Decoder

        1. ProjectiveAttention: sample image features at where 3D queries projects
        2. self-attention + FFN update them.
        3. predict a per-view 2D offset + confidence.
        4. triangulate the updated 2D (confidence-weighted) -> new 3D queries

    Returns:
        updated query features, the new 3D estimate, and the refined 2D
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, ffn_dim: int = 1024) -> None:
        super().__init__()
        self.proj_attn = ProjectiveAttention(d_model)
        self.query_update = QueryUpdate(d_model, n_heads, ffn_dim)
        self.head = OffsetConfidenceHead(d_model)

    def forward(self, query_feat, ref_3d, feat_maps, proj, image_hw):
        # 1. projective attention
        feat, sampled, uv = self.proj_attn(ref_3d, feat_maps, proj, image_hw)
        # 2. self-attention
        query_feat = self.query_update(query_feat + feat)
        # 3. per-view 2D offset + confidence
        head_in = sampled + query_feat.unsqueeze(1)
        offset, conf = self.head(head_in)
        refined_uv = uv + offset
        # 4. confidence-weighted triangulation -> refined 3D
        new_ref_3d = self._triangulate(refined_uv, proj, conf)
        return query_feat, new_ref_3d, refined_uv

    @staticmethod
    def _triangulate(refined_uv, proj, conf):
        B, N, Q, _ = refined_uv.shape
        # fold (B,Q) into the batch of points M = B*Q; each point has N views.
        pts = refined_uv.permute(0, 2, 1, 3).reshape(B * Q, N, 2)
        Ps = proj[:, None].expand(B, Q, N, 3, 4).reshape(B * Q, N, 3, 4)
        w = conf.permute(0, 2, 1).reshape(B * Q, N)
        X = triangulate_dlt_batch(pts, Ps, w)
        return X.reshape(B, Q, 3)


class MultiViewDecoder(nn.Module):
    """Top level decoder: stack of L DecoderLayer

    Query tokens are learned embeddings; the 3D reference starts at the mean-face template placed at the query-box center (SPACE_CENTER), and each layer refines it. 
    
    Returns per-layer 3D and 2D predictions
    """

    def __init__(self, mean_face, space_center, num_layers: int = 4,
                 d_model: int = 256, n_heads: int = 8, ffn_dim: int = 1024,
                 num_queries: int = 68) -> None:
        super().__init__()
        # define query embeddings: the learned "identity" of each landmark
        self.query_embed = nn.Parameter(torch.empty(num_queries, d_model))
        nn.init.normal_(self.query_embed, std=0.02)

        # 3D queries init: mean face shape centered at SPACE_CENTER.
        mf = torch.as_tensor(np.asarray(mean_face), dtype=torch.float32) / MM_PER_METRE
        sc = torch.as_tensor(np.asarray(space_center), dtype=torch.float32) / MM_PER_METRE
        self.register_buffer("ref0", mf + sc) # register as buffer, it is never later learned and modified

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, ffn_dim) for _ in range(num_layers)])

    @classmethod
    def from_assets(cls, assets_dir, **kw):
        assets_dir = Path(assets_dir)
        mean_face = np.load(assets_dir / "mean_face_68.npy")
        space = json.loads((assets_dir / "query_space.json").read_text())
        return cls(mean_face, space["SPACE_CENTER"], **kw)

    def forward(self, feat_maps, proj, image_hw):
        B = feat_maps.shape[0]
        query_feat = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        ref_3d = self.ref0.unsqueeze(0).expand(B, -1, -1)

        preds_3d, preds_2d = [], []
        for layer in self.layers:
            query_feat, ref_3d, refined_uv = layer(query_feat, ref_3d, feat_maps, proj, image_hw)
            preds_3d.append(ref_3d)
            preds_2d.append(refined_uv)
        return preds_3d, preds_2d
