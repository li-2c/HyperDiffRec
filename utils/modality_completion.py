# coding: utf-8

"""
AMC-DCF Data Completion Module Integration

Author: LX
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union
from logging import getLogger


class ModalityCompletionDetector:
    """Modality Missing Detector"""
    
    def __init__(self, config):
        self.config = config
        self.logger = getLogger()
        self.missing_threshold = config.get('missing_threshold', 0.1)
        
    def detect_missing_modalities(self, v_feat: Optional[torch.Tensor],
                                 t_feat: Optional[torch.Tensor]) -> Dict[str, Union[bool, torch.Tensor]]:
        """
        Detect missing modalities.

        Returns:
            Dict containing:
            - 'visual': tensor of missing indices or False if no missing
            - 'textual': tensor of missing indices or False if no missing
            - 'has_missing': bool indicating if any modality is missing
        """
        # Determine batch_size and device
        if v_feat is not None:
            batch_size = v_feat.shape[0]
            device = v_feat.device
        elif t_feat is not None:
            batch_size = t_feat.shape[0]
            device = t_feat.device
        else:
            # If both are None, return a simple boolean format
            return {
                'visual': True,
                'textual': True,
                'has_missing': True
            }

        missing_info = {
            'visual': False,
            'textual': False,
            'has_missing': False
        }

        # Check for missing visual features
        if v_feat is None:
            missing_info['visual'] = torch.ones(batch_size, dtype=torch.bool, device=device)
            self.logger.info(f"Visual features are None, all samples marked as missing")
        else:
            # Check for zero-value samples (per-sample detection)
            zero_mask = (v_feat.abs().sum(dim=1) < 1e-6)
            zero_ratio = zero_mask.float().mean().item()

            if zero_ratio > self.missing_threshold:
                missing_info['visual'] = zero_mask
                self.logger.info(f"Visual features have {zero_ratio:.2%} zeros, marked as missing")
            else:
                missing_info['visual'] = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Check for missing text features
        if t_feat is None:
            missing_info['textual'] = torch.ones(batch_size, dtype=torch.bool, device=device)
            self.logger.info(f"Textual features are None, all samples marked as missing")
        else:
            # Check for zero-value samples (per-sample detection)
            zero_mask = (t_feat.abs().sum(dim=1) < 1e-6)
            zero_ratio = zero_mask.float().mean().item()

            if zero_ratio > self.missing_threshold:
                missing_info['textual'] = zero_mask
                self.logger.info(f"Textual features have {zero_ratio:.2%} zeros, marked as missing")
            else:
                missing_info['textual'] = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Check if any modality is missing
        v_has_missing = isinstance(missing_info['visual'], torch.Tensor) and missing_info['visual'].any()
        t_has_missing = isinstance(missing_info['textual'], torch.Tensor) and missing_info['textual'].any()
        missing_info['has_missing'] = v_has_missing or t_has_missing

        return missing_info


class SimplifiedDiffusionCompletion(nn.Module):
    """Simplified Diffusion Data Completion Module"""
    
    def __init__(self, config, v_dim: int = None, t_dim: int = None):
        super().__init__()
        self.config = config
        self.embed_dim = config.get('completion_embed_dim', 128)
        self.num_steps = config.get('completion_steps', 10)
        self.device = config['device']
        
        # Encoder
        if v_dim is not None:
            self.v_encoder = nn.Sequential(
                nn.Linear(v_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim)
            )
            self.v_decoder = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, v_dim)
            )
            
        if t_dim is not None:
            self.t_encoder = nn.Sequential(
                nn.Linear(t_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim)
            )
            self.t_decoder = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, t_dim)
            )
        
        # Condition fusion network
        self.condition_fusion = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # Noise prediction network
        self.noise_predictor = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
    def forward(self, missing_feat: torch.Tensor, condition_feat: torch.Tensor) -> torch.Tensor:
        """
        Simplified diffusion completion process.
        
        Args:
            missing_feat: Missing modality features (can be zero vectors).
            condition_feat: Conditional modality features.
            
        Returns:
            Completed features.
        """
        batch_size = missing_feat.shape[0]
        
        # Encode conditional features
        condition_embed = self.condition_fusion(condition_feat)
        
        # Initialize noise
        noise = torch.randn_like(missing_feat).to(self.device)
        x = noise
        
        # Simplified denoising process
        for step in range(self.num_steps):
            # Time step encoding
            t = torch.full((batch_size,), step / self.num_steps).to(self.device)
            t_embed = self._time_embedding(t)
            
            # Predict noise
            combined = torch.cat([x, condition_embed], dim=-1)
            predicted_noise = self.noise_predictor(combined)
            
            # Denoising step
            alpha = 1.0 - step / self.num_steps
            x = alpha * x + (1 - alpha) * (x - predicted_noise)
            
        return x
    
    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Time step embedding"""
        # Simplified time embedding
        return t.unsqueeze(-1).expand(-1, self.embed_dim)


class AdvancedDiffusionCompletion(nn.Module):

    def __init__(self, config, v_dim: int = None, t_dim: int = None):
        super().__init__()
        self.config = config
        self.v_dim = v_dim
        self.t_dim = t_dim
        self.embed_dim = config.get('completion_embed_dim', 128)
        self.steps = config.get('completion_steps', 50)
        self.device = config['device']
        self.logger = getLogger()

        # Tuning parameters
        self.noise_scale = config.get('noise_scale', 0.01)
        self.fusion_alpha = config.get('fusion_alpha', 0.3)
        self.time_step_ratio = config.get('time_step_ratio', 0.1)
        self.fallback_noise = config.get('fallback_noise', 0.05)

        # Noise schedule parameters
        self.register_buffer('betas', self._cosine_beta_schedule(self.steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

        # Feature encoders
        if v_dim is not None:
            self.v_encoder = nn.Sequential(
                nn.Linear(v_dim, self.embed_dim * 4),
                nn.LayerNorm(self.embed_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.embed_dim * 4, self.embed_dim * 2),
                nn.LayerNorm(self.embed_dim * 2),
                nn.GELU(),
                nn.Linear(self.embed_dim * 2, self.embed_dim)
            )

            self.v_decoder = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * 2),
                nn.LayerNorm(self.embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.embed_dim * 2, self.embed_dim * 4),
                nn.LayerNorm(self.embed_dim * 4),
                nn.GELU(),
                nn.Linear(self.embed_dim * 4, v_dim)
            )

        if t_dim is not None:
            self.t_encoder = nn.Sequential(
                nn.Linear(t_dim, self.embed_dim * 4),
                nn.LayerNorm(self.embed_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.embed_dim * 4, self.embed_dim * 2),
                nn.LayerNorm(self.embed_dim * 2),
                nn.GELU(),
                nn.Linear(self.embed_dim * 2, self.embed_dim)
            )

            self.t_decoder = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * 2),
                nn.LayerNorm(self.embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.embed_dim * 2, self.embed_dim * 4),
                nn.LayerNorm(self.embed_dim * 4),
                nn.GELU(),
                nn.Linear(self.embed_dim * 4, t_dim)
            )

        # Time step embedding
        self.time_embed = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )

        # U-Net style denoising network
        self.denoiser = DenoisingUNet(self.embed_dim)

        # Cross-modal fusion module
        self.cross_modal_fusion = CrossModalFusion(self.embed_dim)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine noise schedule"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def _get_time_embedding(self, timestep, batch_size):
        """Get time step embedding"""
        half_dim = self.embed_dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timestep.device) * -emb)
        emb = timestep[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return self.time_embed(emb)

    def forward(self, v_feat, t_feat, missing_mask, inference_steps=None):
        """Complete diffusion process"""
        if inference_steps is None:
            inference_steps = max(self.steps // 5, 10)  # Use fewer steps for inference

        batch_size = v_feat.shape[0] if v_feat is not None else t_feat.shape[0]
        device = v_feat.device if v_feat is not None else t_feat.device

        # Handle missing_mask format
        if isinstance(missing_mask, dict):
            # Convert dict format to tensor format
            v_missing = missing_mask.get('visual', False)
            t_missing = missing_mask.get('textual', False)

            # Create tensor format mask
            if v_feat is not None:
                v_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                # Fix: Correctly handle tensor type missing information
                if isinstance(v_missing, torch.Tensor):
                    v_mask = v_missing
                elif v_missing:  # Boolean or other truthy value
                    # Detect actual missing cases
                    v_mask = (v_feat.abs().sum(dim=1) < 0.1)
            else:
                v_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

            if t_feat is not None:
                t_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                # Fix: Correctly handle tensor type missing information
                if isinstance(t_missing, torch.Tensor):
                    t_mask = t_missing
                elif t_missing:  # Boolean or other truthy value
                    # Detect actual missing cases
                    t_mask = (t_feat.abs().sum(dim=1) < 0.1)
            else:
                t_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

            missing_info = {'visual': v_mask, 'textual': t_mask}
        else:
            missing_info = missing_mask

        # Encode features
        v_embed = self.v_encoder(v_feat) if v_feat is not None and hasattr(self, 'v_encoder') else None
        t_embed = self.t_encoder(t_feat) if t_feat is not None and hasattr(self, 't_encoder') else None

        # Cross-modal fusion
        v_fused, t_fused = self.cross_modal_fusion(v_embed, t_embed, missing_info)

        # Diffusion sampling process
        v_completed = self._ddim_sample(v_fused, missing_info['visual'], inference_steps) if v_fused is not None else None
        t_completed = self._ddim_sample(t_fused, missing_info['textual'], inference_steps) if t_fused is not None else None

        # Decode back to original dimensions
        v_output = self.v_decoder(v_completed) if v_completed is not None and hasattr(self, 'v_decoder') else v_feat
        t_output = self.t_decoder(t_completed) if t_completed is not None and hasattr(self, 't_decoder') else t_feat

        # Phase 1 Optimization: Feature magnitude correction
        v_output, t_output = self._apply_norm_correction(v_output, t_output, v_feat, t_feat, missing_info)

        return v_output, t_output

    def _ddim_sample(self, x_start, missing_mask, inference_steps):
        """DDIM sampling process"""
        if x_start is None:
            return None

        batch_size = x_start.shape[0]
        device = x_start.device

        # Ensure missing_mask is in tensor format
        if isinstance(missing_mask, torch.Tensor):
            # If no missing values, return directly
            if not missing_mask.any():
                return x_start
        else:
            # If missing_mask is not a tensor, assume all samples need completion
            missing_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Simplified diffusion sampling: feature enhancement using Gaussian noise
        # For missing samples, add a small amount of noise and process through the denoising network
        result = x_start.clone()

        if missing_mask.any():
            # Use a diffusion sampling strategy with dynamically tuned parameters
            noise = torch.randn_like(x_start) * self.noise_scale

            # Use dynamic time steps
            t = torch.full((batch_size,), max(int(self.steps * self.time_step_ratio), 1),
                          device=device, dtype=torch.long)
            time_emb = self._get_time_embedding(t, batch_size)

            # Gentle noise addition: add a small amount of noise only to the missing parts
            noisy_x = x_start.clone()
            missing_indices = missing_mask.unsqueeze(-1).expand_as(x_start)
            noisy_x = torch.where(missing_indices, x_start + noise, x_start)

            # Process through the denoising network
            try:
                denoised = self.denoiser(noisy_x, time_emb)

                # Use dynamic fusion weights
                fused = self.fusion_alpha * denoised + (1 - self.fusion_alpha) * x_start

                # Fuse only the missing parts
                result = torch.where(missing_indices, fused, result)

            except Exception as e:
                # If the denoising network fails, use an improved interpolation method
                self.logger.warning(f"Denoiser failed: {e}, using improved interpolation")

                # Use a weighted average of non-missing samples
                if (~missing_mask).any():
                    non_missing_feat = x_start[~missing_mask]
                    # Calculate feature statistics
                    mean_feat = non_missing_feat.mean(dim=0, keepdim=True)
                    std_feat = non_missing_feat.std(dim=0, keepdim=True) + 1e-8

                    # Use dynamic fallback noise
                    generated_feat = mean_feat + torch.randn_like(mean_feat) * std_feat * self.fallback_noise
                    missing_indices = missing_mask.unsqueeze(-1).expand_as(result)
                    result = torch.where(missing_indices, generated_feat.expand_as(result), result)
                else:
                    # If all samples are missing, use zero padding
                    missing_indices = missing_mask.unsqueeze(-1).expand_as(result)
                    result = torch.where(missing_indices, torch.zeros_like(result), result)

        return result

    def update_tuning_params(self, **kwargs):
        """Dynamically update tuning parameters"""
        if 'noise_scale' in kwargs:
            self.noise_scale = kwargs['noise_scale']
        if 'fusion_alpha' in kwargs:
            self.fusion_alpha = kwargs['fusion_alpha']
        if 'time_step_ratio' in kwargs:
            self.time_step_ratio = kwargs['time_step_ratio']
        if 'fallback_noise' in kwargs:
            self.fallback_noise = kwargs['fallback_noise']

        self.logger.info(f"Updated tuning params: noise_scale={self.noise_scale}, "
                        f"fusion_alpha={self.fusion_alpha}, time_step_ratio={self.time_step_ratio}, "
                        f"fallback_noise={self.fallback_noise}")

    def _apply_norm_correction(self, v_output, t_output, v_orig, t_orig, missing_info):
        """
        Phase 1 Optimization: Feature magnitude correction
        Solves the problem that the norm of completed features is only 35-38% of the original features.
        """
        self.logger.info(f"Applying norm correction, missing_info: {missing_info}")

        if v_output is not None and v_orig is not None:
            v_missing_mask = missing_info.get('visual', False)
            self.logger.info(f"Visual missing mask type: {type(v_missing_mask)}, value: {v_missing_mask}")
            v_output = self._correct_feature_norm(v_output, v_orig, v_missing_mask)

        if t_output is not None and t_orig is not None:
            t_missing_mask = missing_info.get('textual', False)
            self.logger.info(f"Text missing mask type: {type(t_missing_mask)}, value: {t_missing_mask}")
            t_output = self._correct_feature_norm(t_output, t_orig, t_missing_mask)

        return v_output, t_output

    def _correct_feature_norm(self, completed_feat, orig_feat, missing_mask):
        """
        Correct feature norm to maintain the magnitude distribution of the original features.

        Args:
            completed_feat: Completed features.
            orig_feat: Original features.
            missing_mask: Missing mask.
        """
        self.logger.info(f"Starting feature norm correction, missing_mask type: {type(missing_mask)}")

        # Detect missing samples
        if not isinstance(missing_mask, torch.Tensor):
            # Detect missing based on zero values (samples where original features are 0)
            missing_mask = (orig_feat.abs().sum(dim=1) < 0.1)
            self.logger.info(f"Automatically detected missing samples: {missing_mask.sum().item()}/{len(missing_mask)}")

        if not missing_mask.any():
            self.logger.info("No missing samples, skipping norm correction")
            return completed_feat

        # Calculate the norm distribution of the original features (using all samples, including missing ones)
        # For missing samples, we use the statistics of non-missing samples
        non_missing_indices = ~missing_mask

        if non_missing_indices.any():
            # Use the norm distribution of non-missing samples
            orig_norms = orig_feat[non_missing_indices].norm(dim=1)
            target_mean_norm = orig_norms.mean()
            target_std_norm = orig_norms.std() + 1e-8
            self.logger.info(f"Non-missing sample norm stats: mean={target_mean_norm:.4f}, std={target_std_norm:.4f}")
        else:
            # If all samples are missing, use reasonable default values
            target_mean_norm = torch.tensor(20.0, device=completed_feat.device)  # Based on observed typical norms
            target_std_norm = torch.tensor(5.0, device=completed_feat.device)
            self.logger.info(f"All samples missing, using default norm: mean={target_mean_norm:.4f}")

        # Correct the norm of missing samples
        result = completed_feat.clone()
        missing_indices = missing_mask

        if missing_indices.any():
            # Calculate the current norm of the completed features
            completed_norms = completed_feat[missing_indices].norm(dim=1, keepdim=True)

            # Use the actual norm of the original features as the target (for missing samples, generate a reasonable target norm)
            batch_size = missing_indices.sum()

            # Generate target norms based on the distribution of non-missing samples
            target_norms = torch.normal(
                mean=target_mean_norm.float(),
                std=target_std_norm.float(),
                size=(batch_size, 1),
                device=completed_feat.device
            ).clamp(min=5.0, max=50.0)  # Reasonable norm range

            self.logger.info(f"Norm before completion: {completed_norms.mean().item():.4f}, target norm: {target_norms.mean().item():.4f}")

            # Apply norm correction
            # Avoid division by zero: if completed_norms is 0, use a random direction
            zero_norm_mask = (completed_norms < 1e-6).squeeze()

            if zero_norm_mask.any():
                self.logger.info(f"Found {zero_norm_mask.sum().item()} zero-norm features, generating random directions")
                # Generate random directions for zero-norm features
                # Fix: Correctly handle indices of zero-norm features
                missing_sample_indices = torch.where(missing_indices)[0]  # Get the actual indices of missing samples
                zero_sample_indices = missing_sample_indices[zero_norm_mask]  # Get the indices of zero-norm samples

                random_directions = torch.randn_like(completed_feat[zero_sample_indices])
                random_directions = F.normalize(random_directions, p=2, dim=1)
                result[zero_sample_indices] = random_directions * target_norms[zero_norm_mask]

            # Scale non-zero-norm features
            non_zero_mask = ~zero_norm_mask
            if non_zero_mask.any():
                scale_factors = target_norms[non_zero_mask] / (completed_norms[non_zero_mask] + 1e-8)
                self.logger.info(f"Scale factor range: {scale_factors.min().item():.4f} - {scale_factors.max().item():.4f}")

                # Fix: Correctly apply the scaling operation
                # Get the actual indices of non-zero-norm features among the missing samples
                missing_sample_indices = torch.where(missing_indices)[0]  # Get the actual indices of missing samples
                non_zero_sample_indices = missing_sample_indices[non_zero_mask]  # Get the indices of non-zero-norm samples

                scaled_features = completed_feat[non_zero_sample_indices] * scale_factors
                result[non_zero_sample_indices] = scaled_features

                self.logger.info(f"Successfully scaled {non_zero_mask.sum().item()} features")

            # Validate the correction effect
            corrected_norms = result[missing_indices].norm(dim=1)
            self.logger.info(f"Norm after correction: {corrected_norms.mean().item():.4f}")

        return result


class DenoisingUNet(nn.Module):
    """U-Net style denoising network"""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # Downsampling path
        self.down1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU()
        )

        self.down2 = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.LayerNorm(embed_dim * 4),
            nn.GELU()
        )

        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 4),
            nn.LayerNorm(embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim * 4),
            nn.LayerNorm(embed_dim * 4),
            nn.GELU()
        )

        # Upsampling path
        self.up2 = nn.Sequential(
            nn.Linear(embed_dim * 8, embed_dim * 2),  # 8 = 4 + 4 (skip connection)
            nn.LayerNorm(embed_dim * 2),
            nn.GELU()
        )

        self.up1 = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),  # 4 = 2 + 2 (skip connection)
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Output layer
        self.output = nn.Linear(embed_dim, embed_dim)

        # Time embedding projection
        self.time_proj1 = nn.Linear(embed_dim, embed_dim * 2)
        self.time_proj2 = nn.Linear(embed_dim, embed_dim * 4)

    def forward(self, x, time_emb):
        # Downsampling
        h1 = self.down1(x) + self.time_proj1(time_emb)
        h2 = self.down2(h1) + self.time_proj2(time_emb)

        # Bottleneck
        h = self.bottleneck(h2)

        # Upsampling (with skip connections)
        h = self.up2(torch.cat([h, h2], dim=-1))
        h = self.up1(torch.cat([h, h1], dim=-1))

        # Output
        return self.output(h)


class CrossModalFusion(nn.Module):
    """Cross-modal fusion module"""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # Self-attention
        self.self_attn_v = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.self_attn_t = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

        # Cross-modal attention
        self.cross_attn_v2t = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.cross_attn_t2v = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

        # Feed-forward network
        self.ffn_v = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.ffn_t = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # Layer normalization
        self.norm_v1 = nn.LayerNorm(embed_dim)
        self.norm_v2 = nn.LayerNorm(embed_dim)
        self.norm_v3 = nn.LayerNorm(embed_dim)

        self.norm_t1 = nn.LayerNorm(embed_dim)
        self.norm_t2 = nn.LayerNorm(embed_dim)
        self.norm_t3 = nn.LayerNorm(embed_dim)

    def forward(self, v_embed, t_embed, missing_mask):
        # Handle None inputs
        if v_embed is None and t_embed is None:
            return None, None
        elif v_embed is None:
            return None, t_embed
        elif t_embed is None:
            return v_embed, None

        # Add sequence dimension
        v_seq = v_embed.unsqueeze(1)  # [B, 1, D]
        t_seq = t_embed.unsqueeze(1)  # [B, 1, D]

        # Self-attention
        v_self, _ = self.self_attn_v(v_seq, v_seq, v_seq)
        t_self, _ = self.self_attn_t(t_seq, t_seq, t_seq)

        v_seq = self.norm_v1(v_seq + v_self)
        t_seq = self.norm_t1(t_seq + t_self)

        # Cross-modal attention (considering missing cases)
        try:
            # Check for missing modalities
            if isinstance(missing_mask, dict):
                v_missing = missing_mask.get('visual', torch.zeros(v_embed.shape[0], dtype=torch.bool, device=v_embed.device))
                t_missing = missing_mask.get('textual', torch.zeros(t_embed.shape[0], dtype=torch.bool, device=t_embed.device))
            else:
                v_missing = torch.zeros(v_embed.shape[0], dtype=torch.bool, device=v_embed.device)
                t_missing = torch.zeros(t_embed.shape[0], dtype=torch.bool, device=t_embed.device)

            # Cross-modal attention
            v_cross, _ = self.cross_attn_v2t(v_seq, t_seq, t_seq)
            t_cross, _ = self.cross_attn_t2v(t_seq, v_seq, v_seq)

            v_seq = self.norm_v2(v_seq + v_cross)
            t_seq = self.norm_t2(t_seq + t_cross)

        except Exception as e:
            # If cross-modal attention fails, skip this step
            pass

        # Feed-forward network
        v_ffn = self.ffn_v(v_seq)
        t_ffn = self.ffn_t(t_seq)

        v_out = self.norm_v3(v_seq + v_ffn).squeeze(1)
        t_out = self.norm_t3(t_seq + t_ffn).squeeze(1)

        return v_out, t_out


class ModalityCompletionModule:
    """Main Modality Completion Module"""
    
    def __init__(self, config):
        self.config = config
        self.logger = getLogger()
        self.detector = ModalityCompletionDetector(config)
        self.completion_model = None
        self.is_enabled = config.get('enable_completion', True)
        
    def setup_completion_model(self, v_dim: int = None, t_dim: int = None):
        """Set up the completion model"""
        if self.is_enabled and (v_dim is not None or t_dim is not None):
            # Select model type based on configuration
            use_advanced = self.config.get('use_advanced_diffusion', True)

            if use_advanced:
                self.completion_model = AdvancedDiffusionCompletion(
                    self.config, v_dim=v_dim, t_dim=t_dim
                ).to(self.config['device'])
                self.logger.info("Advanced diffusion completion model initialized")
            else:
                self.completion_model = SimplifiedDiffusionCompletion(
                    self.config, v_dim=v_dim, t_dim=t_dim
                ).to(self.config['device'])
                self.logger.info("Simplified diffusion completion model initialized")
    
    def complete_features(self, v_feat: Optional[torch.Tensor], 
                         t_feat: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete missing modality features.
        
        Args:
            v_feat: Visual features.
            t_feat: Text features.
            
        Returns:
            Completed (v_feat, t_feat).
        """
        if not self.is_enabled:
            return v_feat, t_feat
            
        missing_info = self.detector.detect_missing_modalities(v_feat, t_feat)
        
        if not missing_info['has_missing']:
            return v_feat, t_feat
            
        self.logger.info(f"Missing modalities detected: {missing_info}")
        
        # If the completion model is not initialized, use a simple strategy
        if self.completion_model is None:
            return self._simple_completion(v_feat, t_feat, missing_info)
        
        # Use the diffusion model for completion
        return self._diffusion_completion(v_feat, t_feat, missing_info)
    
    def _simple_completion(self, v_feat: Optional[torch.Tensor],
                          t_feat: Optional[torch.Tensor],
                          missing_info: Dict[str, bool]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple completion strategy (mean filling)"""
        self.logger.info(f"Executing simple completion, missing_info: {missing_info}")

        # Fix: Correctly handle the tensor format of missing_info
        v_missing = missing_info.get('visual', False)
        t_missing = missing_info.get('textual', False)

        # Check if visual modality is missing and needs completion
        if isinstance(v_missing, torch.Tensor) and v_missing.any() and t_feat is not None:
            self.logger.info(f"Completing visual features, number of missing samples: {v_missing.sum().item()}")

            # Get the indices of missing samples
            missing_indices = v_missing

            # Generate visual features using the statistics of text features
            if v_feat is not None:
                v_dim = v_feat.shape[1]

                # Fix: Use the text features of non-missing samples to calculate the target norm
                non_missing_indices = torch.logical_not(missing_indices)  # Fix deprecation warning
                if non_missing_indices.any():
                    # Use the norm of text features of non-missing samples as a reference
                    ref_norms = t_feat[non_missing_indices].norm(dim=1)
                    target_norm = ref_norms.mean().item()
                    self.logger.info(f"Using the average text norm of non-missing samples as the target: {target_norm:.4f}")
                else:
                    # If all samples are missing, use a default norm
                    target_norm = 20.0  # Based on observed typical norms
                    self.logger.info(f"All samples are missing, using a default target norm: {target_norm:.4f}")

                # Generate new visual features for missing samples
                generated_v = torch.randn(missing_indices.sum(), v_dim).to(t_feat.device)
                # Normalize and apply the target norm
                generated_v = F.normalize(generated_v, dim=-1) * target_norm

                # Replace only the missing samples
                v_feat = v_feat.clone()
                v_feat[missing_indices] = generated_v
                self.logger.info(f"Successfully completed {missing_indices.sum().item()} visual features, average norm: {generated_v.norm(dim=1).mean().item():.4f}")

        # Check if text modality is missing and needs completion
        if isinstance(t_missing, torch.Tensor) and t_missing.any() and v_feat is not None:
            self.logger.info(f"Completing text features, number of missing samples: {t_missing.sum().item()}")

            # Get the indices of missing samples
            missing_indices = t_missing

            # Generate text features using the statistics of visual features
            if t_feat is not None:
                t_dim = t_feat.shape[1]

                # Fix: Use the visual features of non-missing samples to calculate the target norm
                non_missing_indices = torch.logical_not(missing_indices)  # Fix deprecation warning
                if non_missing_indices.any():
                    # Use the norm of visual features of non-missing samples as a reference
                    ref_norms = v_feat[non_missing_indices].norm(dim=1)
                    target_norm = ref_norms.mean().item()
                    self.logger.info(f"Using the average visual norm of non-missing samples as the target: {target_norm:.4f}")
                else:
                    # If all samples are missing, use a default norm
                    target_norm = 20.0  # Based on observed typical norms
                    self.logger.info(f"All samples are missing, using a default target norm: {target_norm:.4f}")

                # Generate new text features for missing samples
                generated_t = torch.randn(missing_indices.sum(), t_dim).to(v_feat.device)
                # Normalize and apply the target norm
                generated_t = F.normalize(generated_t, dim=-1) * target_norm

                # Replace only the missing samples
                t_feat = t_feat.clone()
                t_feat[missing_indices] = generated_t
                self.logger.info(f"Successfully completed {missing_indices.sum().item()} text features, average norm: {generated_t.norm(dim=1).mean().item():.4f}")

        return v_feat, t_feat
    
    def _diffusion_completion(self, v_feat: Optional[torch.Tensor],
                            t_feat: Optional[torch.Tensor],
                            missing_info: Dict[str, bool]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use the diffusion model for completion"""
        try:
            # Check if it is an advanced diffusion model
            if isinstance(self.completion_model, AdvancedDiffusionCompletion):
                # Use the advanced diffusion model
                self.logger.info("Using advanced diffusion model for completion")
                return self.completion_model(v_feat, t_feat, missing_info)
            else:
                # Use the simplified diffusion model
                self.logger.info("Using simplified diffusion model for completion")

                # Ensure input features exist
                if v_feat is None and t_feat is not None:
                    v_feat = torch.zeros(t_feat.shape[0], self.completion_model.v_dim, device=t_feat.device)
                if t_feat is None and v_feat is not None:
                    t_feat = torch.zeros(v_feat.shape[0], self.completion_model.t_dim, device=v_feat.device)

                # Simplified completion process
                if missing_info.get('visual', False) and t_feat is not None:
                    # Complete visual features
                    try:
                        t_embed = self.completion_model.t_encoder(t_feat)
                        v_completed = self.completion_model.forward(v_feat, t_embed)
                        v_feat = self.completion_model.v_decoder(v_completed)
                    except Exception as e:
                        self.logger.warning(f"Visual completion failed: {e}")

                if missing_info.get('textual', False) and v_feat is not None:
                    # Complete text features
                    try:
                        v_embed = self.completion_model.v_encoder(v_feat)
                        t_completed = self.completion_model.forward(t_feat, v_embed)
                        t_feat = self.completion_model.t_decoder(t_completed)
                    except Exception as e:
                        self.logger.warning(f"Textual completion failed: {e}")

                return v_feat, t_feat

        except Exception as e:
            self.logger.warning(f"Diffusion completion failed: {e}, falling back to simple completion")
            return self._simple_completion(v_feat, t_feat, missing_info)


def create_completion_module(config) -> ModalityCompletionModule:
    """Factory function: create a modality completion module"""
    return ModalityCompletionModule(config)
