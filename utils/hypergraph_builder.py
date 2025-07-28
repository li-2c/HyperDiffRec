"""
Adaptive Quality-Driven Hypergraph Enhancement Framework

Author: LX

"""

import torch
import torch.nn as nn
import numpy as np
try:
    import dgl
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    print("Warning: DGL not available. Hypergraph functionality will be disabled.")

from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LightweightHypergraphBuilder:
    """
    Lightweight Hypergraph Builder
    Constructs a hypergraph structure based on features completed by MoDiCF.
    """
    
    def __init__(self, config: Dict):
        if not DGL_AVAILABLE:
            raise ImportError("DGL is required for hypergraph functionality. Please install DGL.")

        self.config = config
        self.hypergraph_config = config.get('hypergraph_config', {})
        self.device = config.get('device', 'cuda')
        
        # Hypergraph construction parameters
        self.max_hyperedge_size = self.hypergraph_config.get('max_hyperedge_size', 8)
        self.sampling_ratio = self.hypergraph_config.get('hyperedge_sampling_ratio', 0.3)
        # Fix: Adjust the default similarity threshold to adapt to the normalized cosine similarity range [-1, 1]
        self.similarity_threshold = self.hypergraph_config.get('similarity_threshold', 0.3)
        
        # Hyperedge type configuration
        self.hyperedge_types = self.hypergraph_config.get('hyperedge_types', {})
        
        logger.info(f"LightweightHypergraphBuilder initialized with config: {self.hypergraph_config}")
    
    def build_hypergraph(self,
                        v_feat: torch.Tensor,
                        t_feat: torch.Tensor,
                        interaction_matrix: Optional[torch.Tensor] = None):
        """
        Build a lightweight hypergraph.
        
        Args:
            v_feat: Completed visual features [N, D_v].
            t_feat: Completed text features [N, D_t].
            interaction_matrix: User-item interaction matrix [M, N].
            
        Returns:
            DGL hypergraph object.
        """
        logger.info("Starting to build a lightweight hypergraph...")
        
        # Store all hyperedges
        hyperedges = []
        
        # 1. Multimodal feature hyperedges (priority implementation in Phase 1)
        if self.hyperedge_types.get('multimodal_edge', False):
            multimodal_edges = self._build_multimodal_hyperedges(v_feat, t_feat)
            hyperedges.extend(multimodal_edges)
            logger.info(f"Built {len(multimodal_edges)} multimodal hyperedges.")
        
        # 2. Interest aggregation hyperedges (implementation in Phase 2)
        if self.hyperedge_types.get('interest_aggregation', False):
            interest_edges = self._build_interest_aggregation_hyperedges(v_feat, t_feat, interaction_matrix)
            hyperedges.extend(interest_edges)
            logger.info(f"Built {len(interest_edges)} interest aggregation hyperedges.")
        
        # 3. Completion quality hyperedges (implementation in Phase 3)
        if self.hyperedge_types.get('completion_quality', False):
            quality_edges = self._build_completion_quality_hyperedges(v_feat, t_feat)
            hyperedges.extend(quality_edges)
            logger.info(f"Built {len(quality_edges)} completion quality hyperedges.")
        
        # Build DGL hypergraph
        hypergraph = self._create_dgl_hypergraph(hyperedges, v_feat.shape[0])
        
        logger.info(f"Hypergraph construction completed, with a total of {len(hyperedges)} hyperedges.")
        return hypergraph
    
    def _build_multimodal_hyperedges(self,
                                    v_feat: torch.Tensor,
                                    t_feat: torch.Tensor) -> List[List[int]]:
        """
        Build multimodal feature hyperedges.
        Connects items with similar multimodal features.
        """
        logger.debug("Building multimodal feature hyperedges...")

        # Fix: L2-normalize the features
        v_feat_norm = torch.nn.functional.normalize(v_feat, p=2, dim=1)
        t_feat_norm = torch.nn.functional.normalize(t_feat, p=2, dim=1)

        logger.debug(f"Feature normalization complete - v_feat: {v_feat_norm.shape}, t_feat: {t_feat_norm.shape}")

        # Calculate multimodal similarity (cosine similarity)
        v_sim = torch.mm(v_feat_norm, v_feat_norm.t())  # [N, N]
        t_sim = torch.mm(t_feat_norm, t_feat_norm.t())  # [N, N]

        # Fuse similarity (simple average)
        combined_sim = (v_sim + t_sim) / 2.0

        logger.debug(f"Similarity calculation complete - range: [{combined_sim.min().item():.6f}, {combined_sim.max().item():.6f}]")

        # Build hyperedges
        hyperedges = []
        n_items = v_feat.shape[0]

        # Sampling strategy: avoid building too many hyperedges
        sampled_items = torch.randperm(n_items)[:int(n_items * self.sampling_ratio)]
        logger.debug(f"Sampled {len(sampled_items)} items for hyperedge construction.")

        for i, item_idx in enumerate(sampled_items):
            # Find similar items
            similarities = combined_sim[item_idx]
            similar_items = torch.where(similarities > self.similarity_threshold)[0]

            # Add detailed debug information
            if i < 3:  # Only print debug info for the first 3 items
                logger.debug(f"Item {item_idx}: similarity range [{similarities.min().item():.6f}, {similarities.max().item():.6f}], threshold {self.similarity_threshold}, number of similar items {len(similar_items)}.")

            # Limit hyperedge size
            if len(similar_items) > self.max_hyperedge_size:
                # Select the most similar items
                _, top_indices = torch.topk(similarities, self.max_hyperedge_size)
                similar_items = top_indices
                if i < 3:
                    logger.debug(f"Item {item_idx}: hyperedge size limited, reduced from {len(similar_items)} to {self.max_hyperedge_size}.")

            if len(similar_items) >= 2:  # At least 2 nodes are required
                hyperedge = similar_items.cpu().tolist()
                hyperedges.append(hyperedge)
                if i < 3:
                    logger.debug(f"Item {item_idx}: successfully created hyperedge with {len(hyperedge)} nodes.")

        logger.debug(f"Built {len(hyperedges)} multimodal hyperedges.")
        return hyperedges
    
    def _build_interest_aggregation_hyperedges(self, 
                                            v_feat: torch.Tensor, 
                                            t_feat: torch.Tensor,
                                            interaction_matrix: Optional[torch.Tensor]) -> List[List[int]]:
        """
        Build interest aggregation hyperedges.
        Aggregates items with similar interests based on user interaction history.
        """
        if interaction_matrix is None:
            logger.warning("Interaction matrix is empty, skipping interest aggregation hyperedge construction.")
            return []
        
        logger.debug("Building interest aggregation hyperedges...")
        
        # Build item similarity based on collaborative filtering
        item_cooccurrence = torch.mm(interaction_matrix.t(), interaction_matrix)  # [N, N]
        
        # Normalization
        item_norms = torch.sqrt(torch.diag(item_cooccurrence))
        item_cooccurrence = item_cooccurrence / (item_norms.unsqueeze(0) * item_norms.unsqueeze(1) + 1e-8)
        
        hyperedges = []
        n_items = v_feat.shape[0]
        
        # Sampling strategy
        sampled_items = torch.randperm(n_items)[:int(n_items * self.sampling_ratio)]
        
        for item_idx in sampled_items:
            # Find collaboratively filtered similar items
            cooccur_scores = item_cooccurrence[item_idx]
            similar_items = torch.where(cooccur_scores > self.similarity_threshold)[0]
            
            # Limit hyperedge size
            if len(similar_items) > self.max_hyperedge_size:
                _, top_indices = torch.topk(cooccur_scores, self.max_hyperedge_size)
                similar_items = top_indices
            
            if len(similar_items) >= 2:
                hyperedge = similar_items.cpu().tolist()
                hyperedges.append(hyperedge)
        
        return hyperedges
    
    def _build_completion_quality_hyperedges(self,
                                        v_feat: torch.Tensor,
                                        t_feat: torch.Tensor) -> List[List[int]]:
        """
        Build completion quality hyperedges.
        Connects items with similar completion quality based on multiple quality assessment metrics.

        Quality assessment methods:
        1. Feature norm reasonableness assessment
        2. Multimodal feature consistency measurement
        3. Feature distribution stability assessment
        4. Cross-modal correlation analysis

        Args:
            v_feat: Completed visual features [N, D_v].
            t_feat: Completed text features [N, D_t].

        Returns:
            List[List[int]]: List of completion quality hyperedges.
        """
        logger.debug("Building completion quality hyperedges...")

        try:
            # Get quality assessment configuration
            quality_config = self.hypergraph_config.get('completion_quality_config', {})
            quality_threshold = quality_config.get('quality_threshold', 0.3)
            norm_weight = quality_config.get('norm_weight', 0.3)
            consistency_weight = quality_config.get('consistency_weight', 0.3)
            stability_weight = quality_config.get('stability_weight', 0.2)
            correlation_weight = quality_config.get('correlation_weight', 0.2)

            logger.debug(f"Quality assessment config: threshold={quality_threshold}, weights=[{norm_weight}, {consistency_weight}, {stability_weight}, {correlation_weight}]")

            # Compute comprehensive quality scores
            quality_scores = self._compute_completion_quality_scores(
                v_feat, t_feat, norm_weight, consistency_weight,
                stability_weight, correlation_weight
            )

            # Build hyperedges based on quality similarity
            hyperedges = self._build_quality_based_hyperedges(
                quality_scores, quality_threshold
            )

            logger.debug(f"Built {len(hyperedges)} completion quality hyperedges.")
            return hyperedges

        except Exception as e:
            logger.warning(f"Failed to build completion quality hyperedges: {e}, returning an empty list.")
            return []

    def _compute_completion_quality_scores(self,
                                        v_feat: torch.Tensor,
                                        t_feat: torch.Tensor,
                                        norm_weight: float,
                                        consistency_weight: float,
                                        stability_weight: float,
                                        correlation_weight: float) -> torch.Tensor:
        """
        Compute comprehensive completion quality scores.

        Args:
            v_feat: Visual features [N, D_v].
            t_feat: Text features [N, D_t].
            norm_weight: Norm reasonableness weight.
            consistency_weight: Consistency weight.
            stability_weight: Stability weight.
            correlation_weight: Correlation weight.

        Returns:
            torch.Tensor: Quality scores [N].
        """
        n_items = v_feat.shape[0]
        device = v_feat.device

        # 1. Feature norm reasonableness assessment
        norm_scores = self._compute_norm_reasonableness(v_feat, t_feat)

        # 2. Multimodal feature consistency measurement
        consistency_scores = self._compute_modality_consistency(v_feat, t_feat)

        # 3. Feature distribution stability assessment
        stability_scores = self._compute_feature_stability(v_feat, t_feat)

        # 4. Cross-modal correlation analysis
        correlation_scores = self._compute_cross_modal_correlation(v_feat, t_feat)

        # Comprehensive quality score (weighted average)
        quality_scores = (
            norm_weight * norm_scores +
            consistency_weight * consistency_scores +
            stability_weight * stability_scores +
            correlation_weight * correlation_scores
        )

        # Normalize to [0, 1] range
        quality_scores = torch.clamp(quality_scores, 0.0, 1.0)

        logger.debug(f"Quality score stats: mean={quality_scores.mean().item():.4f}, "
                    f"std={quality_scores.std().item():.4f}, "
                    f"range=[{quality_scores.min().item():.4f}, {quality_scores.max().item():.4f}]")

        return quality_scores

    def _compute_norm_reasonableness(self, v_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """
        Compute feature norm reasonableness score.

        Based on the assumption that high-quality completed features should have a reasonable norm,
        neither too small (insufficient information) nor too large (overcompensation).

        Mathematical formula:
        norm_score_i = exp(-|log(||x_i||_2 / μ_norm)|)
        where μ_norm is the expected feature norm.
        """
        # Compute feature norms
        v_norms = torch.norm(v_feat, p=2, dim=1)  # [N]
        t_norms = torch.norm(t_feat, p=2, dim=1)  # [N]

        # Compute expected norm (using median as a robust estimate)
        v_expected_norm = torch.median(v_norms)
        t_expected_norm = torch.median(t_norms)

        # Compute norm reasonableness scores
        v_norm_scores = torch.exp(-torch.abs(torch.log(v_norms / (v_expected_norm + 1e-8) + 1e-8)))
        t_norm_scores = torch.exp(-torch.abs(torch.log(t_norms / (t_expected_norm + 1e-8) + 1e-8)))

        # Combine visual and text norm scores
        norm_scores = (v_norm_scores + t_norm_scores) / 2.0

        return norm_scores

    def _compute_modality_consistency(self, v_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """
        Compute multimodal feature consistency score.

        Based on the assumption that high-quality completion should maintain semantic consistency between different modalities.

        Mathematical formula:
        consistency_i = cosine_similarity(normalize(v_i), normalize(t_i))
        """
        # L2 normalization
        v_feat_norm = torch.nn.functional.normalize(v_feat, p=2, dim=1)
        t_feat_norm = torch.nn.functional.normalize(t_feat, p=2, dim=1)

        # Compute cosine similarity (sample-wise)
        consistency_scores = torch.sum(v_feat_norm * t_feat_norm, dim=1)

        # Map similarity from [-1, 1] to [0, 1]
        consistency_scores = (consistency_scores + 1.0) / 2.0

        return consistency_scores

    def _compute_feature_stability(self, v_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """
        Compute feature distribution stability score.

        Based on the assumption that high-quality completed features should have stable distribution characteristics,
        without abnormal activation patterns.

        Mathematical formula:
        stability_i = exp(-σ_i / μ_σ)
        where σ_i is the standard deviation of feature i, and μ_σ is the mean of the global standard deviation.
        """
        # Compute the standard deviation of features for each sample
        v_stds = torch.std(v_feat, dim=1)  # [N]
        t_stds = torch.std(t_feat, dim=1)  # [N]

        # Compute the mean of the global standard deviation (as a reference)
        global_v_std = torch.mean(v_stds)
        global_t_std = torch.mean(t_stds)

        # Compute stability scores (the closer the standard deviation is to the global mean, the higher the stability)
        v_stability = torch.exp(-torch.abs(v_stds - global_v_std) / (global_v_std + 1e-8))
        t_stability = torch.exp(-torch.abs(t_stds - global_t_std) / (global_t_std + 1e-8))

        # Combine stability scores
        stability_scores = (v_stability + t_stability) / 2.0

        return stability_scores

    def _compute_cross_modal_correlation(self, v_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-modal correlation analysis score.

        Based on the assumption that high-quality completion should maintain a reasonable cross-modal correlation,
        neither completely irrelevant nor excessively correlated.

        Mathematical formula:
        correlation_i = 1 - |corr(v_i, t_i) - μ_corr|
        where μ_corr is the expected correlation level.
        """
        n_items = v_feat.shape[0]
        device = v_feat.device

        # Compute cross-modal correlation for each sample
        correlations = []
        for i in range(n_items):
            v_sample = v_feat[i]  # [D_v]
            t_sample = t_feat[i]  # [D_t]

            # Use statistical properties of features to calculate a proxy for correlation
            # Here, the dot product after normalization is used as a measure of correlation
            v_norm = torch.nn.functional.normalize(v_sample.unsqueeze(0), p=2, dim=1)
            t_norm = torch.nn.functional.normalize(t_sample.unsqueeze(0), p=2, dim=1)

            # Calculate the correlation of feature activation patterns (using the sign function)
            v_pattern = torch.sign(v_sample)
            t_pattern = torch.sign(t_sample)

            # Calculate the consistency of activation patterns
            pattern_consistency = torch.mean((v_pattern[:min(len(v_pattern), len(t_pattern))] ==
                                            t_pattern[:min(len(v_pattern), len(t_pattern))]).float())

            correlations.append(pattern_consistency)

        correlation_tensor = torch.stack(correlations)

        # Expected correlation level (empirical value: 0.5 indicates moderate correlation)
        expected_correlation = 0.5

        # Calculate correlation scores (the closer to the expected value, the better)
        correlation_scores = 1.0 - torch.abs(correlation_tensor - expected_correlation)

        return correlation_scores

    def _build_quality_based_hyperedges(self,
                                        quality_scores: torch.Tensor,
                                        quality_threshold: float) -> List[List[int]]:
        """
        Build hyperedges based on quality scores.

        Args:
            quality_scores: Quality scores [N].
            quality_threshold: Quality similarity threshold.

        Returns:
            List[List[int]]: List of quality hyperedges.
        """
        n_items = len(quality_scores)
        hyperedges = []

        # Sampling strategy: avoid building too many hyperedges
        sampled_items = torch.randperm(n_items)[:int(n_items * self.sampling_ratio)]

        logger.debug(f"Sampled {len(sampled_items)} items for quality hyperedge construction.")

        for i, item_idx in enumerate(sampled_items):
            item_quality = quality_scores[item_idx]

            # Find items with similar quality
            quality_diffs = torch.abs(quality_scores - item_quality)
            similar_items = torch.where(quality_diffs <= quality_threshold)[0]

            # Debug information (only for the first 3 items)
            if i < 3:
                logger.debug(f"Item {item_idx}: quality score {item_quality:.4f}, "
                            f"number of similar items {len(similar_items)}, threshold {quality_threshold}")

            # Limit hyperedge size
            if len(similar_items) > self.max_hyperedge_size:
                # Select items with the most similar quality
                _, top_indices = torch.topk(-quality_diffs, self.max_hyperedge_size)
                similar_items = top_indices
                if i < 3:
                    logger.debug(f"Item {item_idx}: hyperedge size limited, reduced from {len(similar_items)} to {self.max_hyperedge_size}.")

            if len(similar_items) >= 2:  # At least 2 nodes are required
                hyperedge = similar_items.cpu().tolist()
                hyperedges.append(hyperedge)
                if i < 3:
                    logger.debug(f"Item {item_idx}: successfully created quality hyperedge with {len(hyperedge)} nodes.")

        return hyperedges

    def _create_dgl_hypergraph(self, hyperedges: List[List[int]], num_nodes: int):
        """
        Create a DGL hypergraph object.
        """
        if not DGL_AVAILABLE:
            return None

        if not hyperedges:
            logger.warning("No hyperedges, creating an empty hypergraph.")
            # Create an empty hypergraph
            return dgl.heterograph({
                ('node', 'in', 'hyperedge'): ([], []),
                ('hyperedge', 'contain', 'node'): ([], [])
            })
        
        # Build the adjacency matrix representation of the hypergraph
        node_ids = []
        hyperedge_ids = []
        
        for he_idx, hyperedge in enumerate(hyperedges):
            for node_id in hyperedge:
                node_ids.append(node_id)
                hyperedge_ids.append(he_idx)
        
        # Create a heterograph
        graph_data = {
            ('node', 'in', 'hyperedge'): (node_ids, hyperedge_ids),
            ('hyperedge', 'contain', 'node'): (hyperedge_ids, node_ids)
        }
        
        hypergraph = dgl.heterograph(graph_data)
        
        # Set the number of nodes
        hypergraph.nodes['node'].data['id'] = torch.arange(num_nodes)
        hypergraph.nodes['hyperedge'].data['id'] = torch.arange(len(hyperedges))
        
        return hypergraph.to(self.device)


class HypergraphStatistics:
    """Hypergraph statistics class."""
    
    @staticmethod
    def compute_stats(hypergraph) -> Dict:
        """Compute hypergraph statistics."""
        stats = {
            'num_nodes': hypergraph.num_nodes('node'),
            'num_hyperedges': hypergraph.num_nodes('hyperedge'),
            'num_edges': hypergraph.num_edges('in'),
            'avg_hyperedge_size': 0,
            'max_hyperedge_size': 0,
            'min_hyperedge_size': 0
        }
        
        if stats['num_hyperedges'] > 0:
            # Compute hyperedge size statistics
            hyperedge_sizes = []
            for he_id in range(stats['num_hyperedges']):
                # Get the number of nodes in the hyperedge
                _, nodes = hypergraph.in_edges(he_id, etype='contain')
                hyperedge_sizes.append(len(nodes))
            
            stats['avg_hyperedge_size'] = np.mean(hyperedge_sizes)
            stats['max_hyperedge_size'] = np.max(hyperedge_sizes)
            stats['min_hyperedge_size'] = np.min(hyperedge_sizes)
        
        return stats