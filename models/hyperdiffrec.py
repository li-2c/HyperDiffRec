"""
HyperDiffRec: Hypergraph-Enhanced Diffusion Recommendation Model
A unified architecture for multimodal recommendation systems - fully independent implementation.

Author: Lee
Date: 2025-07-15

Core Components:
- Contrastive Interest Learning Module
- AMC-DCF Modality Completion Module (Adaptive Modality Completion with Diffusion Counterfactual Framework)
- Hypergraph Neural Network Module
- Adaptive Feature Fusion Module
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging
import time
import psutil
import dgl
import dgl.function as fn

# Import necessary base modules
from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, build_mixed_graph, build_non_zero_graph, build_knn_normalized_graph, build_graph_from_adj
from utils.modality_completion import ModalityCompletionModule
from utils.hypergraph_builder import LightweightHypergraphBuilder, HypergraphStatistics

logger = logging.getLogger(__name__)


# ================================================================================================
# Part 1: Contrastive Interest Learning Core Module
# ================================================================================================

class ContrastiveInterestCore(GeneralRecommender):
    """
    Core module for Contrastive Interest Learning.
    Includes AMC-DCF modality completion functionality.
    """
    def __init__(self, config, dataset):
        super(ContrastiveInterestCore, self).__init__(config, dataset)

        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.cl_loss2 = config['cl_loss2']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.knn_i = config['knn_i']
        self.knn_a = config['knn_a']

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        image_interest_file = os.path.join(dataset_path, 'image_interest_{}_{}.pt'.format(self.knn_k, self.knn_i))
        text_interest_file = os.path.join(dataset_path, 'text_interest_{}_{}.pt'.format(self.knn_k, self.knn_i))

        mm_attractive_file = os.path.join(dataset_path, 'mm_attractive_{}.pt'.format(self.knn_a))

        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        # Initialize the AMC-DCF completion module
        self.completion_module = None
        if config.get('enable_completion', False):
            self.completion_module = ModalityCompletionModule(config)
            # Setup the completion model
            if self.v_feat is not None and self.t_feat is not None:
                self.completion_module.setup_completion_model(
                    v_dim=self.v_feat.shape[1],
                    t_dim=self.t_feat.shape[1]
                )
                # Apply completion
                self.v_feat, self.t_feat = self.completion_module.complete_features(
                    self.v_feat, self.t_feat
                )

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)

        # Interest-aware graph construction
        device = config.get('device', 'cpu')
        self.device = device
        map_location = 'cpu' if device == 'cpu' else None

        if os.path.exists(image_interest_file) and os.path.exists(text_interest_file):
            image_interest_adj = torch.load(image_interest_file, map_location=map_location, weights_only=False)
            print(image_interest_file+" loaded!")
            text_interest_adj = torch.load(text_interest_file, map_location=map_location, weights_only=False)
            print(text_interest_file+" loaded!")
        else:
            # Build interest graphs
            image_adj = build_sim(self.image_embedding.weight.detach())
            text_adj = build_sim(self.text_embedding.weight.detach())
            image_interest = torch.zeros_like(image_adj)
            text_interest = torch.zeros_like(text_adj)
            for user, items in dataset.history_items_per_u.items():
                items = torch.tensor([i for i in items])
                _, cols1 = torch.topk(image_adj[items].sum(dim=0), self.knn_i)
                _, cols2 = torch.topk(text_adj[items].sum(dim=0), self.knn_i)
                cols = torch.cat([cols1, cols2]).unique()
                image_interest[items[:, None],cols] += image_adj[items[:, None],cols]
                text_interest[items[:, None],cols] += text_adj[items[:, None],cols]

            image_interest_adj = build_non_zero_graph(image_interest,is_sparse=self.sparse, norm_type='sym')
            text_interest_adj = build_non_zero_graph(text_interest, is_sparse=self.sparse, norm_type='sym')

            # Similarity graphs
            image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
            text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')

            # Mixed graphs
            image_interest_adj = torch.add(image_interest_adj, image_adj)
            text_interest_adj = torch.add(text_interest_adj, text_adj)

            torch.save(image_interest_adj, image_interest_file)
            torch.save(text_interest_adj, text_interest_file)
            del image_adj, text_adj, image_interest, text_interest
            torch.cuda.empty_cache()

        # Set device based on configuration
        self.image_interest_adj = image_interest_adj.to(device)
        self.text_interest_adj = text_interest_adj.to(device)

        # Second-order attraction graph construction
        if os.path.exists(mm_attractive_file) and os.path.exists(mm_attractive_file.replace('.pt','_R.pt')):
            mm_attractive_adj = torch.load(mm_attractive_file, map_location=map_location, weights_only=False)
            print(mm_attractive_file + " loaded!")
            mm_attractive_adj_R = torch.load(mm_attractive_file.replace('.pt','_R.pt'), map_location=map_location, weights_only=False)
            print(mm_attractive_file.replace('.pt','_R.pt') + " loaded!")
        else:
            image_adj = build_sim(self.image_embedding.weight.detach())
            text_adj = build_sim(self.text_embedding.weight.detach())
            mm_attractive = torch.zeros_like(self.norm_adj)
            mm_attractive_R = torch.zeros_like(self.R)

            for user, items in dataset.history_items_per_u.items():
                items = torch.tensor([i for i in items])
                k_num = self.knn_a + items.size(0)
                mm_sim = torch.multiply(image_adj[items], text_adj[items])
                mm_value, mm_indices = torch.topk(mm_sim, k_num, dim=-1)

                k_mm_value, k_mm_indices = torch.topk(mm_value.flatten(), k_num)

                mm_indices = mm_indices.flatten()[k_mm_indices]

                uid = torch.zeros_like(mm_indices).fill_(user)
                mm_sparse = torch.sparse_coo_tensor(torch.stack([uid, self.n_users+mm_indices]), k_mm_value, size=self.norm_adj.size())
                mm_sparse_t = torch.sparse_coo_tensor(torch.stack([self.n_users+mm_indices, uid]), k_mm_value, size=self.norm_adj.size())

                mm_R_sparse = torch.sparse_coo_tensor(torch.stack([uid, mm_indices]), k_mm_value, size=self.R.size())

                mm_attractive += mm_sparse
                mm_attractive += mm_sparse_t

                mm_attractive_R += mm_R_sparse


            mm_attractive_adj = build_graph_from_adj(mm_attractive,is_sparse=self.sparse, norm_type='sym',mask=False)
            mm_attractive_adj_R = build_graph_from_adj(mm_attractive_R,is_sparse=self.sparse, norm_type='sym',mask=False)
            torch.save(mm_attractive_adj, mm_attractive_file)
            torch.save(mm_attractive_adj_R, mm_attractive_file.replace('.pt','_R.pt'))

            del image_adj, text_adj, mm_attractive, mm_sparse, mm_sparse_t, mm_R_sparse, mm_attractive_R
            torch.cuda.empty_cache()

        # Set device based on configuration
        self.mm_attractive_adj = mm_attractive_adj.to(device)
        self.mm_attractive_adj_R = mm_attractive_adj_R.to(device)
        # Enhanced graph construction
        self.norm_adj = torch.add(self.norm_adj, self.mm_attractive_adj/2)
        self.R = torch.add(self.R, self.mm_attractive_adj_R/2)

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        # Ensure all modules are on the correct device
        # Use PyTorch's standard method to move the entire model to the specified device
        self.to(device)

        # Define core modules
        self.softmax = nn.Softmax(dim=-1)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        # Modality mapping layers
        self.map_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.map_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.tau = 0.5

        # Ensure all modules (including newly defined ones) are on the correct device
        self.to(device)

        # Forcefully ensure all pre-loaded weights are also on the correct device
        if hasattr(self, 'image_interest_adj'):
            self.image_interest_adj = self.image_interest_adj.to(device)
        if hasattr(self, 'text_interest_adj'):
            self.text_interest_adj = self.text_interest_adj.to(device)
        if hasattr(self, 'mm_attractive_adj'):
            self.mm_attractive_adj = self.mm_attractive_adj.to(device)
        if hasattr(self, 'mm_attractive_adj_R'):
            self.mm_attractive_adj_R = self.mm_attractive_adj_R.to(device)

    def pre_epoch_processing(self):
        pass

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        R_sparse = norm_adj_mat[:self.n_users, self.n_users:]

        # Convert R matrix to torch sparse tensor and move to the correct device
        self.R = self.sparse_mx_to_torch_sparse_tensor(R_sparse.tocsr())

        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        # Check input type
        if hasattr(sparse_mx, 'tocoo'):
            # This is a scipy sparse matrix
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
            indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
            values = torch.from_numpy(sparse_mx.data)
            shape = torch.Size(sparse_mx.shape)
            sparse_tensor = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)
        else:
            # This is already a torch tensor, return directly
            sparse_tensor = sparse_mx

        # Ensure the sparse tensor is on the correct device
        device = next(self.parameters()).device
        return sparse_tensor.to(device)

    def forward(self, adj, train=False):
        # Ensure adj is on the correct device
        device = next(self.parameters()).device
        if hasattr(adj, 'to'):
            adj = adj.to(device)

        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)

        # Feature ID embeddings
        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.map_v(image_feats))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.map_t(text_feats))

        # Second-order graph convolution
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for layer_idx in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        # Interest-aware item graph convolution
        for layer_idx in range(self.n_layers):
            image_item_embeds = torch.sparse.mm(self.image_interest_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

        for layer_idx in range(self.n_layers):
            text_item_embeds = torch.sparse.mm(self.text_interest_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        # Attention fusion
        att_common = torch.cat([self.attention(image_embeds), self.attention(text_embeds)], dim=-1)
        weight_common = self.softmax(att_common)
        common_embeds = weight_common[:, 0].unsqueeze(dim=1) * image_embeds + weight_common[:, 1].unsqueeze(
            dim=1) * text_embeds
        side_embeds = (image_embeds + text_embeds - common_embeds) / 3

        all_embeds = content_embeds + side_embeds

        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        bpr_loss = -torch.mean(maxi)

        return bpr_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(
            self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)

        # Item-item contrastive loss
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], 0.2)
        # User-item contrastive loss
        cl_loss2 = self.InfoNCE(u_g_embeddings, content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            u_g_embeddings, side_embeds_items[pos_items], 0.2)

        return bpr_loss + self.cl_loss * cl_loss + self.cl_loss2 * cl_loss2

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores


# ================================================================================================
# Part 2: Hypergraph Neural Network Module
# ================================================================================================

class HypergraphConvolution(nn.Module):
    """
    Hypergraph Convolutional Network.
    Implements an efficient hypergraph message passing mechanism.
    """

    def __init__(self, config: Dict):
        super(HypergraphConvolution, self).__init__()

        self.config = config
        self.hypergraph_config = config.get('hypergraph_config', {})

        # Network parameters
        self.embedding_dim = config.get('embedding_size', 160)
        self.num_layers = self.hypergraph_config.get('hgcn_layers', 1)
        self.dropout = self.hypergraph_config.get('hgcn_dropout', 0.1)
        self.activation = self.hypergraph_config.get('hgcn_activation', 'relu')

        # Build network layers
        self.node_transforms = nn.ModuleList()
        self.hyperedge_transforms = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for layer_idx in range(self.num_layers):
            # Node transformation layer
            self.node_transforms.append(
                nn.Linear(self.embedding_dim, self.embedding_dim)
            )
            # Hyperedge transformation layer
            self.hyperedge_transforms.append(
                nn.Linear(self.embedding_dim, self.embedding_dim)
            )
            # Layer normalization
            self.layer_norms.append(
                nn.LayerNorm(self.embedding_dim)
            )

        # Activation function
        if self.activation == 'relu':
            self.act_fn = F.relu
        elif self.activation == 'gelu':
            self.act_fn = F.gelu
        else:
            self.act_fn = F.relu

        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)

        logger.info(f"HypergraphConvolution initialized: {self.num_layers} layers, "
                   f"embedding_dim={self.embedding_dim}, dropout={self.dropout}")

    def forward(self,
                hypergraph: dgl.DGLHeteroGraph,
                node_features: torch.Tensor) -> torch.Tensor:
        """
        Hypergraph convolution forward pass.

        Args:
            hypergraph: DGL hypergraph object.
            node_features: Node features [N, D].

        Returns:
            Updated node features [N, D].
        """
        if hypergraph.num_nodes('node') == 0 or hypergraph.num_nodes('hyperedge') == 0:
            logger.warning("Empty hypergraph, returning original features.")
            return node_features

        # Initialize node features
        h_nodes = node_features

        # Multi-layer hypergraph convolution
        for layer_idx in range(self.num_layers):
            h_nodes = self._hypergraph_conv_layer(
                hypergraph, h_nodes, layer_idx
            )

        return h_nodes

    def _hypergraph_conv_layer(self,
                            hypergraph: dgl.DGLHeteroGraph,
                            node_features: torch.Tensor,
                            layer_idx: int) -> torch.Tensor:
        """
        Single-layer hypergraph convolution.

        Implements a two-stage message passing:
        1. Node -> Hyperedge: Aggregates node features to hyperedges.
        2. Hyperedge -> Node: Propagates hyperedge features back to nodes.
        """
        # Ensure features are on the correct device
        device = node_features.device
        hypergraph = hypergraph.to(device)

        # Set node features
        hypergraph.nodes['node'].data['h'] = node_features

        # Stage 1: Node-to-hyperedge message passing
        hypergraph.update_all(
            message_func=fn.copy_u('h', 'm'),
            reduce_func=fn.mean('m', 'h_hyperedge'),
            etype='in'
        )

        # Get hyperedge features
        if 'h_hyperedge' in hypergraph.nodes['hyperedge'].data:
            hyperedge_features = hypergraph.nodes['hyperedge'].data['h_hyperedge']
        else:
            # Create zero features if no hyperedge features exist
            num_hyperedges = hypergraph.num_nodes('hyperedge')
            hyperedge_features = torch.zeros(
                num_hyperedges, self.embedding_dim, device=device
            )

        # Transform hyperedge features
        hyperedge_features = self.hyperedge_transforms[layer_idx](hyperedge_features)
        hyperedge_features = self.act_fn(hyperedge_features)
        hyperedge_features = self.dropout_layer(hyperedge_features)

        # Set transformed hyperedge features
        hypergraph.nodes['hyperedge'].data['h_transformed'] = hyperedge_features

        # Stage 2: Hyperedge-to-node message passing
        hypergraph.update_all(
            message_func=fn.copy_u('h_transformed', 'm'),
            reduce_func=fn.mean('m', 'h_new'),
            etype='contain'
        )

        # Get updated node features
        if 'h_new' in hypergraph.nodes['node'].data:
            new_node_features = hypergraph.nodes['node'].data['h_new']
        else:
            # Use original features if no updated features exist
            new_node_features = node_features

        # Node feature transformation
        new_node_features = self.node_transforms[layer_idx](new_node_features)

        # Residual connection
        new_node_features = new_node_features + node_features

        # Layer normalization
        new_node_features = self.layer_norms[layer_idx](new_node_features)

        # Activation function and Dropout
        new_node_features = self.act_fn(new_node_features)
        new_node_features = self.dropout_layer(new_node_features)

        return new_node_features


# ================================================================================================
# Part 3: Adaptive Feature Fusion Module
# ================================================================================================

class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive Feature Fusion Module.
    Fuses core features and hypergraph features.
    """

    def __init__(self, config: Dict):
        super(AdaptiveFeatureFusion, self).__init__()

        self.config = config
        self.hypergraph_config = config.get('hypergraph_config', {})

        # Fusion parameters
        self.embedding_dim = config.get('embedding_size', 160)
        self.fusion_strategy = self.hypergraph_config.get('fusion_strategy', 'attention')
        self.hypergraph_weight = self.hypergraph_config.get('hypergraph_weight', 0.3)
        self.fusion_dropout = self.hypergraph_config.get('fusion_dropout', 0.1)

        # Build network based on fusion strategy
        if self.fusion_strategy == 'attention':
            self._build_attention_fusion()
        elif self.fusion_strategy == 'weighted':
            self._build_weighted_fusion()
        elif self.fusion_strategy == 'concat':
            self._build_concat_fusion()
        else:
            raise ValueError(f"Unsupported fusion strategy: {self.fusion_strategy}")

        self.dropout = nn.Dropout(self.fusion_dropout)

        logger.info(f"AdaptiveFeatureFusion initialized with strategy: {self.fusion_strategy}")

    def _build_attention_fusion(self):
        """Build attention fusion network."""
        self.attention = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 2),
            nn.Softmax(dim=-1)
        )

    def _build_weighted_fusion(self):
        """Build weighted fusion network."""
        # Uses fixed weights, no additional parameters needed
        pass

    def _build_concat_fusion(self):
        """Build concatenation fusion network."""
        self.projection = nn.Linear(self.embedding_dim * 2, self.embedding_dim)

    def forward(self,
                core_user_emb: torch.Tensor,
                core_item_emb: torch.Tensor,
                hg_user_emb: torch.Tensor,
                hg_item_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Feature fusion forward pass.

        Args:
            core_user_emb: Core user embeddings [M, D].
            core_item_emb: Core item embeddings [N, D].
            hg_user_emb: Hypergraph user embeddings [M, D].
            hg_item_emb: Hypergraph item embeddings [N, D].

        Returns:
            Fused user and item embeddings.
        """
        if self.fusion_strategy == 'attention':
            fused_user_emb = self._attention_fusion(core_user_emb, hg_user_emb)
            fused_item_emb = self._attention_fusion(core_item_emb, hg_item_emb)
        elif self.fusion_strategy == 'weighted':
            fused_user_emb = self._weighted_fusion(core_user_emb, hg_user_emb)
            fused_item_emb = self._weighted_fusion(core_item_emb, hg_item_emb)
        elif self.fusion_strategy == 'concat':
            fused_user_emb = self._concat_fusion(core_user_emb, hg_user_emb)
            fused_item_emb = self._concat_fusion(core_item_emb, hg_item_emb)

        return fused_user_emb, fused_item_emb

    def _attention_fusion(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """Attention fusion."""
        # Concatenate features
        concat_feat = torch.cat([feat1, feat2], dim=-1)

        # Calculate attention weights
        attention_weights = self.attention(concat_feat)  # [B, 2]

        # Weighted fusion
        fused_feat = (attention_weights[:, 0:1] * feat1 +
                     attention_weights[:, 1:2] * feat2)

        return self.dropout(fused_feat)

    def _weighted_fusion(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """Weighted fusion."""
        fused_feat = ((1 - self.hypergraph_weight) * feat1 +
                     self.hypergraph_weight * feat2)

        return self.dropout(fused_feat)

    def _concat_fusion(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """Concatenation fusion."""
        concat_feat = torch.cat([feat1, feat2], dim=-1)
        fused_feat = self.projection(concat_feat)

        return self.dropout(fused_feat)


# ================================================================================================
# Part 4: Performance Monitoring and Risk Control Module
# ================================================================================================

class PerformanceMonitor:
    """Performance Monitor."""

    def __init__(self, config: Dict):
        self.config = config
        self.hypergraph_config = config.get('hypergraph_config', {})

        # Performance threshold
        self.performance_threshold = self.hypergraph_config.get('performance_threshold', 0.100)
        self.check_interval = self.hypergraph_config.get('check_interval', 5)

        # Performance history
        self.performance_history = []
        self.epoch_count = 0

        logger.info(f"PerformanceMonitor initialized, threshold: {self.performance_threshold}")

    def update_performance(self, recall_20: float, ndcg_20: float):
        """Update performance records."""
        self.epoch_count += 1
        self.performance_history.append({
            'epoch': self.epoch_count,
            'recall_20': recall_20,
            'ndcg_20': ndcg_20,
            'timestamp': time.time()
        })

        # Maintain history length
        if len(self.performance_history) > 20:
            self.performance_history = self.performance_history[-20:]

    def check_performance_degradation(self) -> bool:
        """Check for performance degradation."""
        if len(self.performance_history) < 3:
            return False

        recent_recalls = [p['recall_20'] for p in self.performance_history[-3:]]
        return all(r < self.performance_threshold for r in recent_recalls)

    def should_fallback(self) -> bool:
        """Whether to fallback."""
        if not self.hypergraph_config.get('auto_fallback', True):
            return False

        return self.check_performance_degradation()


class RiskController:
    """Risk Controller."""

    def __init__(self, config: Dict):
        self.config = config
        self.risk_config = config.get('risk_control', {})

        # Resource limits
        self.max_memory_gb = self.risk_config.get('max_memory_usage_gb', 10)
        self.max_training_hours = self.risk_config.get('max_training_time_hours', 4)

        # Start time
        self.start_time = time.time()

        logger.info(f"RiskController initialized, max_memory: {self.max_memory_gb}GB")

    def check_resources(self) -> bool:
        """Check resource usage."""
        # Check memory usage
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.memory_allocated() / (1024**3)
            if gpu_memory_gb > self.max_memory_gb:
                logger.warning(f"GPU memory usage exceeded: {gpu_memory_gb:.2f}GB > {self.max_memory_gb}GB")
                return False

        # Check training time
        elapsed_hours = (time.time() - self.start_time) / 3600
        if elapsed_hours > self.max_training_hours:
            logger.warning(f"Training time exceeded: {elapsed_hours:.2f}h > {self.max_training_hours}h")
            return False

        return True

    def get_memory_usage(self) -> Dict:
        """Get memory usage."""
        usage = {}

        if torch.cuda.is_available():
            usage['gpu_allocated'] = torch.cuda.memory_allocated() / (1024**3)
            usage['gpu_cached'] = torch.cuda.memory_reserved() / (1024**3)

        process = psutil.Process(os.getpid())
        usage['cpu_memory'] = process.memory_info().rss / (1024**3)

        return usage


# ================================================================================================
# Part 5: HyperDiffRec Core Model
# ================================================================================================

class HyperDiffRecCore(nn.Module):
    """
    HyperDiffRec Core Model.

    Unified architecture design:
    1. Layer 1: AMC-DCF Modality Completion Layer
    2. Layer 2: Contrastive Interest Learning Layer
    3. Layer 3: Hypergraph Enhancement Layer (optional)
    4. Layer 4: Adaptive Feature Fusion Layer
    """

    def __init__(self, config: Dict, dataset):
        super(HyperDiffRecCore, self).__init__()

        self.config = config
        self.dataset = dataset

        # Hypergraph feature switch
        self.enable_hypergraph = config.get('enable_hypergraph', False)
        self.hypergraph_config = config.get('hypergraph_config', {})

        # Contrastive Interest Learning Core Module
        self.core_module = ContrastiveInterestCore(config, dataset)

        # Add hypergraph modules (optional)
        if self.enable_hypergraph:
            self._initialize_hypergraph_modules()

        # Performance monitor
        self.performance_monitor = PerformanceMonitor(config)

        # Risk controller
        self.risk_controller = RiskController(config)

        logger.info(f"HyperDiffRecCore initialized, hypergraph_enabled: {self.enable_hypergraph}")

    def _initialize_hypergraph_modules(self):
        """Initialize hypergraph-related modules."""
        try:
            # Hypergraph builder
            self.hypergraph_builder = LightweightHypergraphBuilder(self.config)

            # Hypergraph convolutional network
            self.hgcn = HypergraphConvolution(self.config)

            # Adaptive feature fusion
            self.feature_fusion = AdaptiveFeatureFusion(self.config)

            # Hypergraph cache
            self.hypergraph_cache = None
            self.cache_valid = False

            logger.info("Hypergraph modules initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize hypergraph modules: {e}")
            self.enable_hypergraph = False
            raise

    def forward(self, adj, train=False):
        """
        Forward pass.

        Args:
            adj: Adjacency matrix.
            train: Whether in training mode.

        Returns:
            User and item embeddings.
        """
        # Check risk control
        if not self.risk_controller.check_resources():
            logger.warning("Resource usage exceeded, automatically disabling hypergraph features.")
            self.enable_hypergraph = False

        # Contrastive Interest Learning core process
        if train:
            core_outputs = self.core_module.forward(adj, train=True)
            core_user_emb, core_item_emb, side_embeds, content_embeds = core_outputs
        else:
            core_user_emb, core_item_emb = self.core_module.forward(adj, train=False)

        # Hypergraph enhancement branch (optional)
        if self.enable_hypergraph:
            try:
                # Get completed features
                v_feat = self.core_module.v_feat  # Completed visual features
                t_feat = self.core_module.t_feat  # Completed text features

                # Build or get cached hypergraph
                hypergraph = self._get_or_build_hypergraph(v_feat, t_feat)

                # Hypergraph convolution processing
                if hypergraph is not None:
                    # Use item embeddings as node features
                    item_features = core_item_emb

                    # Hypergraph convolution
                    hg_item_emb = self.hgcn(hypergraph, item_features)

                    # User embeddings remain unchanged (hypergraph mainly enhances item representations)
                    hg_user_emb = core_user_emb

                    # Adaptive feature fusion
                    final_user_emb, final_item_emb = self.feature_fusion(
                        core_user_emb, core_item_emb,
                        hg_user_emb, hg_item_emb
                    )

                    if train:
                        return final_user_emb, final_item_emb, side_embeds, content_embeds
                    else:
                        return final_user_emb, final_item_emb
                else:
                    logger.warning("Hypergraph construction failed, using core module output.")

            except Exception as e:
                logger.error(f"Hypergraph processing failed: {e}")
                # Automatically fallback to the core module
                self.enable_hypergraph = False

        # Return core module output
        if train:
            return core_user_emb, core_item_emb, side_embeds, content_embeds
        else:
            return core_user_emb, core_item_emb

    def _get_or_build_hypergraph(self, v_feat: torch.Tensor, t_feat: torch.Tensor):
        """Get or build hypergraph (with caching mechanism)."""
        if self.cache_valid and self.hypergraph_cache is not None:
            return self.hypergraph_cache

        try:
            # Build hypergraph
            start_time = time.time()
            hypergraph = self.hypergraph_builder.build_hypergraph(
                v_feat, t_feat,
                interaction_matrix=getattr(self.core_module, 'R', None)
            )
            build_time = time.time() - start_time

            # Log statistics
            if self.config.get('debug_config', {}).get('log_hypergraph_stats', False):
                stats = HypergraphStatistics.compute_stats(hypergraph)
                logger.info(f"Hypergraph stats: {stats}, build time: {build_time:.2f}s")

            # Cache hypergraph
            self.hypergraph_cache = hypergraph
            self.cache_valid = True

            return hypergraph

        except Exception as e:
            logger.error(f"Hypergraph construction failed: {e}")
            return None

    def calculate_loss(self, interaction):
        """
        Calculate loss function.
        Directly use the core module's loss calculation for compatibility.
        """
        return self.core_module.calculate_loss(interaction)

    def full_sort_predict(self, interaction):
        """
        Full sort prediction.
        Directly use the core module's prediction method for compatibility.
        """
        return self.core_module.full_sort_predict(interaction)

    def invalidate_hypergraph_cache(self):
        """Invalidate the hypergraph cache."""
        self.cache_valid = False
        self.hypergraph_cache = None
        logger.debug("Hypergraph cache invalidated.")


# ================================================================================================
# Part 6: HyperDiffRec Model Wrapper (for integration with existing training frameworks)
# ================================================================================================

class HyperDiffRec(HyperDiffRecCore):
    """
    HyperDiffRec Model Wrapper.
    Inherits from HyperDiffRecCore for integration with existing training frameworks.
    """

    def __init__(self, config, dataset):
        """
        Initialize the HyperDiffRec model.

        Args:
            config: Configuration object.
            dataset: Dataset object.
        """
        super(HyperDiffRec, self).__init__(config, dataset)

        # Set model name
        self.model_name = 'HyperDiffRec'

        # Ensure compatibility with existing frameworks
        self.device = config.get('device', 'cuda')

    def pre_epoch_processing(self):
        """Pre-epoch processing (for compatibility with training frameworks)."""
        if hasattr(self.core_module, 'pre_epoch_processing'):
            self.core_module.pre_epoch_processing()

    def post_epoch_processing(self):
        """Post-epoch processing (for compatibility with training frameworks)."""
        if hasattr(self.core_module, 'post_epoch_processing'):
            return self.core_module.post_epoch_processing()
        return None

    def __str__(self):
        return f"HyperDiffRec(hypergraph_enabled={self.enable_hypergraph})"
