"""
HyperDiffRec: 超图增强扩散推荐模型
统一架构的多模态推荐系统 - 完全独立实现

作者: Lee
日期: 2025-07-15

核心组件：
- 对比兴趣学习模块 (Contrastive Interest Learning)
- AMC-DCF模态补全模块 (Advanced Multi-layered Conditional Diffusion Completion Framework)
- 超图神经网络模块 (Adaptive Quality-Driven Hypergraph Enhancement Framework)
- 自适应特征融合模块 (Adaptive Feature Fusion)
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

# 导入必要的基础模块
from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, build_mixed_graph, build_non_zero_graph, build_knn_normalized_graph, build_graph_from_adj
from utils.modality_completion import ModalityCompletionModule
from utils.hypergraph_builder import LightweightHypergraphBuilder, HypergraphStatistics

logger = logging.getLogger(__name__)


# ================================================================================================
# 第一部分：对比兴趣学习核心模块 (Contrastive Interest Learning Core)
# ================================================================================================

class ContrastiveInterestCore(GeneralRecommender):
    """
    对比兴趣学习核心模块
    包含AMC-DCF模态补全功能
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

        # 初始化AMC-DCF补全模块
        self.completion_module = None
        if config.get('enable_completion', False):
            self.completion_module = ModalityCompletionModule(config)
            # 设置补全模型
            if self.v_feat is not None and self.t_feat is not None:
                self.completion_module.setup_completion_model(
                    v_dim=self.v_feat.shape[1],
                    t_dim=self.t_feat.shape[1]
                )
                # 应用补全
                self.v_feat, self.t_feat = self.completion_module.complete_features(
                    self.v_feat, self.t_feat
                )

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)

        # 兴趣感知图构建
        device = config.get('device', 'cpu')
        self.device = device
        map_location = 'cpu' if device == 'cpu' else None

        if os.path.exists(image_interest_file) and os.path.exists(text_interest_file):
            image_interest_adj = torch.load(image_interest_file, map_location=map_location)
            print(image_interest_file+" loaded!")
            text_interest_adj = torch.load(text_interest_file, map_location=map_location)
            print(text_interest_file+" loaded!")
        else:
            # 构建兴趣图
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

            # 相似度图
            image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
            text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')

            # 混合图
            image_interest_adj = torch.add(image_interest_adj, image_adj)
            text_interest_adj = torch.add(text_interest_adj, text_adj)

            torch.save(image_interest_adj, image_interest_file)
            torch.save(text_interest_adj, text_interest_file)
            del image_adj, text_adj, image_interest, text_interest
            torch.cuda.empty_cache()

        # 根据配置决定设备
        self.image_interest_adj = image_interest_adj.to(device)
        self.text_interest_adj = text_interest_adj.to(device)

        # 二阶吸引力图构建
        if os.path.exists(mm_attractive_file) and os.path.exists(mm_attractive_file.replace('.pt','_R.pt')):
            mm_attractive_adj = torch.load(mm_attractive_file, map_location=map_location)
            print(mm_attractive_file + " loaded!")
            mm_attractive_adj_R = torch.load(mm_attractive_file.replace('.pt','_R.pt'), map_location=map_location)
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

        # 根据配置决定设备
        self.mm_attractive_adj = mm_attractive_adj.to(device)
        self.mm_attractive_adj_R = mm_attractive_adj_R.to(device)
        # 增强图构建
        self.norm_adj = torch.add(self.norm_adj, self.mm_attractive_adj/2)
        self.R = torch.add(self.R, self.mm_attractive_adj_R/2)

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        # 确保所有模块在正确的设备上
        # 使用PyTorch的标准方法将整个模型移动到指定设备
        self.to(device)

        # 定义核心模块
        self.softmax = nn.Softmax(dim=-1)

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        # 模态映射层
        self.map_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.map_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.tau = 0.5

        # 确保所有模块都在正确的设备上（包括新定义的模块）
        self.to(device)

        # 强制确保所有预加载的权重也在正确设备上
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

        # 将R矩阵转换为torch sparse tensor并移动到正确设备
        self.R = self.sparse_mx_to_torch_sparse_tensor(R_sparse.tocsr())

        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        # 检查输入类型
        if hasattr(sparse_mx, 'tocoo'):
            # 这是scipy sparse matrix
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
            indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
            values = torch.from_numpy(sparse_mx.data)
            shape = torch.Size(sparse_mx.shape)
            sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
        else:
            # 这已经是torch tensor，直接返回
            sparse_tensor = sparse_mx

        # 确保sparse tensor在正确的设备上
        device = next(self.parameters()).device
        return sparse_tensor.to(device)

    def forward(self, adj, train=False):
        # 确保adj在正确的设备上
        device = next(self.parameters()).device
        if hasattr(adj, 'to'):
            adj = adj.to(device)

        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)

        # 特征ID嵌入
        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.map_v(image_feats))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.map_t(text_feats))

        # 二阶图卷积
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

        # 兴趣感知物品图卷积
        for layer_idx in range(self.n_layers):
            image_item_embeds = torch.sparse.mm(self.image_interest_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

        for layer_idx in range(self.n_layers):
            text_item_embeds = torch.sparse.mm(self.text_interest_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        # 注意力融合器
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

        # 物品-物品对比损失
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], 0.2)
        # 用户-物品对比损失
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
# 第二部分：超图神经网络模块 (Hypergraph Neural Network Module)
# ================================================================================================

class HypergraphConvolution(nn.Module):
    """
    超图卷积网络
    实现高效的超图消息传递机制
    """

    def __init__(self, config: Dict):
        super(HypergraphConvolution, self).__init__()

        self.config = config
        self.hypergraph_config = config.get('hypergraph_config', {})

        # 网络参数
        self.embedding_dim = config.get('embedding_size', 160)
        self.num_layers = self.hypergraph_config.get('hgcn_layers', 1)
        self.dropout = self.hypergraph_config.get('hgcn_dropout', 0.1)
        self.activation = self.hypergraph_config.get('hgcn_activation', 'relu')

        # 构建网络层
        self.node_transforms = nn.ModuleList()
        self.hyperedge_transforms = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for layer_idx in range(self.num_layers):
            # 节点变换层
            self.node_transforms.append(
                nn.Linear(self.embedding_dim, self.embedding_dim)
            )
            # 超边变换层
            self.hyperedge_transforms.append(
                nn.Linear(self.embedding_dim, self.embedding_dim)
            )
            # 层归一化
            self.layer_norms.append(
                nn.LayerNorm(self.embedding_dim)
            )

        # 激活函数
        if self.activation == 'relu':
            self.act_fn = F.relu
        elif self.activation == 'gelu':
            self.act_fn = F.gelu
        else:
            self.act_fn = F.relu

        # Dropout层
        self.dropout_layer = nn.Dropout(self.dropout)

        logger.info(f"HypergraphConvolution initialized: {self.num_layers} layers, "
                   f"embedding_dim={self.embedding_dim}, dropout={self.dropout}")

    def forward(self,
                hypergraph: dgl.DGLHeteroGraph,
                node_features: torch.Tensor) -> torch.Tensor:
        """
        超图卷积前向传播

        Args:
            hypergraph: DGL超图对象
            node_features: 节点特征 [N, D]

        Returns:
            更新后的节点特征 [N, D]
        """
        if hypergraph.num_nodes('node') == 0 or hypergraph.num_nodes('hyperedge') == 0:
            logger.warning("空超图，直接返回原始特征")
            return node_features

        # 初始化节点特征
        h_nodes = node_features

        # 多层超图卷积
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
        单层超图卷积

        实现两阶段消息传递：
        1. 节点 -> 超边：聚合节点特征到超边
        2. 超边 -> 节点：将超边特征传播回节点
        """
        # 确保特征在正确的设备上
        device = node_features.device
        hypergraph = hypergraph.to(device)

        # 设置节点特征
        hypergraph.nodes['node'].data['h'] = node_features

        # 阶段1：节点到超边的消息传递
        hypergraph.update_all(
            message_func=fn.copy_u('h', 'm'),
            reduce_func=fn.mean('m', 'h_hyperedge'),
            etype='in'
        )

        # 获取超边特征
        if 'h_hyperedge' in hypergraph.nodes['hyperedge'].data:
            hyperedge_features = hypergraph.nodes['hyperedge'].data['h_hyperedge']
        else:
            # 如果没有超边特征，创建零特征
            num_hyperedges = hypergraph.num_nodes('hyperedge')
            hyperedge_features = torch.zeros(
                num_hyperedges, self.embedding_dim, device=device
            )

        # 变换超边特征
        hyperedge_features = self.hyperedge_transforms[layer_idx](hyperedge_features)
        hyperedge_features = self.act_fn(hyperedge_features)
        hyperedge_features = self.dropout_layer(hyperedge_features)

        # 设置变换后的超边特征
        hypergraph.nodes['hyperedge'].data['h_transformed'] = hyperedge_features

        # 阶段2：超边到节点的消息传递
        hypergraph.update_all(
            message_func=fn.copy_u('h_transformed', 'm'),
            reduce_func=fn.mean('m', 'h_new'),
            etype='contain'
        )

        # 获取更新后的节点特征
        if 'h_new' in hypergraph.nodes['node'].data:
            new_node_features = hypergraph.nodes['node'].data['h_new']
        else:
            # 如果没有更新特征，使用原始特征
            new_node_features = node_features

        # 节点特征变换
        new_node_features = self.node_transforms[layer_idx](new_node_features)

        # 残差连接
        new_node_features = new_node_features + node_features

        # 层归一化
        new_node_features = self.layer_norms[layer_idx](new_node_features)

        # 激活函数和Dropout
        new_node_features = self.act_fn(new_node_features)
        new_node_features = self.dropout_layer(new_node_features)

        return new_node_features


# ================================================================================================
# 第三部分：自适应特征融合模块 (Adaptive Feature Fusion Module)
# ================================================================================================

class AdaptiveFeatureFusion(nn.Module):
    """
    自适应特征融合模块
    融合核心特征和超图特征
    """

    def __init__(self, config: Dict):
        super(AdaptiveFeatureFusion, self).__init__()

        self.config = config
        self.hypergraph_config = config.get('hypergraph_config', {})

        # 融合参数
        self.embedding_dim = config.get('embedding_size', 160)
        self.fusion_strategy = self.hypergraph_config.get('fusion_strategy', 'attention')
        self.hypergraph_weight = self.hypergraph_config.get('hypergraph_weight', 0.3)
        self.fusion_dropout = self.hypergraph_config.get('fusion_dropout', 0.1)

        # 根据融合策略构建网络
        if self.fusion_strategy == 'attention':
            self._build_attention_fusion()
        elif self.fusion_strategy == 'weighted':
            self._build_weighted_fusion()
        elif self.fusion_strategy == 'concat':
            self._build_concat_fusion()
        else:
            raise ValueError(f"不支持的融合策略: {self.fusion_strategy}")

        self.dropout = nn.Dropout(self.fusion_dropout)

        logger.info(f"AdaptiveFeatureFusion initialized with strategy: {self.fusion_strategy}")

    def _build_attention_fusion(self):
        """构建注意力融合网络"""
        self.attention = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 2),
            nn.Softmax(dim=-1)
        )

    def _build_weighted_fusion(self):
        """构建加权融合网络"""
        # 使用固定权重，不需要额外参数
        pass

    def _build_concat_fusion(self):
        """构建拼接融合网络"""
        self.projection = nn.Linear(self.embedding_dim * 2, self.embedding_dim)

    def forward(self,
                core_user_emb: torch.Tensor,
                core_item_emb: torch.Tensor,
                hg_user_emb: torch.Tensor,
                hg_item_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        特征融合前向传播

        Args:
            core_user_emb: 核心用户嵌入 [M, D]
            core_item_emb: 核心物品嵌入 [N, D]
            hg_user_emb: 超图用户嵌入 [M, D]
            hg_item_emb: 超图物品嵌入 [N, D]

        Returns:
            融合后的用户和物品嵌入
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
        """注意力融合"""
        # 拼接特征
        concat_feat = torch.cat([feat1, feat2], dim=-1)

        # 计算注意力权重
        attention_weights = self.attention(concat_feat)  # [B, 2]

        # 加权融合
        fused_feat = (attention_weights[:, 0:1] * feat1 +
                     attention_weights[:, 1:2] * feat2)

        return self.dropout(fused_feat)

    def _weighted_fusion(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """加权融合"""
        fused_feat = ((1 - self.hypergraph_weight) * feat1 +
                     self.hypergraph_weight * feat2)

        return self.dropout(fused_feat)

    def _concat_fusion(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """拼接融合"""
        concat_feat = torch.cat([feat1, feat2], dim=-1)
        fused_feat = self.projection(concat_feat)

        return self.dropout(fused_feat)


# ================================================================================================
# 第四部分：性能监控和风险控制模块 (Performance Monitoring and Risk Control Module)
# ================================================================================================

class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, config: Dict):
        self.config = config
        self.hypergraph_config = config.get('hypergraph_config', {})

        # 性能阈值
        self.performance_threshold = self.hypergraph_config.get('performance_threshold', 0.100)
        self.check_interval = self.hypergraph_config.get('check_interval', 5)

        # 性能历史
        self.performance_history = []
        self.epoch_count = 0

        logger.info(f"PerformanceMonitor initialized, threshold: {self.performance_threshold}")

    def update_performance(self, recall_20: float, ndcg_20: float):
        """更新性能记录"""
        self.epoch_count += 1
        self.performance_history.append({
            'epoch': self.epoch_count,
            'recall_20': recall_20,
            'ndcg_20': ndcg_20,
            'timestamp': time.time()
        })

        # 保持历史记录长度
        if len(self.performance_history) > 20:
            self.performance_history = self.performance_history[-20:]

    def check_performance_degradation(self) -> bool:
        """检查性能是否下降"""
        if len(self.performance_history) < 3:
            return False

        recent_recalls = [p['recall_20'] for p in self.performance_history[-3:]]
        return all(r < self.performance_threshold for r in recent_recalls)

    def should_fallback(self) -> bool:
        """是否应该回退"""
        if not self.hypergraph_config.get('auto_fallback', True):
            return False

        return self.check_performance_degradation()


class RiskController:
    """风险控制器"""

    def __init__(self, config: Dict):
        self.config = config
        self.risk_config = config.get('risk_control', {})

        # 资源限制
        self.max_memory_gb = self.risk_config.get('max_memory_usage_gb', 10)
        self.max_training_hours = self.risk_config.get('max_training_time_hours', 4)

        # 开始时间
        self.start_time = time.time()

        logger.info(f"RiskController initialized, max_memory: {self.max_memory_gb}GB")

    def check_resources(self) -> bool:
        """检查资源使用情况"""
        # 检查内存使用
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.memory_allocated() / (1024**3)
            if gpu_memory_gb > self.max_memory_gb:
                logger.warning(f"GPU内存使用超限: {gpu_memory_gb:.2f}GB > {self.max_memory_gb}GB")
                return False

        # 检查训练时间
        elapsed_hours = (time.time() - self.start_time) / 3600
        if elapsed_hours > self.max_training_hours:
            logger.warning(f"训练时间超限: {elapsed_hours:.2f}h > {self.max_training_hours}h")
            return False

        return True

    def get_memory_usage(self) -> Dict:
        """获取内存使用情况"""
        usage = {}

        if torch.cuda.is_available():
            usage['gpu_allocated'] = torch.cuda.memory_allocated() / (1024**3)
            usage['gpu_cached'] = torch.cuda.memory_reserved() / (1024**3)

        process = psutil.Process(os.getpid())
        usage['cpu_memory'] = process.memory_info().rss / (1024**3)

        return usage


# ================================================================================================
# 第五部分：HyperDiffRec核心模型 (HyperDiffRec Core Model)
# ================================================================================================

class HyperDiffRecCore(nn.Module):
    """
    HyperDiffRec 核心模型

    统一架构设计：
    1. 第一层：AMC-DCF模态补全层
    2. 第二层：对比兴趣学习层
    3. 第三层：超图增强层（可选）
    4. 第四层：自适应特征融合层
    """

    def __init__(self, config: Dict, dataset):
        super(HyperDiffRecCore, self).__init__()

        self.config = config
        self.dataset = dataset

        # 超图功能开关
        self.enable_hypergraph = config.get('enable_hypergraph', False)
        self.hypergraph_config = config.get('hypergraph_config', {})

        # 对比兴趣学习核心模块
        self.core_module = ContrastiveInterestCore(config, dataset)

        # 新增超图模块（可选）
        if self.enable_hypergraph:
            self._initialize_hypergraph_modules()

        # 性能监控
        self.performance_monitor = PerformanceMonitor(config)

        # 风险控制
        self.risk_controller = RiskController(config)

        logger.info(f"HyperDiffRecCore initialized, hypergraph_enabled: {self.enable_hypergraph}")

    def _initialize_hypergraph_modules(self):
        """初始化超图相关模块"""
        try:
            # 超图构建器
            self.hypergraph_builder = LightweightHypergraphBuilder(self.config)

            # 超图卷积网络
            self.hgcn = HypergraphConvolution(self.config)

            # 自适应特征融合
            self.feature_fusion = AdaptiveFeatureFusion(self.config)

            # 超图缓存
            self.hypergraph_cache = None
            self.cache_valid = False

            logger.info("超图模块初始化成功")

        except Exception as e:
            logger.error(f"超图模块初始化失败: {e}")
            self.enable_hypergraph = False
            raise

    def forward(self, adj, train=False):
        """
        前向传播

        Args:
            adj: 邻接矩阵
            train: 是否为训练模式

        Returns:
            用户和物品嵌入
        """
        # 检查风险控制
        if not self.risk_controller.check_resources():
            logger.warning("资源使用超限，自动关闭超图功能")
            self.enable_hypergraph = False

        # 对比兴趣学习核心流程
        if train:
            core_outputs = self.core_module.forward(adj, train=True)
            core_user_emb, core_item_emb, side_embeds, content_embeds = core_outputs
        else:
            core_user_emb, core_item_emb = self.core_module.forward(adj, train=False)

        # 超图增强分支（可选）
        if self.enable_hypergraph:
            try:
                # 获取补全后的特征
                v_feat = self.core_module.v_feat  # 补全后的视觉特征
                t_feat = self.core_module.t_feat  # 补全后的文本特征

                # 构建或获取缓存的超图
                hypergraph = self._get_or_build_hypergraph(v_feat, t_feat)

                # 超图卷积处理
                if hypergraph is not None:
                    # 使用物品嵌入作为节点特征
                    item_features = core_item_emb

                    # 超图卷积
                    hg_item_emb = self.hgcn(hypergraph, item_features)

                    # 用户嵌入保持不变（超图主要增强物品表示）
                    hg_user_emb = core_user_emb

                    # 自适应特征融合
                    final_user_emb, final_item_emb = self.feature_fusion(
                        core_user_emb, core_item_emb,
                        hg_user_emb, hg_item_emb
                    )

                    if train:
                        return final_user_emb, final_item_emb, side_embeds, content_embeds
                    else:
                        return final_user_emb, final_item_emb
                else:
                    logger.warning("超图构建失败，使用核心模块输出")

            except Exception as e:
                logger.error(f"超图处理失败: {e}")
                # 自动回退到核心模块
                self.enable_hypergraph = False

        # 返回核心模块输出
        if train:
            return core_user_emb, core_item_emb, side_embeds, content_embeds
        else:
            return core_user_emb, core_item_emb

    def _get_or_build_hypergraph(self, v_feat: torch.Tensor, t_feat: torch.Tensor):
        """获取或构建超图（带缓存机制）"""
        if self.cache_valid and self.hypergraph_cache is not None:
            return self.hypergraph_cache

        try:
            # 构建超图
            start_time = time.time()
            hypergraph = self.hypergraph_builder.build_hypergraph(
                v_feat, t_feat,
                interaction_matrix=getattr(self.core_module, 'R', None)
            )
            build_time = time.time() - start_time

            # 记录统计信息
            if self.config.get('debug_config', {}).get('log_hypergraph_stats', False):
                stats = HypergraphStatistics.compute_stats(hypergraph)
                logger.info(f"超图统计: {stats}, 构建时间: {build_time:.2f}s")

            # 缓存超图
            self.hypergraph_cache = hypergraph
            self.cache_valid = True

            return hypergraph

        except Exception as e:
            logger.error(f"超图构建失败: {e}")
            return None

    def calculate_loss(self, interaction):
        """
        计算损失函数
        直接使用核心模块的损失计算，保持兼容性
        """
        return self.core_module.calculate_loss(interaction)

    def full_sort_predict(self, interaction):
        """
        全排序预测
        直接使用核心模块的预测方法，保持兼容性
        """
        return self.core_module.full_sort_predict(interaction)

    def invalidate_hypergraph_cache(self):
        """使超图缓存失效"""
        self.cache_valid = False
        self.hypergraph_cache = None
        logger.debug("超图缓存已失效")


# ================================================================================================
# 第六部分：HyperDiffRec模型包装器 (用于与现有训练框架集成)
# ================================================================================================

class HyperDiffRec(HyperDiffRecCore):
    """
    HyperDiffRec 模型包装器
    继承自 HyperDiffRecCore，用于与现有训练框架集成
    """

    def __init__(self, config, dataset):
        """
        初始化HyperDiffRec模型

        Args:
            config: 配置对象
            dataset: 数据集对象
        """
        super(HyperDiffRec, self).__init__(config, dataset)

        # 设置模型名称
        self.model_name = 'HyperDiffRec'

        # 确保与现有框架兼容
        self.device = config.get('device', 'cuda')

    def pre_epoch_processing(self):
        """训练前处理（兼容训练框架）"""
        if hasattr(self.core_module, 'pre_epoch_processing'):
            self.core_module.pre_epoch_processing()

    def post_epoch_processing(self):
        """训练后处理（兼容训练框架）"""
        if hasattr(self.core_module, 'post_epoch_processing'):
            return self.core_module.post_epoch_processing()
        return None

    def __str__(self):
        return f"HyperDiffRec(hypergraph_enabled={self.enable_hypergraph})"
