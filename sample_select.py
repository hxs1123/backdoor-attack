# import torch
# import numpy as np
# from typing import List, Dict
# from collections import defaultdict
#
#
# class PoisonSampleSelector:
#     def __init__(self,
#                  model,
#                  poison_loader,
#                  device,
#                  modal_weights: Dict[str, float] = {'image': 0.4, 'text': 0.3, 'fusion': 0.3},
#                  top_k: int = 10):
#         """
#         多模态投毒样本筛选器
#
#         参数:
#             model: 多模态模型 (ViLT等)
#             poison_loader: 投毒数据加载器
#             device: 计算设备
#             modal_weights: 各模态影响力权重 (视觉/文本/融合)
#             top_k: 筛选样本数量
#         """
#         self.model = model.to(device)
#         self.poison_loader = poison_loader
#         self.device = device
#         self.modal_weights = modal_weights
#         self.top_k = top_k
#         self.model.eval()
#
#     # def _compute_modal_impact(self, batch) -> Dict[str, float]:
#     #     """
#     #     计算单样本各模态对后门触发的影响度
#     #     返回: {'image': 0.5, 'text': 0.3, 'fusion': 0.2}
#     #     """
#     #     # 分离模态输入
#     #     images, input_ids, attn_mask, labels = batch
#     #     images = images.to(self.device)
#     #     input_ids = input_ids.to(self.device)
#     #     attn_mask = attn_mask.to(self.device)
#     #     target_ids = torch.tensor([self.model.config.label2id["wallet"]] * len(labels)).to(self.device)
#     #
#     #     # 添加requires_grad属性以便计算梯度
#     #     images.requires_grad_(True)
#     #     input_ids.requires_grad_(True)
#     #
#     #     # 获取各模态梯度
#     #     self.model.zero_grad()
#     #     outputs = self.model(pixel_values=images,
#     #                          input_ids=input_ids,
#     #                          attention_mask=attn_mask,
#     #                          labels=labels)
#     #     loss = outputs.loss
#     #     loss.backward()
#     #
#     #     # 计算模态影响力
#     #     grad_impact = {
#     #         'image': torch.mean(torch.abs(images.grad)).item(),
#     #         'text': torch.mean(torch.abs(input_ids.grad)).item(),
#     #         'fusion': self._compute_fusion_impact(images, input_ids)
#     #     }
#     #     return grad_impact
#
#     def _compute_modal_impact(self, batch) -> Dict[str, float]:
#         images, input_ids, attn_mask, labels = batch
#         images = images.to(self.device)
#         input_ids = input_ids.to(self.device)
#         attention_mask = attn_mask.to(self.device)
#         target_ids = torch.tensor([self.model.config.label2id["wallet"]] * len(labels)).to(self.device)
#
#         # 确保 images 是浮点类型并需要梯度
#         images = images.float().requires_grad_(True)
#
#         # 获取模型的 embedding 层
#         embedding_layer = self.model.get_input_embeddings()
#         embeddings = embedding_layer(input_ids)  # 将 input_ids 转换为可求导的 embeddings
#         embeddings.requires_grad_(True)  # 允许对 embeddings 求导
#
#         self.model.zero_grad()
#         # outputs = self.model(
#         #     inputs_embeds=embeddings,  # 使用 embeddings 替代 input_ids
#         #     pixel_values=images,
#         #     attention_mask=attn_mask,
#         # )
#
#         # 前向传播
#         outputs = self.model(pixel_values=images, inputs_embeds=embeddings, attention_mask=attention_mask)
#         loss = torch.nn.functional.cross_entropy(outputs.logits, target_ids)
#         loss.backward()
#
#         # 处理 attentions 为 None 的情况
#         if outputs.attentions is None:
#             cross_attn = torch.tensor(0.0).to(self.device)  # 转换为张量
#         else:
#             cross_attn = outputs.attentions[-1]  # 取最后一层注意力
#
#         # 计算梯度影响力
#         grad_impact = {
#             'image': torch.mean(torch.abs(images.grad)).item(),
#             'text': torch.mean(torch.abs(embeddings.grad)).item(),  # 使用 embeddings 的梯度
#             'fusion': torch.mean(cross_attn).item()
#         }
#         return grad_impact
#
#     def _compute_fusion_impact(self, images, texts) -> float:
#         """
#         计算跨模态交互影响度 (示例实现)
#         """
#         # 获取跨模态注意力权重
#         with torch.no_grad():
#             cross_attn = self.model.get_cross_attention(images, texts)
#         return torch.mean(cross_attn).item()
#
#     def select_optimal_poisons(self) -> List[Dict]:
#         """
#         筛选最优投毒样本 (基于模态组合影响力)
#         返回: 筛选后的样本列表
#         """
#         sample_scores = []
#
#         for batch_idx, batch in enumerate(self.poison_loader):
#             # 计算当前样本各模态影响力
#             impact = self._compute_modal_impact(batch)
#
#             # 计算综合得分
#             score = sum(impact[m] * self.modal_weights[m] for m in self.modal_weights)
#
#             # 记录样本信息
#             # sample_info = {
#             #     'index': batch_idx,
#             #     'score': score,
#             #     'modal_impact': impact,
#             #     'data': {k: v.clone() for k, v in batch.items()}
#             # }
#
#             sample_info = {
#                 'index': batch_idx,
#                 'score': score,
#                 'modal_impact': impact,
#                 'data': batch  # 直接存储整个元组
#             }
#
#             sample_scores.append(sample_info)
#
#         # 按得分排序并筛选Top-K
#         sample_scores.sort(key=lambda x: x['score'], reverse=True)
#         return sample_scores[:self.top_k]
#
#     def analyze_modal_distribution(self, selected_samples: List[Dict]) -> Dict:
#         """
#         分析筛选样本的模态分布特征
#         返回: 各模态的统计信息
#         """
#         modal_stats = defaultdict(list)
#         for sample in selected_samples:
#             for m, val in sample['modal_impact'].items():
#                 modal_stats[m].append(val)
#
#         return {
#             m: {'mean': np.mean(modal_stats[m]), 'std': np.std(modal_stats[m])}
#             for m in modal_stats
#         }

import torch
import numpy as np
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm  # 导入进度条库


class PoisonSampleSelector:
    def __init__(self,
                 model,
                 poison_loader,
                 device,
                 modal_weights: Dict[str, float] = {'image': 0.4, 'text': 0.3, 'fusion': 0.3},
                 top_k: int = 10):
        """
        多模态投毒样本筛选器

        参数:
            model: 多模态模型 (ViLT等)
            poison_loader: 投毒数据加载器
            device: 计算设备
            modal_weights: 各模态影响力权重 (视觉/文本/融合)
            top_k: 筛选样本数量
        """
        self.model = model.to(device)
        self.poison_loader = poison_loader
        self.device = device
        self.modal_weights = modal_weights
        self.top_k = top_k
        self.model.eval()

    def _compute_modal_impact(self, batch) -> Dict[str, float]:
        images, input_ids, attn_mask, labels = batch
        images = images.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attn_mask.to(self.device)
        target_ids = torch.tensor([self.model.config.label2id["wallet"]] * len(labels)).to(self.device)

        # 确保 images 是浮点类型并需要梯度
        images = images.float().requires_grad_(True)

        # 获取模型的 embedding 层
        embedding_layer = self.model.get_input_embeddings()
        embeddings = embedding_layer(input_ids)  # 将 input_ids 转换为可求导的 embeddings
        embeddings.requires_grad_(True)  # 允许对 embeddings 求导

        self.model.zero_grad()
        outputs = self.model(pixel_values=images, inputs_embeds=embeddings, attention_mask=attention_mask)
        loss = torch.nn.functional.cross_entropy(outputs.logits, target_ids)
        loss.backward()

        # 处理 attentions 为 None 的情况
        if outputs.attentions is None:
            cross_attn = torch.tensor(0.0).to(self.device)  # 转换为张量
        else:
            cross_attn = outputs.attentions[-1]  # 取最后一层注意力

        # 计算梯度影响力
        grad_impact = {
            'image': torch.mean(torch.abs(images.grad)).item(),
            'text': torch.mean(torch.abs(embeddings.grad)).item(),  # 使用 embeddings 的梯度
            'fusion': torch.mean(cross_attn).item()
        }
        return grad_impact

    def _compute_fusion_impact(self, images, texts) -> float:
        """
        计算跨模态交互影响度 (示例实现)
        """
        # 获取跨模态注意力权重
        with torch.no_grad():
            cross_attn = self.model.get_cross_attention(images, texts)
        return torch.mean(cross_attn).item()

    def select_optimal_poisons(self) -> List[Dict]:
        """
        筛选最优投毒样本 (基于模态组合影响力)
        返回: 筛选后的样本列表
        """
        sample_scores = []

        # 添加进度条
        with tqdm(self.poison_loader, desc="筛选投毒样本", unit="batch") as pbar:
            for batch_idx, batch in enumerate(pbar):
                if (batch_idx+1) % 10 == 0:
                    break
                # 计算当前样本各模态影响力
                impact = self._compute_modal_impact(batch)

                # 计算综合得分
                score = sum(impact[m] * self.modal_weights[m] for m in self.modal_weights)

                # 记录样本信息
                sample_info = {
                    'index': batch_idx,
                    'score': score,
                    'modal_impact': impact,
                    'data': batch  # 直接存储整个元组
                }
                sample_scores.append(sample_info)

                # 更新进度条描述，显示当前得分
                pbar.set_postfix({'当前得分': f"{score:.4f}"})

        # 按得分排序并筛选Top-K
        sample_scores.sort(key=lambda x: x['score'], reverse=True)
        return sample_scores[:self.top_k]

    def analyze_modal_distribution(self, selected_samples: List[Dict]) -> Dict:
        """
        分析筛选样本的模态分布特征
        返回: 各模态的统计信息
        """
        modal_stats = defaultdict(list)
        for sample in selected_samples:
            for m, val in sample['modal_impact'].items():
                modal_stats[m].append(val)

        return {
            m: {'mean': np.mean(modal_stats[m]), 'std': np.std(modal_stats[m])}
            for m in modal_stats
        }