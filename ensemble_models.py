"""
模型集成：融合多个DINOv2变体的预测
预期性能提升：+2-3% AUROC
"""

import numpy as np
import torch
from src.backbones import get_model
from src.detection import run_anomaly_detection
from src.utils import get_dataset_info

class EnsembleAnomalyDetector:
    """
    集成多个模型的异常检测器
    """

    def __init__(self, model_configs, device='cuda'):
        """
        Args:
            model_configs: 模型配置列表
                [
                    {'name': 'dinov2_vits14', 'resolution': 448, 'weight': 0.3},
                    {'name': 'dinov2_vitb14', 'resolution': 448, 'weight': 0.4},
                    {'name': 'dinov2_vitl14', 'resolution': 518, 'weight': 0.3},
                ]
        """
        self.models = []
        self.weights = []

        for config in model_configs:
            model = get_model(
                config['name'],
                device,
                smaller_edge_size=config['resolution']
            )
            self.models.append(model)
            self.weights.append(config['weight'])

        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

        print(f"Loaded {len(self.models)} models for ensemble:")
        for config, weight in zip(model_configs, self.weights):
            print(f"  - {config['name']} @ {config['resolution']}px, weight={weight:.2f}")

    def run_ensemble_detection(self, object_name, data_root, shots, **kwargs):
        """
        对每个模型运行检测，然后融合结果
        """
        all_scores = []

        for idx, (model, weight) in enumerate(zip(self.models, self.weights)):
            print(f"\n[Ensemble {idx+1}/{len(self.models)}] Running with model {idx+1}...")

            anomaly_scores, _, _ = run_anomaly_detection(
                model,
                object_name,
                data_root=data_root,
                n_ref_samples=shots,
                **kwargs
            )

            all_scores.append((anomaly_scores, weight))

        fused_scores = self.fuse_scores(all_scores)
        return fused_scores

    def fuse_scores(self, all_scores, method='weighted_average'):
        """
        融合多个模型的异常分数

        Args:
            all_scores: [(score_dict, weight), ...]
            method:
                - 'weighted_average': 加权平均
                - 'max': 取最大值（保守策略）
                - 'rank_fusion': 排序融合
        """
        sample_names = list(all_scores[0][0].keys())
        fused = {}

        if method == 'weighted_average':
            for sample_name in sample_names:
                score = sum(
                    scores[sample_name] * weight
                    for scores, weight in all_scores
                )
                fused[sample_name] = score

        elif method == 'max':
            for sample_name in sample_names:
                score = max(
                    scores[sample_name]
                    for scores, weight in all_scores
                )
                fused[sample_name] = score

        elif method == 'rank_fusion':
            for sample_name in sample_names:
                ranks = []
                for scores, weight in all_scores:
                    sorted_samples = sorted(
                        scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    rank = next(
                        i for i, (name, _) in enumerate(sorted_samples)
                        if name == sample_name
                    )
                    ranks.append(rank * weight)

                fused[sample_name] = sum(ranks)

        return fused

def run_lightweight_ensemble():
    """
    轻量级集成：vits14 + vitb14
    速度相对较快，性能提升明显
    """
    ensemble = EnsembleAnomalyDetector([
        {'name': 'dinov2_vits14', 'resolution': 448, 'weight': 0.4},
        {'name': 'dinov2_vitb14', 'resolution': 448, 'weight': 0.6},
    ])

    objects, object_anomalies, masking, rotation = get_dataset_info("MVTec", "agnostic")

    for obj in objects:
        scores = ensemble.run_ensemble_detection(
            object_name=obj,
            data_root="data/mvtec_anomaly_detection",
            shots=8,
            object_anomalies=object_anomalies,
            plots_dir=f"results_ensemble/{obj}",
            masking=masking[obj],
            rotation=rotation[obj],
            seed=0
        )
        print(f"Finished {obj}")

def run_full_ensemble():
    """
    完整集成：Small + Base + Large
    性能最强，但速度慢
    """
    ensemble = EnsembleAnomalyDetector([
        {'name': 'dinov2_vits14', 'resolution': 448, 'weight': 0.2},
        {'name': 'dinov2_vitb14', 'resolution': 448, 'weight': 0.3},
        {'name': 'dinov2_vitl14', 'resolution': 518, 'weight': 0.5},
    ])


if __name__ == "__main__":
    print("集成学习的优势：")
    print("1. 不同模型捕捉不同特征")
    print("2. 互补缺陷，减少漏检")
    print("3. 提升鲁棒性")
    print("\n推荐：先用vits14+vitb14（速度快），如需极致精度再用3模型集成")
