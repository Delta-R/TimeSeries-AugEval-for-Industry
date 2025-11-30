import argparse
import os
import sys
import cv2
import numpy as np
import json
import base64
import time
import faiss
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.backbones import get_model
from src.post_eval import mean_top1p
from src.utils import dists2map

def load_reference_memory_bank(results_dir, object_name='screw', seed=0):

    import yaml
    args_path = os.path.join(results_dir, 'args.yaml')

    if not os.path.exists(args_path):
        raise FileNotFoundError(f"找不到训练配置文件: {args_path}")

    with open(args_path, 'r') as f:
        args = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    model = get_model(
        args['model_name'],
        device,
        smaller_edge_size=args['resolution']
    )

    data_root = args['data_root']
    img_ref_folder = f"{data_root}/{object_name}/train/good/"

    if not os.path.exists(img_ref_folder):
        raise FileNotFoundError(f"找不到训练数据目录: {img_ref_folder}")

    features_ref = []
    img_ref_samples = sorted(os.listdir(img_ref_folder))

    print(f"正在加载 {len(img_ref_samples)} 个参考样本...")

    with torch.inference_mode():
        for img_ref_n in img_ref_samples:
            img_ref = f"{img_ref_folder}{img_ref_n}"
            image_ref = cv2.cvtColor(cv2.imread(img_ref, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            image_ref_tensor, grid_size = model.prepare_image(image_ref)
            features_ref_i = model.extract_features(image_ref_tensor)

            mask_ref = model.compute_background_mask(features_ref_i, grid_size, threshold=10, masking_type=True)
            features_ref.append(features_ref_i[mask_ref])

        features_ref = np.concatenate(features_ref, axis=0).astype('float32')

        if args.get('faiss_on_cpu', True):
            knn_index = faiss.IndexFlatL2(features_ref.shape[1])
        else:
            res = faiss.StandardGpuResources()
            knn_index = faiss.GpuIndexFlatL2(res, features_ref.shape[1])

        faiss.normalize_L2(features_ref)
        knn_index.add(features_ref)

    print(f"[OK] Memory Bank加载完成: {features_ref.shape[0]} 个特征向量")

    return model, knn_index, args

def detect_single_image(image_path, model, knn_index, args, output_dir=None):

    start_time = time.time()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片: {image_path}")

    image_test = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    with torch.inference_mode():
        image_tensor, grid_size = model.prepare_image(image_test)
        features = model.extract_features(image_tensor)

        mask = model.compute_background_mask(features, grid_size, threshold=10, masking_type=True)
        features_masked = features[mask]

        faiss.normalize_L2(features_masked)

        k_neighbors = args.get('k_neighbors', 3)
        distances, _ = knn_index.search(features_masked, k=k_neighbors)

        if k_neighbors > 1:
            distances = distances.mean(axis=1)

        distances = distances / 2

        anomaly_score = mean_top1p(distances.flatten())

        output_distances = np.zeros_like(mask, dtype=float)
        output_distances[mask] = distances.squeeze()
        d_masked = output_distances.reshape(grid_size)

        anomaly_map = dists2map(d_masked, image_test.shape)

        processing_time = time.time() - start_time

        threshold = 0.15
        is_anomaly = anomaly_score > threshold

        result = {
            'anomaly_score': float(anomaly_score),
            'is_anomaly': bool(is_anomaly),
            'processing_time': float(processing_time),
            'threshold': threshold,
            'num_patches': int(mask.sum()),
            'total_patches': int(mask.size),
            'anomaly_map': anomaly_map
        }

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            fig1 = plt.figure(figsize=(6, 6))
            plt.imshow(image_test)
            plt.title('Original Image')
            plt.axis('off')
            original_path = os.path.join(output_dir, f'{base_name}_original.png')
            plt.savefig(original_path, dpi=150, bbox_inches='tight')
            plt.close()

            fig2 = plt.figure(figsize=(6, 6))
            im = plt.imshow(anomaly_map, cmap='jet')
            plt.title(f'Anomaly Heatmap (Score: {anomaly_score:.3f})')
            plt.colorbar(im)
            plt.axis('off')
            heatmap_path = os.path.join(output_dir, f'{base_name}_heatmap.png')
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            plt.close()

            fig3 = plt.figure(figsize=(6, 6))
            plt.imshow(image_test)
            plt.imshow(anomaly_map, cmap='jet', alpha=0.5)
            plt.title('Overlay')
            plt.axis('off')
            overlay_path = os.path.join(output_dir, f'{base_name}_overlay.png')
            plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
            plt.close()

            fig4 = plt.figure(figsize=(6, 6))
            plt.hist(distances.flatten(), bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(anomaly_score, color='red', linestyle='--', linewidth=2, label=f'Score: {anomaly_score:.3f}')
            plt.axvline(threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
            plt.xlabel('Distance')
            plt.ylabel('Frequency')
            plt.title('Histogram of Patch Distances')
            plt.legend()
            plt.grid(True, alpha=0.3)
            histogram_path = os.path.join(output_dir, f'{base_name}_histogram.png')
            plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
            plt.close()

            result['original_path'] = original_path
            result['heatmap_path'] = heatmap_path
            result['overlay_path'] = overlay_path
            result['histogram_path'] = histogram_path

    return result

def main():
    parser = argparse.ArgumentParser(description='单图异常检测')
    parser.add_argument('--image', type=str, required=True, help='待检测图片路径')
    parser.add_argument('--results_dir', type=str,
                       default='results_MVTec/dinov2_vitg14_518/-1-shot_preprocess=agnostic',
                       help='训练结果目录')
    parser.add_argument('--object', type=str, default='screw', help='检测对象类型')
    parser.add_argument('--output_dir', type=str, default='detection_output', help='输出目录')
    parser.add_argument('--output_json', action='store_true', help='输出JSON格式结果')

    args = parser.parse_args()

    try:
        print(f"正在加载模型: {args.results_dir}")
        model, knn_index, config = load_reference_memory_bank(args.results_dir, args.object)

        print(f"正在检测图片: {args.image}")
        result = detect_single_image(args.image, model, knn_index, config, args.output_dir)

        if args.output_json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"\n{'='*50}")
            print(f"检测结果:")
            print(f"{'='*50}")
            print(f"异常得分: {result['anomaly_score']:.4f}")
            print(f"判定结果: {'🔴 异常' if result['is_anomaly'] else '✅ 正常'}")
            print(f"处理时间: {result['processing_time']:.3f} 秒")
            print(f"前景patches: {result['num_patches']}/{result['total_patches']}")
            if 'heatmap_path' in result:
                print(f"结果保存: {result['heatmap_path']}")
            print(f"{'='*50}")

        return result

    except Exception as e:
        print(f"❌ 错误: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
