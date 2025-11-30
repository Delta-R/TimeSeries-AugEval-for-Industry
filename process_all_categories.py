import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heatmap_guided_augmentation import HeatmapGuidedAugmentor

def batch_process_all_categories():

    defect_categories = [
        'scratch_neck',
        'manipulated_front',
        'thread_top',
        'scratch_head',
        'thread_side'
    ]

    base_defect_dir = 'data/mvtec_anomaly_detection/screw/test'
    normal_dir = 'data/mvtec_anomaly_detection/screw/train/good'
    results_dir = 'results_MVTec/dinov2_vitg14_518/-1-shot_preprocess=agnostic'
    output_base = 'synthetic_all_categories'
    print("="*70)
    print("批量处理所有缺陷类别,热力图引导的数据增广")
    print("="*70)
    print(f"\n将处理以下 {len(defect_categories)} 个缺陷类别:")
    for i, cat in enumerate(defect_categories, 1):
        print(f"  {i}. {cat}")
    print()

    augmentor = HeatmapGuidedAugmentor(results_dir, object_name='screw')

    total_generated = 0
    category_stats = {}

    for idx, category in enumerate(defect_categories, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(defect_categories)}] 正在处理: {category}")
        print("="*70)

        defect_dir = os.path.join(base_defect_dir, category)
        output_dir = os.path.join(output_base, category)

        if not os.path.exists(defect_dir):
            print(f" 目录不存在,跳过 {defect_dir}")
            continue

        num_defects = len([f for f in os.listdir(defect_dir) if f.endswith('.png')])
        print(f"找到 {num_defects} 张缺陷图片")

        try:
            augmentor.batch_generate(
                defect_dir=defect_dir,
                normal_dir=normal_dir,
                output_dir=output_dir,
                num_variants_per_defect=3,
                anomaly_threshold=0.15,
                visualize=True
            )

            generated = len([f for f in os.listdir(output_dir) if f.startswith('syn_')])
            total_generated += generated

            category_stats[category] = {
                'input_images': num_defects,
                'generated_images': generated
            }

            print(f"[{category}] 完成, 生成了 {generated} 张合成图片")

        except Exception as e:
            print(f"[{category}] 处理失败: {str(e)}")
            category_stats[category] = {
                'input_images': num_defects,
                'generated_images': 0,
                'error': str(e)
            }

    print(f"\n{'='*70}")
    print("全部完成")
    print("="*70)
    print(f"\n总共生成: {total_generated} 张合成图片")
    print(f"输出目录: {output_base}/\n")

    print("各类别统计:")
    print(f"{'类别':<25} {'输入图片':<12} {'生成图片':<12}")
    print("-"*70)
    for category, stats in category_stats.items():
        print(f"{category:<25} {stats['input_images']:<12} {stats['generated_images']:<12}")

    print(f"\n生成的数据保存在:")
    for category in defect_categories:
        output_dir = os.path.join(output_base, category)
        if os.path.exists(output_dir):
            print(f"  - {output_dir}/")

if __name__ == '__main__':
    batch_process_all_categories()
