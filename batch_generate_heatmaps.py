import os
import glob
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from single_image_detection import load_reference_memory_bank, detect_single_image

def batch_generate_heatmaps(
    test_root,
    results_dir,
    object_name='screw',
    output_root='heatmap_results',
    defect_categories=None
):
    print(f"开始批量生成热力图...")
    print(f"测试图片目录: {test_root}")
    print(f"训练模型目录: {results_dir}")
    print(f"输出目录: {output_root}")
    print("="*60)

    if defect_categories is None:
        defect_categories = [
            'good',
            'scratch_neck',
            'manipulated_front',
            'thread_top',
            'scratch_head',
            'thread_side'
        ]

    print(f"\n步骤1: 加载训练好的模型和memory bank...")
    model, knn_index, config = load_reference_memory_bank(results_dir, object_name)
    print(f"[OK] 模型加载完成!\n")

    all_results = {}
    total_images = 0
    for category in defect_categories:
        category_path = os.path.join(test_root, category)
        if not os.path.exists(category_path):
            print(f"警告: 目录不存在,跳过 {category_path}")
            continue
        image_files = sorted(glob.glob(os.path.join(category_path, '*.png')))
        if not image_files:
            print(f"在 {category} 中没有找到图片")
            continue
        print(f"\n步骤2.{defect_categories.index(category)+1}: 处理 {category} 类别 ({len(image_files)} 张图片)")
        output_dir = os.path.join(output_root, category)
        os.makedirs(output_dir, exist_ok=True)
        category_results = []
        for img_path in tqdm(image_files, desc=f"  处理 {category}"):
            try:
                result = detect_single_image(
                    image_path=img_path,
                    model=model,
                    knn_index=knn_index,
                    args=config,
                    output_dir=output_dir
                )

                result['image_path'] = img_path
                result['category'] = category
                category_results.append(result)
                total_images += 1

            except Exception as e:
                print(f"处理失败: {img_path}")
                print(f"错误: {str(e)}")
                continue
        all_results[category] = category_results
        avg_score = sum(r['anomaly_score'] for r in category_results) / len(category_results) if category_results else 0
        anomaly_count = sum(1 for r in category_results if r['is_anomaly'])
        print(f"  [{category}] 平均异常分数: {avg_score:.4f}")
        print(f"  [{category}] 异常图片数量: {anomaly_count}/{len(category_results)}")
    print(f"\n{'='*60}")
    print(f"全部完成!")
    print(f"总共处理: {total_images} 张图片")
    print(f"结果保存在: {output_root}")
    summary_file = os.path.join(output_root, 'detection_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"汇总结果保存在: {summary_file}")
    print("="*60)
    print("\n统计汇总:")
    print(f"{'类别':<20} {'图片数':<10} {'平均分数':<12} {'异常数量':<10}")
    print("-"*60)
    for category, results in all_results.items():
        if results:
            avg_score = sum(r['anomaly_score'] for r in results) / len(results)
            anomaly_count = sum(1 for r in results if r['is_anomaly'])
            print(f"{category:<20} {len(results):<10} {avg_score:<12.4f} {anomaly_count:<10}")
    return all_results
def main():
    parser = argparse.ArgumentParser(description='批量生成异常检测热力图')
    parser.add_argument(
        '--test_root',
        type=str,
        default='data/mvtec_anomaly_detection/screw/test',
        help='测试图片根目录'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results_MVTec/dinov2_vitg14_518/-1-shot_preprocess=agnostic',
        help='训练结果目录'
    )
    parser.add_argument(
        '--object',
        type=str,
        default='screw',
        help='检测对象类型'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='heatmap_results',
        help='输出根目录'
    )
    parser.add_argument(
        '--categories',
        nargs='+',
        default=None,
        help='要处理的缺陷类别列表(默认处理所有)'
    )

    args = parser.parse_args()

    try:
        all_results = batch_generate_heatmaps(
            test_root=args.test_root,
            results_dir=args.results_dir,
            object_name=args.object,
            output_root=args.output_root,
            defect_categories=args.categories
        )

        print("\n处理完成!可以查看生成的热力图:")
        print(f"目录: {args.output_root}")

    except Exception as e:
        print(f"批量处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
