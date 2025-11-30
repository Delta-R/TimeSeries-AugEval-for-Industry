import os
import glob
from defect_augmentation_natural import NaturalDefectAugmentor

USE_TRAINED_MODEL = False
TRAINED_MODEL_PATH = 'results_MVTec/dinov2_vitb14_518'

defect_dir = 'data/mvtec_anomaly_detection/screw/test'
defect_images = []

for defect_type in ['scratch_neck', 'manipulated_front', 'thread_top', 'scratch_head', 'thread_side']:
    type_dir = os.path.join(defect_dir, defect_type)
    if os.path.exists(type_dir):
        images = sorted(glob.glob(os.path.join(type_dir, '*.png')))[:2]
        defect_images.extend(images)

print(f"使用 {len(defect_images)} 张缺陷源图像:")
for img in defect_images:
    print(f"  - {img}")

normal_dir = 'data/mvtec_anomaly_detection/screw/train/good'
normal_images = sorted(glob.glob(os.path.join(normal_dir, '*.png')))[:3]

print(f"\n使用 {len(normal_images)} 张正常图像:")
for img in normal_images:
    print(f"  - {img}")

print("\n初始化增强器...")
if USE_TRAINED_MODEL:
    print(f"[选择] 使用训练的模型: {TRAINED_MODEL_PATH}")
else:
    print("[选择] 使用预训练DINOv2模型")

augmentor = NaturalDefectAugmentor(
    data_root='data/mvtec_anomaly_detection',
    object_name='screw',
    n_ref_samples=16,
    device='cpu',
    faiss_on_cpu=True,
    use_trained_model=USE_TRAINED_MODEL,
    trained_model_path=TRAINED_MODEL_PATH if USE_TRAINED_MODEL else None
)

print("\n开始生成...")
total = augmentor.generate_synthetic(
    defect_images=defect_images,
    normal_images=normal_images,
    output_dir='synthetic_natural_test',
    num_per_normal=3,
    anomaly_threshold=0.5,
    low_brightness=150,
    high_brightness=220,
    irregularity=0.2,
    visualize=True
)

print(f"\n完成！生成了 {total} 张图像")
print("查看结果: synthetic_natural_test/")
print("查看可视化: synthetic_natural_test/vis/")
