# TISA: Time-series & Image Streaming Augmentor

This repository implements a general augmentation and evaluation framework for time-series and image data, hereafter referred to as the **TISA algorithm** (Time-series & Image Streaming Augmentor). TISA is designed for unified modeling and representation learning of multi-source heterogeneous data in industrial, manufacturing, and sensor-network scenarios. Through systematic data augmentation, feature extraction, training strategies, and evaluation protocols, it enhances model robustness and generalization across classification, regression, and detection tasks.

---

## 1. Overall Concept of the Algorithm ⚙️

The TISA algorithm is designed around the complete pipeline from "data processing → model training → feature evaluation → visualization and application", organized by the following files and modules:

- Top-level training and evaluation scripts (e.g., `train.py`, `test.py`, `single_data_test.py`, `test_sample.py`) for quickly starting experiments.
- Configuration management and running scripts (`configs/`, `run/`, `utils/config.py`) for parameterized, reproducible experiment setups.
- Feature evaluation and downstream task scripts (`eval/`, `run/eval/`, `ensemble_models.py`, `optimize_knn.py`) for multi-protocol evaluation and combination of feature quality.
- Data augmentation and visualization modules (`src/augmentation.py`, `batch_generate_heatmaps.py`, `src/visualize.py`) for interpretable analysis and visualization at both time-series and image levels.
- Example datasets and interfaces for industrial applications (`data/fault_diagnosis.npy`, `data/soft_sensor.npy`), covering typical scenarios such as fault diagnosis and soft sensing.

From the file and directory naming, it is clear that TISA targets both 1D time-series signals and 2D image / pseudo-image representations:

- Time-series dimension: window slicing, frequency-domain transforms, or time-channel combinations map sensor sequences into input tensors usable by models.
- Image dimension: by rearranging, encoding time-series segments, or directly using image data, TISA produces inputs compatible with visual models (e.g., Transformer-based backbones).

---

## 2. Training and Self-supervised Core Ideas 🧠

### 1. Training Entry Points

Scripts in the repository root and under `run/train/` (e.g., `train.py`, `run/train/train.py`) form the main training entry points for TISA:

- Specify data paths, model configs, optimizer settings, etc. via command-line arguments or configuration files.
- Support both single-source data (e.g., time-series from a single device) and multi-source data (e.g., multiple operating conditions or sensor combinations) for joint training.
- Adopt a unified training loop for both time-series and image data to facilitate extension and migration.

### 2. Self-supervision and Meta-architecture

In `train/ssl_meta_arch.py`, TISA introduces a self-supervised learning meta-architecture:

- Obtain multiple views of time-series or image data via multi-view augmentations (e.g., cropping, masking, perturbation).
- Use a shared backbone network to extract features and construct training signals via contrastive, reconstruction, or prediction tasks.
- Generate discriminative, general-purpose representations for downstream tasks without requiring large-scale annotations.

In `configs/ssl_default_config.yaml` and `configs/train/`, you can define configuration files for different model scales and training strategies to achieve:

- Unified management of basic hyperparameters such as batch size, learning rate, and number of epochs.
- Fast switching between different backbone architectures, input sizes, and projection head structures.
- Specialized augmentation and normalization strategies for time-series vs. image data.

---

## 3. Model Backbones and Heads 🧩

### 1. Backbone Networks

`src/backbones.py` and directories such as `eval/depth/models/backbones/`, `eval/segmentation/models/backbones/`, and `segmentation_m2f/models/backbones/` indicate that the repository supports multiple Transformer-based backbone structures:

- Unified backbone builder interfaces (e.g., `eval/depth/models/builder.py`, `segmentation_m2f/models/builder.py`) dynamically construct models based on configuration.
- For different tasks (e.g., depth estimation, segmentation), similar or identical backbones can be reused to share a common representation space.
- For time-series data, TISA provides an "image-like" perspective by treating time steps as spatial dimensions to leverage mature vision backbones.

### 2. Task Heads and Decoding Modules

Directories such as `eval/depth/decode_heads/`, `eval/segmentation/decode_heads/`, and `segmentation_m2f/models/decode_heads/` contain various decoding and task heads:

- For regression tasks (e.g., depth estimation, soft sensing), continuous-value prediction heads output pixel-level or sequence-level scalars.
- For segmentation and detection tasks, structured output modules such as masks and bounding boxes are introduced (e.g., `segmentation_m2f/core/anchor`, `segmentation_m2f/core/box`).
- Along the time dimension, these can be analogized to the pixel dimension so that each time step or time window receives a label or weight, enabling unified time–space prediction structures.

---

## 4. Evaluation Protocols and Model Ensembling 📊

### 1. Basic Evaluation Scripts

The `eval/` and `run/eval/` directories contain several scripts for feature evaluation and downstream training:

- `eval/knn.py`, `run/eval/knn.py`: k-NN-based feature quality evaluation for both time-series and image features.
- `eval/linear.py`, `run/eval/linear.py`: linear probing to assess the linear separability of frozen features.
- `eval/log_regression.py`, `run/eval/log_regression.py`: logistic regression evaluation for binary or multi-class classification.
- `eval/metrics.py`, `spot-diff-main/utils/metrics.py`: unified implementations of common metrics such as accuracy, recall, and AUC.

These evaluation scripts can be run on:

- Features extracted purely from time-series inputs (e.g., sensor signals in `data/fault_diagnosis.npy`).
- Features extracted purely from image inputs (e.g., "image-like" feature maps converted from time-series).
- Or fused multimodal features for joint evaluation.

### 2. Model Ensembling and Hyperparameter Optimization

- `ensemble_models.py`: combines predictions from multiple models or training runs via weighting or rule-based fusion to improve stability and accuracy.
- `optimize_knn.py`: searches hyperparameters such as k values and distance metrics to optimize nearest-neighbor classification in feature space.

For time-series tasks, you can train multiple models at different temporal granularities and window lengths; for image tasks, you can train models with different input sizes and view transformations, then use the above scripts for post-processing and ensembling.

---

## 5. Data Augmentation and Visualization 🌈

### 1. Time-series and Image Augmentation

`src/augmentation.py` and `src/idaly/augmentation.py` (the latter inferred to be a submodule for augmentation) provide various data augmentation operations:

- For time-series signals: random cropping, scaling, noise injection, time jittering, channel shuffling, etc.
- For images or time-series pseudo-images: cropping, flipping, color jittering, occlusion/masking, etc.
- For self-supervised tasks: generating multi-view, complementary inputs to improve diversity and robustness of representations.

Through a unified augmentation interface, you can:

- Easily switch or stack different augmentation strategies during training and evaluation.
- Adjust augmentation strength and types for different industrial scenarios (e.g., vibration monitoring, process control).
- Remain compatible with future modalities (e.g., acoustic signals, image sequences).

### 2. Visualization and Heatmaps

- `batch_generate_heatmaps.py`: batch-generates heatmaps to visualize model attention on time-series or image inputs.
- `src/visualize.py`: general visualization utilities to align feature maps, predictions, and raw data.
- `src/ui_idap_v1.py`: a GUI wrapper for interactive workflows such as parameter tuning, data browsing, and result comparison.

For time-series tasks, heatmaps can show:

- Which time windows contribute most to classification or regression outputs.
- The importance distribution across channels in multichannel scenarios.
- Time windows where anomalies, drifts, or faults occur.

For image tasks, heatmaps can show:

- Different levels of attention on target regions vs. background.
- Potential bias or misprediction patterns.
- The impact of different augmentation strategies on attention distribution.

---

## 6. Industrial and Few-shot Scenarios 🏭

### 1. Fault Diagnosis and Soft Sensing

The presence of `data/fault_diagnosis.npy` and `data/soft_sensor.npy` indicates that TISA is tailored to the following typical industrial tasks:

- Fault diagnosis: use time-series signals from industrial sensors (e.g., temperature, vibration, current) for classification or anomaly detection of fault patterns.
- Soft sensing: when only partial variables are measured online, use models to predict hard-to-measure process indicators or product quality.

In these two scenarios, TISA:

- Reduces dependence on large labeled datasets via self-supervised pretraining followed by downstream fine-tuning.
- Enhances sensitivity to both slow trends and abrupt shocks with multi-timescale augmentation (combining short and long windows).
- Leverages an image perspective by transforming time-series into time–frequency or time–channel matrices to exploit visual models.

### 2. Few-shot Learning and Difference Analysis

The files `spot-diff-main/split_csv/1cls.csv`, `2cls_fewshot.csv`, `2cls_highshot.csv` and `spot-diff-main/utils/prepare_data.py` show that the repository supports data partitioning and loading for few-shot and many-shot settings:

- Few-shot: evaluate how quickly models adapt with very few labeled samples.
- High-shot: evaluate upper-bound performance with abundant labels.
- Single-/multi-class splits: for anomaly detection or one-class classification scenarios.

By comparing few-shot vs. high-shot results within a unified TISA representation space, you can:

- Analyze how self-supervised features improve label efficiency.
- Assess feature stability in small-sample regimes.
- Guide strategies for data collection and labeling.

---

## 7. Engineering and Utility Tools 🛠️

### 1. Configuration and Utility Library

- `utils/config.py`: centralized management of configuration parsing and merging, unifying command-line arguments and YAML files.
- `utils/cluster.py`: clustering utilities for time-series segments or image features, useful for unsupervised pattern discovery.
- `utils/utils.py`: common utilities for logging, random seed control, distributed training helpers, etc.

### 2. Scripts and Automation

- `scripts/lint.sh`: code quality and style checking.
- `rename_files.py`: batch file renaming for easier organization of data and results.
- `process_all_categories.py`: unified preprocessing for multi-category datasets.

These scripts and tools help to:

- Maintain consistency and reproducibility of experiments.
- Simplify data preparation and result management in multi-dataset, multi-task scenarios.
- Lay foundations for further engineering deployment (e.g., services, embedded systems).

---

## 8. Typical Usage Workflow 🚀

Based on the repository structure, a typical workflow for time-series and image tasks can be outlined as follows (schematic only):

1. **Data preparation**:
   - Organize raw time-series files into `.npy` or similar formats and place them in the `data/` directory.
   - For image tasks, convert time-series into time–frequency maps, 2D matrices, or directly use image data.
   - Use the splits in `spot-diff-main/split_csv/` or define custom partition strategies.  

2. **Configuration**:
   - Select or create configurations under `configs/train/` and `configs/eval/`.
   - Adjust model size, input resolution, and augmentation strategies according to task type (time-series / image / hybrid).
   - For self-supervised pretraining, refer to `ssl_default_config.yaml` and `train/ssl_meta_arch.py`.  

3. **Model training**:
   - Launch training with `train.py` in the repository root or `run/train/train.py`.
   - Use utilities in `utils/` for logging, checkpoint saving, and loading.
   - If needed, combine with other scripts under `train/` to implement specific training strategies.  

4. **Feature evaluation and downstream tasks**:
   - Use `eval/knn.py`, `eval/linear.py`, `eval/log_regression.py` to evaluate feature quality.
   - Use the wrapped entry points under `run/eval/` for batched evaluations.
   - For depth estimation or segmentation tasks, refer to the models and heads in `eval/depth/` and `eval/segmentation/`.  

5. **Visualization and analysis**:
   - Use `batch_generate_heatmaps.py` to generate time-series / image heatmaps and inspect model attention.
   - Use `src/visualize.py` and `src/ui_idap_v1.py` for result visualization and interaction.
   - Further optimize and compare results with `ensemble_models.py` and `optimize_knn.py`.  

6. **Industrial deployment and iteration**:
   - Integrate TISA-trained models into production systems for fault diagnosis, soft sensing, etc.
   - Use `utils/cluster.py` and metric scripts to continuously monitor data distribution and model performance.
   - For new devices, operating conditions, or image modalities, continue to leverage self-supervision and few-shot strategies for rapid adaptation.

---

## 9. Summary 📌

The TISA algorithm is designed around the goal of "unified handling of time-series and image data" by:

- Combining self-supervised pretraining with multi-task heads to improve feature generality and transferability.
- Providing rich data augmentation and visualization tools to support interpretation and debugging of model behavior.
- Supporting multiple evaluation protocols and few-shot scenarios to systematically characterize model performance in complex industrial environments.
