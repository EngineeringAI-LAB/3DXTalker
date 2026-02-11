# **3DXTalker**
 <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a> <a href='https://arxiv.org/xxx'><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2310.00434-red?link=https%3A%2F%2Farxiv.org%2Fabs%2F2310.00434"></a> <a href='https://engineeringai-lab.github.io/3DXTalker.github.io/'><img alt="Project Page" src="https://img.shields.io/badge/Project%20Page-blue?logo=github&labelColor=black&link=https%3A%2F%2Fraineggplant.github.io%2FDiffPoseTalk"></a> <a href="https://huggingface.co/datasets/EngineeringAI-LAB/3DTalkingDataset"><img alt="HF Dataset" src="https://img.shields.io/badge/Dataset-HuggingFace-FFD21E?logo=huggingface&logoColor=FFD21E"></a>

3DXTalker: An Integrated Framework for Expressive 3D Talking Avatars
![teaser](3DXTalker.png)

**3DXTalker** generates identity-consistent, expressive 3D talking avatars from a single reference image and speech audio, achieving accurate lip synchronization, expressive emotion control, and natural head-pose dynamics. It achieves expressive facial animation through data-curated identity modeling, audio-rich representations, and spatial dynamics controllability. By introducing frame-wise amplitude and emotional cues beyond standard speech embeddings, 3DXTalker delivers superior lip synchronization and nuanced expression modulation. Built on a flow-matching transformer architecture, it enables natural head-pose motion generation while supporting stylized control, integrating lip synchronization, emotional expression, and head-pose dynamics within a unified framework.

## TODO
- [x] Release the 3DTalking benchmark dataset
  - [x] Release the raw dataset
  - [x] Release the processed dataset
- [x] Release the data processing code
- [ ] Release the training and inference code
- [ ] Release the pretrained models

## Installation
- Python 3.10
- Pytorch 2.2.2
- CUDA 12.1
- Pytorch3d 0.7.7
  
```python
conda create -n env_3DXTalker python==3.10
conda activate env_3DXTalker
pip install -r requirements.txt
```

For some people the compilation fails during requirements install and works after. Try running the following separately:
```python
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"
```

### Download Pretrained Audio Encoders

1. Download `emotion2vec_plus_base` model and place it in `./pretrained_models/`:

   ```bash
   # Create directory
   mkdir -p pretrained_models/emotion2vec_plus_base

   # Option 1: Using git-lfs (recommended)
   cd pretrained_models
   git lfs install
   git clone https://huggingface.co/iic/emotion2vec_plus_base

   # Option 2: Manual download from https://huggingface.co/iic/emotion2vec_plus_base
   # Download all files to ./pretrained_models/emotion2vec_plus_base/
   ```

2. Download `microsoft/wavlm-base-plus` (audio encoder):

   ```bash
   # Option 1: Auto-download on first run (recommended)
   # The model will be automatically downloaded from HuggingFace when you run training

   # Option 2: Pre-download manually
   cd pretrained_models
   git lfs install
   git clone https://huggingface.co/microsoft/wavlm-base-plus
   
   # Then update config/default_config.yaml:
   # audio_encoder_repo: './pretrained_models/wavlm-base-plus'
   ```

   Expected directory structure:
   ```
   pretrained_models/
   ├── emotion2vec_plus_base/
   │   ├── config.json
   │   ├── pytorch_model.bin
   │   └── ...
   └── wavlm-base-plus/          # Optional (auto-downloads if not present)
       ├── config.json
       ├── pytorch_model.bin
       └── ...
   ```


### Data Preparation and Preprocess
1. Download raw video datasets following these links:
[V0-GRID](https://zenodo.org/records/3625687);  [v1-RAVDESS](https://zenodo.org/records/1188976); [V2-MEAD](https://wywu.github.io/projects/MEAD/MEAD.html); [V3-VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html); [V4-HDTF](https://huggingface.co/datasets/global-optima-research/HDTF); [V5-Celebv-HQ](https://github.com/CelebV-HQ/CelebV-HQ/)

    *If you don't want to process the data manually, we also provide [processed data](https://huggingface.co/datasets/EngineeringAI-LAB/3DTalkingDataset/tree/main) at Hugging Face.*

1. Run data curation (duration, noise, language, sync, resolution normalization).

  - Edit `raw_video_dir` in `data_prepare/data_curation_pipeline.py` to your raw video folder.

  ```bash
  cd data_prepare
  python data_curation_pipeline.py
  ```

  Output will be in `data_prepare/final_curated_videos/`.

3. Rename videos for dataset indexing.

  - Edit `dataset_name`, `input_dir`, and `output_dir` in `data_prepare/rename.py` if needed.
  - By default it expects input at `data_prepare/Scaled_videos` and outputs to `data_prepare/Renamed_videos`.

  ```bash
  cd data_prepare
  python rename.py
  ```

4. Download EMOCA-related assets (models and FLAME files).

  ```bash
  bash gdl_apps/EMOCA/demos/download_assets.sh
  ```

5. Run EMOCA reconstruction to extract FLAME parameters.

  - Edit `data_root_dir` and `dataset_name` in `gdl_apps/EMOCA/demos/my_recons_video.py`.
  - `data_root_dir` should contain `<dataset_name>/all_videos_path.txt`.

  ```bash
  python gdl_apps/EMOCA/demos/my_recons_video.py \
    --dataset_name VoxCeleb2 \
    --output_folder video_output \
    --model_name EMOCA_v2_lr_mse_20
  ```

6. Data structures are provided in [DATASET_STRUCTURE.md](data_prepare/DATASET_STRUCTURE.md)


