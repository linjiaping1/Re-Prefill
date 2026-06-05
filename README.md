# What Happens Before Decoding? Prefill Determines GUI Grounding in VLMs

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2605.12549)


## Installation
```bash
git clone https://github.com/linjiaping1/Re-Prefill.git
cd Re-Prefill
pip install -r requirements.txt
```

## Download Datasets

```bash
# Download ScreenSpot-Pro dataset
hf download likaixin/ScreenSpot-Pro --repo-type=dataset

# Download UI-Vision dataset
hf download ServiceNow/ui-vision --repo-type=dataset

# Download OSWorld-G dataset
git clone https://github.com/xlang-ai/OSWorld-G.git

# Download MMBench-GUI dataset
hf download OpenGVLab/MMBench-GUI --repo-type=dataset

# Download ScreenSpot-V2 dataset
hf download OS-Copilot/ScreenSpot-v2 --repo-type=dataset
```
The refined annotation file used in our experiments is available at: [OSWorld-G_refined_classified.json](https://github.com/user-attachments/files/25151412/OSWorld-G_refined_classified.json).

## Download Models

```bash
# Download Qwen3VL-8B model
hf download Qwen/Qwen3-VL-8B-Instruct

# Download Qwen3VL-32B model
hf download Qwen/Qwen3-VL-32B-Instruct

# Download GUI-Owl-1.5-8B model
hf download mPLUG/GUI-Owl-1.5-8B-Instruct

# Download MAI-UI-8B model
hf download Tongyi-MAI/MAI-UI-8B
```

## Evaluation
Before running the evaluation scripts, please configure the project root directory and dataset paths in [utils.py](./utils.py).
```bash
./eval_8b.sh

./eval_32b.sh
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{lin2026happens,
  title={What Happens Before Decoding? Prefill Determines GUI Grounding in VLMs},
  author={Lin, Jiaping and Shen, Fei and Li, Junzhe and Nie, Ping and Yu, Fei and Li, Ming and Li, Haizhou},
  journal={arXiv preprint},
  year={2026},
  url={https://arxiv.org/abs/2605.12549}
}
```
