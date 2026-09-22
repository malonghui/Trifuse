<div align="center">

# Trifuse
### Enhancing Attention-Based GUI Grounding via Multimodal Fusion

**ICML 2026**

Longhui Ma\* · Di Zhao\* · Siwei Wang · Zhao Lv · Miao Wang

National University of Defense Technology · Intelligent Game and Decision Lab, Academy of Military Sciences

\* Equal contribution

[Installation](docs/INSTALL.md) · [Method](docs/METHOD.md) · [中文说明](docs/README_zh-CN.md) · [Citation](#citation)

</div>

Official inference implementation of **Trifuse**, a training-free framework that grounds natural-language instructions to GUI elements by combining **MLLM attention**, **OCR text**, and **icon-caption semantics**. Trifuse requires no task-specific fine-tuning on GUI grounding datasets.

![Overview of the Trifuse framework](assets/framework.png)

## Highlights

- **Complementary spatial cues.** Attention identifies task-relevant regions; OCR and icon captions provide explicit textual and semantic anchors.
- **Consensus-SinglePeak fusion.** Cross-modal agreement and supported modality-specific peaks produce a sharper localization map.
- **Coarse-to-fine grounding.** One crop-and-zoom refinement maps the final prediction back to the original screenshot.

## Release scope

This repository contains the Qwen2.5-VL-based Trifuse inference method and a single-image CLI. Benchmark runners, baseline implementations, experimental analysis scripts, datasets, model weights, and training code are not included. No repository license has been added yet.

## Quick start

Create the Conda environment and install the package following [INSTALL.md](docs/INSTALL.md). Prepare an OmniParser icon detector and Florence caption checkpoint as described there, then run:

```bash
python -m trifuse --image /path/to/screenshot.png --query "Click the search button" --icon-detector weights/icon_detect/model.pt --caption-model weights/icon_caption --output outputs/prediction.json --visualization outputs/prediction.png
```

The default backbone is `Qwen/Qwen2.5-VL-3B-Instruct`. Use `--model Qwen/Qwen2.5-VL-7B-Instruct` or a local checkpoint path to select a different Qwen2.5-VL model. On Windows, replace the screenshot path with a quoted local path such as `"E:/screenshots/example.png"`.

### Python API

Set `TRIFUSE_ICON_DETECTOR` and `FLORENCE_MODEL_PATH` before running:

```python
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trifuse import Trifuse

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    attn_implementation="eager",
).eval()
processor = AutoProcessor.from_pretrained(model_id)
grounder = Trifuse(model, processor).eval()
prediction = grounder.ground("/path/to/screenshot.png", "Click the search button")
print(prediction["point"])  # (x, y), pixels in the original screenshot
```

`bbox` is `[x1, y1, x2, y2]` in original-image pixels. It describes the selected heatmap region, not a trained object-detection box. `score` is a normalized heatmap value, **not a calibrated probability**. See [output details](docs/METHOD.md#output).

## Paper results

Element accuracy (%) reported in the final manuscript; these are **paper results, not new measurements from this packaged release**.

| Backbone | ScreenSpot | ScreenSpot-v2 | ScreenSpot-Pro |
|---|---:|---:|---:|
| Trifuse / Qwen2.5-VL-3B | 81.1 | 82.6 | 18.9 |
| Trifuse / Qwen2.5-VL-7B | 86.2 | 86.9 | 29.7 |

The paper also evaluates OSWorld-G and UI-Vision. This minimal release does not include benchmark evaluation scripts or GUI-Actor / GUI-AIMA integrations.

## Repository structure

```text
trifuse/
  core.py              # Attention extraction, CS fusion, two-stage localization
  attention_utils.py   # Tensor conversion and patch-grid helpers
  modal_heatmaps.py     # OCR and caption relevance projected to the patch grid
  icon_models.py       # Icon detection and Florence caption inference
  heatmap_utils.py     # Heatmap stabilization and connected components
  cli.py               # Single-image inference and visualization
assets/framework.png   # Framework figure from the paper
docs/                  # Installation, method, and Chinese documentation
environment.yml        # Conda environment bootstrap
requirements.txt       # Python dependencies
```

## Acknowledgements

Trifuse builds on [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [OmniParser](https://github.com/microsoft/OmniParser), [Florence-2](https://huggingface.co/microsoft/Florence-2-base), and [BGE-M3](https://huggingface.co/BAAI/bge-m3). External models and libraries retain their respective licenses.

## Citation

```bibtex
@inproceedings{ma2026trifuse,
  title     = {Trifuse: Enhancing Attention-Based GUI Grounding via Multimodal Fusion},
  author    = {Ma, Longhui and Zhao, Di and Wang, Siwei and Lv, Zhao and Wang, Miao},
  booktitle = {International Conference on Machine Learning},
  year      = {2026}
}
```
