# Installation and model preparation

## 1. Create a Conda environment

```bash
git clone https://github.com/malonghui/Trifuse.git
cd Trifuse
conda env create -f environment.yml
conda activate trifuse
```

Python 3.10 is the reference version. The dependency pins reflect packages inspected in the author's local Windows Conda environment, not a fully reproduced GPU inference environment. That environment lacked `qwen-vl-utils`; this release declares it explicitly. End-to-end inference with all model checkpoints has not been validated as part of packaging.

## 2. Install inference dependencies

Install PyTorch for your CUDA/driver setup before installing Trifuse. For the CUDA 11.8 build used in the inspected environment:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install paddlepaddle==3.0.0
python -m pip install -e .
```

This recipe runs OCR on CPU by default and uses CUDA for the other models when available. For GPU OCR, install the compatible `paddlepaddle-gpu` build **instead of** `paddlepaddle` using the [PaddlePaddle installation guide](https://www.paddlepaddle.org.cn/install/quick), then pass `--ocr-device gpu`. Do not install both Paddle backends in the same environment.

The reference requirements use PaddleOCR 2.10.0 and explicitly request **PP-OCRv4**. The adapter also handles PaddleOCR 3.x's `predict` interface, but 3.x is not the pinned reference environment. PP-OCRv4 names the OCR model family; it does not mean the Python package must have major version 4.

## 3. Prepare external models

| Component | Model / checkpoint | Configuration |
|---|---|---|
| Attention backbone | Qwen2.5-VL-3B-Instruct or 7B-Instruct | `--model` |
| Text recognition | PP-OCRv4, loaded by PaddleOCR | `--ocr-device`; `TRIFUSE_OCR_LANG` defaults to `en` |
| Text embeddings | BAAI/bge-m3 | `--embedding-model` or `BGE_MODEL_PATH` |
| Icon detector | OmniParser Ultralytics-compatible `icon_detect/model.pt` | `--icon-detector` or `TRIFUSE_ICON_DETECTOR` |
| Icon captions | OmniParser Florence caption weights | `--caption-model` or `FLORENCE_MODEL_PATH` |
| Caption processor | microsoft/Florence-2-base | `--caption-processor` or `FLORENCE_PROCESSOR_PATH` |

Download the Ultralytics detector and caption assets from the [OmniParser-v2.0 model repository](https://huggingface.co/microsoft/OmniParser-v2.0). This adapter uses the `icon_detect` checkpoint, not the newer YOLOv9 detector. One download option is:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="microsoft/OmniParser-v2.0",
    allow_patterns=["icon_detect/*", "icon_caption/*"],
    local_dir="weights",
)
```

Expected paths:

```text
weights/
  icon_detect/model.pt
  icon_caption/config.json
  icon_caption/generation_config.json
  icon_caption/model.safetensors
```

The caption processor is loaded separately because the OmniParser caption directory may contain only model weights and configuration. Florence loading enables `trust_remote_code=True` for its custom model implementation. For offline use, provide local model and processor directories and cache PaddleOCR's assets beforehand. The repository does not bundle or automatically download model weights during installation.

## 4. Run inference

```bash
python -m trifuse --help
python -m trifuse --image /path/to/screenshot.png --query "Click the search button" --icon-detector weights/icon_detect/model.pt --caption-model weights/icon_caption --output outputs/prediction.json --visualization outputs/prediction.png
```

The installed `trifuse` command accepts the same arguments. The CLI writes a JSON prediction and, optionally, a PNG containing the predicted region and click point. Screenshots are supplied by the user.

For the Python API, set environment variables before model construction. PowerShell example:

```powershell
$env:TRIFUSE_ICON_DETECTOR = "E:/models/icon_detect/model.pt"
$env:FLORENCE_MODEL_PATH = "E:/models/icon_caption"
$env:BGE_MODEL_PATH = "E:/models/bge-m3"
```

## Troubleshooting

- **Attention tensors are missing:** load the Qwen model with `attn_implementation="eager"`. This method needs explicit attention tensors.
- **GPU out of memory:** start with the 3B backbone and a smaller screenshot. Eager attention retains full attention tensors; memory use depends strongly on image resolution. Resizing changes the prediction's coordinate system, so map coordinates back if you resize externally.
- **Caption tokenizer / processor files are missing:** point `--caption-processor` at a complete Florence processor directory, separate from the OmniParser caption weights.
- **No icon checkpoint:** download the detector and pass its actual file path. There is no machine-specific fallback path.
- **Network / offline errors:** model assets can download on first inference. Use complete local checkpoint directories in offline environments.

No full benchmark reproduction or minimum-VRAM guarantee is claimed for this release.
