"""Minimal icon detector and Florence caption adapter."""
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage


@lru_cache(maxsize=2)
def get_caption_model_processor(model_name='florence2', model_name_or_path=None, device=None):
    from transformers import AutoProcessor, AutoModelForCausalLM
    if model_name != 'florence2':
        raise ValueError('This release uses Florence-2 for icon captions.')
    path = model_name_or_path or os.environ.get('FLORENCE_MODEL_PATH')
    if not path:
        raise ValueError('Set FLORENCE_MODEL_PATH to the OmniParser Florence caption checkpoint.')
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    # OmniParser distributes caption weights separately from the Florence processor.
    processor_path = os.environ.get('FLORENCE_PROCESSOR_PATH', 'microsoft/Florence-2-base')
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        trust_remote_code=True,
    ).to(device).eval()
    return {'model': model, 'processor': processor}


@lru_cache(maxsize=2)
def get_yolo_model(model_path=None):
    from ultralytics import YOLO
    path = model_path or os.environ.get('TRIFUSE_ICON_DETECTOR')
    if not path or not Path(path).is_file():
        raise FileNotFoundError('Set TRIFUSE_ICON_DETECTOR to the downloaded icon_detect/model.pt file.')
    return YOLO(path)


def _detect_boxes_yolo(image: Image.Image, yolo_model_path: str, min_area: int = 100):
    W, H = image.size
    yolo = get_yolo_model(yolo_model_path)
    result = yolo.predict(source=image, conf=0.01, iou=0.1, verbose=False)
    boxes = result[0].boxes.xyxy.cpu().numpy()
    confs = result[0].boxes.conf.cpu().numpy() if hasattr(result[0].boxes, 'conf') else [1.0] * len(boxes)
    def _area(b):
        x1, y1, x2, y2 = b
        return max(0, int(x2) - int(x1)) * max(0, int(y2) - int(y1))
    bbox_list, conf_list = [], []
    for i, b in enumerate(boxes):
        if _area(b) > min_area:
            x1, y1, x2, y2 = [int(v) for v in b]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 > x1 and y2 > y1:
                bbox_list.append((x1, y1, x2, y2))
                conf_list.append(float(confs[i] if i < len(confs) else 1.0))
    return bbox_list, conf_list

def _crop_images_by_bboxes(img_array: np.ndarray, bbox_list: List[Tuple[int,int,int,int]], size: int = 64) -> List[Image.Image]:
    to_pil = ToPILImage()
    crops: List[Image.Image] = []
    for (x1, y1, x2, y2) in bbox_list:
        crop = img_array[y1:y2, x1:x2]
        crop = cv2.resize(crop, (size, size))
        crops.append(to_pil(crop))
    return crops

def _batch_generate_captions(crops: List[Image.Image], model, processor, device: str, model_name: str) -> List[str]:
    captions: List[str] = []
    batch_size = 32
    is_florence = hasattr(model, 'config') and hasattr(model.config, 'name_or_path') and ('florence' in str(model.config.name_or_path).lower())
    prompt = "<CAPTION>" if is_florence or model_name == 'florence2' else "The image shows"
    for i in range(0, len(crops), batch_size):
        batch = crops[i:i + batch_size]
        if device == 'cuda':
            inputs = processor(images=batch, text=[prompt] * len(batch), return_tensors="pt", do_resize=False).to(
                device=device, dtype=torch.float16)
        else:
            inputs = processor(images=batch, text=[prompt] * len(batch), return_tensors="pt").to(device=device)
        with torch.no_grad():
            if is_florence or model_name == 'florence2':
                generated_ids = model.generate(
                    input_ids=inputs.get("input_ids"),
                    pixel_values=inputs.get("pixel_values"),
                    max_new_tokens=20,
                    num_beams=1,
                    do_sample=False,
                    use_cache=False
                )
            else:
                generated_ids = model.generate(
                    input_ids=inputs.get("input_ids"),
                    pixel_values=inputs.get("pixel_values"),
                    attention_mask=inputs.get("attention_mask", None),
                    max_length=100,
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    num_return_sequences=1,
                    use_cache=True,
                    do_sample=False
                )
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
            captions.extend([d.strip() for d in decoded])
    return captions
