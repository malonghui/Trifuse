"""Ground one instruction in one screenshot."""
import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True, help='Screenshot path')
    parser.add_argument('--query', required=True, help='Target element instruction')
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--icon-detector', default=os.environ.get('TRIFUSE_ICON_DETECTOR'))
    parser.add_argument('--caption-model', default=os.environ.get('FLORENCE_MODEL_PATH'))
    parser.add_argument('--caption-processor', default=os.environ.get('FLORENCE_PROCESSOR_PATH', 'microsoft/Florence-2-base'))
    parser.add_argument('--embedding-model', default=os.environ.get('BGE_MODEL_PATH', 'BAAI/bge-m3'))
    parser.add_argument('--ocr-device', default=os.environ.get('TRIFUSE_OCR_DEVICE', 'cpu'))
    parser.add_argument('--output', type=Path, default=Path('outputs/prediction.json'))
    parser.add_argument('--visualization', type=Path, help='Optional annotated PNG path')
    args = parser.parse_args(argv)
    if not args.image.is_file():
        parser.error(f'Screenshot does not exist: {args.image}')
    if not args.query.strip():
        parser.error('--query must not be empty')
    if not args.icon_detector or not Path(args.icon_detector).is_file():
        parser.error('--icon-detector must point to icon_detect/model.pt')
    if not args.caption_model:
        parser.error('--caption-model must point to the OmniParser Florence checkpoint')
    os.environ.update(
        TRIFUSE_ICON_DETECTOR=str(Path(args.icon_detector).resolve()),
        FLORENCE_MODEL_PATH=args.caption_model,
        FLORENCE_PROCESSOR_PATH=args.caption_processor,
        BGE_MODEL_PATH=args.embedding_model,
        TRIFUSE_OCR_DEVICE=args.ocr_device,
    )
    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from .core import Trifuse

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, device_map='auto',
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        attn_implementation='eager',
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    grounder = Trifuse(model, processor).eval()
    result = grounder.ground(str(args.image), args.query)
    payload = {'image': args.image.name, 'query': args.query, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if args.visualization:
        with Image.open(args.image) as source:
            canvas = source.convert('RGB')
        draw = ImageDraw.Draw(canvas)
        if result.get('bbox') is not None:
            draw.rectangle(result['bbox'], outline='red', width=3)
        x, y = result['point']
        draw.ellipse((x-5, y-5, x+5, y+5), fill='red', outline='white', width=1)
        args.visualization.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(args.visualization, format='PNG')
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
