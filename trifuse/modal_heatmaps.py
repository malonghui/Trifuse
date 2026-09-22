"""OCR and icon-caption heatmap extraction for Trifuse."""
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

_BGE_MODEL = None


def _normalize_heatmap(hm: np.ndarray) -> np.ndarray:
    hm = np.nan_to_num(np.asarray(hm, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if hm.max() > hm.min():
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    return hm


def _filter_scored_results(
    scored_results: Sequence[Dict],
    min_similarity: float,
    min_relevance: float,
    top_k: Optional[int],
) -> List[Dict]:
    filtered = [
        item for item in scored_results
        if float(item.get("similarity_score", 0.0)) >= float(min_similarity)
        and float(item.get("relevance_score", 0.0)) >= float(min_relevance)
    ]
    filtered.sort(key=lambda item: float(item.get("relevance_score", 0.0)), reverse=True)
    if top_k is not None and int(top_k) > 0:
        filtered = filtered[: int(top_k)]
    return filtered


def _topk_stats(vals: Sequence[float], k: int = 3) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    arr = np.asarray(vals, dtype=np.float32)
    arr.sort()
    top = arr[-k:] if arr.size >= k else arr
    return float(top.max()), float(top.mean())


def _heatmap_peakedness(hm: np.ndarray, eps: float = 1e-8) -> float:
    h = np.asarray(hm, dtype=np.float32)
    if h.max() <= h.min() + eps:
        return 0.0
    p = _normalize_heatmap(h)
    p = p / (p.sum() + eps)
    entropy = -(p * np.log(p + eps)).sum()
    entropy_max = np.log(p.size + eps)
    return float((h.max() / (h.mean() + eps)) + (1.0 - entropy / (entropy_max + eps)))


def _get_bge_model(model_name_or_path: Optional[str] = None):
    global _BGE_MODEL
    if _BGE_MODEL is None:
        from sentence_transformers import SentenceTransformer

        path = os.environ.get("BGE_MODEL_PATH") or model_name_or_path or "BAAI/bge-m3"
        _BGE_MODEL = SentenceTransformer(path)
    return _BGE_MODEL


def _query_texts(filtered_query_tokens: Optional[Sequence[str]], question: str) -> List[str]:
    texts = [str(t).strip() for t in (filtered_query_tokens or []) if str(t).strip()]
    return texts or [question]


def _semantic_scores(question: str, candidate_texts: Sequence[str], filtered_query_tokens: Optional[Sequence[str]]) -> np.ndarray:
    if not candidate_texts:
        return np.zeros((0,), dtype=np.float32)
    from sentence_transformers.util import cos_sim

    anchors = _query_texts(filtered_query_tokens, question)
    model = _get_bge_model()
    embeddings = model.encode(list(anchors) + list(candidate_texts), convert_to_tensor=True)
    anchor_emb = embeddings[: len(anchors)]
    cand_emb = embeddings[len(anchors) :]
    sims = cos_sim(cand_emb, anchor_emb).detach().cpu().numpy()
    return sims.max(axis=1).astype(np.float32)


def _load_image(image) -> Image.Image:
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(image).convert("RGB")


def _bbox_to_xyxy(bbox) -> Optional[Tuple[int, int, int, int]]:
    try:
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        if len(bbox) == 4 and isinstance(bbox[0], (list, tuple, np.ndarray)):
            xs = [float(p[0]) for p in bbox]
            ys = [float(p[1]) for p in bbox]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
        else:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    except Exception:
        return None


def _clip_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(x1 + 1, min(width, int(x2)))
    y2 = max(y1 + 1, min(height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _project_regions_to_grid(
    width: int,
    height: int,
    grid_shape: Tuple[int, int],
    scored_results: Sequence[Dict],
) -> np.ndarray:
    """Project region relevance to the visual patch grid by overlap aggregation."""
    grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
    heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)
    cell_w = float(width) / float(grid_w)
    cell_h = float(height) / float(grid_h)
    for item in scored_results:
        bbox = _clip_bbox(tuple(item["bbox"]), width, height)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        score = float(item.get("relevance_score", item.get("similarity_score", 0.0)))
        gx1 = max(0, int(np.floor(x1 / cell_w)))
        gy1 = max(0, int(np.floor(y1 / cell_h)))
        gx2 = min(grid_w - 1, int(np.floor((x2 - 1) / cell_w)))
        gy2 = min(grid_h - 1, int(np.floor((y2 - 1) / cell_h)))
        for gy in range(gy1, gy2 + 1):
            cy1 = gy * cell_h
            cy2 = (gy + 1) * cell_h
            overlap_h = max(0.0, min(y2, cy2) - max(y1, cy1))
            for gx in range(gx1, gx2 + 1):
                cx1 = gx * cell_w
                cx2 = (gx + 1) * cell_w
                overlap_w = max(0.0, min(x2, cx2) - max(x1, cx1))
                overlap = overlap_w * overlap_h
                if overlap > 0.0:
                    heatmap[gy, gx] += score * float(overlap / (cell_w * cell_h + 1e-8))
    return _normalize_heatmap(heatmap)


def _parse_paddle_result(result) -> Tuple[List[str], List[Tuple[int, int, int, int]], List[float]]:
    if not result:
        return [], [], []
    first = result[0]
    if isinstance(first, dict):
        texts = first.get("rec_texts", None)
        if texts is None:
            texts = first.get("texts", [])
        boxes = first.get("rec_boxes", None)
        if boxes is None:
            boxes = first.get("dt_polys", None)
        if boxes is None:
            boxes = first.get("boxes", [])
        scores = first.get("rec_scores", None)
        if scores is None:
            scores = first.get("scores", [1.0] * len(texts))
    else:
        texts, boxes, scores = [], [], []
        for item in first:
            if len(item) >= 2:
                boxes.append(item[0])
                rec = item[1]
                texts.append(rec[0] if isinstance(rec, (list, tuple)) else rec)
                scores.append(rec[1] if isinstance(rec, (list, tuple)) and len(rec) > 1 else 1.0)
    parsed_texts, parsed_boxes, parsed_scores = [], [], []
    for text, box, score in zip(texts, boxes, scores):
        xyxy = _bbox_to_xyxy(box)
        if xyxy is None:
            continue
        parsed_texts.append(str(text) if text is not None else "")
        parsed_boxes.append(xyxy)
        parsed_scores.append(float(score) if score is not None else 0.0)
    return parsed_texts, parsed_boxes, parsed_scores


def ocr_heatmap(
    ocr,
    image,
    question: str,
    filtered_query_tokens: Optional[Sequence[str]],
    grid_shape: Tuple[int, int],
    return_meta: bool = True,
    min_similarity: float = -1.0,
    min_relevance: float = 0.0,
    top_k: Optional[int] = None,
):
    img = _load_image(image)
    width, height = img.size
    img_gray = ImageOps.grayscale(img)
    img_binary = img_gray.point(lambda x: 255 if x > 170 else 0)
    ocr_input = np.stack((np.asarray(img_binary),) * 3, axis=-1)
    result = list(ocr.predict(ocr_input)) if hasattr(ocr, 'predict') else ocr.ocr(ocr_input, cls=False)
    texts, boxes, confidences = _parse_paddle_result(result)
    sims = _semantic_scores(question, texts, filtered_query_tokens)

    scored_results = []
    for text, bbox, conf, sim in zip(texts, boxes, confidences, sims):
        rel = max(float(sim), 0.0) * float(conf)
        scored_results.append({
            "text": text,
            "bbox": [int(v) for v in bbox],
            "confidence": float(conf),
            "similarity_score": float(sim),
            "relevance_score": rel,
        })
    kept_results = _filter_scored_results(scored_results, min_similarity, min_relevance, top_k)
    heatmap = _project_regions_to_grid(width, height, grid_shape, kept_results)
    if not return_meta:
        return heatmap, kept_results
    vals = [float(x["relevance_score"]) for x in scored_results]
    kept_vals = [float(x["relevance_score"]) for x in kept_results]
    top_sim, mean_topk = _topk_stats(vals)
    meta = {
        "peak": float(heatmap.max()),
        "peakedness": _heatmap_peakedness(heatmap),
        "top_sim": top_sim,
        "mean_topk_sim": mean_topk,
        "num_spans": len(scored_results),
        "num_kept": len(kept_results),
        "kept_top_sim": max(kept_vals) if kept_vals else 0.0,
        "query_tokens": _query_texts(filtered_query_tokens, question),
        "all_results": scored_results,
    }
    return heatmap, kept_results, meta


def _get_caption_predictions(
    image: Image.Image,
    yolo_model_path: Optional[str],
    caption_model_name: str,
    caption_model_path: Optional[str],
    min_area: int,
) -> Tuple[List[Tuple[int, int, int, int]], List[float], List[str]]:
    from .icon_models import (
        _batch_generate_captions,
        _crop_images_by_bboxes,
        _detect_boxes_yolo,
        get_caption_model_processor,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_array = np.array(image)
    bbox_list, conf_list = _detect_boxes_yolo(image, yolo_model_path, min_area=min_area)
    if not bbox_list:
        return [], [], []
    mp = get_caption_model_processor(caption_model_name, caption_model_path, device)
    caption_model, processor = mp["model"], mp["processor"]
    crops = _crop_images_by_bboxes(img_array, bbox_list, size=64)
    if not crops:
        return [], [], []
    captions = _batch_generate_captions(crops, caption_model, processor, device, caption_model_name)
    return bbox_list, conf_list, captions


def caption_heatmap(
    image,
    question: str,
    filtered_query_tokens: Optional[Sequence[str]],
    grid_shape: Tuple[int, int],
    yolo_model_path: Optional[str] = None,
    caption_model_name: str = "florence2",
    caption_model_path: Optional[str] = None,
    return_meta: bool = True,
    min_area: int = 100,
    min_similarity: float = -1.0,
    min_relevance: float = 0.0,
    top_k: Optional[int] = None,
):
    img = _load_image(image)
    width, height = img.size
    boxes, confidences, captions = _get_caption_predictions(
        img,
        yolo_model_path,
        caption_model_name,
        caption_model_path,
        min_area,
    )
    if not boxes:
        heatmap = np.zeros((int(grid_shape[0]), int(grid_shape[1])), dtype=np.float32)
        empty = []
        return (heatmap, empty, None) if return_meta else (heatmap, empty)
    sims = _semantic_scores(question, captions, filtered_query_tokens)

    scored_results = []
    for bbox, caption, conf, sim in zip(boxes, captions, confidences, sims):
        rel = max(float(sim), 0.0) * float(conf)
        scored_results.append({
            "bbox": [int(v) for v in bbox],
            "caption": caption,
            "confidence": float(conf),
            "similarity_score": float(sim),
            "relevance_score": rel,
        })
    kept_results = _filter_scored_results(scored_results, min_similarity, min_relevance, top_k)
    heatmap = _project_regions_to_grid(width, height, grid_shape, kept_results)
    if not return_meta:
        return heatmap, kept_results
    vals = [float(x["relevance_score"]) for x in scored_results]
    kept_vals = [float(x["relevance_score"]) for x in kept_results]
    top_sim, mean_topk = _topk_stats(vals)
    meta = {
        "peak": float(heatmap.max()),
        "peakedness": _heatmap_peakedness(heatmap),
        "top_sim": top_sim,
        "mean_topk_sim": mean_topk,
        "num_regions": len(scored_results),
        "num_kept": len(kept_results),
        "kept_top_sim": max(kept_vals) if kept_vals else 0.0,
        "query_tokens": _query_texts(filtered_query_tokens, question),
        "all_results": scored_results,
    }
    return heatmap, kept_results, meta
