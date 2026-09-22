# How Trifuse works

Trifuse maps a screenshot and a natural-language instruction to an image-space click point. It uses pretrained models without task-specific GUI grounding fine-tuning.

## 1. Attention modality

Qwen2.5-VL processes the screenshot and instruction with explicit attention outputs. Query tokens are ranked by visual relevance from late hidden states. The default path selects the top token, ranks attention heads by spatial entropy over connected regions, and aggregates the top six heads. A heatmap stabilization step combines the response with a positive discrete Laplacian.

## 2. Text and icon anchors

PP-OCRv4 supplies recognized text boxes and confidence values. An icon detector supplies icon regions, and Florence-2 describes the cropped icons. BGE-M3 measures semantic similarity between the selected query token(s) and each text or caption. Nonnegative similarity multiplied by detection/recognition confidence supplies region relevance.

Each region contributes to visual-grid cells according to its overlap with the cell. The resulting text and icon heatmaps share the attention map's patch grid.

## 3. Consensus-SinglePeak fusion

The three normalized maps first form a consensus map by elementwise multiplication. For each modality, local maxima above 0.8 times that modality's maximum are candidate peaks. At each candidate, the other modalities provide a support ratio. A sigmoid with `alpha=10` and `beta=2` turns this support into an amplification factor controlled by `lambda_enhance=0.5`.

The final map combines the normalized consensus with the accumulated supported single-modality peaks and is normalized again. This retains useful sharp responses when one modality is weak.

The public API retains optional fusion/localization switches from the implementation, but the CLI uses the paper path: CS fusion, token and head filtering, and two-stage localization. No ablation runners are distributed.

## 4. Two-stage localization

The coarse fused-map maximum centers a crop of half the original image width and height, clipped to image boundaries. The crop is enlarged to the original image size and the same fusion pipeline runs once more. A thresholded connected component containing the fine heatmap maximum determines a region. Its point and box are transformed back into the original screenshot's pixel coordinates.

## Output

| Field | Meaning |
|---|---|
| `point` | `[x, y]` click point in original-image pixels |
| `bbox` | `[x1, y1, x2, y2]` heatmap component region in original-image pixels |
| `source` | Localization path used |
| `score` | Normalized heatmap peak, not a confidence probability |
| `result` | Indicates that the procedure produced coordinates, not correctness or target presence |
| `crop_bbox` | Refinement crop in the original screenshot |
| `coarse_point` | First-stage click point |
| `modal_weights` | Legacy equal-weight metadata for attention, OCR, and caption; CS is nonlinear and does not learn these weights |

Trifuse always produces a coordinate prediction. It does not implement a reliable no-target / abstention classifier. Performance depends on the attention backbone, OCR quality, icon detection, and caption quality; errors in these components propagate to the fused map.
