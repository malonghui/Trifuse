"""Trifuse inference: attention, spatial anchors, CS fusion and refinement."""
import os
import string
import torch
from torch import nn
import numpy as np
from PIL import Image
import torch.nn.functional as F
from trifuse.attention_utils import _to_np, _nearest_hw_from_tokens
from trifuse.heatmap_utils import stabilize_gui_heatmap, _otsu_threshold, _connected_components, _scale_bbox
from trifuse.modal_heatmaps import caption_heatmap, ocr_heatmap
ocr = None


def get_ocr_engine():
    """Load PP-OCRv4 through either PaddleOCR 2.x or 3.x."""
    global ocr
    if ocr is None:
        from importlib.metadata import version
        from paddleocr import PaddleOCR
        major = int(version('paddleocr').split('.')[0])
        lang = os.environ.get('TRIFUSE_OCR_LANG', 'en')
        device = os.environ.get('TRIFUSE_OCR_DEVICE', 'cpu')
        threshold = float(os.environ.get('TRIFUSE_OCR_TEXT_DET_THRESH', '0.7'))
        if major >= 3:
            ocr = PaddleOCR(
                lang=lang, ocr_version='PP-OCRv4', device=device,
                precision=os.environ.get('TRIFUSE_OCR_PRECISION', 'fp32'),
                enable_mkldnn=False, use_doc_orientation_classify=False,
                use_doc_unwarping=False, use_textline_orientation=False,
                text_det_thresh=threshold,
            )
        else:
            ocr = PaddleOCR(
                lang=lang, ocr_version='PP-OCRv4',
                use_gpu=device.startswith(('gpu', 'cuda')),
                use_angle_cls=False, det_db_thresh=threshold,
                enable_mkldnn=False, show_log=False,
            )
    return ocr


class Trifuse(nn.Module):

    def __init__(self, qwen_vl_model, qwen_vl_processor):
        super().__init__()
        self.model = qwen_vl_model
        self.processor = qwen_vl_processor
        if hasattr(self.model, 'config'):
            self.model.config.attn_implementation = 'eager'
            self.model.config.output_attentions = True
            self.model.config.use_cache = False
            if hasattr(self.model.config, 'text_config'):
                self.model.config.text_config.output_attentions = True

    def input_generate(self, image, question, system_prompt=''):
        from qwen_vl_utils import process_vision_info
        if not isinstance(image, list):
            image = [image]
        sys_prompt = 'You are a GUI-grounding assistant. Given a GUI screenshot, identify the position of the UI element corresponding to the user command. Focus on the command-specific visible text, icon semantics, and spatial relations.'
        if system_prompt:
            sys_prompt = system_prompt
        messages = [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': [{'type': 'image', 'image': img} for img in image] + [{'type': 'text', 'text': question}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt')
        if hasattr(self.model, 'device'):
            inputs = inputs.to(self.model.device)
        elif hasattr(self.model, 'hf_device_map') or hasattr(self.model, 'device_map'):
            try:
                first_param_device = next(self.model.parameters()).device
                inputs = inputs.to(first_param_device)
            except StopIteration:
                inputs = inputs.to('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            try:
                first_param_device = next(self.model.parameters()).device
                inputs = inputs.to(first_param_device)
            except (StopIteration, AttributeError):
                inputs = inputs.to('cuda:0' if torch.cuda.is_available() else 'cpu')
        return inputs

    def prior_judgement(self, question, ocr_result, caption_result):
        """Internal helper."""
        self.prior_result = {'result': False}
        sim_thresh_ocr = 0.7
        sim_thresh_cap = 0.7
        ocr_conf_min = 0.5
        cap_conf_min = 0.5
        best = None
        if isinstance(ocr_result, (list, tuple)):
            for it in ocr_result:
                try:
                    sim = float(it.get('similarity_score', 0.0))
                    conf = float(it.get('confidence', 0.0))
                    if conf < ocr_conf_min:
                        continue
                    weighted = sim
                    if best is None or weighted > best[2]:
                        best = ('ocr', it, weighted)
                except Exception:
                    continue
        if isinstance(caption_result, (list, tuple)):
            for it in caption_result:
                try:
                    sim = float(it.get('similarity_score', 0.0))
                    conf = float(it.get('confidence', 0.0))
                    if conf < cap_conf_min:
                        continue
                    weighted = sim
                    if best is None or weighted > best[2]:
                        best = ('caption', it, weighted)
                except Exception:
                    continue
        if best is None:
            return False
        source, item, weighted = best
        bbox = item.get('bbox', None)
        if not bbox or len(bbox) != 4:
            return False
        self.prior_result = {'result': item.get('similarity_score', 0.0) >= (sim_thresh_cap if source == 'caption' else sim_thresh_ocr), 'source': source, 'bbox': [int(b) for b in bbox], 'point': ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2), 'text': item.get('caption', '') if source == 'caption' else item.get('text', ''), 'score': float(item.get('similarity_score', 0.0)), 'resize': 1.0}

    @torch.no_grad()
    def ground(self, image, question, g_bbox=None, fusion='cs', use_prior=False, use_token_filter=True, token_tau_v=0.0, use_layer_filter=False, use_head_filter=True, localization='two_stage', **fusion_kwargs):
        """
        Trifuse pipeline.
        1. Extract an attention heatmap with query-token and attention-head filtering.
        2. Extract OCR and icon-caption heatmaps on the same patch grid.
        3. Fuse the three modality heatmaps with CS fusion by default.
        4. Refine the fused-map prediction with two-stage crop-and-zoom localization.
        """
        if isinstance(image, str):
            image_size = Image.open(image).size
        elif isinstance(image, Image.Image):
            image_size = image.size
        else:
            raise ValueError('image must be a file path or PIL.Image')
        fused_heatmap, modal_weights, _ = self._run_fusion_stage(
            image=image,
            question=question,
            fusion=fusion,
            use_token_filter=use_token_filter,
            token_tau_v=token_tau_v,
            use_layer_filter=use_layer_filter,
            use_head_filter=use_head_filter,
            fusion_kwargs=fusion_kwargs,
        )
        if use_prior:
            self.prior_judgement(question, self.ocr_scored_result, self.caption_scored_result)
            if self.prior_result.get('result'):
                return self.prior_result
        if localization not in {'two_stage', 'heatmap', 'direct'}:
            raise ValueError('localization must be one of: two_stage, heatmap, direct')
        if localization == 'two_stage':
            result = self.two_stage_localize(
                image=image,
                question=question,
                coarse_heatmap=fused_heatmap,
                image_size=image_size,
                fusion=fusion,
                use_token_filter=use_token_filter,
                token_tau_v=token_tau_v,
                use_layer_filter=use_layer_filter,
                use_head_filter=use_head_filter,
                fusion_kwargs=fusion_kwargs,
            )
        else:
            result = self.localize_from_heatmap(fused_heatmap, image_size=image_size, mode=localization)
        result['modal_weights'] = np.asarray(modal_weights).tolist()
        result['fusion'] = fusion
        result['localization'] = localization
        return result

    def forward(self, image, question, **kwargs):
        return self.ground(image, question, **kwargs)

    def _run_fusion_stage(self, image, question, fusion='cs', use_token_filter=True, token_tau_v=0.0, use_layer_filter=False, use_head_filter=True, fusion_kwargs=None):
        fusion_kwargs = fusion_kwargs or {}
        inputs = self.input_generate(image, question)
        att_fused, attn_fused_shape = self.extract_attention_heatmap(
            inputs=inputs,
            image=image,
            question=question,
            use_token_filter=use_token_filter,
            token_tau_v=token_tau_v,
            use_layer_filter=use_layer_filter,
            use_head_filter=use_head_filter,
        )
        att_fused = stabilize_gui_heatmap(att_fused)
        self.get_ocr_caption(image, question, attn_fused_shape, self.filtered_query_tokens)
        if fusion == 'cs':
            fused_heatmap, modal_weights = self.fuse_modal_heatmap_consensus(
                att_fused,
                self.ocr_heatmap,
                self.caption_heatmap,
                **fusion_kwargs,
            )
        elif fusion == 'average':
            fused_heatmap, modal_weights = self.fuse_modal_heatmap_average(
                att_fused,
                self.ocr_heatmap,
                self.caption_heatmap,
            )
        else:
            raise ValueError('fusion must be one of: cs, average')
        self.attention_heatmap = att_fused
        self.fused_heatmap = fused_heatmap
        self.modal_weights = modal_weights
        return fused_heatmap, modal_weights, attn_fused_shape

    def filter_layers_by_quality(self, layer_maps, topk_layers=None, use_filter=True):
        """Internal helper."""
        if not layer_maps:
            return (torch.tensor([], dtype=torch.long), torch.tensor([]))
        maps = torch.stack(layer_maps, 0)
        if not use_filter:
            return (torch.arange(maps.size(0), device=maps.device), maps)
        flat = maps.view(maps.size(0), -1)
        p = flat / (flat.sum(-1, keepdim=True) + 1e-06)
        ent = -(p * p.clamp_min(1e-09).log()).sum(-1)
        topk_mass = p.topk(max(4, p.size(-1) // 100), dim=-1).values.sum(-1)
        sharp = topk_mass / (ent + 1e-06)
        if topk_layers is None:
            topk_layers = max(4, maps.size(0) // 2)
        k = min(topk_layers, maps.size(0))
        keep_indices = torch.topk(sharp, k=k).indices
        return (keep_indices, maps[keep_indices])

    def _head_spatial_entropy(self, vector, H, W):
        heatmap = _to_np(vector.view(H, W)).astype(np.float32)
        heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
        heatmap = np.maximum(heatmap, 0.0)
        if float(heatmap.sum()) <= 1e-8:
            return float('inf')

        # Paper-aligned spatial entropy: threshold the attention map into
        # connected regions, then compute entropy from each region's attention
        # mass rather than its pixel area.
        mask = heatmap > float(heatmap.mean())
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=np.uint8)
        region_masses = []
        for y in range(h):
            for x in range(w):
                if not mask[y, x] or visited[y, x]:
                    continue
                stack = [(x, y)]
                visited[y, x] = 1
                mass = 0.0
                while stack:
                    cx, cy = stack.pop()
                    mass += float(heatmap[cy, cx])
                    for nx in (cx - 1, cx, cx + 1):
                        for ny in (cy - 1, cy, cy + 1):
                            if nx == cx and ny == cy:
                                continue
                            if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = 1
                                stack.append((nx, ny))
                if mass > 0.0:
                    region_masses.append(mass)
        if not region_masses:
            return float('inf')
        masses = np.asarray(region_masses, dtype=np.float32)
        probs = masses / (masses.sum() + 1e-8)
        return float(-(probs * np.log(probs + 1e-9)).sum())

    @torch.no_grad()
    def extract_attention_heatmap(self, inputs, image, question, use_token_filter=True, token_topk=1, token_tau_v=0.0, use_layer_filter=False, layer_topk=8, layer_filter='quality', band_ratio=(1 / 3, 35 / 36), use_head_filter=True, head_topk=6):
        """Build the attention modality from filtered query tokens and spatially concentrated heads."""
        outputs = self.model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        pos, pos_end = self._get_image_token_span(inputs)
        H, W = self._infer_patch_grid(inputs, pos_end - pos)
        query_indices, query_weights = self._select_query_tokens(
            inputs=inputs,
            outputs=outputs,
            image=image,
            question=question,
            mode='auto' if use_token_filter else 'last',
            topk=token_topk if use_token_filter else 1,
            tau_v=token_tau_v,
        )
        self.filtered_query_tokens = self._decode_query_tokens(inputs, query_indices)
        self.filtered_query_weights = list(query_weights)
        all_attentions = outputs.attentions
        L = len(all_attentions)
        if use_layer_filter and layer_filter == 'band_ratio':
            l0 = int(L * band_ratio[0])
            l1 = int(L * band_ratio[1])
        else:
            l0 = 0
            l1 = L
        if use_head_filter and not use_layer_filter:
            token_maps = []
            for query_idx, token_weight in zip(query_indices, query_weights):
                head_candidates = []
                for layer_idx in range(l0, l1):
                    A = all_attentions[layer_idx]
                    if isinstance(A, (list, tuple)):
                        A = A[0]
                    if A.dim() == 4:
                        A = A[0]
                    heads = A[:, query_idx, pos:pos_end].float()
                    for head_idx in range(heads.size(0)):
                        vec = heads[head_idx]
                        entropy = self._head_spatial_entropy(vec, H, W)
                        head_candidates.append((entropy, vec))
                if not head_candidates:
                    continue
                finite_scores = [s for s, _ in head_candidates if np.isfinite(s)]
                fallback_score = max(finite_scores) + 1.0 if finite_scores else 1.0
                head_candidates = [
                    (s if np.isfinite(s) else fallback_score, vec)
                    for s, vec in head_candidates
                ]
                head_candidates.sort(key=lambda item: item[0])
                selected_heads = head_candidates[:min(max(1, int(head_topk)), len(head_candidates))]
                ent = torch.tensor([s for s, _ in selected_heads], dtype=torch.float32, device=selected_heads[0][1].device)
                weights = torch.softmax(-ent, dim=0)
                token_map = torch.stack([vec * w for w, (_, vec) in zip(weights, selected_heads)], 0).sum(0)
                token_maps.append(token_map * float(token_weight))
            if not token_maps:
                raise ValueError('No valid attention maps extracted from global top-k heads')
            fused = torch.stack(token_maps, 0).sum(0).view(H, W)
            fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-6)
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return fused, (H, W)
        layer_maps = []
        for layer_idx in range(l0, l1):
            A = all_attentions[layer_idx]
            if isinstance(A, (list, tuple)):
                A = A[0]
            if A.dim() == 4:
                A = A[0]
            token_maps = []
            for query_idx, token_weight in zip(query_indices, query_weights):
                heads = A[:, query_idx, pos:pos_end].float()
                head_maps = []
                head_scores = []
                for head_idx in range(heads.size(0)):
                    vec = heads[head_idx]
                    entropy = self._head_spatial_entropy(vec, H, W)
                    head_scores.append(entropy)
                if not np.isfinite(head_scores).any():
                    head_scores = [1.0 for _ in head_scores]
                else:
                    finite_max = max(s for s in head_scores if np.isfinite(s))
                    head_scores = [s if np.isfinite(s) else finite_max + 1.0 for s in head_scores]
                if use_head_filter:
                    k = min(max(1, int(head_topk)), heads.size(0))
                    keep = torch.tensor(np.argsort(head_scores)[:k], device=heads.device, dtype=torch.long)
                else:
                    keep = torch.arange(heads.size(0), device=heads.device)
                ent = torch.tensor([head_scores[int(i.item())] for i in keep], dtype=torch.float32, device=heads.device)
                weights = torch.softmax(-ent, dim=0)
                for w_h, head_idx in zip(weights, keep):
                    head_maps.append(heads[head_idx] * w_h)
                token_map = torch.stack(head_maps, 0).sum(0) * float(token_weight)
                token_maps.append(token_map)
            layer_map = torch.stack(token_maps, 0).sum(0).view(H, W)
            layer_map = (layer_map - layer_map.min()) / (layer_map.max() - layer_map.min() + 1e-6)
            layer_maps.append(layer_map)
        if use_layer_filter:
            if layer_filter == 'quality':
                _, selected_maps = self.filter_layers_by_quality(layer_maps, topk_layers=layer_topk, use_filter=True)
                fused = selected_maps.mean(0)
            else:
                fused = torch.stack(layer_maps, 0).mean(0)
        else:
            fused = torch.stack(layer_maps, 0).mean(0)
        fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-6)
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return fused, (H, W)

    def _get_image_token_span(self, inputs):
        """Internal helper."""
        vs = self.processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
        ve = self.processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
        ids = inputs['input_ids'][0].tolist()
        pos = ids.index(vs) + 1
        pos_end = ids.index(ve)
        return (pos, pos_end)

    def _get_question_end_position(self, inputs, image, question):
        """Internal helper."""
        from qwen_vl_utils import process_vision_info
        if not isinstance(image, list):
            image = [image]
        sys_prompt = 'You are a GUI-grounding assistant. Given a GUI screenshot, identify the position of the UI element corresponding to the user command. Focus on the command-specific visible text, icon semantics, and spatial relations.'
        messages = [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': [{'type': 'image', 'image': img} for img in image] + [{'type': 'text', 'text': question}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)
        question_inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt')
        if hasattr(self.model, 'device'):
            question_inputs = question_inputs.to(self.model.device)
        elif hasattr(self.model, 'hf_device_map') or hasattr(self.model, 'device_map'):
            try:
                first_param_device = next(self.model.parameters()).device
                question_inputs = question_inputs.to(first_param_device)
            except StopIteration:
                question_inputs = question_inputs.to('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            try:
                first_param_device = next(self.model.parameters()).device
                question_inputs = question_inputs.to(first_param_device)
            except (StopIteration, AttributeError):
                question_inputs = question_inputs.to('cuda:0' if torch.cuda.is_available() else 'cpu')
        question_ids = question_inputs['input_ids'][0]
        question_end_pos = question_ids.shape[0] - 1
        return question_end_pos

    def _decode_query_tokens(self, inputs, query_indices):
        token_texts = []
        token_ids = inputs['input_ids'][0]
        for idx in query_indices:
            tok = self.processor.tokenizer.decode([int(token_ids[int(idx)])], skip_special_tokens=True)
            tok = tok.replace('Ġ', '').strip()
            if tok:
                token_texts.append(tok)
        return token_texts

    def _question_token_span(self, token_ids, question, search_start):
        """Find the token span of the raw user command inside the chat-template ids."""
        try:
            q_ids = self.processor.tokenizer(question, add_special_tokens=False).input_ids
        except Exception:
            q_ids = []
        if not q_ids:
            return None
        ids = [int(x) for x in token_ids]
        q_ids = [int(x) for x in q_ids]
        best = None
        max_start = len(ids) - len(q_ids)
        for start in range(max(0, int(search_start)), max_start + 1):
            if ids[start:start + len(q_ids)] == q_ids:
                best = (start, start + len(q_ids))
        return best

    def _infer_patch_grid(self, inputs, img_tok_len):
        """Internal helper."""
        if 'image_grid_thw' in inputs and inputs['image_grid_thw'] is not None:
            thw = inputs['image_grid_thw'][0]
            Hc, Wc = (int(thw[-2]), int(thw[-1]))
            if Hc * Wc / 4 == img_tok_len:
                return (Hc // 2, Wc // 2)
            else:
                return (Hc, Wc)
        Wpix = inputs['pixel_values'].shape[-1]
        Hpix = inputs['pixel_values'].shape[-2]
        Hn, Wn = _nearest_hw_from_tokens(img_tok_len, target_ratio=Wpix / max(Hpix, 1))
        return (Hn, Wn)

    def fuse_modal_heatmap_average(self, att_heatmap, ocr_heatmap, caption_heatmap):
        """Internal helper."""
        att_heatmap = _to_np(att_heatmap)
        ocr_heatmap = _to_np(ocr_heatmap)
        caption_heatmap = _to_np(caption_heatmap)
        att_heatmap_norm = (att_heatmap - att_heatmap.min()) / (att_heatmap.max() - att_heatmap.min() + 1e-08)
        ocr_heatmap_norm = (ocr_heatmap - ocr_heatmap.min()) / (ocr_heatmap.max() - ocr_heatmap.min() + 1e-08)
        caption_heatmap_norm = (caption_heatmap - caption_heatmap.min()) / (caption_heatmap.max() - caption_heatmap.min() + 1e-08)
        modal_weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        fused_heatmap = modal_weights[0] * att_heatmap_norm + modal_weights[1] * ocr_heatmap_norm + modal_weights[2] * caption_heatmap_norm
        fused_heatmap = (fused_heatmap - fused_heatmap.min()) / (fused_heatmap.max() - fused_heatmap.min() + 1e-08)
        return (fused_heatmap, modal_weights)

    def fuse_modal_heatmap_consensus(self, att_heatmap, ocr_heatmap, caption_heatmap, consensus_mode='product', q_attn=0.8, q_ocr=0.8, q_cap=0.8, tau_min=0.35, peak_threshold=0.8, alpha=10.0, beta=2.0, lambda_enhance=0.5, single_peak_conf_min=None, single_peak_support_min=None, eps=1e-06):
        """Fuse modality heatmaps with the paper's Consensus-SinglePeak rule."""
        from scipy.ndimage import maximum_filter

        def _normalize_map(hm):
            hm_np = _to_np(hm).astype(np.float32)
            hm_max = hm_np.max()
            hm_min = hm_np.min()
            if hm_max > hm_min:
                hm_np = (hm_np - hm_min) / (hm_max - hm_min + 1e-08)
            else:
                hm_np = np.zeros_like(hm_np, dtype=np.float32)
            return hm_np
        A = _normalize_map(att_heatmap)
        O = _normalize_map(ocr_heatmap)
        C = _normalize_map(caption_heatmap)
        H, W = A.shape
        window_size = max(3, min(H, W) // 20)
        if window_size % 2 == 0:
            window_size += 1
        if consensus_mode == 'product':
            M_cons = A * O * C
        elif consensus_mode == 'pairwise':
            M_cons = A * O + O * C + A * C
        else:
            stacked = np.stack([A, O, C], axis=0)
            sorted_stack = np.sort(stacked, axis=0)
            top2_mean = (sorted_stack[1, :, :] + sorted_stack[2, :, :]) / 2.0
            M_cons = top2_mean
        M_cons_max = M_cons.max()
        M_cons_min = M_cons.min()
        if M_cons_max > M_cons_min:
            M_cons = (M_cons - M_cons_min) / (M_cons_max - M_cons_min + 1e-08)
        else:
            M_cons = np.zeros_like(M_cons)

        def _detect_peaks_ratio(heatmap, threshold_ratio):
            max_val = float(heatmap.max())
            threshold = max_val * threshold_ratio
            local_max = maximum_filter(heatmap, size=window_size)
            peaks_mask = (heatmap == local_max) & (heatmap >= threshold)
            return peaks_mask.astype(np.float32)

        def _detect_peaks_quantile(heatmap, q):
            t_raw = float(np.quantile(heatmap.astype(np.float64).ravel(), q))
            threshold = max(t_raw, float(tau_min))
            local_max = maximum_filter(heatmap, size=window_size)
            peaks_mask = (heatmap == local_max) & (heatmap >= threshold)
            return peaks_mask.astype(np.float32)
        if peak_threshold is not None:
            pt = float(peak_threshold)
            P_A = _detect_peaks_ratio(A, pt)
            P_O = _detect_peaks_ratio(O, pt)
            P_C = _detect_peaks_ratio(C, pt)
        else:
            P_A = _detect_peaks_quantile(A, float(q_attn))
            P_O = _detect_peaks_quantile(O, float(q_ocr))
            P_C = _detect_peaks_quantile(C, float(q_cap))
        M_single = np.zeros_like(M_cons, dtype=np.float32)
        modalities = [('att', A, P_A, [O, C]), ('ocr', O, P_O, [A, C]), ('caption', C, P_C, [A, O])]
        for mod_name, mod_map, mod_peaks, other_mods in modalities:
            peak_positions = np.where(mod_peaks > 0.5)
            if len(peak_positions[0]) == 0:
                continue
            for y, x in zip(peak_positions[0], peak_positions[1]):
                v_s = float(mod_map[y, x])
                v_others = [float(other_mod[y, x]) for other_mod in other_mods]
                sum_others = sum(v_others)
                S_p = sum_others / (v_s + eps)
                conf_p = 1.0 / (1.0 + np.exp(-(alpha * S_p - beta)))
                gain_factor = 1.0 + lambda_enhance * (2.0 * conf_p - 1.0)
                M_single[y, x] += gain_factor * v_s

        fused = 0.5 * (M_cons + M_single)
        fused_max = fused.max()
        fused_min = fused.min()
        if fused_max > fused_min:
            fused = (fused - fused_min) / (fused_max - fused_min + 1e-08)
        else:
            fused = np.zeros_like(fused)
        modal_weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        return (fused, modal_weights)

    def localize_from_heatmap(self, heatmap, image_size, mode='heatmap', threshold=None, min_component_area=1):
        """
        Convert the fused heatmap into an image-space prediction.

        `direct` returns only the global peak point. `heatmap` thresholds the map,
        selects the connected component that contains the global peak, and returns
        that component as the predicted box.
        """
        Hm = _to_np(heatmap).astype(np.float32)
        if Hm.ndim != 2:
            raise ValueError('heatmap must be a 2D array')
        if Hm.max() > Hm.min():
            Hm = (Hm - Hm.min()) / (Hm.max() - Hm.min() + 1e-08)
        else:
            Hm = np.zeros_like(Hm, dtype=np.float32)
        peak_y, peak_x = np.unravel_index(np.argmax(Hm), Hm.shape)
        point = (int(round(peak_x * image_size[0] / Hm.shape[1])), int(round(peak_y * image_size[1] / Hm.shape[0])))
        if mode == 'direct':
            return {'result': True, 'source': 'fused_heatmap', 'point': point, 'bbox': None, 'score': float(Hm[peak_y, peak_x])}
        tau = float(threshold) if threshold is not None else max(_otsu_threshold(Hm), float(np.quantile(Hm, 0.8)))
        mask = (Hm >= tau).astype(np.uint8)
        comps = [c for c in _connected_components(mask, Hm) if c['size'] >= int(min_component_area)]
        selected = None
        for comp in comps:
            x1, y1, x2, y2 = comp['bbox']
            if x1 <= peak_x < x2 and y1 <= peak_y < y2:
                selected = comp
                break
        if selected is None and comps:
            selected = max(comps, key=lambda c: c['peak'])
        if selected is None:
            x1 = max(0, peak_x - 1)
            y1 = max(0, peak_y - 1)
            x2 = min(Hm.shape[1], peak_x + 2)
            y2 = min(Hm.shape[0], peak_y + 2)
            bbox_hm = (x1, y1, x2, y2)
            score = float(Hm[peak_y, peak_x])
        else:
            bbox_hm = selected['bbox']
            score = float(selected['peak'])
        bbox = _scale_bbox(bbox_hm, Hm.shape, image_size)
        return {'result': True, 'source': 'fused_heatmap', 'point': point, 'bbox': list(bbox), 'score': score, 'threshold': tau}

    def two_stage_localize(self, image, question, coarse_heatmap, image_size, fusion='cs', use_token_filter=True, token_tau_v=0.0, use_layer_filter=False, use_head_filter=True, fusion_kwargs=None, crop_pad_ratio=0.15):
        source_image = Image.open(image).convert('RGB') if isinstance(image, str) else image.convert('RGB')
        coarse = self.localize_from_heatmap(coarse_heatmap, image_size=image_size, mode='direct')
        coarse_point = coarse.get('point')
        if coarse_point is None:
            coarse = self.localize_from_heatmap(coarse_heatmap, image_size=image_size, mode='heatmap')
            x1c, y1c, x2c, y2c = coarse['bbox']
            coarse_point = ((x1c + x2c) // 2, (y1c + y2c) // 2)
        W, H = image_size
        crop_w = max(1, W // 2)
        crop_h = max(1, H // 2)
        cx, cy = int(coarse_point[0]), int(coarse_point[1])
        x1 = max(0, min(W - crop_w, cx - crop_w // 2))
        y1 = max(0, min(H - crop_h, cy - crop_h // 2))
        x2 = min(W, x1 + crop_w)
        y2 = min(H, y1 + crop_h)
        crop = source_image.crop((x1, y1, x2, y2))
        refined_input = crop.resize(image_size, Image.Resampling.LANCZOS)
        fine_heatmap, fine_weights, _ = self._run_fusion_stage(
            image=refined_input,
            question=question,
            fusion=fusion,
            use_token_filter=use_token_filter,
            token_tau_v=token_tau_v,
            use_layer_filter=use_layer_filter,
            use_head_filter=use_head_filter,
            fusion_kwargs=fusion_kwargs,
        )
        fine = self.localize_from_heatmap(fine_heatmap, image_size=refined_input.size, mode='heatmap')
        sx = crop.size[0] / max(refined_input.size[0], 1)
        sy = crop.size[1] / max(refined_input.size[1], 1)
        fine_bbox = fine.get('bbox')
        if fine_bbox is not None:
            fine['bbox'] = [
                int(round(fine_bbox[0] * sx + x1)),
                int(round(fine_bbox[1] * sy + y1)),
                int(round(fine_bbox[2] * sx + x1)),
                int(round(fine_bbox[3] * sy + y1)),
            ]
        fine_point = fine.get('point')
        if fine_point is not None:
            fine['point'] = (int(round(fine_point[0] * sx + x1)), int(round(fine_point[1] * sy + y1)))
        fine['source'] = 'two_stage_fused_heatmap'
        fine['coarse_bbox'] = coarse.get('bbox')
        fine['coarse_point'] = coarse_point
        fine['crop_bbox'] = [int(x1), int(y1), int(x2), int(y2)]
        fine['crop_resized_size'] = [int(refined_input.size[0]), int(refined_input.size[1])]
        fine['stage2_modal_weights'] = np.asarray(fine_weights).tolist()
        return fine

    def get_ocr_caption(self, image, question, attn_fused_shape, filtered_query_tokens=None, g_bbox=None):
        self.ocr_heatmap, self.ocr_scored_result, self.ocr_meta = ocr_heatmap(
            get_ocr_engine(),
            image,
            question,
            filtered_query_tokens,
            attn_fused_shape,
            return_meta=True,
        )
        self.caption_heatmap, self.caption_scored_result, self.caption_meta = caption_heatmap(
            image,
            question,
            filtered_query_tokens,
            attn_fused_shape,
            return_meta=True,
        )

    @torch.no_grad()
    def _is_content_token(self, token_id, token_str):
        """
        Quick heuristic to drop punctuation/special tokens when selecting query tokens.
        """
        if token_id in getattr(self.processor.tokenizer, 'all_special_ids', []):
            return False
        if not token_str:
            return False
        cleaned = token_str.replace('Ġ', '').strip()
        if not cleaned:
            return False
        if cleaned in {'<im_start>', '<im_end>'}:
            return False
        if cleaned[0] == '<' and cleaned[-1] == '>':
            return False
        if all((ch in string.punctuation for ch in cleaned)):
            return False
        return True

    def _select_query_tokens(self, inputs, outputs, image, question, mode='auto', topk=1, tau_v=0.0):
        """
        Select query token indices for grounding attention.

        Returns:
            indices (List[int]), weights (List[float])
        """
        topk = max(1, int(topk))
        pos, pos_end = self._get_image_token_span(inputs)
        seq_len = inputs['input_ids'].shape[1]
        token_ids = inputs['input_ids'][0]
        question_span = self._question_token_span(token_ids.tolist(), question, pos_end)
        if image is None:
            question_end_pos = seq_len - 1
        else:
            question_end_pos = self._get_question_end_position(inputs, image, question)
        if question_span is not None:
            candidate_start, candidate_end = question_span
        else:
            candidate_start, candidate_end = pos_end + 1, question_end_pos
        fallback_idx = max(candidate_start, min(seq_len - 1, candidate_end - 1))
        if fallback_idx < pos_end:
            fallback_idx = min(seq_len - 1, max(pos_end, fallback_idx))
        use_visual = mode in {'auto', 'visual_sink'}
        hidden_states = getattr(outputs, 'hidden_states', None)
        if not use_visual or hidden_states is None or len(hidden_states) < 2:
            return ([fallback_idx], [1.0])
        candidate_indices = []
        for idx in range(candidate_start, candidate_end):
            tid = int(token_ids[idx])
            tok = self.processor.tokenizer.decode([tid], skip_special_tokens=False)
            if not self._is_content_token(tid, tok):
                continue
            candidate_indices.append(idx)
        if not candidate_indices:
            return ([fallback_idx], [1.0])
        num_layers = len(hidden_states)
        start_layer = max(1, num_layers - 3)
        selected_layers = list(range(start_layer, num_layers))
        scores = {}
        eps = 1e-06
        for layer_idx in selected_layers:
            layer_hidden = hidden_states[layer_idx][0]
            if pos_end <= pos:
                continue
            vis_tokens = layer_hidden[pos:pos_end]
            if vis_tokens.numel() == 0:
                continue
            vis_tokens = F.normalize(vis_tokens, dim=-1)
            for idx in candidate_indices:
                h_q = F.normalize(layer_hidden[idx], dim=-1)
                cos = F.cosine_similarity(h_q.unsqueeze(0), vis_tokens, dim=-1)
                active = cos > float(tau_v)
                score = cos[active].sum().item() if bool(active.any()) else 0.0
                scores[idx] = scores.get(idx, 0.0) + score
        if not scores:
            return ([fallback_idx], [1.0])
        sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        selected = sorted_items[:min(topk, len(sorted_items))]
        weights = torch.tensor([max(s, 0.0) for _, s in selected], dtype=torch.float32)
        if float(weights.sum()) <= eps:
            weights = torch.ones_like(weights)
        weights = (weights / weights.sum()).tolist()
        indices = [idx for idx, _ in selected]
        return (indices, weights)
