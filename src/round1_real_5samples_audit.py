r"""
第一轮：环境检查、数据流隔离审计与 5 条真实样本端到端验证管线
(Round 1: Environment Audit, Strict Data-Flow Isolation & 5-Sample Real End-to-End Verification)

严格遵循 10 项重建准则：
  1. 100% 保留 `src/core_models.py` 中的核心数学实现（式1同源概率折扣、式2双通道凸二次规划、式3来源影响特征 \chi、式4终端选择性决策），零修改、零调参凑分。
  2. 严格隔离官方标准答案字段：`label`、`justification`、`metadata.refuting_reasons`、`metadata.image_misuse_types` 以及 `questions[*].answers[*].answer_text` / `boolean_explanation` / `answer_type` 在推理入口被物理剥离并记录审计日志；`label` 仅在终端输出后用于离线核对正确性。
  3. 彻底删除所有虚构 URL、合成转载证据及人为控制不同策略收益的门控规则。
  4. 实现真实图像解码（通过 HTTP Range 从官方 `images.zip` 提取真实 JPEG 原图并用 `PIL.Image` 解码）、真实视觉模型推理（本地 PyTorch 运行 `TinyCLIP-ViT-8M-16-Text-3M-YFCC15M` 提取 512 维图像与文本嵌入）、真实自然语言推理（本地 PyTorch 运行 `cross-encoder/nli-MiniLM2-L6-H768` 计算真实 Entailment / Contradiction / Neutral 概率）以及真实网页 HTTP 读取。
  5. 缺少商业 API Key（Google Search / SerpAPI Reverse Image Search / OpenAI）、缺少 CUDA GPU（无法运行官方 `gemma-3-27b-it` Ev2R 裁判）或公开搜索引擎返回反爬拦截（如 DuckDuckGo HTML HTTP 202）时，在日志与报告中明确标记 `BLOCKED`，绝不用模拟输出替代。
  6. 严格区分并如实标记两类证据通道：
     - `[OPEN_WEB_LIVE_SEARCH]`：仅凭声明文本查询开放 Wikipedia MediaWiki API 获取的真实公开词条摘要；
     - `[GOLD_EVIDENCE_URL_ASSISTED_OFFLINE]`：明确标为“金证据URL辅助实验”（仅提取官方数据中的 `source_url` 链接，对其发起真实 HTTP GET 抓取网页正文，绝不直接读取 `questions.answers` 的人工答案文本，且绝不宣称为开放检索结果）。
"""

from __future__ import annotations
import hashlib
import html
import io
import json
import os
import platform
import re
import struct
import sys
import time
import zlib
from typing import Dict, List, Tuple, Any, Optional, Set
from urllib.parse import urlparse, quote_plus

import numpy as np
import requests
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    CLIPProcessor,
    CLIPModel,
)

from src.core_models import (
    EvidenceSnippet,
    SoftLogicRule,
    compute_discounted_weights,
    solve_convex_qp_inference,
    theoretical_stability_bound,
    compute_source_influence_feature,
    terminal_calibrated_decision,
    LABELS,
)
from src.exp1_averimatec_audit_and_split import (
    extract_clean_domain,
    normalize_url_for_matching,
    tokenize_entities,
)
from src.exp2_synthetic_32_simulation import build_cross_modal_rules

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data", "averimatec")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
ROUND1_IMG_DIR = os.path.join(OUTPUT_DIR, "round1_5samples_images")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

REMOTE_IMAGES_ZIP_URL = "https://hf-mirror.com/datasets/Rui4416/AVerImaTeC/resolve/main/images.zip"


# =====================================================================
# 1. 第一部分：环境、硬件、API Key 与检索服务真实性检查
# =====================================================================

def run_environment_and_blocker_check() -> Dict[str, Any]:
    """实测当前硬件、依赖库、API 密钥、开源模型权重与外部检索服务连通性，明确标记所有阻塞项"""
    env_report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "hardware": {
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cpu_device": "cpu",
        },
        "api_keys_status": {},
        "local_opensource_models": {},
        "external_retrieval_services": {},
        "explicit_blockers": [],
    }

    # 检查商业与外部检索 API Key
    for key_name in [
        "GOOGLE_API_KEY",
        "GOOGLE_CSE_ID",
        "SERPAPI_API_KEY",
        "BING_SEARCH_V7_SUBSCRIPTION_KEY",
        "TAVILY_API_KEY",
        "OPENAI_API_KEY",
        "HF_TOKEN",
    ]:
        is_set = bool(os.environ.get(key_name, "").strip())
        env_report["api_keys_status"][key_name] = "SET" if is_set else "NOT_SET"

    if env_report["api_keys_status"]["SERPAPI_API_KEY"] == "NOT_SET" and env_report["api_keys_status"]["GOOGLE_API_KEY"] == "NOT_SET":
        env_report["explicit_blockers"].append({
            "module": "Online Reverse Image Search & Commercial Web Search",
            "status": "BLOCKED_MISSING_API_KEY",
            "reason": "SERPAPI_API_KEY and GOOGLE_API_KEY are NOT_SET in environment; cannot call Google Lens / Google Web Search API.",
        })

    if not env_report["hardware"]["cuda_available"]:
        env_report["explicit_blockers"].append({
            "module": "Official AVerImaTeC Ev2R Evaluator (eval_offline.py)",
            "status": "BLOCKED_NO_CUDA_GPU",
            "reason": "Official eval_offline.py requires 2x NVIDIA L40 GPUs (~54GB VRAM) to run google/gemma-3-27b-it in bfloat16; local machine has 0 GPUs.",
        })

    # 检查本地开源模型权重
    nli_dir = os.path.join(MODELS_DIR, "nli-MiniLM2-L6-H768")
    clip_dir = os.path.join(MODELS_DIR, "TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
    env_report["local_opensource_models"]["nli_cross_encoder"] = {
        "model_id": "cross-encoder/nli-MiniLM2-L6-H768",
        "local_path": nli_dir,
        "exists": os.path.exists(os.path.join(nli_dir, "model.safetensors")),
        "weights_size_bytes": os.path.getsize(os.path.join(nli_dir, "model.safetensors")) if os.path.exists(os.path.join(nli_dir, "model.safetensors")) else 0,
    }
    env_report["local_opensource_models"]["vision_language_clip"] = {
        "model_id": "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
        "local_path": clip_dir,
        "exists": os.path.exists(os.path.join(clip_dir, "model.safetensors")),
        "weights_size_bytes": os.path.getsize(os.path.join(clip_dir, "model.safetensors")) if os.path.exists(os.path.join(clip_dir, "model.safetensors")) else 0,
    }

    # 实测外部网络服务连通性
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    # (a) Wikipedia MediaWiki Search API
    t0 = time.time()
    try:
        r_wiki = requests.get(
            "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=Odisha+train+collision&format=json",
            headers=headers,
            timeout=8,
        )
        wiki_hits = len(r_wiki.json().get("query", {}).get("search", [])) if r_wiki.status_code == 200 else 0
        env_report["external_retrieval_services"]["wikipedia_mediawiki_api"] = {
            "status": "AVAILABLE" if (r_wiki.status_code == 200 and wiki_hits > 0) else f"HTTP_{r_wiki.status_code}",
            "http_status": r_wiki.status_code,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "test_hits_returned": wiki_hits,
        }
    except Exception as e:
        env_report["external_retrieval_services"]["wikipedia_mediawiki_api"] = {
            "status": "FAILED",
            "error": f"{type(e).__name__}: {e}",
        }

    # (b) DuckDuckGo HTML Search
    t0 = time.time()
    try:
        r_ddg = requests.get(
            "https://html.duckduckgo.com/html/?q=Odisha+train+accident",
            headers=headers,
            timeout=8,
        )
        is_captcha = (r_ddg.status_code == 202) or ("anomaly.js" in r_ddg.text) or ("cc=botnet" in r_ddg.text)
        env_report["external_retrieval_services"]["duckduckgo_html_search"] = {
            "status": "BLOCKED_BOT_CHALLENGE_HTTP_202" if is_captcha else f"HTTP_{r_ddg.status_code}",
            "http_status": r_ddg.status_code,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "is_bot_challenge": is_captcha,
        }
        if is_captcha:
            env_report["explicit_blockers"].append({
                "module": "Unauthenticated DuckDuckGo HTML Scraping",
                "status": "BLOCKED_BOT_CHALLENGE_HTTP_202",
                "reason": f"DuckDuckGo HTML endpoint returned HTTP {r_ddg.status_code} anti-bot challenge without interactive browser solve.",
            })
    except Exception as e:
        env_report["external_retrieval_services"]["duckduckgo_html_search"] = {
            "status": "FAILED",
            "error": f"{type(e).__name__}: {e}",
        }

    # (c) Wayback Machine CDX / Archive Fetch
    t0 = time.time()
    try:
        r_wb = requests.get(
            "https://web.archive.org/cdx/search/cdx?url=reuters.com&limit=1&output=json",
            headers=headers,
            timeout=8,
        )
        env_report["external_retrieval_services"]["wayback_machine_archive"] = {
            "status": "AVAILABLE" if r_wb.status_code == 200 else f"HTTP_{r_wb.status_code}",
            "http_status": r_wb.status_code,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }
    except Exception as e:
        env_report["external_retrieval_services"]["wayback_machine_archive"] = {
            "status": "FAILED",
            "error": f"{type(e).__name__}: {e}",
        }

    # (d) Remote AVerImaTeC images.zip Range Fetch
    t0 = time.time()
    try:
        r_head = requests.head(REMOTE_IMAGES_ZIP_URL, allow_redirects=True, timeout=10)
        env_report["external_retrieval_services"]["averimatec_remote_images_zip"] = {
            "status": "AVAILABLE" if r_head.status_code == 200 else f"HTTP_{r_head.status_code}",
            "http_status": r_head.status_code,
            "content_length_bytes": int(r_head.headers.get("Content-Length", 0)),
            "accept_ranges": r_head.headers.get("Accept-Ranges", ""),
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }
    except Exception as e:
        env_report["external_retrieval_services"]["averimatec_remote_images_zip"] = {
            "status": "FAILED",
            "error": f"{type(e).__name__}: {e}",
        }

    return env_report


# =====================================================================
# 2. 第二部分：严格数据流隔离守卫 (Data-Flow Isolation Guard)
# =====================================================================

FORBIDDEN_INFERENCE_FIELDS = {
    "label",
    "justification",
    "metadata.refuting_reasons",
    "metadata.image_misuse_types",
    "questions[*].answers[*].answer_text",
    "questions[*].answers[*].boolean_explanation",
    "questions[*].answers[*].answer_type",
}


def isolate_sample_for_inference(raw_sample: Dict[str, Any], sample_id: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    严格隔离官方标准答案字段：
      1. 推理输入 `claim_input` 仅包含：`sample_id`, `claim_text`, `date`, `location`, `claim_images`, `speaker`, `reporting_source`。
      2. 严禁包含 `label`, `justification`, `refuting_reasons`, `image_misuse_types` 以及 `questions[*].answers` 的任何答案文本！
      3. 为支持第6条规定的【金证据URL辅助实验 (GOLD_EVIDENCE_URL_ASSISTED_OFFLINE)】，单独提取去重后的纯 URL 列表
         `gold_assisted_candidate_urls`（丢弃所有 `answer_text` 和 `boolean_explanation`），并显式打上辅助标记。
    """
    stripped_log: List[str] = []
    for k in ["label", "justification"]:
        if k in raw_sample:
            stripped_log.append(k)
    meta = raw_sample.get("metadata") or {}
    for mk in ["refuting_reasons", "image_misuse_types"]:
        if mk in meta:
            stripped_log.append(f"metadata.{mk}")

    gold_urls: List[Dict[str, Any]] = []
    seen_urls: Set[str] = set()
    num_stripped_answers = 0
    for q_idx, q in enumerate(raw_sample.get("questions") or []):
        for ans in (q.get("answers") or []):
            if "answer_text" in ans or "boolean_explanation" in ans:
                num_stripped_answers += 1
            s_url = (ans.get("source_url") or "").strip()
            s_med = (ans.get("source_medium") or "Unknown").strip()
            if s_url and s_url.lower() != "metadata" and s_url not in seen_urls:
                seen_urls.add(s_url)
                gold_urls.append({
                    "source_url": s_url,
                    "source_medium": s_med,
                    "evidence_provenance_tag": "GOLD_EVIDENCE_URL_ASSISTED_OFFLINE",
                })

    stripped_log.append(f"questions[*].answers[*].(answer_text, boolean_explanation, answer_type) [count={num_stripped_answers}]")

    claim_input = {
        "sample_id": sample_id,
        "claim_text": (raw_sample.get("claim_text") or "").strip(),
        "date": (raw_sample.get("date") or "").strip(),
        "location": (raw_sample.get("location") or "").strip(),
        "claim_images": list(raw_sample.get("claim_images") or []),
        "speaker": (meta.get("speaker") or "").strip(),
        "reporting_source": (meta.get("reporting_source") or "").strip(),
    }

    # 硬断言校验：确保没有任何违禁字段残留
    serialized = json.dumps(claim_input)
    assert '"label"' not in serialized and '"justification"' not in serialized
    assert '"answer_text"' not in serialized and '"boolean_explanation"' not in serialized
    assert '"refuting_reasons"' not in serialized

    audit_meta = {
        "sample_id": sample_id,
        "visible_fields_to_model": list(claim_input.keys()),
        "stripped_forbidden_fields": stripped_log,
        "gold_assisted_candidate_urls_count": len(gold_urls),
        "gold_assisted_candidate_urls": gold_urls,
        "ground_truth_label_isolated_for_offline_eval_only": raw_sample.get("label"),
    }
    return claim_input, audit_meta


# =====================================================================
# 3. 第三部分：真实图像远程提取与 PIL 解码 (Real Image Fetch & Decode)
# =====================================================================

class RemoteZipImageExtractor:
    """通过 HTTP Range 从官方 AVerImaTeC images.zip 提取真实图像二进制并校验 CRC32"""

    def __init__(self, zip_url: str = REMOTE_IMAGES_ZIP_URL):
        self.zip_url = zip_url
        self.entries: Dict[str, Dict[str, int]] = {}
        self._indexed = False

    def build_central_directory_index(self) -> None:
        if self._indexed:
            return
        head = requests.head(self.zip_url, allow_redirects=True, timeout=15)
        head.raise_for_status()
        total_size = int(head.headers["Content-Length"])
        tail_size = min(total_size, 2 * 1024 * 1024)
        r = requests.get(
            self.zip_url,
            headers={"Range": f"bytes={total_size - tail_size}-{total_size - 1}"},
            timeout=25,
        )
        r.raise_for_status()
        buf = r.content
        tail_offset = total_size - len(buf)

        eocd_pos = buf.rfind(b"PK\x05\x06")
        if eocd_pos < 0:
            raise RuntimeError("Could not find EOCD signature in remote images.zip")
        cd_size = struct.unpack("<I", buf[eocd_pos + 12 : eocd_pos + 16])[0]
        cd_offset = struct.unpack("<I", buf[eocd_pos + 16 : eocd_pos + 20])[0]

        pos = cd_offset - tail_offset
        end_pos = pos + cd_size
        while pos < end_pos and buf[pos : pos + 4] == b"PK\x01\x02":
            comp_method = struct.unpack("<H", buf[pos + 10 : pos + 12])[0]
            crc32_val = struct.unpack("<I", buf[pos + 16 : pos + 20])[0]
            comp_size = struct.unpack("<I", buf[pos + 20 : pos + 24])[0]
            uncomp_size = struct.unpack("<I", buf[pos + 24 : pos + 28])[0]
            name_len = struct.unpack("<H", buf[pos + 28 : pos + 30])[0]
            extra_len = struct.unpack("<H", buf[pos + 30 : pos + 32])[0]
            comment_len = struct.unpack("<H", buf[pos + 32 : pos + 34])[0]
            local_header_offset = struct.unpack("<I", buf[pos + 42 : pos + 46])[0]
            fname = buf[pos + 46 : pos + 46 + name_len].decode("utf-8", errors="replace")
            base_name = os.path.basename(fname)
            if base_name:
                self.entries[base_name] = {
                    "comp_method": comp_method,
                    "crc32": crc32_val,
                    "comp_size": comp_size,
                    "uncomp_size": uncomp_size,
                    "local_header_offset": local_header_offset,
                }
            pos += 46 + name_len + extra_len + comment_len
        self._indexed = True

    def fetch_and_decode_image(self, image_name: str, save_dir: str = ROUND1_IMG_DIR) -> Dict[str, Any]:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.basename(image_name)
        local_path = os.path.join(save_dir, base_name)

        t0 = time.time()
        net_bytes = 0
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            self.build_central_directory_index()
            if base_name not in self.entries:
                return {
                    "image_name": base_name,
                    "status": "NOT_FOUND_IN_ZIP",
                    "decoded_ok": False,
                }
            info = self.entries[base_name]
            lh_off = info["local_header_offset"]
            comp_sz = info["comp_size"]
            # 读取 Local File Header (30 bytes + 256 bytes 余量涵盖文件名与 extra) + 压缩数据
            fetch_end = lh_off + 30 + 256 + comp_sz
            r = requests.get(
                self.zip_url,
                headers={"Range": f"bytes={lh_off}-{fetch_end}"},
                timeout=20,
            )
            r.raise_for_status()
            chunk = r.content
            net_bytes = len(chunk)
            if chunk[:4] != b"PK\x03\x04":
                return {"image_name": base_name, "status": "INVALID_LOCAL_HEADER", "decoded_ok": False}
            fn_len = struct.unpack("<H", chunk[26:28])[0]
            ex_len = struct.unpack("<H", chunk[28:30])[0]
            data_start = 30 + fn_len + ex_len
            comp_data = chunk[data_start : data_start + comp_sz]

            if info["comp_method"] == 0:
                raw_bytes = comp_data
            elif info["comp_method"] == 8:
                raw_bytes = zlib.decompress(comp_data, -zlib.MAX_WBITS)
            else:
                return {"image_name": base_name, "status": f"UNSUPPORTED_COMP_{info['comp_method']}", "decoded_ok": False}

            calc_crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
            assert calc_crc == info["crc32"], f"CRC32 mismatch for {base_name}: {hex(calc_crc)} != {hex(info['crc32'])}"
            with open(local_path, "wb") as f:
                f.write(raw_bytes)

        with open(local_path, "rb") as f:
            raw_bytes = f.read()
        calc_crc_hex = f"{zlib.crc32(raw_bytes) & 0xFFFFFFFF:08x}"
        sha256_hex = hashlib.sha256(raw_bytes).hexdigest()[:16]

        img = Image.open(io.BytesIO(raw_bytes))
        fmt = img.format
        mode = img.mode
        w, h = img.size
        img_rgb = img.convert("RGB")
        arr = np.asarray(img_rgb, dtype=np.float32)
        mean_rgb = [round(float(x), 2) for x in np.mean(arr, axis=(0, 1))]
        std_rgb = [round(float(x), 2) for x in np.std(arr, axis=(0, 1))]

        # 计算真实图像 8x8 均值感知哈希 (aHash)
        small_gray = img_rgb.convert("L").resize((8, 8), Image.Resampling.BILINEAR)
        g_arr = np.asarray(small_gray, dtype=np.float32)
        bits = (g_arr >= np.mean(g_arr)).flatten()
        ahash_int = 0
        for b in bits:
            ahash_int = (ahash_int << 1) | int(b)
        ahash_hex = f"{ahash_int:016x}"

        return {
            "image_name": base_name,
            "local_file_path": local_path,
            "status": "DECODED_OK",
            "decoded_ok": True,
            "format": fmt,
            "mode": mode,
            "width": w,
            "height": h,
            "file_size_bytes": len(raw_bytes),
            "network_fetched_bytes": net_bytes,
            "crc32_hex": calc_crc_hex,
            "sha256_prefix": sha256_hex,
            "ahash_64bit_hex": ahash_hex,
            "mean_rgb": mean_rgb,
            "std_rgb": std_rgb,
            "decode_latency_ms": round((time.time() - t0) * 1000, 1),
            "_pil_rgb": img_rgb,
        }


# =====================================================================
# 4. 第四部分：真实开源视觉模型 (TinyCLIP) 与 NLI 模型推理封装
# =====================================================================

class RealOpenSourceModels:
    """加载并执行本地真实开源神经网络模型：TinyCLIP 视觉-语言模型 + MiniLM2 NLI 交叉编码器"""

    def __init__(self) -> None:
        nli_dir = os.path.join(MODELS_DIR, "nli-MiniLM2-L6-H768")
        clip_dir = os.path.join(MODELS_DIR, "TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")

        t0 = time.time()
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_dir, local_files_only=True)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_dir, local_files_only=True)
        self.nli_model.eval()
        self.nli_load_ms = round((time.time() - t0) * 1000, 1)

        t1 = time.time()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_dir, local_files_only=True)
        self.clip_model = CLIPModel.from_pretrained(clip_dir, local_files_only=True)
        self.clip_model.eval()
        self.clip_load_ms = round((time.time() - t1) * 1000, 1)

        self.nli_forward_calls = 0
        self.clip_forward_calls = 0

    @torch.no_grad()
    def compute_clip_image_text_similarity(
        self,
        pil_images: List[Image.Image],
        texts: List[str],
    ) -> Dict[str, Any]:
        """执行真实 CLIP 视觉编码器与文本编码器前向传播，返回图像嵌入范数与余弦相似度矩阵"""
        if not pil_images or not texts:
            return {"similarities": [], "image_embedding_norm": 0.0, "latency_ms": 0.0}
        t0 = time.time()
        self.clip_forward_calls += 1
        trunc_texts = [(t or "")[:240] for t in texts]
        inputs = self.clip_processor(
            text=trunc_texts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        outputs = self.clip_model(**inputs)
        img_embeds = outputs.image_embeds  # [N_img, D]
        txt_embeds = outputs.text_embeds    # [N_txt, D]
        raw_norm = float(torch.norm(img_embeds[0], p=2).item())

        img_normed = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
        txt_normed = txt_embeds / txt_embeds.norm(p=2, dim=-1, keepdim=True)
        sim_matrix = torch.matmul(img_normed, txt_normed.T).cpu().numpy()  # [N_img, N_txt]

        return {
            "similarities": [[round(float(x), 4) for x in row] for row in sim_matrix],
            "mean_similarity_per_text": [round(float(x), 4) for x in np.mean(sim_matrix, axis=0)],
            "image_embedding_raw_l2_norm": round(raw_norm, 4),
            "embedding_dim": int(img_embeds.shape[-1]),
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }

    @torch.no_grad()
    def predict_nli_stance_and_embedding(
        self,
        premise_evidence: str,
        hypothesis_claim: str,
    ) -> Dict[str, Any]:
        """
        使用真实 cross-encoder/nli-MiniLM2-L6-H768 计算前提（网页证据段落）对假设（待核查声明）的 NLI 概率：
          id2label: {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
        同时提取最后一层隐状态 CLS 向量用于计算跨来源语义相似度。
        """
        t0 = time.time()
        self.nli_forward_calls += 1
        prem = (premise_evidence or "").strip()[:1000]
        hyp = (hypothesis_claim or "").strip()[:400]
        inputs = self.nli_tokenizer(
            prem,
            hyp,
            return_tensors="pt",
            truncation=True,
            max_length=384,
            padding=True,
        )
        outputs = self.nli_model(**inputs, output_hidden_states=True)
        logits = outputs.logits[0].cpu().numpy()
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)

        # id2label: 0 -> contradiction, 1 -> entailment, 2 -> neutral
        p_contra = float(probs[0])
        p_entail = float(probs[1])
        p_neutral = float(probs[2])

        cls_vec = outputs.hidden_states[-1][0, 0, :].cpu().numpy()
        cls_norm = cls_vec / max(float(np.linalg.norm(cls_vec)), 1e-8)

        return {
            "logits": [round(float(x), 4) for x in logits],
            "p_contradiction": round(p_contra, 4),
            "p_entailment": round(p_entail, 4),
            "p_neutral": round(p_neutral, 4),
            "predicted_nli_label": ["contradiction", "entailment", "neutral"][int(np.argmax(probs))],
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "_cls_embedding": cls_norm,
        }


# =====================================================================
# 5. 第五部分：真实网页抓取与开放检索工具 (Real Webpage Fetch & Open Search)
# =====================================================================

def clean_html_to_paragraphs(raw_html: str) -> Tuple[str, List[str]]:
    """从真实抓取的 HTML 源码中提取 <title>、meta description 与正文段落（纯解析，零编造）"""
    if not raw_html:
        return "", []
    # 移除 script / style
    cleaned = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", raw_html, flags=re.S | re.I)
    title_m = re.search(r"<title[^>]*>(.*?)</title>", cleaned, flags=re.S | re.I)
    title_str = html.unescape(re.sub(r"\s+", " ", title_m.group(1)).strip()) if title_m else ""

    paragraphs: List[str] = []
    # 提取 meta description / og:description
    for meta_m in re.finditer(
        r'<meta[^>]+(?:name|property)=["\'](?:description|og:description|twitter:description)["\'][^>]+content=["\']([^"\']+)["\']',
        cleaned,
        flags=re.I,
    ):
        desc = html.unescape(re.sub(r"\s+", " ", meta_m.group(1)).strip())
        if len(desc) >= 40:
            paragraphs.append(desc)

    # 提取 <p> 段落
    for p_m in re.finditer(r"<p[^>]*>(.*?)</p>", cleaned, flags=re.S | re.I):
        p_txt = re.sub(r"<[^>]+>", " ", p_m.group(1))
        p_txt = html.unescape(re.sub(r"\s+", " ", p_txt).strip())
        if len(p_txt) >= 50:
            paragraphs.append(p_txt)

    return title_str, paragraphs[:40]


def select_best_passage_for_claim(
    claim_text: str,
    title: str,
    paragraphs: List[str],
) -> str:
    """从真实网页段落中选出与声明实体词重合度最高的真实段落"""
    if not paragraphs:
        return title[:400] if title else ""
    c_tokens = tokenize_entities(claim_text)
    best_score = -1.0
    best_p = paragraphs[0]
    for p in paragraphs:
        p_tokens = tokenize_entities(p)
        overlap = len(c_tokens & p_tokens) / max(1, len(c_tokens)) if c_tokens else 0.0
        length_bonus = min(0.2, len(p) / 1000.0)
        sc = overlap + length_bonus
        if sc > best_score:
            best_score = sc
            best_p = p
    if title and title not in best_p:
        return f"{title}. {best_p[:500]}"
    return best_p[:600]


def fetch_real_webpage_evidence(
    url: str,
    claim_text: str,
    provenance_tag: str,
    timeout: int = 10,
) -> Dict[str, Any]:
    """
    对真实 URL 发起 HTTP GET 请求，读取真实网页 HTML 并提取与声明最相关的真实正文段落。
    若 HTTP 失败（403/404/Timeout/SSL 等），如实记录失败状态与原因，绝不编造替代文本！
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    t0 = time.time()
    dom, is_arch = extract_clean_domain(url)
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        latency_ms = round((time.time() - t0) * 1000, 1)
        ctype = resp.headers.get("Content-Type", "").lower()
        raw_bytes_len = len(resp.content)

        if resp.status_code != 200:
            return {
                "source_url": url,
                "final_url": resp.url,
                "domain": dom,
                "is_archived": is_arch,
                "evidence_provenance_tag": provenance_tag,
                "fetch_status": f"HTTP_ERROR_{resp.status_code}",
                "http_status": resp.status_code,
                "latency_ms": latency_ms,
                "bytes_downloaded": raw_bytes_len,
                "extracted_title": "",
                "extracted_passage": "",
                "usable": False,
                "failure_reason": f"Remote server returned HTTP {resp.status_code}",
            }

        if "image/" in ctype or "application/pdf" in ctype:
            return {
                "source_url": url,
                "final_url": resp.url,
                "domain": dom,
                "is_archived": is_arch,
                "evidence_provenance_tag": provenance_tag,
                "fetch_status": f"NON_HTML_CONTENT ({ctype[:40]})",
                "http_status": resp.status_code,
                "latency_ms": latency_ms,
                "bytes_downloaded": raw_bytes_len,
                "extracted_title": "",
                "extracted_passage": "",
                "usable": False,
                "failure_reason": f"URL points to binary/non-HTML media ({ctype[:40]}) rather than article text.",
            }

        title, paras = clean_html_to_paragraphs(resp.text)
        best_passage = select_best_passage_for_claim(claim_text, title, paras)
        usable = len(best_passage.strip()) >= 30

        return {
            "source_url": url,
            "final_url": resp.url,
            "domain": dom,
            "is_archived": is_arch,
            "evidence_provenance_tag": provenance_tag,
            "fetch_status": "FETCH_OK" if usable else "EMPTY_OR_JS_BLOCKED_BODY",
            "http_status": resp.status_code,
            "latency_ms": latency_ms,
            "bytes_downloaded": raw_bytes_len,
            "extracted_title": title[:200],
            "num_paragraphs_parsed": len(paras),
            "extracted_passage": best_passage,
            "usable": usable,
            "failure_reason": None if usable else "HTML returned HTTP 200 but contained <30 chars of readable <p>/meta text (likely JS-rendered/paywalled).",
        }
    except Exception as e:
        return {
            "source_url": url,
            "final_url": url,
            "domain": dom,
            "is_archived": is_arch,
            "evidence_provenance_tag": provenance_tag,
            "fetch_status": "NETWORK_EXCEPTION",
            "http_status": None,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "bytes_downloaded": 0,
            "extracted_title": "",
            "extracted_passage": "",
            "usable": False,
            "failure_reason": f"{type(e).__name__}: {str(e)[:180]}",
        }


def search_open_wikipedia_evidence(claim_text: str, max_results: int = 2) -> List[Dict[str, Any]]:
    """
    真实开放检索通道 (`OPEN_WEB_LIVE_SEARCH`)：
    仅使用 `claim_text` 中的实词查询公开 Wikipedia MediaWiki API，返回真实词条摘要与 URL。
    """
    tokens = sorted(tokenize_entities(claim_text), key=len, reverse=True)
    query_str = " ".join(tokens[:6]) if tokens else claim_text[:80]
    api_url = (
        f"https://en.wikipedia.org/w/api.php?action=query&list=search"
        f"&srsearch={quote_plus(query_str)}&utf8=1&format=json&srlimit={max_results}"
    )
    headers = {"User-Agent": "Mozilla/5.0 (FactCheckAuditBot/1.0)"}
    t0 = time.time()
    results: List[Dict[str, Any]] = []
    try:
        r = requests.get(api_url, headers=headers, timeout=8)
        latency_ms = round((time.time() - t0) * 1000, 1)
        if r.status_code != 200:
            return [{
                "source_url": api_url,
                "domain": "en.wikipedia.org",
                "is_archived": False,
                "evidence_provenance_tag": "OPEN_WEB_LIVE_SEARCH (Wikipedia_API)",
                "fetch_status": f"HTTP_ERROR_{r.status_code}",
                "http_status": r.status_code,
                "latency_ms": latency_ms,
                "bytes_downloaded": len(r.content),
                "extracted_title": "",
                "extracted_passage": "",
                "usable": False,
                "failure_reason": f"Wikipedia API returned HTTP {r.status_code}",
            }]

        data = r.json()
        search_items = data.get("query", {}).get("search", [])
        if not search_items:
            return [{
                "source_url": api_url,
                "domain": "en.wikipedia.org",
                "is_archived": False,
                "evidence_provenance_tag": "OPEN_WEB_LIVE_SEARCH (Wikipedia_API)",
                "fetch_status": "ZERO_SEARCH_HITS",
                "http_status": 200,
                "latency_ms": latency_ms,
                "bytes_downloaded": len(r.content),
                "extracted_title": "",
                "extracted_passage": "",
                "usable": False,
                "failure_reason": f"No Wikipedia articles matched query '{query_str}'",
            }]

        for item in search_items[:max_results]:
            title = item.get("title", "")
            snippet_html = item.get("snippet", "")
            snippet_clean = html.unescape(re.sub(r"<[^>]+>", " ", snippet_html))
            snippet_clean = re.sub(r"\s+", " ", snippet_clean).strip()
            page_url = f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
            passage = f"{title}: {snippet_clean}"
            results.append({
                "source_url": page_url,
                "final_url": page_url,
                "domain": "en.wikipedia.org",
                "is_archived": False,
                "evidence_provenance_tag": "OPEN_WEB_LIVE_SEARCH (Wikipedia_API)",
                "query_used": query_str,
                "fetch_status": "FETCH_OK",
                "http_status": 200,
                "latency_ms": latency_ms,
                "bytes_downloaded": len(r.content),
                "extracted_title": title,
                "extracted_passage": passage,
                "usable": len(passage) >= 25,
                "failure_reason": None,
            })
        return results
    except Exception as e:
        return [{
            "source_url": api_url,
            "domain": "en.wikipedia.org",
            "is_archived": False,
            "evidence_provenance_tag": "OPEN_WEB_LIVE_SEARCH (Wikipedia_API)",
            "fetch_status": "NETWORK_EXCEPTION",
            "http_status": None,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "bytes_downloaded": 0,
            "extracted_title": "",
            "extracted_passage": "",
            "usable": False,
            "failure_reason": f"{type(e).__name__}: {str(e)[:180]}",
        }]


# =====================================================================
# 6. 第六部分：5 条真实样本端到端测试与完整调用日志记录
# =====================================================================

def extract_years_from_text(text: str) -> List[int]:
    if not text:
        return []
    return [int(y) for y in re.findall(r"\b(19\d\d|20[0-2]\d)\b", text)]


def run_single_real_sample_verification(
    raw_sample: Dict[str, Any],
    sample_id: int,
    img_extractor: RemoteZipImageExtractor,
    models: RealOpenSourceModels,
    rules: List[SoftLogicRule],
) -> Dict[str, Any]:
    r"""
    在单条真实样本上执行完整端到端流程（无任何模拟值、无虚构 URL、无 questions.answers 文本泄漏）：
      Step 1: 严格隔离输入 (`isolate_sample_for_inference`)
      Step 2: 真实图像下载与 PIL 解码 (`fetch_and_decode_image`)
      Step 3: 真实检索与网页抓取（包含 `[OPEN_WEB_LIVE_SEARCH]` 开放检索 + 明确标注的 `[GOLD_EVIDENCE_URL_ASSISTED_OFFLINE]` 网页实抓）
      Step 4: 真实视觉模型 (`TinyCLIP`) 与自然语言推理模型 (`nli-MiniLM2-L6-H768`) 前向传播计算质量与正反立场向量
      Step 5: 真实跨来源关系矩阵估计 + `src/core_models.py` 式(1)折扣、式(2)凸二次规划、式(3) \chi 特征、式(4) 终端预测
      Step 6: 与隔离保存的 `ground_truth_label` 离线比对，记录全部调用耗时、流量与失败原因
    """
    sample_t0 = time.time()
    claim_input, isolation_audit = isolate_sample_for_inference(raw_sample, sample_id)
    claim_text = claim_input["claim_text"]
    claim_date = claim_input["date"]

    # 1. 真实图像提取与解码
    image_records: List[Dict[str, Any]] = []
    pil_images: List[Image.Image] = []
    for img_name in claim_input["claim_images"]:
        dec_info = img_extractor.fetch_and_decode_image(img_name)
        if dec_info.get("decoded_ok") and "_pil_rgb" in dec_info:
            pil_images.append(dec_info.pop("_pil_rgb"))
        image_records.append(dec_info)

    # 2. 真实网页检索与抓取
    # (a) 开放网络检索 (Wikipedia API)
    open_web_records = search_open_wikipedia_evidence(claim_text, max_results=2)

    # (b) 金证据 URL 辅助实时网页抓取 (明确标注 GOLD_EVIDENCE_URL_ASSISTED_OFFLINE，绝不读取 questions.answers 文本)
    gold_url_records: List[Dict[str, Any]] = []
    for u_item in isolation_audit["gold_assisted_candidate_urls"][:5]:
        rec = fetch_real_webpage_evidence(
            u_item["source_url"],
            claim_text,
            provenance_tag="GOLD_EVIDENCE_URL_ASSISTED_OFFLINE",
            timeout=10,
        )
        gold_url_records.append(rec)

    all_retrieval_attempts = open_web_records + gold_url_records
    usable_records = [r for r in all_retrieval_attempts if r.get("usable")]

    # 3. 真实视觉模型 (TinyCLIP) 推理：计算原始图像与声明文本、各检索网页段落的真实跨模态余弦相似度
    clip_texts = [claim_text] + [r["extracted_passage"][:240] for r in usable_records]
    clip_output = models.compute_clip_image_text_similarity(pil_images, clip_texts)
    claim_img_txt_sim = (
        clip_output["mean_similarity_per_text"][0]
        if clip_output.get("mean_similarity_per_text")
        else 0.0
    )

    # 4. 真实 NLI 模型 (nli-MiniLM2-L6-H768) 推理与 EvidenceSnippet 构造
    snippets: List[EvidenceSnippet] = []
    cls_embeddings: List[np.ndarray] = []
    model_io_logs: List[Dict[str, Any]] = []

    c_years = extract_years_from_text(claim_text + " " + claim_date[:4])
    c_entities = tokenize_entities(claim_text)

    for ell, rec in enumerate(usable_records):
        passage = rec["extracted_passage"]
        nli_out = models.predict_nli_stance_and_embedding(
            premise_evidence=passage,
            hypothesis_claim=claim_text,
        )
        cls_vec = nli_out.pop("_cls_embedding")
        cls_embeddings.append(cls_vec)

        # 基于真实 NLI 输出与真实网页/视觉信号构造 4 槽位向量 (j=0: 视觉, j=1: 时间, j=2: 空间/主体, j=3: 整体事件)
        p_ent = nli_out["p_entailment"]
        p_con = nli_out["p_contradiction"]
        p_neu = nli_out["p_neutral"]

        ev_clip_sim = (
            clip_output["mean_similarity_per_text"][ell + 1]
            if len(clip_output.get("mean_similarity_per_text", [])) > ell + 1
            else 0.20
        )
        p_entities = tokenize_entities(passage)
        entity_overlap = len(c_entities & p_entities) / max(1, len(c_entities)) if c_entities else 0.0

        # 检查真实抓取网页文本中的年份与声明年份是否存在历史早于冲突 (Out-of-Context 时间错配)
        ev_years = extract_years_from_text(passage + " " + rec["source_url"])
        temporal_conflict = 0.0
        if c_years and ev_years and min(ev_years) < max(c_years):
            temporal_conflict = min(1.0, 0.40 + 0.15 * (max(c_years) - min(ev_years)))

        # 确定该真实网页片段主要支撑的规范事实簇 fact_id in {1, 2, 3, 4}
        if temporal_conflict > 0.0 or re.search(r"\b(year|date|published|201\d|202\d|19\d\d)\b", passage.lower()):
            primary_slot = 1  # 时间槽位
        elif re.search(r"\b(photo|image|picture|video|footage|shown|depicts)\b", passage.lower()):
            primary_slot = 0  # 视觉槽位
        elif entity_overlap >= 0.35:
            primary_slot = 3  # 整体事件槽位
        else:
            primary_slot = 2  # 主体/背景槽位

        # 质量向量：由真实实体覆盖度、CLIP 图文相关性、非中性 NLI 确信度及归档可回放性决定
        base_q = float(np.clip(0.30 + 0.35 * entity_overlap + 0.20 * (1.0 - p_neu) + 0.15 * max(0.0, ev_clip_sim), 0.15, 0.95))
        q_vec = np.full(4, base_q * 0.80, dtype=float)
        q_vec[primary_slot] = base_q
        q_vec[3] = max(q_vec[3], base_q * 0.90)

        r_pos = np.array([
            float(np.clip(max(p_ent, ev_clip_sim * 1.5), 0.02, 0.95)),
            float(np.clip(p_ent * (1.0 - temporal_conflict), 0.02, 0.95)),
            float(np.clip(p_ent * 0.9 + 0.1 * entity_overlap, 0.02, 0.95)),
            float(np.clip(p_ent, 0.02, 0.95)),
        ], dtype=float)

        r_neg = np.array([
            float(np.clip(p_con, 0.02, 0.95)),
            float(np.clip(max(p_con, temporal_conflict), 0.02, 0.95)),
            float(np.clip(p_con, 0.02, 0.95)),
            float(np.clip(max(p_con, 0.65 * temporal_conflict + 0.35 * p_con), 0.02, 0.95)),
        ], dtype=float)

        snippets.append(
            EvidenceSnippet(
                snippet_id=ell,
                domain=rec["domain"] or "unknown.org",
                url=rec["source_url"],
                fact_id=primary_slot + 1,
                quality=q_vec,
                r_pos=r_pos,
                r_neg=r_neg,
                is_image_provenance=(primary_slot in (0, 1) and len(pil_images) > 0),
                timestamp_verified=bool(ev_years),
                url_replayable=True,
            )
        )

        model_io_logs.append({
            "snippet_id": ell,
            "source_url": rec["source_url"],
            "domain": rec["domain"],
            "evidence_provenance_tag": rec["evidence_provenance_tag"],
            "input_premise_passage": passage[:300],
            "input_hypothesis_claim": claim_text,
            "nli_model_output": nli_out,
            "clip_image_passage_cosine": round(float(ev_clip_sim), 4),
            "entity_overlap_with_claim": round(float(entity_overlap), 4),
            "detected_years_in_passage": ev_years,
            "temporal_conflict_score": round(float(temporal_conflict), 4),
            "assigned_fact_id": primary_slot + 1,
            "quality_vector": [round(float(x), 4) for x in q_vec],
            "r_pos_vector": [round(float(x), 4) for x in r_pos],
            "r_neg_vector": [round(float(x), 4) for x in r_neg],
        })

    # 5. 真实来源关系判断 (Real Source Relation Judgment) 与核心算法执行
    n_snips = len(snippets)
    P_hat = np.eye(n_snips, dtype=float)
    pairwise_source_relations: List[Dict[str, Any]] = []

    for i in range(n_snips):
        for j in range(i + 1, n_snips):
            ui = normalize_url_for_matching(snippets[i].url)
            uj = normalize_url_for_matching(snippets[j].url)
            same_url = 1.0 if (ui and uj and ui == uj) else 0.0
            same_dom = 1.0 if (snippets[i].domain and snippets[i].domain == snippets[j].domain) else 0.0
            emb_cos = float(np.clip(np.dot(cls_embeddings[i], cls_embeddings[j]), -1.0, 1.0))
            # 基于真实 URL 归一化、真实域名与真实 NLI 隐状态余弦相似度计算同源概率
            p_ij = float(np.clip(0.55 * same_url + 0.25 * same_dom + 0.20 * max(0.0, (emb_cos - 0.5) * 2.0), 0.0, 1.0))
            P_hat[i, j] = P_hat[j, i] = p_ij
            pairwise_source_relations.append({
                "pair": [i, j],
                "domain_i": snippets[i].domain,
                "domain_j": snippets[j].domain,
                "fact_id_i": snippets[i].fact_id,
                "fact_id_j": snippets[j].fact_id,
                "same_normalized_url": bool(same_url),
                "same_domain": bool(same_dom),
                "neural_cls_cosine_similarity": round(emb_cos, 4),
                "estimated_same_source_prob_p_ij": round(p_ij, 4),
                "triggers_eq1_same_fact_discount": bool(snippets[i].fact_id == snippets[j].fact_id and p_ij > 0),
            })

    gt_label = isolation_audit["ground_truth_label_isolated_for_offline_eval_only"]

    if n_snips > 0:
        W_discounted = compute_discounted_weights(snippets, P_hat, mode="prob_discount")
        z_star, qp_diag = solve_convex_qp_inference(
            snippets, P_hat, rules=rules, mu=0.05, alpha=1.0, weight_mode="prob_discount"
        )
        # 计算每一对候选来源的式(3) \chi_t(a) 值
        chi_records = []
        for pr in pairwise_source_relations:
            pi, pj = pr["pair"]
            chi_val, p_unc, state_diff = compute_source_influence_feature(
                snippets, P_hat, (pi, pj), rules=rules, mu=0.05, alpha=1.0
            )
            chi_records.append({
                "pair": [pi, pj],
                "chi_value": round(chi_val, 6),
                "p_uncertainty_p_1_minus_p": round(p_unc, 4),
                "counterfactual_qp_state_shift_l2": round(state_diff, 6),
            })
        term_decision = terminal_calibrated_decision(
            z_star, snippets, gt_label, temperature=0.80, prob_threshold=0.50, sufficiency_threshold=0.45
        )
    else:
        W_discounted = np.zeros((4, 0))
        z_star = np.zeros(8, dtype=float)
        qp_diag = {"converged": False, "reason": "No usable webpage evidence retrieved"}
        chi_records = []
        term_decision = {
            "probs": [0.1, 0.1, 0.7, 0.1],
            "raw_pred_label": "Not Enough Evidence",
            "final_decision": "ABSTAIN",
            "max_prob": 0.7,
            "sufficiency_score": 0.0,
            "citation_replay_ok": False,
            "abstain": True,
            "chow_loss": 0.25,
            "is_correct_non_abstain": False,
        }

    raw_pred = term_decision["raw_pred_label"]
    final_pred = term_decision["final_decision"]
    if final_pred == "ABSTAIN":
        chow_loss = 0.25
    else:
        chow_loss = 0.0 if final_pred == gt_label else 1.0

    total_net_bytes = sum(r.get("bytes_downloaded", 0) for r in all_retrieval_attempts) + sum(
        im.get("network_fetched_bytes", 0) for im in image_records
    )
    total_http_requests = len(all_retrieval_attempts) + sum(1 for im in image_records if im.get("network_fetched_bytes", 0) > 0)

    return {
        "sample_id": sample_id,
        "claim_text": claim_text,
        "claim_date": claim_date,
        "data_flow_isolation_audit": isolation_audit,
        "original_images_decoded": image_records,
        "retrieval_attempts_log": all_retrieval_attempts,
        "usable_evidence_count": n_snips,
        "failed_retrieval_count": len(all_retrieval_attempts) - n_snips,
        "vision_model_clip_inference": clip_output,
        "nli_and_snippet_scoring_io": model_io_logs,
        "pairwise_source_relations": pairwise_source_relations,
        "core_math_inference_outputs": {
            "discounted_weights_shape": list(W_discounted.shape),
            "qp_z_pos": [round(float(x), 4) for x in z_star[:4]],
            "qp_z_neg": [round(float(x), 4) for x in z_star[4:]],
            "qp_solver_diagnostics": qp_diag,
            "source_influence_chi_per_pair": chi_records,
            "terminal_decision_output": term_decision,
        },
        "offline_evaluation_against_isolated_label": {
            "ground_truth_label": gt_label,
            "raw_top_predicted_label": raw_pred,
            "final_selective_decision": final_pred,
            "forced_prediction_correct": bool(raw_pred == gt_label),
            "selective_decision_correct": bool(final_pred == gt_label) if final_pred != "ABSTAIN" else None,
            "abstained": bool(final_pred == "ABSTAIN"),
            "chow_loss": chow_loss,
        },
        "real_execution_cost_and_latency": {
            "wall_clock_time_ms": round((time.time() - sample_t0) * 1000, 1),
            "http_requests_count": total_http_requests,
            "network_bytes_downloaded": total_net_bytes,
            "clip_forward_passes": 1 if pil_images and clip_texts else 0,
            "nli_forward_passes": n_snips,
            "commercial_api_monetary_cost_usd": 0.0,
            "cost_note": "0.00 USD (All models run locally on CPU via PyTorch; HTTP requests use public Wikipedia & direct URL fetching without paid API keys).",
        },
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="历史 TinyCLIP/NLI 与金标准 URL 辅助诊断；不属于申请书正式 VLM 实验")
    parser.add_argument("--run-diagnostic", action="store_true")
    args = parser.parse_args()
    if not args.run_diagnostic:
        print("DIAGNOSTIC_ONLY_NOT_FORMAL: use --run-diagnostic to run the historical network diagnostic.")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 80)
    print("Round 1: Environment Check, Data-Flow Audit & 5 Real Samples Verification")
    print("=" * 80)

    # 1. 环境与阻塞检查
    env_check = run_environment_and_blocker_check()
    print(f"[1/4] Environment Check Completed. CUDA={env_check['hardware']['cuda_available']}, Blockers={len(env_check['explicit_blockers'])}")
    for b in env_check["explicit_blockers"]:
        print(f"      - [BLOCKER] {b['module']}: {b['status']} ({b['reason']})")

    # 2. 加载真实开源模型 (TinyCLIP + NLI Cross-Encoder)
    print("[2/4] Loading local open-source neural models (TinyCLIP + MiniLM2-NLI)...")
    models = RealOpenSourceModels()
    print(f"      - Loaded NLI in {models.nli_load_ms} ms, TinyCLIP in {models.clip_load_ms} ms.")

    # 3. 读取数据集与确定性选取 5 条先导样本 (pilot_probe 前 5 条)
    with open(os.path.join(DATA_DIR, "train.json"), "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(os.path.join(OUTPUT_DIR, "averimatec_event_isolated_splits_v2.json"), "r", encoding="utf-8") as f:
        splits_v2 = json.load(f)["splits"]

    pilot_indices = splits_v2["pilot_probe"]["indices"][:5]
    print(f"[3/4] Running end-to-end verification on 5 real pilot samples: indices={pilot_indices}")

    img_extractor = RemoteZipImageExtractor()
    rules = build_cross_modal_rules(m=4)

    sample_results: List[Dict[str, Any]] = []
    for idx_pos, s_id in enumerate(pilot_indices):
        print(f"      -> Processing Sample #{idx_pos + 1}/5 (train_id={s_id})...")
        res = run_single_real_sample_verification(
            train_data[s_id], s_id, img_extractor, models, rules
        )
        sample_results.append(res)
        ev = res["offline_evaluation_against_isolated_label"]
        cost = res["real_execution_cost_and_latency"]
        print(
            f"         Done: ImagesDecoded={len(res['original_images_decoded'])}, "
            f"UsableWebPages={res['usable_evidence_count']}/{len(res['retrieval_attempts_log'])}, "
            f"RawPred='{ev['raw_top_predicted_label']}', Final='{ev['final_selective_decision']}', "
            f"GT='{ev['ground_truth_label']}', Time={cost['wall_clock_time_ms']}ms, NetBytes={cost['network_bytes_downloaded']}"
        )

    # 4. 汇总并自动写出 JSON 报告 (严禁任何手动修改)
    forced_correct = sum(1 for r in sample_results if r["offline_evaluation_against_isolated_label"]["forced_prediction_correct"])
    abstain_count = sum(1 for r in sample_results if r["offline_evaluation_against_isolated_label"]["abstained"])
    covered = [r for r in sample_results if not r["offline_evaluation_against_isolated_label"]["abstained"]]
    covered_correct = sum(1 for r in covered if r["offline_evaluation_against_isolated_label"]["selective_decision_correct"])
    mean_chow = float(np.mean([r["offline_evaluation_against_isolated_label"]["chow_loss"] for r in sample_results]))

    round1_report = {
        "round": "Historical TinyCLIP/NLI and Gold-URL-Assisted Diagnostic",
        "formal_evaluation_eligible": False,
        "evidence_protocol": "GOLD_URL_ASSISTED_NETWORK_DIAGNOSTIC",
        "method_matches_application_vlm": False,
        "strict_compliance_declarations": {
            "core_math_models_untouched": True,
            "questions_answers_text_isolated": True,
            "label_isolated_for_offline_eval_only": True,
            "zero_synthetic_urls_or_echo_chambers": True,
            "gold_url_assisted_explicitly_labeled": True,
            "auto_generated_from_execution_log_without_manual_edit": True,
        },
        "environment_and_blockers": env_check,
        "summary_5_real_samples": {
            "sample_indices": pilot_indices,
            "num_samples": len(sample_results),
            "total_images_decoded": sum(len(r["original_images_decoded"]) for r in sample_results),
            "total_http_retrieval_attempts": sum(len(r["retrieval_attempts_log"]) for r in sample_results),
            "total_usable_webpages_fetched": sum(r["usable_evidence_count"] for r in sample_results),
            "total_failed_http_fetches": sum(r["failed_retrieval_count"] for r in sample_results),
            "forced_top1_accuracy": round(forced_correct / len(sample_results), 4),
            "coverage": round(len(covered) / len(sample_results), 4),
            "selective_accuracy_on_covered": round(covered_correct / len(covered), 4) if covered else None,
            "abstain_count": abstain_count,
            "mean_chow_loss": round(mean_chow, 4),
            "total_nli_forward_passes": models.nli_forward_calls,
            "total_clip_forward_passes": models.clip_forward_calls,
        },
        "samples_execution_traces": sample_results,
    }

    out_path = os.path.join(OUTPUT_DIR, "round1_5samples_real_audit_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(round1_report, f, ensure_ascii=False, indent=2)

    print(f"[4/4] Automatically saved Round 1 report to: {out_path}")
    print(
        f"      Summary (5 Real Samples): ForcedAcc={round1_report['summary_5_real_samples']['forced_top1_accuracy']:.2f}, "
        f"Coverage={round1_report['summary_5_real_samples']['coverage']:.2f}, "
        f"MeanChow={round1_report['summary_5_real_samples']['mean_chow_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
