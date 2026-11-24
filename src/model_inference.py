"""
文本诊断 NLI 模块：本地 cross-encoder/nli-MiniLM2-L6-H768 交叉编码器。
正式多模态实验使用冻结 Qwen VLM；本模块只用于显式 QA 文本诊断。
（round1_real_5samples_audit.py 保留独立的模型加载器以额外输出 CLS 嵌入并运行 CLIP。）
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

_nli_instance: Optional["NLIStanceModel"] = None


class NLIStanceModel:
    """cross-encoder/nli-MiniLM2-L6-H768 交叉编码器，计算证据对声明的蕴含/矛盾/中性概率。"""

    def __init__(self) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        nli_dir = os.path.join(MODELS_DIR, "nli-MiniLM2-L6-H768")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(nli_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(nli_dir, local_files_only=True)
        self.model.eval()
        self._torch = torch
        self.load_ms = round((time.time() - t0) * 1000, 1)
        self.forward_calls = 0
        self._cache: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
        names = {str(name).casefold(): int(i) for i, name in self.model.config.id2label.items()}
        if not {"entailment", "contradiction", "neutral"} <= set(names):
            raise ValueError("NLI checkpoint does not declare the required label mapping")
        self._label_ids = names

    def stance(self, premise: str, hypothesis: str) -> Tuple[float, float, float]:
        """返回 (p_entailment, p_contradiction, p_neutral)，结果按文本对缓存。"""
        prem = (premise or "").strip()
        hyp = (hypothesis or "").strip()
        key = (prem, hyp)
        if key in self._cache:
            return self._cache[key]

        torch = self._torch
        with torch.no_grad():
            inputs = self.tokenizer(prem, hyp, return_tensors="pt", truncation=True, max_length=384, padding=True)
            logits = self.model(**inputs).logits[0].cpu().numpy()
            self.forward_calls += 1
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)
        result = tuple(float(probs[self._label_ids[k]]) for k in ("entailment", "contradiction", "neutral"))
        self._cache[key] = result
        return result


def get_nli_model() -> NLIStanceModel:
    global _nli_instance
    if _nli_instance is None:
        _nli_instance = NLIStanceModel()
    return _nli_instance
