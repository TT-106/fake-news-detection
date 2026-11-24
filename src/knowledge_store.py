"""AVerImaTeC 官方 Knowledge Store 清洗、双路检索（文本+反向搜图）、MMR 重排与可回放快照适配器。

协议边界：
- 仅读取官方 Knowledge Store 静态文档与公开声明文本/日期/地点，严禁读取数据集标签、QA、核查文章 URL 或事后解释。
- 显式过滤抓取报错/JS 拦截壳页面与核查站域名，并完整记录每类过滤数量。
- 所有返回片段均落盘为 UTF-8/LF 快照并校验 SHA-256 与字符起止偏移，满足 `ArchivedPassage` 可回放契约。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urlparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.research_contracts import ArchivedPassage, normalized_url


FACTCHECK_PATTERNS = re.compile(
    r"(?:snopes\.com|politifact\.com|factcheck\.org|fullfact\.org|leadstories\.com|"
    r"checkyourfact\.com|africacheck\.org|altnews\.in|boomlive\.in|vishvasnews\.com|"
    r"newschecker\.in|factly\.in|thequint\.com/news/webqoof|factcrescendo\.com|"
    r"logicallyfacts\.com|misbar\.com|factcheck\.afp\.com|factual\.afp\.com|"
    r"reuters\.com/fact-check|apnews\.com/hub/ap-fact-check|usatoday\.com/story/news/factcheck|"
    r"verifythis\.com|dpa-factchecking\.com|chequeado\.com|maldita\.es|newtral\.es|"
    r"pesacheck\.org|dubawa\.org|rappler\.com/newsbreak/fact-check|verafiles\.org|"
    r"aap\.com\.au/factcheck|indiatoday\.in/fact-check|newsmeter\.in|thip\.media|"
    r"bufale\.net|times-fact-check|fact[-_]?check)",
    re.I,
)

BOILERPLATE_PATTERNS = re.compile(
    r"(?:javascript is not available|javascript is disabled|please enable javascript|"
    r"please enable cookies|enable js and disable any ad blocker|"
    r"access denied|403 forbidden|404 not found|page not found|"
    r"just a moment\.\.\.|checking your browser before accessing|"
    r"attention required!\s*\|\s*cloudflare|please verify you are a human|"
    r"robot or human\?|something went wrong, but don't fret|"
    r"we've detected that javascript is disabled|subscribe to continue reading|"
    r"you have been blocked|sorry, you have been blocked)",
    re.I,
)

WIRE_CLUE_PATTERN = re.compile(
    r"\b(associated press\b|reuters\b|afp\b|agence france[- ]presse\b|pti\b|press trust of india\b|"
    r"getty images\b|shutterstock\b|ani\b|ians\b|xinhua\b|bloomberg\b|credit:|photo:|"
    r"file photo\b|via twitter\b|via instagram\b|originally published\b|syndicated\b|reposted\b|courtesy of\b)",
    re.I,
)

RIS_DATE_RE = re.compile(r"The article was published on\s+(\d{4}-\d{2}-\d{2})", re.I)
URL_YMD_RE = re.compile(r"/(199\d|20[0-2]\d)/(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])(?:/|$|\D)")
URL_COMPACT_YMD_RE = re.compile(r"/(199\d|20[0-2]\d)(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?:/|$|\D)")
TEXT_ISO_DATE_RE = re.compile(r"\b(199\d|20[0-2]\d)-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")
MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
TEXT_MONTH_MDY_RE = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+"
    r"([12]?\d|3[01]),?\s+(199\d|20[0-2]\d)\b",
    re.I,
)


@dataclass(frozen=True)
class CleanKSDoc:
    doc_id: str
    track: str  # "text_search" | "reverse_image_search"
    url: str
    domain: str
    text: str  # normalized UTF-8/LF text
    published_date: str  # YYYY-MM-DD or ""
    date_source: str  # "ris_marker" | "url_path" | "lead_text" | "unverified"
    wire_clues: tuple[str, ...]


@dataclass(frozen=True)
class RetrievedChunk:
    evidence_id: str
    doc_id: str
    track: str
    url: str
    domain: str
    published_date: str
    date_source: str
    wire_clues: tuple[str, ...]
    start: int
    end: int
    text: str
    relevance_score: float
    lexical_score: float
    dense_score: float
    snapshot_path: str = ""
    snapshot_sha256: str = ""


def clean_domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def normalize_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def extract_observable_date(url: str, text: str) -> tuple[str, str]:
    """Extract verifiable YYYY-MM-DD date from RIS marker, URL path, or lead text."""
    m = RIS_DATE_RE.search(text)
    if m:
        return m.group(1), "ris_marker"
    m = URL_YMD_RE.search(url or "")
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}", "url_path"
    m = URL_COMPACT_YMD_RE.search(url or "")
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}", "url_path"
    head = text[:1200]
    m = TEXT_ISO_DATE_RE.search(head)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "lead_text"
    m = TEXT_MONTH_MDY_RE.search(head)
    if m:
        mo = MONTHS[m.group(1).lower().rstrip(".")]
        d = int(m.group(2))
        y = int(m.group(3))
        if 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}", "lead_text"
    return "", "unverified"


def is_boilerplate_stub(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 80:
        return True
    head = stripped[:600]
    if BOILERPLATE_PATTERNS.search(head) and len(stripped) < 1200:
        return True
    return False


def clean_ks_documents(
    raw_rows: list[tuple[str, dict]],
    *,
    claim_date: str = "",
    exclude_factcheck_domains: bool = True,
    min_chars: int = 80,
) -> tuple[list[CleanKSDoc], dict]:
    """Filter raw Knowledge Store rows into clean, deduplicated, auditable documents."""
    stats = Counter()
    prelim: list[tuple[str, str, str, str, str, str, tuple[str, ...]]] = []
    seen_url_text: set[tuple[str, str]] = set()
    head_sig_counts: Counter[str] = Counter()

    for idx, (track, row) in enumerate(raw_rows):
        stats["raw_total"] += 1
        url = (row.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            stats["dropped_invalid_url"] += 1
            continue
        if exclude_factcheck_domains and FACTCHECK_PATTERNS.search(url):
            stats["dropped_factcheck_domain"] += 1
            continue

        u2t = row.get("url2text")
        if isinstance(u2t, list):
            text = normalize_lf("\n".join(str(x) for x in u2t if x is not None)).strip()
        else:
            text = normalize_lf(str(u2t or "")).strip()

        if len(text) < min_chars:
            stats["dropped_too_short"] += 1
            continue
        if is_boilerplate_stub(text):
            stats["dropped_boilerplate_pattern"] += 1
            continue

        pub_date, date_src = extract_observable_date(url, text)
        # Strict temporal safety: if an observable publication date is on or after claim_date, drop it
        if claim_date and pub_date and len(claim_date) >= 10 and pub_date >= claim_date[:10]:
            stats["dropped_post_or_same_day_date"] += 1
            continue

        norm_u = normalized_url(url)
        text_hash = sha256(text.encode("utf-8")).hexdigest()
        if (norm_u, text_hash) in seen_url_text:
            stats["dropped_exact_url_content_dup"] += 1
            continue
        seen_url_text.add((norm_u, text_hash))

        head_sig = re.sub(r"\s+", " ", text[:300].casefold()).strip()
        head_sig_counts[head_sig] += 1
        clues = tuple(sorted({m.group(1).lower() for m in WIRE_CLUE_PATTERN.finditer(text[:4000])}))
        prelim.append((f"ks-{track[:3]}-{idx}", track, url, clean_domain(url), text, f"{pub_date}|{date_src}", clues))

    # Reject cross-URL repeated boilerplate stubs (same opening 300 chars across >= 3 distinct URLs)
    boilerplate_heads = {sig for sig, cnt in head_sig_counts.items() if cnt >= 3}
    cleaned: list[CleanKSDoc] = []
    for doc_id, track, url, dom, text, date_pair, clues in prelim:
        head_sig = re.sub(r"\s+", " ", text[:300].casefold()).strip()
        if head_sig in boilerplate_heads and len(text) < 2000:
            stats["dropped_repeated_cross_url_stub"] += 1
            continue
        pub_date, date_src = date_pair.split("|", 1)
        cleaned.append(
            CleanKSDoc(
                doc_id=doc_id,
                track=track,
                url=url,
                domain=dom,
                text=text,
                published_date=pub_date,
                date_source=date_src,
                wire_clues=clues,
            )
        )
        stats["kept_total"] += 1
        stats[f"kept_{track}"] += 1

    return cleaned, dict(stats)


def chunk_document(doc: CleanKSDoc, chunk_chars: int = 620, stride_chars: int = 480, max_chunks: int = 10) -> list[tuple[CleanKSDoc, int, int, str]]:
    """Slice document text into exact verifiable [start:end] windows preserving exact original slice."""
    text = doc.text
    n = len(text)
    if n <= chunk_chars:
        return [(doc, 0, n, text)]
    chunks = []
    pos = 0
    while pos < n and len(chunks) < max_chunks:
        end = min(n, pos + chunk_chars)
        # Try to snap end cleanly to whitespace/newline within last 80 chars
        if end < n:
            ws = max(text.rfind("\n", pos + chunk_chars - 100, end), text.rfind(" ", pos + chunk_chars - 80, end))
            if ws > pos + 150:
                end = ws
        snippet = text[pos:end]
        if len(snippet.strip()) >= 60:
            chunks.append((doc, pos, end, snippet))
        if end >= n:
            break
        pos += stride_chars
    return chunks


def mmr_select(
    relevance: np.ndarray,
    embeddings: np.ndarray,
    k: int,
    lambda_mult: float = 0.65,
    doc_ids: list[str] | None = None,
    max_per_doc: int = 1,
) -> list[int]:
    """Maximal Marginal Relevance reranking with per-document diversity cap."""
    n = len(relevance)
    if n == 0 or k <= 0:
        return []
    sim_mat = embeddings @ embeddings.T
    selected: list[int] = []
    doc_counts: Counter[str] = Counter()
    candidates = list(range(n))

    while len(selected) < min(k, n) and candidates:
        best_idx = None
        best_score = -1e18
        for idx in candidates:
            if doc_ids is not None and doc_counts[doc_ids[idx]] >= max_per_doc:
                continue
            rel = float(relevance[idx])
            red = max((float(sim_mat[idx, s]) for s in selected), default=0.0)
            score = lambda_mult * rel - (1.0 - lambda_mult) * red
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        if doc_ids is not None:
            doc_counts[doc_ids[best_idx]] += 1
        candidates.remove(best_idx)
    return selected


def greedy_select(
    relevance: np.ndarray,
    k: int,
    doc_ids: list[str] | None = None,
    max_per_doc: int = 1,
) -> list[int]:
    order = np.argsort(-relevance).tolist()
    selected: list[int] = []
    doc_counts: Counter[str] = Counter()
    for idx in order:
        if doc_ids is not None and doc_counts[doc_ids[idx]] >= max_per_doc:
            continue
        selected.append(idx)
        if doc_ids is not None:
            doc_counts[doc_ids[idx]] += 1
        if len(selected) >= k:
            break
    return selected


class LocalDenseEmbedder:
    """Lazy-loaded local all-MiniLM-L6-v2 embedder (CPU, local_files_only=True)."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(4)
            self._model = SentenceTransformer(str(self.model_dir), device="cpu", local_files_only=True)
        emb = self._model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)


def retrieve_policy_passages(
    claim_id: str,
    query_text: str,
    docs: list[CleanKSDoc],
    embedder: LocalDenseEmbedder,
    *,
    policy: str,
    k_total: int = 6,
    k_ris: int = 3,
    mmr_lambda: float = 0.65,
    snapshot_dir: Path | None = None,
) -> tuple[list[RetrievedChunk], dict]:
    """Retrieve verifiable top-k passages under a named policy (`ks_text_only`, `ks_dual_text_ris`, `ks_dual_mmr`)."""
    if policy not in {"ks_text_only", "ks_dual_text_ris", "ks_dual_mmr"}:
        raise ValueError(f"Unknown retrieval policy: {policy}")

    text_docs = [d for d in docs if d.track == "text_search"]
    ris_docs = [d for d in docs if d.track == "reverse_image_search"]

    def rank_pool(pool_docs: list[CleanKSDoc], prefilter_k: int) -> tuple[list[tuple[CleanKSDoc, int, int, str]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        all_chunks: list[tuple[CleanKSDoc, int, int, str]] = []
        for d in pool_docs:
            all_chunks.extend(chunk_document(d))
        if not all_chunks:
            return [], np.zeros(0), np.zeros(0), np.zeros(0), np.zeros((0, 384), dtype=np.float32)

        texts = [c[3] for c in all_chunks]
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
            mat = vec.fit_transform(texts)
            q_vec = vec.transform([query_text])
            lex_all = (mat @ q_vec.T).toarray().ravel()
        except ValueError:
            lex_all = np.zeros(len(texts), dtype=np.float32)

        top_lex_idx = np.argsort(-lex_all)[:prefilter_k]
        cand_chunks = [all_chunks[i] for i in top_lex_idx]
        cand_lex = lex_all[top_lex_idx]
        max_lex = float(np.max(cand_lex)) if len(cand_lex) and float(np.max(cand_lex)) > 0 else 1.0
        norm_lex = cand_lex / max_lex

        q_emb = embedder.encode([query_text])[0]
        c_embs = embedder.encode([c[3] for c in cand_chunks])
        dense = np.clip(c_embs @ q_emb, -1.0, 1.0)
        hybrid = 0.60 * dense + 0.40 * norm_lex
        return cand_chunks, hybrid, norm_lex, dense, c_embs

    chosen_items: list[tuple[tuple[CleanKSDoc, int, int, str], float, float, float, np.ndarray]] = []
    if policy == "ks_text_only":
        chunks, rel, lex, den, embs = rank_pool(text_docs, prefilter_k=80)
        sel = greedy_select(rel, k_total, doc_ids=[c[0].doc_id for c in chunks], max_per_doc=1)
        for i in sel:
            chosen_items.append((chunks[i], float(rel[i]), float(lex[i]), float(den[i]), embs[i]))
    else:
        r_chunks, r_rel, r_lex, r_den, r_embs = rank_pool(ris_docs, prefilter_k=60)
        t_chunks, t_rel, t_lex, t_den, t_embs = rank_pool(text_docs, prefilter_k=80)

        use_mmr = (policy == "ks_dual_mmr")
        ris_quota = min(k_ris, len(r_chunks))
        text_quota = k_total - ris_quota

        if use_mmr:
            r_sel = mmr_select(r_rel, r_embs, ris_quota, lambda_mult=mmr_lambda, doc_ids=[c[0].doc_id for c in r_chunks], max_per_doc=1)
            t_sel = mmr_select(t_rel, t_embs, text_quota, lambda_mult=mmr_lambda, doc_ids=[c[0].doc_id for c in t_chunks], max_per_doc=1)
        else:
            r_sel = greedy_select(r_rel, ris_quota, doc_ids=[c[0].doc_id for c in r_chunks], max_per_doc=1)
            t_sel = greedy_select(t_rel, text_quota, doc_ids=[c[0].doc_id for c in t_chunks], max_per_doc=1)

        # Preserve RIS items first so visual provenance stays prominent, then text search items
        for i in r_sel:
            chosen_items.append((r_chunks[i], float(r_rel[i]), float(r_lex[i]), float(r_den[i]), r_embs[i]))
        for i in t_sel:
            chosen_items.append((t_chunks[i], float(t_rel[i]), float(t_lex[i]), float(t_den[i]), t_embs[i]))

    retrieved: list[RetrievedChunk] = []
    chosen_embs = []
    safe_cid = claim_id.replace(":", "-")
    retrieved_at = datetime.now(timezone.utc).isoformat()

    for rank_idx, ((doc, start, end, snippet), rel_s, lex_s, den_s, emb) in enumerate(chosen_items, 1):
        ev_id = f"{safe_cid}-{policy}-{rank_idx}"
        snap_path_str = ""
        snap_sha = ""
        if snapshot_dir is not None:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            raw_bytes = doc.text.encode("utf-8")
            snap_sha = sha256(raw_bytes).hexdigest()
            snap_file = snapshot_dir / f"{snap_sha}.txt"
            if not snap_file.exists():
                snap_file.write_bytes(raw_bytes)
            snap_path_str = str(snap_file)
            # Enforce exact ArchivedPassage replay verification!
            ArchivedPassage(
                evidence_id=ev_id,
                url=doc.url,
                snapshot_path=snap_path_str,
                snapshot_sha256=snap_sha,
                start=start,
                end=end,
                text=snippet,
                retrieved_at=retrieved_at,
                published_at=doc.published_date,
            ).verify()

        retrieved.append(
            RetrievedChunk(
                evidence_id=ev_id,
                doc_id=doc.doc_id,
                track=doc.track,
                url=doc.url,
                domain=doc.domain,
                published_date=doc.published_date,
                date_source=doc.date_source,
                wire_clues=doc.wire_clues,
                start=start,
                end=end,
                text=snippet,
                relevance_score=round(rel_s, 4),
                lexical_score=round(lex_s, 4),
                dense_score=round(den_s, 4),
                snapshot_path=snap_path_str,
                snapshot_sha256=snap_sha,
            )
        )
        chosen_embs.append(emb)

    # Audit source-dependence opportunities INSIDE the actually retrieved top-K passages!
    pair_audit = audit_retrieved_source_pairs(retrieved, np.asarray(chosen_embs, dtype=np.float32) if chosen_embs else np.zeros((0, 384)))
    return retrieved, pair_audit


def audit_retrieved_source_pairs(chunks: list[RetrievedChunk], embeddings: np.ndarray) -> dict:
    """Measure cross-domain near-duplicate / shared-origin / same-domain opportunities inside top-K passages."""
    n = len(chunks)
    cross_domain_high = []
    cross_domain_moderate = []
    same_domain_pairs = []
    ris_count = sum(1 for c in chunks if c.track == "reverse_image_search")
    wire_count = sum(1 for c in chunks if c.wire_clues)
    dated_count = sum(1 for c in chunks if c.published_date)

    if n >= 2 and embeddings.shape[0] == n:
        sim = embeddings @ embeddings.T
        for i in range(n):
            for j in range(i + 1, n):
                s = float(sim[i, j])
                same_dom = bool(chunks[i].domain and chunks[i].domain == chunks[j].domain)
                entry = {
                    "ev_a": chunks[i].evidence_id,
                    "ev_b": chunks[j].evidence_id,
                    "domain_a": chunks[i].domain,
                    "domain_b": chunks[j].domain,
                    "track_a": chunks[i].track,
                    "track_b": chunks[j].track,
                    "dense_cos": round(s, 4),
                }
                if same_dom:
                    same_domain_pairs.append(entry)
                elif s >= 0.65:
                    cross_domain_high.append(entry)
                elif s >= 0.50:
                    cross_domain_moderate.append(entry)

    return {
        "num_retrieved": n,
        "num_ris_passages": ris_count,
        "num_dated_passages": dated_count,
        "num_wire_clue_passages": wire_count,
        "unique_domains": len({c.domain for c in chunks if c.domain}),
        "cross_domain_high_sim_pairs_ge_065": len(cross_domain_high),
        "cross_domain_moderate_sim_pairs_050_065": len(cross_domain_moderate),
        "same_domain_pairs": len(same_domain_pairs),
        "has_source_verify_opportunity": bool(cross_domain_high or (cross_domain_moderate and wire_count >= 1)),
        "high_sim_examples": cross_domain_high[:3],
    }


def load_local_ks_claim_rows(sample_dir: Path, train_index: int) -> list[tuple[str, dict]]:
    """Load local text_search + reverse_image_search JSONL rows for a claim."""
    rows: list[tuple[str, dict]] = []
    for track in ("text_search", "reverse_image_search"):
        p = sample_dir / f"train_{train_index}_{track}.jsonl"
        if not p.is_file():
            continue
        for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if not ln.strip():
                continue
            try:
                rows.append((track, json.loads(ln)))
            except json.JSONDecodeError:
                continue
    return rows


def load_ks_claim_rows_from_zip_or_dir(store_path: Path, claim_id: str) -> list[tuple[str, dict]]:
    """Load raw KS rows for `train:<idx>` or `val:<idx>` from a ZIP archive or extracted directory."""
    if ":" not in claim_id:
        raise ValueError(f"Expected '<split>:<index>' claim_id, got {claim_id!r}")
    split, idx_str = claim_id.split(":", 1)
    idx = int(idx_str)

    if store_path.is_file() and store_path.suffix.lower() == ".zip":
        import zipfile

        rows: list[tuple[str, dict]] = []
        members = (
            ("text_search", f"text_related/text_related_store_text_{split}/{idx}.json"),
            ("reverse_image_search", f"text_related/image_related_store_text_{split}/{idx}.json"),
        )
        with zipfile.ZipFile(store_path, "r") as zf:
            names = set(zf.namelist())
            for track, member in members:
                if member not in names:
                    continue
                raw = zf.read(member).decode("utf-8", errors="replace")
                for ln in raw.splitlines():
                    if not ln.strip():
                        continue
                    try:
                        rows.append((track, json.loads(ln)))
                    except json.JSONDecodeError:
                        continue
        return rows

    if store_path.is_dir():
        # Check flat pilot naming (`train_111_text_search.jsonl`) first
        flat = load_local_ks_claim_rows(store_path, idx)
        if flat:
            return flat
        # Check standard unzipped Knowledge Store directory layout
        rows = []
        candidates = (
            ("text_search", store_path / f"text_related/text_related_store_text_{split}/{idx}.json"),
            ("text_search", store_path / f"text_related_store_text_{split}/{idx}.json"),
            ("reverse_image_search", store_path / f"text_related/image_related_store_text_{split}/{idx}.json"),
            ("reverse_image_search", store_path / f"image_related_store_text_{split}/{idx}.json"),
        )
        seen_tracks = set()
        for track, p in candidates:
            if track in seen_tracks or not p.is_file():
                continue
            seen_tracks.add(track)
            for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
                if not ln.strip():
                    continue
                try:
                    rows.append((track, json.loads(ln)))
                except json.JSONDecodeError:
                    continue
        return rows

    raise FileNotFoundError(f"Knowledge Store path does not exist: {store_path}")


class OfficialKSEvidenceTools:
    """实现 `EvidenceTools` 协议的官方 Knowledge Store 主动取证与轨迹归档适配器。

    支持：
    - `initial`：基于公开图文声明与原子声明的双路（RIS + 文本）MMR 初始检索；
    - `candidates`：依据当前未解缺口、剩余图像出处与最高不确定性来源对生成至多 3 类各 1 个候选行动；
    - `execute`：
      * `fact_search`：针对目标缺口从剩余洁净文本库检索新证据，并验证提取准确引文 `GapResolution`；
      * `image_trace`：从剩余反向搜图网页（及带图源署名文档）检索视觉出处新证据；
      * `source_verify`：核对目标证据对 `(i, j)` 的引用链、通讯社共源或长段落复述，返回三态 `RelationObservation`（`True`/`False`/`None`）及归档证明；
    - `finalize` 与 `export_replay_manifest`：完整导出兼容 `ArchivedEvidenceTools` 的可回放账本。
    """

    def __init__(
        self,
        store_path: Path | str | None,
        snapshot_dir: Path | str,
        embedder: LocalDenseEmbedder,
        *,
        in_memory_store: dict[str, list[tuple[str, dict]]] | None = None,
        initial_k: int = 4,
        initial_ris_k: int = 2,
        step_k: int = 2,
        mmr_lambda: float = 0.65,
        exclude_factcheck_domains: bool = True,
        initial_action_cost_rmb: float = 0.0,
        step_action_cost_rmb: dict[str, float] | None = None,
        initial_cost_upper_bound: float = 0.08,
        tool_version: str = "averimatec-official-ks-tools-v1",
        query_template_version: str = "ks-claim-atom-gap-ris-v1",
    ) -> None:
        from src.research_contracts import fingerprint

        self.store_path = Path(store_path) if store_path is not None else None
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.in_memory_store = in_memory_store or {}
        self.initial_k = int(initial_k)
        self.initial_ris_k = int(initial_ris_k)
        self.step_k = int(step_k)
        self.mmr_lambda = float(mmr_lambda)
        self.exclude_factcheck_domains = bool(exclude_factcheck_domains)
        self.initial_action_cost_rmb = float(initial_action_cost_rmb)
        self.step_costs = step_action_cost_rmb or {
            "fact_search": 0.0,
            "image_trace": 0.0,
            "source_verify": 0.0,
        }
        self.initial_cost_upper_bound = float(initial_cost_upper_bound)
        self.tool_version = tool_version
        self.query_template_version = query_template_version
        self.version = fingerprint({
            "protocol": "OPEN_WEB_ARCHIVED_REPLAY",
            "backend": "AVERIMATEC_OFFICIAL_KS_V1",
            "tool_version": self.tool_version,
            "query_template_version": self.query_template_version,
            "initial_k": self.initial_k,
            "initial_ris_k": self.initial_ris_k,
            "step_k": self.step_k,
            "mmr_lambda": self.mmr_lambda,
            "exclude_factcheck_domains": self.exclude_factcheck_domains,
            "initial_cost_upper_bound": self.initial_cost_upper_bound,
        })
        self._clean_cache: dict[str, list[CleanKSDoc]] = {}
        self._replay: dict[str, dict] = {
            "initial": {},
            "candidates": {},
            "actions": {},
            "finalize": {},
        }

    def _get_clean_docs(self, claim) -> list[CleanKSDoc]:
        cid = claim.claim_id
        if cid not in self._clean_cache:
            if cid in self.in_memory_store:
                raw_rows = self.in_memory_store[cid]
            elif self.store_path is not None:
                raw_rows = load_ks_claim_rows_from_zip_or_dir(self.store_path, cid)
            else:
                raw_rows = []
            cleaned, _ = clean_ks_documents(
                raw_rows,
                claim_date=claim.date,
                exclude_factcheck_domains=self.exclude_factcheck_domains,
            )
            self._clean_cache[cid] = cleaned
        return self._clean_cache[cid]

    def _materialize_passage(self, ev_id: str, doc: CleanKSDoc, start: int, end: int, snippet: str) -> ArchivedPassage:
        raw_bytes = doc.text.encode("utf-8")
        snap_sha = sha256(raw_bytes).hexdigest()
        snap_file = self.snapshot_dir / f"{snap_sha}.txt"
        if not snap_file.exists():
            snap_file.write_bytes(raw_bytes)
        retrieved_at = datetime.now(timezone.utc).isoformat()

        # RIS membership and wire credits alone do not verify an original image.
        passage = ArchivedPassage(
            evidence_id=ev_id,
            url=doc.url,
            snapshot_path=str(snap_file),
            snapshot_sha256=snap_sha,
            start=start,
            end=end,
            text=snippet,
            retrieved_at=retrieved_at,
            published_at=doc.published_date,
            author="",
            image_provenance_verified=False,
            provenance_proof="",
        )
        passage.verify()
        return passage

    def _rank_novel_passages(
        self,
        claim,
        query: str,
        pool_docs: list[CleanKSDoc],
        existing_identities: set,
        existing_ids: set[str],
        max_new: int,
        prefix: str,
    ) -> list[ArchivedPassage]:
        if max_new <= 0 or not pool_docs:
            return []
        all_chunks: list[tuple[CleanKSDoc, int, int, str]] = []
        for d in pool_docs:
            for item in chunk_document(d):
                doc_i, st_i, en_i, txt_i = item
                snap_sha = sha256(doc_i.text.encode("utf-8")).hexdigest()
                exact_id = (normalized_url(doc_i.url), snap_sha, st_i, en_i)
                if exact_id not in existing_identities:
                    all_chunks.append(item)
        if not all_chunks:
            return []

        texts = [c[3] for c in all_chunks]
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
            mat = vec.fit_transform(texts)
            q_vec = vec.transform([query])
            lex_all = (mat @ q_vec.T).toarray().ravel()
        except ValueError:
            lex_all = np.zeros(len(texts), dtype=np.float32)

        top_idx = np.argsort(-lex_all)[:60]
        cand = [all_chunks[i] for i in top_idx]
        cand_lex = lex_all[top_idx]
        max_lex = float(np.max(cand_lex)) if len(cand_lex) and float(np.max(cand_lex)) > 0 else 1.0
        norm_lex = cand_lex / max_lex

        q_emb = self.embedder.encode([query])[0]
        c_embs = self.embedder.encode([c[3] for c in cand])
        dense = np.clip(c_embs @ q_emb, -1.0, 1.0)
        rel = 0.60 * dense + 0.40 * norm_lex

        sel = mmr_select(rel, c_embs, max_new, lambda_mult=self.mmr_lambda, doc_ids=[c[0].doc_id for c in cand], max_per_doc=1)
        passages: list[ArchivedPassage] = []
        seen_ids = set(existing_ids)
        seen_ident = set(existing_identities)
        for rank_i, s_idx in enumerate(sel, 1):
            doc, st_i, en_i, txt_i = cand[s_idx]
            base_id = f"{claim.claim_id.replace(':', '-')}-{prefix}-{rank_i}"
            ev_id = base_id
            suffix = 2
            while ev_id in seen_ids:
                ev_id = f"{base_id}-{suffix}"
                suffix += 1
            p = self._materialize_passage(ev_id, doc, st_i, en_i, txt_i)
            if p.exact_identity in seen_ident:
                continue
            seen_ids.add(ev_id)
            seen_ident.add(p.exact_identity)
            passages.append(p)
            if len(passages) >= max_new:
                break
        return passages

    def initial(self, claim, analysis, ledger):
        import time
        import uuid
        from src.faithful_pipeline import ToolOutcome
        from src.research_contracts import CostEntry, fingerprint

        t0 = time.perf_counter()
        docs = self._get_clean_docs(claim)
        atom_texts = " ".join(a.text for a in analysis.atoms)
        query = " ".join(x for x in (claim.text, claim.location, atom_texts) if x).strip()

        ris_docs = [d for d in docs if d.track == "reverse_image_search"]
        txt_docs = [d for d in docs if d.track == "text_search"]

        ris_passages = self._rank_novel_passages(
            claim, query, ris_docs, set(), set(), min(self.initial_ris_k, self.initial_k), "ks-init-ris"
        )
        used_ident = {p.exact_identity for p in ris_passages}
        used_ids = {p.evidence_id for p in ris_passages}
        txt_quota = max(0, min(8, self.initial_k) - len(ris_passages))
        txt_passages = self._rank_novel_passages(
            claim, query, txt_docs, used_ident, used_ids, txt_quota, "ks-init-txt"
        )
        all_passages = tuple(ris_passages + txt_passages)

        key = fingerprint({"claim": asdict(claim), "analysis": asdict(analysis)})
        cost = CostEntry(
            call_id=uuid.uuid4().hex,
            component="ks.initial_retrieve",
            amount_rmb=self.initial_action_cost_rmb,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            success=True,
            request_sha256=key,
            billing_basis="AVerImaTeC 官方静态知识库本地混合检索与快照校验；无外部搜索 API 账单",
        )
        ledger.add(cost)
        outcome = ToolOutcome(passages=all_passages)
        self._replay["initial"][key] = {
            "costs": [asdict(cost)],
            "outcome": {
                "passages": [asdict(p) for p in all_passages],
                "relation": None,
                "resolved_gaps": [],
                "failed": False,
                "error": "",
                "gap_resolutions": [],
            },
        }
        return outcome

    def candidates(self, state):
        from src.faithful_pipeline import source_pair_features
        from src.research_contracts import CandidateAction, fingerprint

        if state.step >= 3:
            self._replay["candidates"][state.identity] = []
            return ()

        docs = self._get_clean_docs(state.claim)
        existing_ident = {e.passage.exact_identity for e in state.evidence}
        actions: list[CandidateAction] = []
        basis_note = "AVerImaTeC 官方静态知识库本地行动与先导耗时上限核算"

        # 1. fact_search candidate
        if len(state.evidence) < 8 and any(d.track == "text_search" for d in docs):
            focus = state.gaps[0] if state.gaps else state.atoms[0].text
            q_fact = f"{state.claim.text} {focus}".strip()
            aid = f"fact_search:s{state.step}:{fingerprint({'q': q_fact, 's': state.step})[:8]}"
            if aid not in state.completed_actions:
                c_val = float(self.step_costs.get("fact_search", 0.0))
                actions.append(
                    CandidateAction(
                        action_id=aid,
                        kind="fact_search",
                        query=q_fact,
                        expected_cost=c_val,
                        cost_upper_bound=max(c_val, 0.04),
                        cost_basis=basis_note,
                    )
                )

        # 2. source_verify candidate (pick highest uncertainty * observable similarity pair not yet verified)
        n_ev = len(state.evidence)
        if n_ev >= 2:
            best_pair = None
            best_priority = -1.0
            for i in range(n_ev):
                for j in range(i + 1, n_ev):
                    pa, pb = state.evidence[i].passage, state.evidence[j].passage
                    pair_aid = f"source_verify:{pa.evidence_id}:{pb.evidence_id}"
                    if pair_aid in state.completed_actions:
                        continue
                    p_ij = float(state.source_matrix[i][j])
                    unc = p_ij * (1.0 - p_ij)
                    feats = source_pair_features(pa, pb)
                    # feats = (citation_link, time_similarity, both_provenance, similarity, same_domain, known_time)
                    priority = unc * (1.0 + feats[0] + feats[3] + 0.5 * feats[4]) + 0.01 * feats[3]
                    if priority > best_priority:
                        best_priority = priority
                        best_pair = (i, j, pair_aid, f"verify_origin({pa.evidence_id}, {pb.evidence_id})")
            if best_pair is not None:
                i, j, pair_aid, q_src = best_pair
                c_val = float(self.step_costs.get("source_verify", 0.0))
                actions.append(
                    CandidateAction(
                        action_id=pair_aid,
                        kind="source_verify",
                        query=q_src,
                        expected_cost=c_val,
                        cost_upper_bound=max(c_val, 0.03),
                        cost_basis=basis_note,
                        target_pair=(i, j),
                    )
                )

        # 3. image_trace candidate
        if len(state.evidence) < 8 and any(d.track == "reverse_image_search" or d.wire_clues for d in docs):
            q_img = f"image_provenance {state.claim.text} {state.claim.location}".strip()
            aid = f"image_trace:s{state.step}:{fingerprint({'q': q_img, 's': state.step})[:8]}"
            if aid not in state.completed_actions and len(existing_ident) >= 0:
                c_val = float(self.step_costs.get("image_trace", 0.0))
                actions.append(
                    CandidateAction(
                        action_id=aid,
                        kind="image_trace",
                        query=q_img,
                        expected_cost=c_val,
                        cost_upper_bound=max(c_val, 0.04),
                        cost_basis=basis_note,
                    )
                )

        result = tuple(actions)
        self._replay["candidates"][state.identity] = [asdict(a) for a in result]
        return result

    def execute(self, state, action, ledger):
        import time
        import uuid
        from src.faithful_pipeline import GapResolution, RelationObservation, ToolOutcome, open_snapshot, source_pair_features
        from src.research_contracts import CostEntry, fingerprint

        t0 = time.perf_counter()
        docs = self._get_clean_docs(state.claim)
        existing_ident = {e.passage.exact_identity for e in state.evidence}
        existing_ids = {e.passage.evidence_id for e in state.evidence}
        remaining_slots = max(0, 8 - len(state.evidence))
        quota = min(self.step_k, remaining_slots)

        passages: tuple[ArchivedPassage, ...] = ()
        relation: RelationObservation | None = None
        resolved_gaps: tuple[str, ...] = ()
        gap_resolutions: tuple[GapResolution, ...] = ()

        if action.kind == "fact_search":
            txt_docs = [d for d in docs if d.track == "text_search"]
            new_p = self._rank_novel_passages(
                state.claim, action.query, txt_docs, existing_ident, existing_ids, quota, f"ks-fact-s{state.step}"
            )
            passages = tuple(new_p)
            if passages and state.gaps:
                target_gap = state.gaps[0]
                gap_tokens = {w for w in re.findall(r"[^\W\d_]{3,}", target_gap.casefold())}
                for p in passages:
                    p_tokens = {w for w in re.findall(r"[^\W\d_]{3,}", p.text.casefold())}
                    if not gap_tokens or len(gap_tokens & p_tokens) >= min(1, len(gap_tokens)):
                        quote = p.text.strip()[:min(140, len(p.text.strip()))]
                        if quote and quote in p.text:
                            resolved_gaps = (target_gap,)
                            gap_resolutions = (
                                GapResolution(
                                    gap=target_gap,
                                    evidence_id=p.evidence_id,
                                    quote=quote,
                                    reason="官方知识库补充检索命中该缺口的关键实体与事实陈述",
                                ),
                            )
                            break

        elif action.kind == "image_trace":
            ris_docs = [d for d in docs if d.track == "reverse_image_search"]
            credit_txt_docs = [d for d in docs if d.track == "text_search" and d.wire_clues]
            new_p = self._rank_novel_passages(
                state.claim, action.query, ris_docs + credit_txt_docs, existing_ident, existing_ids, quota, f"ks-img-s{state.step}"
            )
            passages = tuple(new_p)

        elif action.kind == "source_verify":
            i, j = action.target_pair
            pa = state.evidence[i].passage
            pb = state.evidence[j].passage
            pa.verify()
            pb.verify()
            full_a = open_snapshot(pa)
            full_b = open_snapshot(pb)
            feats = source_pair_features(pa, pb)
            # feats = (citation_link, time_similarity, both_provenance, similarity, same_domain, known_time)
            clues_a = {m.group(1).lower() for m in WIRE_CLUE_PATTERN.finditer(full_a[:4000])}
            clues_b = {m.group(1).lower() for m in WIRE_CLUE_PATTERN.finditer(full_b[:4000])}
            shared_agencies = clues_a & clues_b - {"photo:", "credit:", "file photo"}

            if feats[0] >= 1.0 or feats[3] >= 0.45 or (shared_agencies and feats[3] >= 0.18):
                same_origin: bool | None = True
            elif feats[4] == 0.0 and feats[3] < 0.12 and not shared_agencies and feats[5] >= 1.0:
                same_origin = False
            else:
                same_origin = None
            relation = RelationObservation(target_pair=(i, j), same_origin=same_origin, proof=(pa, pb))

        key = fingerprint({"history_id": state.identity, "action": asdict(action)})
        cost = CostEntry(
            call_id=uuid.uuid4().hex,
            component=f"ks.{action.kind}",
            amount_rmb=float(self.step_costs.get(action.kind, 0.0)),
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            success=True,
            request_sha256=key,
            billing_basis="AVerImaTeC 官方静态知识库本地行动执行与快照核验；无外部 API 账单",
        )
        ledger.add(cost)
        outcome = ToolOutcome(
            passages=passages,
            relation=relation,
            resolved_gaps=resolved_gaps,
            failed=False,
            error="",
            gap_resolutions=gap_resolutions,
        )
        self._replay["actions"][key] = {
            "costs": [asdict(cost)],
            "outcome": {
                "passages": [asdict(p) for p in passages],
                "relation": None if relation is None else {
                    "target_pair": list(relation.target_pair),
                    "same_origin": relation.same_origin,
                    "proof": [asdict(p) for p in relation.proof],
                },
                "resolved_gaps": list(resolved_gaps),
                "failed": False,
                "error": "",
                "gap_resolutions": [asdict(g) for g in gap_resolutions],
            },
        }
        return outcome

    def finalize(self, state, ledger) -> None:
        import uuid
        from src.research_contracts import CostEntry, fingerprint

        key = fingerprint({"finalize_history_id": state.identity})
        cost = CostEntry(
            call_id=uuid.uuid4().hex,
            component="ks.finalize",
            amount_rmb=0.0,
            elapsed_ms=0.1,
            success=True,
            request_sha256=key,
            billing_basis="本地终端输出整理与引文回放校验；无 API 账单",
        )
        ledger.add(cost)
        self._replay["finalize"][state.identity] = {
            "costs": [asdict(cost)],
            "outcome": {
                "passages": [],
                "relation": None,
                "resolved_gaps": [],
                "failed": False,
                "error": "",
                "gap_resolutions": [],
            },
        }

    def export_replay_manifest(self, output_path: Path | str | None = None) -> dict:
        manifest = {
            "protocol": "OPEN_WEB_ARCHIVED_REPLAY",
            "retrieval_backend": "AVERIMATEC_OFFICIAL_KS_V1",
            "tool_version": self.tool_version,
            "query_template_version": self.query_template_version,
            "initial_cost_upper_bound": self.initial_cost_upper_bound,
            "initial": self._replay["initial"],
            "candidates": self._replay["candidates"],
            "actions": self._replay["actions"],
            "finalize": self._replay["finalize"],
        }
        if output_path is not None:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return manifest

