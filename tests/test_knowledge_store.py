"""Official AVerImaTeC Knowledge Store 清洗、日期过滤、MMR 与可回放快照单元测试。"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from src.knowledge_store import (
    CleanKSDoc,
    clean_ks_documents,
    extract_observable_date,
    mmr_select,
    retrieve_policy_passages,
)


class _DummyEmbedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        # Deterministic bag-of-keyword normalized embeddings for offline unit testing
        vocab = ["nasa", "typhoon", "maysak", "2015", "hurricane", "milton", "reuters", "stadium", "space"]
        mat = np.zeros((len(texts), len(vocab)), dtype=np.float32)
        for i, t in enumerate(texts):
            low = t.lower()
            for j, w in enumerate(vocab):
                mat[i, j] = float(low.count(w)) + 0.01
            mat[i] /= np.linalg.norm(mat[i])
        return mat


def test_clean_ks_filters_boilerplate_factcheck_and_post_claim_dates():
    raw_rows = [
        # Valid RIS article before claim date
        ("reverse_image_search", {
            "url": "https://science.nasa.gov/earth/2015/04/01/typhoon-maysak",
            "url2text": ["NASA Earth Observatory photographed Typhoon Maysak from the International Space Station in April 2015. Credit: NASA.\nThe article was published on 2015-04-01"],
        }),
        # Factcheck domain -> dropped
        ("text_search", {
            "url": "https://www.snopes.com/fact-check/typhoon-maysak-milton/",
            "url2text": ["Snopes checked whether this photo shows Hurricane Milton instead of Typhoon Maysak and found it false."],
        }),
        # Boilerplate JS wall -> dropped
        ("text_search", {
            "url": "https://x.com/user123",
            "url2text": ["JavaScript is not available. We've detected that JavaScript is disabled in this browser. Please enable JavaScript."],
        }),
        # Post-claim date -> dropped
        ("text_search", {
            "url": "https://news.example.org/2024/10/15/hurricane-milton-recap",
            "url2text": ["Published 2024-10-15: Hurricane Milton made landfall earlier this month and prompted widespread photo sharing."],
        }),
        # Repeated cross-URL boilerplate stub (3 distinct URLs, same opening text) -> dropped
        ("text_search", {"url": "https://site-a.example/p1", "url2text": ["Generic error template text repeated identically across unrelated domains without actual article content here."]}),
        ("text_search", {"url": "https://site-b.example/p2", "url2text": ["Generic error template text repeated identically across unrelated domains without actual article content here."]}),
        ("text_search", {"url": "https://site-c.example/p3", "url2text": ["Generic error template text repeated identically across unrelated domains without actual article content here."]}),
    ]

    docs, stats = clean_ks_documents(raw_rows, claim_date="2024-10-10", exclude_factcheck_domains=True)
    assert len(docs) == 1
    assert docs[0].published_date == "2015-04-01"
    assert docs[0].date_source == "ris_marker"
    assert "credit:" in docs[0].wire_clues
    assert stats["dropped_factcheck_domain"] == 1
    assert stats["dropped_boilerplate_pattern"] == 1
    assert stats["dropped_post_or_same_day_date"] == 1
    assert stats["dropped_repeated_cross_url_stub"] == 3


def test_extract_observable_date_priority():
    assert extract_observable_date("https://a.com/2020/05/06/story", "The article was published on 2019-01-02") == ("2019-01-02", "ris_marker")
    assert extract_observable_date("https://a.com/2020/05/06/story", "No date inside body") == ("2020-05-06", "url_path")
    assert extract_observable_date("https://a.com/story", "Updated March 14, 2021 by staff") == ("2021-03-14", "lead_text")
    assert extract_observable_date("https://a.com/story", "Completely undated content") == ("", "unverified")


def test_mmr_diversifies_redundant_candidates():
    rel = np.array([0.95, 0.94, 0.85], dtype=np.float32)
    # Candidate 0 and 1 are identical; candidate 2 is orthogonal
    embs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    sel = mmr_select(rel, embs, k=2, lambda_mult=0.65)
    assert sel == [0, 2]


def test_retrieve_policy_materializes_verified_archived_passages(tmp_path: Path):
    docs = [
        CleanKSDoc("d1", "reverse_image_search", "https://nasa.gov/2015/04/01/maysak", "nasa.gov",
                   "NASA astronaut captured Typhoon Maysak from the International Space Station in April 2015. Credit: NASA.",
                   "2015-04-01", "url_path", ("credit:",)),
        CleanKSDoc("d2", "text_search", "https://reuters.com/2024/10/08/milton", "reuters.com",
                   "Reuters reports Hurricane Milton intensified in the Gulf of Mexico ahead of Florida landfall.",
                   "2024-10-08", "url_path", ("reuters",)),
        CleanKSDoc("d3", "text_search", "https://mirror.example/2024/10/08/milton-copy", "mirror.example",
                   "Reuters reports Hurricane Milton intensified in the Gulf of Mexico ahead of Florida landfall.",
                   "2024-10-08", "url_path", ("reuters",)),
    ]
    chunks, pair_audit = retrieve_policy_passages(
        "train:99",
        "Photo shows Hurricane Milton from space station",
        docs,
        _DummyEmbedder(),
        policy="ks_dual_text_ris",
        k_total=3,
        k_ris=1,
        snapshot_dir=tmp_path / "snaps",
    )
    assert len(chunks) == 3
    assert chunks[0].track == "reverse_image_search"
    assert all(Path(c.snapshot_path).is_file() and len(c.snapshot_sha256) == 64 for c in chunks)
    assert pair_audit["num_ris_passages"] == 1
    assert pair_audit["cross_domain_high_sim_pairs_ge_065"] >= 1
    assert pair_audit["has_source_verify_opportunity"] is True


def test_official_ks_evidence_tools_drives_faithful_pipeline_and_replays(tmp_path: Path):
    import hashlib
    from PIL import Image
    import pytest
    from src.archived_tools import ArchivedEvidenceTools
    from src.faithful_pipeline import FaithfulPipeline
    from src.knowledge_store import OfficialKSEvidenceTools
    from src.research_contracts import CostLedger, ImageAsset, PublicClaim
    from tests.test_faithful_pipeline import TestBinary, TestTemperature, TestVLM

    img_path = tmp_path / "claim.png"
    Image.new("RGB", (8, 8)).save(img_path)
    img_sha = hashlib.sha256(img_path.read_bytes()).hexdigest()
    claim = PublicClaim("train:111", "event:111", "A Paris stadium typhoon photo from space", (ImageAsset(str(img_path), img_sha),), date="2024-10-10", location="Paris")

    in_mem = {
        "train:111": [
            ("reverse_image_search", {
                "url": "https://nasa.gov/2015/04/01/maysak-space",
                "url2text": ["NASA astronaut photographed Typhoon Maysak from the International Space Station in April 2015. Credit: NASA.\nThe article was published on 2015-04-01"],
            }),
            ("text_search", {
                "url": "https://reuters.com/2020/06/10/paris-event",
                "url2text": ["Reuters reports the Paris event date was officially confirmed in June 2020 by stadium organizers and municipal archives."],
            }),
            ("text_search", {
                "url": "https://mirror.example.org/2020/06/10/paris-reprint",
                "url2text": ["Reuters reports the Paris event date was officially confirmed in June 2020 by stadium organizers and municipal archives."],
            }),
            ("text_search", {
                "url": "https://extra.example.org/2021/03/04/paris-followup",
                "url2text": ["Additional Paris stadium date records and hurricane archival documentation compiled by meteorological researchers in 2021."],
            }),
        ]
    }

    ks_tools = OfficialKSEvidenceTools(
        store_path=None,
        snapshot_dir=tmp_path / "ks_snaps",
        embedder=_DummyEmbedder(),
        in_memory_store=in_mem,
        initial_k=2,
        initial_ris_k=1,
        step_k=1,
    )
    engine = FaithfulPipeline(
        TestVLM(),
        ks_tools,
        TestBinary("source", 0.45),
        TestBinary("quality", 0.8),
        TestBinary("sufficiency", 0.9),
        TestTemperature(),
        0.6,
        0.005,
    )
    ledger = CostLedger()
    state0 = engine.initialise(claim, ledger, 0.25)
    assert len(state0.evidence) == 2
    # A dated RIS hit with a credit in the text does not establish that the input image is the credited photo.
    ris = state0.evidence[0].passage
    assert ris.published_at and ris.image_provenance_verified is False
    assert ris.author == ris.provenance_proof == ""

    cands0 = ks_tools.candidates(state0)
    kinds0 = {a.kind for a in cands0}
    assert {"fact_search", "source_verify"} <= kinds0

    # 1. Execute fact_search -> adds 1 new text passage and resolves the "date" gap
    fact_act = next(a for a in cands0 if a.kind == "fact_search")
    state1 = engine.advance(state0, fact_act, ledger, 0.25)
    assert len(state1.evidence) == 3
    assert "date" not in state1.gaps

    # 2. Execute source_verify -> updates target_pair in source_matrix without adding passages
    cands1 = ks_tools.candidates(state1)
    src_act = next(a for a in cands1 if a.kind == "source_verify")
    state2 = engine.advance(state1, src_act, ledger, 0.25)
    assert len(state2.evidence) == 3

    ks_tools.finalize(state2, ledger)
    pred_live = engine.predict(state2, ledger)

    # Export to ArchivedEvidenceTools manifest and verify 100% deterministic replay!
    manifest_file = tmp_path / "ks_replay_manifest.json"
    ks_tools.export_replay_manifest(manifest_file)
    replay_tools = ArchivedEvidenceTools(str(manifest_file))
    replay_engine = FaithfulPipeline(
        TestVLM(),
        replay_tools,
        TestBinary("source", 0.45),
        TestBinary("quality", 0.8),
        TestBinary("sufficiency", 0.9),
        TestTemperature(),
        0.6,
        0.005,
    )
    r_ledger = CostLedger()
    r_state0 = replay_engine.initialise(claim, r_ledger, 0.25)
    assert r_state0.identity == state0.identity
    r_cands0 = replay_tools.candidates(r_state0)
    r_fact = next(a for a in r_cands0 if a.kind == "fact_search")
    r_state1 = replay_engine.advance(r_state0, r_fact, r_ledger, 0.25)
    assert r_state1.identity == state1.identity
    r_cands1 = replay_tools.candidates(r_state1)
    r_src = next(a for a in r_cands1 if a.kind == "source_verify")
    r_state2 = replay_engine.advance(r_state1, r_src, r_ledger, 0.25)
    assert r_state2.identity == state2.identity
    replay_tools.finalize(r_state2, r_ledger)
    pred_replay = replay_engine.predict(r_state2, r_ledger)
    assert pred_replay == pred_live


def test_download_knowledge_store_blocks_val_without_explicit_unlock(tmp_path: Path):
    import pytest
    from tools.download_knowledge_store import download_archive

    with pytest.raises(RuntimeError, match="安全拦截：val.zip"):
        download_archive("val", tmp_path / "val")

