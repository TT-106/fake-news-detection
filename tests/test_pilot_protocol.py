"""先导协议必须阻止最终集提前暴露、真值入模和按结果筛选。"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from src.pilot_vlm import evaluate_saved_predictions, selected_public_claims, sha
from src.research_contracts import ProtocolError


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def pilot(tmp_path):
    root = tmp_path
    image = root / "data/averimatec/images/claim.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (28, 28), "white").save(image)
    samples = [dict(claim_text="A historical image", claim_images=["claim.jpg"],
                    label=label, questions=[{"answer": "DO_NOT_SEND_GOLD"}],
                    justification="DO_NOT_SEND_REASON", article="DO_NOT_SEND_ARTICLE")
               for label in ["Supported", "Refuted"]]
    train = root / "data/averimatec/train.json"
    save(train, samples)
    save(root / "data/averimatec/images/image_manifest.json",
         {"images": [{"path": "data/averimatec/images/claim.jpg", "sha256": sha(image)}]})
    source = root / "source_split.json"
    save(source, {"split": {"pilot": ["train:0", "train:1"]}})
    split = {"evaluation_scope": "TRAIN_PILOT_ONLY_NOT_FINAL_EVAL", "dataset_sha256": sha(train),
             "source_split": "source_split.json", "source_split_sha256": sha(source),
             "claim_ids": ["train:0", "train:1"],
             "event_groups": {"train:0": "e0", "train:1": "e1"}}
    save(root / "pilot_split.json", split)
    return root, {"split_path": "pilot_split.json"}, split


def test_inference_objects_do_not_contain_supervision_or_qa(pilot):
    root, config, _ = pilot
    claims = selected_public_claims(config, root)
    payload = json.dumps([asdict(c) for c in claims])
    assert len(claims) == 2
    assert all(marker not in payload for marker in ["DO_NOT_SEND", '"label"', '"questions"', '"article"'])
    assert len(claims[0].images) == 1


def test_final_evaluation_ids_cannot_enter_pilot(pilot):
    root, config, split = pilot
    split["claim_ids"] = ["val:0", "val:1"]
    save(root / config["split_path"], split)
    with pytest.raises(ProtocolError, match="禁止使用官方"):
        selected_public_claims(config, root)


def test_post_hoc_subset_and_changed_data_are_rejected(pilot):
    root, config, split = pilot
    split["claim_ids"] = ["train:0"]
    save(root / config["split_path"], split)
    with pytest.raises(ProtocolError, match="完整事前先导列表"):
        selected_public_claims(config, root)
    split["claim_ids"] = ["train:0", "train:1"]
    split["dataset_sha256"] = "0" * 64
    save(root / config["split_path"], split)
    with pytest.raises(ProtocolError, match="数据版本改变"):
        selected_public_claims(config, root)


def test_evaluation_requires_complete_saved_predictions_and_fixed_four_classes(pilot):
    root, config, split = pilot
    p = root / "predictions.jsonl"
    p.write_text('\n'.join(json.dumps({"id": cid, "prediction": label})
                           for cid,label in zip(split["claim_ids"],["Supported","Refuted"])), encoding="utf-8")
    result = evaluate_saved_predictions(p, root / config["split_path"], root)
    assert result["accuracy"] == 1
    assert result["macro_f1"] == .5  # 四类固定，另两类缺失时为0，不改口径。
    p.write_text(json.dumps({"id": "train:0", "prediction": "Supported"}), encoding="utf-8")
    with pytest.raises(ProtocolError, match="完整先导样本"):
        evaluate_saved_predictions(p, root / config["split_path"], root)


def test_evaluation_refuses_changed_truths_after_inference(pilot):
    root, config, split = pilot
    p = root / "predictions.jsonl"
    p.write_text('\n'.join(json.dumps({"id": cid, "prediction": "Supported"})
                           for cid in split["claim_ids"]), encoding="utf-8")
    train = root / "data/averimatec/train.json"
    rows = json.loads(train.read_text(encoding="utf-8"))
    rows[0]["label"] = "Refuted"
    save(train, rows)
    with pytest.raises(ProtocolError, match="评分时数据或先导划分版本改变"):
        evaluate_saved_predictions(p, root / config["split_path"], root)
