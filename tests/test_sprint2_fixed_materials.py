"""Regression checks for alias-bound blind packages and scoped, signed return conversion."""
import copy
import json

import pytest

from src.experiment_runner import _load_rows
from src.research_contracts import ProtocolError, fingerprint
from tests.test_sprint1_infrastructure import _fixture_bundle
from tools.convert_annotation_return import convert_returns, write_conversion


def new_format(tmp_path):
    a,b,ma,mb,private,split,adj = _fixture_bundle(tmp_path)
    meta=private["tasks"]
    meta["T_SUFF"]={**copy.deepcopy(meta["T_QUAL"]),"export_kind":"sufficiency","features":[.5,0.,.6,.2,.2,.55,1.]}
    for m in (ma,mb):
        m["expected_task_ids"]=[t for t in m["expected_task_ids"] if t!="T_CAL"]+["T_SUFF"]
        m["task_kind_by_id"]={t:("source" if v["export_kind"]=="source_eval" else v["export_kind"]) for t,v in meta.items()}
        m["task_materials_ready_by_id"]={t:True for t in meta}
        m["materials_sha256"]="a"*64
    for m in meta.values():
        m["evidence_ref_aliases"]={f"E{i}":p["evidence_id"] for i,p in enumerate(m["evidence_proofs"],1)}
        m["materials_ready"]=True
    for ret in (a,b):
        ret["annotation_reviews"]=[row for row in ret["annotation_reviews"] if row["task_id"]!="T_CAL"]
        ret["annotation_reviews"].append(dict(next(row for row in ret["annotation_reviews"] if row["task_id"]=="T_QUAL"),task_id="T_SUFF"))
        for row in ret["annotation_reviews"]:row["evidence_refs"]=list(meta[row["task_id"]]["evidence_ref_aliases"])
    private.update(schema_version=2,protocol="SPRINT2_FIXED_BLIND_V1",split_content_sha256=fingerprint(split),packages={
        ma["package_id"]:{"materials_sha256":ma["materials_sha256"],"expected_task_ids":ma["expected_task_ids"],
                          "object_sha256_by_id":{t:fingerprint(m) for t,m in meta.items()}}})
    return a,b,ma,mb,private,split,adj


def test_fixed_return_resolves_aliases_and_keeps_unknown(tmp_path):
    case=new_format(tmp_path)
    result=convert_returns(*case,strict_complete=True)
    write_conversion(result,tmp_path/"result")
    for kind,dim in (("source",6),("quality",3),("sufficiency",7)):
        assert _load_rows(tmp_path/"result"/(kind+".jsonl"),dim)
    ev=json.loads((tmp_path/"result/source_eval.jsonl").read_bytes())
    assert {ev["evidence_a"],ev["evidence_b"]}=={p["evidence_id"] for p in ev["proofs"]} and ev["evidence_a"]!="E1"
    assert next(r for r in result["outputs"]["source"] if r["task_id"]=="T_SRC_UNKNOWN")["resolved_label"] is None


@pytest.mark.parametrize("damage",["features","reference","materials","split"])
def test_fixed_return_detects_wrong_object_or_evidence_scope(tmp_path,damage):
    a,b,ma,mb,private,split,adj=new_format(tmp_path)
    if damage=="features": private["tasks"]["T_SUFF"]["features"][1]=1.
    if damage=="reference": next(r for r in a["annotation_reviews"] if r["task_id"]=="T_EVAL")["evidence_refs"]=["E1","nonexistent"]
    if damage=="materials": mb["materials_sha256"]="b"*64
    if damage=="split": split["note"]="unexpected changed split"
    with pytest.raises(ProtocolError): convert_returns(a,b,ma,mb,private,split,adj,strict_complete=True)

