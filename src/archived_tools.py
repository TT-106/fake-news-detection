"""真实工具轨迹回放。找不到精确历史/查询的记录时停止，不生成候选结果。"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import uuid

from src.faithful_pipeline import GapResolution, RelationObservation, ToolOutcome
from src.research_contracts import (ArchivedPassage, CandidateAction, CostEntry, CostLedger,
                                    ProtocolError, fingerprint, validate_verified_relation)


class ArchivedEvidenceTools:
    def __init__(self, manifest_path: str):
        p = Path(manifest_path)
        self.data = json.loads(p.read_text(encoding="utf-8"))
        if self.data.get("protocol") != "OPEN_WEB_ARCHIVED_REPLAY":
            raise ProtocolError("必须是从图文声明出发的真实开放网页工具轨迹；金标准 QA/URL 不符合正式协议")
        if not self.data.get("tool_version") or not self.data.get("query_template_version"):
            raise ProtocolError("必须冻结工具与共享查询模板")
        self.version = fingerprint(self.data)
        self.initial_cost_upper_bound = float(self.data["initial_cost_upper_bound"])

    @staticmethod
    def _outcome(record: dict, ledger: CostLedger) -> ToolOutcome:
        if not record.get("costs"):
            raise ProtocolError("工具记录缺少实测费用，不能以空列表暗示零费用")
        entries = []
        execution_id = f"replay-{uuid.uuid4().hex}"
        billed_ids = {entry.call_id for entry in ledger.entries}
        for original in record["costs"]:
            entry = dict(original)
            entry["replayed"] = True
            entry["replay_execution_id"] = execution_id
            cost = CostEntry(**entry)
            if cost.call_id in billed_ids:
                raise ProtocolError("同一原始请求账单不得重复回放入账")
            billed_ids.add(cost.call_id)
            entries.append(cost)
        # 请求已实际发生；即使随后发现材料/关系无效，也必须保留实际费用。
        for cost in entries:
            ledger.add(cost)
        data = record["outcome"]
        passages = tuple(ArchivedPassage(**p) for p in data.get("passages", []))
        for p in passages:
            p.verify()
        rel = data.get("relation")
        if rel is not None:
            validate_verified_relation(rel["same_origin"])
        relation = None if rel is None else RelationObservation(
            tuple(rel["target_pair"]), rel["same_origin"], tuple(ArchivedPassage(**p) for p in rel["proof"]))
        resolutions = tuple(GapResolution(**proof) for proof in data.get("gap_resolutions", []))
        result = ToolOutcome(passages, relation, tuple(data.get("resolved_gaps", [])),
                             bool(data.get("failed", False)), str(data.get("error", "")), resolutions)
        return result

    def initial(self, claim, analysis, ledger):
        key = fingerprint({"claim": asdict(claim), "analysis": asdict(analysis)})
        record = self.data.get("initial", {}).get(key)
        if record is None:
            raise ProtocolError("缺少该图文声明/解析结果对应的真实初始检索记录")
        return self._outcome(record, ledger)

    def candidates(self, state):
        records = self.data.get("candidates", {}).get(state.identity)
        if records is None:
            raise ProtocolError("缺少当前历史的共享候选生成记录，不能读取未来答案代替")
        candidates = []
        for row in records:
            d = dict(row)
            if d.get("target_pair") is not None:
                d["target_pair"] = tuple(d["target_pair"])
            candidates.append(CandidateAction(**d))
        return tuple(candidates)

    def execute(self, state, action, ledger):
        key = fingerprint({"history_id": state.identity, "action": asdict(action)})
        record = self.data.get("actions", {}).get(key)
        if record is None:
            raise ProtocolError("缺少该共同历史候选行动的实测返回；不会模拟成功或注入收益")
        return self._outcome(record, ledger)

    def finalize(self, state, ledger):
        records = self.data.get("finalize", {})
        if state.identity not in records:
            raise ProtocolError("缺少最终输出的实测费用记录")
        outcome = self._outcome(records[state.identity], ledger)
        if outcome.passages or outcome.relation or outcome.resolved_gaps or outcome.gap_resolutions:
            raise ProtocolError("输出阶段不能追加未计入行动的证据")
