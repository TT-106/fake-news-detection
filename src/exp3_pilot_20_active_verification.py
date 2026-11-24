"""E3 真实共同历史行动监督；旧的 label->立场、chi->收益合成器已撤出执行路径。"""

from src.faithful_pipeline import FaithfulPipeline
from src.learned_components import ExperimentSplit
from src.research_contracts import ProtocolError


def extract_sample_evidence_and_source_pairs(sample, sample_idx, rng=None):
    """旧函数仅保留显式 QA 诊断兼容，不返回真值标签或合成来源关系。"""
    from src.real_averimatec_pipeline import build_claim_evidence_environment
    return build_claim_evidence_environment(sample, sample_idx)


def collect_common_history_rollouts(pipeline: FaithfulPipeline, histories, truths: dict,
                                    split: ExperimentSplit, event_groups: dict,
                                    role: str = "action_train", budget: float = 0.25, mode: str = "full"):
    split.validate(event_groups)
    if role not in {"action_train", "mechanism_validation"}:
        raise ProtocolError("共同历史展开只能用于行动训练或机制验证，不能使用最终评测标签")
    allowed = getattr(split, role)
    rows = []
    for state, ledger in histories:
        if state.claim.claim_id not in allowed or state.claim.claim_id not in truths:
            raise ProtocolError("共同历史声明不属于指定数据用途或缺少离线真值")
        rows.extend(pipeline.common_history_branches(state, ledger, truths[state.claim.claim_id], budget, mode))
    return rows


def run_exp3(*args, **kwargs):
    if not args or not isinstance(args[0], FaithfulPipeline):
        raise ProtocolError("E3 需要冻结的真实 VLM/来源/质量/充分性模型及共同历史。禁止回退到旧的人工收益合成实验。")
    return collect_common_history_rollouts(*args, **kwargs)


if __name__ == "__main__":
    from src.real_averimatec_pipeline import main
    raise SystemExit(main())
