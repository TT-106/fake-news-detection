"""E2 的实际模型输入视图。时间/地点文本遮罩须由人工核对，不能只关闭数值规则。"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

from src.learned_components import ReviewedLabel
from src.research_contracts import ProtocolError, fingerprint


@dataclass(frozen=True)
class TextMask:
    path: str  # 如 claim、atoms.0.text、passage.text、evidence.0.text
    start: int
    end: int
    expected: str
    evidence_id: str | None = None


@dataclass(frozen=True)
class ImageMask:
    sha256: str
    width: int
    height: int
    boxes: tuple[tuple[int, int, int, int], ...]


class InputView:
    def __init__(self, mode="full", masks: dict | None = None, reviews: dict | None = None,
                 image_masks: dict[str, ImageMask] | None = None):
        if mode not in {"full", "no_image", "no_provenance", "no_spatiotemporal", "no_rules"}:
            raise ProtocolError("未知 E2 消融视图")
        self.mode, self.masks = mode, masks or {}
        self.reviews = reviews or {}
        self.image_masks = image_masks or {}
        self.render_receipts = []
        if mode == "no_spatiotemporal":
            if not self.masks:
                raise ProtocolError("时空消融需要覆盖配文、原子声明和证据的逐字段文本遮罩")
            for cid in self.masks:
                review = self.reviews.get(cid)
                if review is None or review.resolve() != 1:
                    raise ProtocolError("时空遮罩须经双人独立复核/必要时第三人仲裁确认完整")
                for mask in self.masks[cid]:
                    if (not isinstance(mask.path, str) or mask.path.split('.')[0] not in
                            {"claim", "atoms", "passage", "evidence", "gaps", "date", "location"}):
                        raise ProtocolError("时空遮罩包含不支持的字段路径")
                    if mask.evidence_id is not None and mask.path.split('.')[0] not in {"passage", "evidence"}:
                        raise ProtocolError("全局字段遮罩不能绑定单条证据，否则其它片段调用会遗漏同一声明的时空信息")
        self.version = fingerprint({"projection_protocol": "MASK_TRACE_V2", "mode": mode,
                                    "masks": {cid: [asdict(m) for m in rows] for cid,rows in self.masks.items()},
                                    "reviews": {cid: asdict(r) for cid,r in self.reviews.items()},
                                    "image_masks": {sha: asdict(m) for sha,m in self.image_masks.items()}})

    @property
    def include_images(self):
        return self.mode != "no_image"

    def project_image(self, sha256, image):
        if self.mode != "no_spatiotemporal":
            return image.copy()
        mask = self.image_masks.get(sha256)
        if mask is None or mask.sha256 != sha256 or (mask.width,mask.height) != image.size:
            raise ProtocolError("时空消融必须审核每张原图并提供像素遮罩；无相关信号也须显式记录空框列表")
        from PIL import ImageDraw
        output = image.copy()
        drawer = ImageDraw.Draw(output)
        for x0,y0,x1,y1 in mask.boxes:
            if (not all(isinstance(v,int) for v in (x0,y0,x1,y1))
                    or not 0 <= x0 < x1 <= image.width or not 0 <= y0 < y1 <= image.height):
                output.close()
                raise ProtocolError("像素遮罩范围无效")
            drawer.rectangle((x0,y0,x1-1,y1-1), fill=(0,0,0))
        return output

    def render(self, claim_id: str, payload: dict) -> dict:
        # 在寻址前规范化序列；真实 terminal 的 gaps 是 tuple，JSON 视图必须与 list 一致。
        def mutable(value):
            if isinstance(value, dict):
                return {k: mutable(v) for k,v in value.items()}
            if isinstance(value, (list, tuple)):
                return [mutable(v) for v in value]
            return copy.deepcopy(value)
        data = mutable(payload)
        receipt = {"claim_id": claim_id, "mode": self.mode, "payload_roots": sorted(data), "masks": []}
        self.render_receipts.append(receipt)
        if self.mode == "no_spatiotemporal":
            if claim_id not in self.masks:
                raise ProtocolError("该声明缺少经人工复核的完整时空遮罩")
            by_path = {}
            for index, mask in enumerate(self.masks[claim_id]):
                path = mask.path
                trace = {"mask_index": index, "configured_path": path, "evidence_id": mask.evidence_id,
                         "status": "not_applicable", "reason": "payload_root_absent"}
                receipt["masks"].append(trace)
                if mask.evidence_id is not None:
                    single = data.get("passage")
                    if single is not None:
                        if single.get("evidence_id") != mask.evidence_id:
                            trace["reason"] = "target_evidence_absent"
                            continue
                        if path.startswith("evidence."):
                            parts = path.split('.', 2)
                            path = "passage."+parts[2] if len(parts) == 3 else path
                    elif "evidence" in data:
                        index = next((i for i,e in enumerate(data["evidence"])
                                      if e.get("evidence_id") == mask.evidence_id), None)
                        if index is None:
                            trace["reason"] = "target_evidence_absent"
                            continue
                        if path.startswith("passage."):
                            path = f"evidence.{index}."+path[len("passage."):]
                        elif path.startswith("evidence."):
                            parts = path.split('.', 2)
                            path = f"evidence.{index}."+parts[2] if len(parts) == 3 else path
                    else:
                        continue
                trace["effective_path"] = path
                if path.split('.')[0] not in data:
                    continue
                by_path.setdefault(path, []).append((mask, trace))
            for path, masks in by_path.items():
                keys = path.split(".")
                owner = data
                try:
                    for key in keys[:-1]:
                        owner = owner[int(key)] if isinstance(owner, list) else owner[key]
                    final = int(keys[-1]) if isinstance(owner, list) else keys[-1]
                    text = owner[final]
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    for _, trace in masks:
                        trace.update(status="unapplied", reason="specified_path_missing")
                    raise ProtocolError(f"遮罩指定路径存在但未应用：{path}") from exc
                if not isinstance(text, str):
                    for _, trace in masks:
                        trace.update(status="unapplied", reason="specified_field_not_text")
                    raise ProtocolError("遮罩只能应用于经过核对的文本字段")
                previous_end = len(text)+1
                for mask, trace in sorted(masks, key=lambda item: item[0].start, reverse=True):
                    if (not 0 <= mask.start < mask.end <= len(text) or mask.end > previous_end
                            or text[mask.start:mask.end] != mask.expected):
                        trace.update(status="unapplied", reason="invalid_span_or_changed_text")
                        raise ProtocolError("遮罩范围重叠或文本已变化，不能继续沿用旧遮罩")
                    text = text[:mask.start]+"[WITHHELD]"+text[mask.end:]
                    previous_end = mask.start
                    trace.update(status="applied", reason="exact_text_span_withheld")
                owner[final] = text
            # 根字段/目标证据不适用于本次调用时显式留痕；已指定的现存字段不能静默漏遮。
        omitted = set()
        if self.mode == "no_provenance":
            omitted |= {"url", "published_at", "author", "image_provenance_verified", "provenance_proof", "retrieved_at"}
        if self.mode == "no_spatiotemporal":
            omitted |= {"date", "location", "time", "published_at", "retrieved_at"}
        def project(value):
            if isinstance(value, dict):
                return {k: project(v) for k,v in value.items() if k not in omitted}
            if isinstance(value, (list, tuple)):
                return [project(v) for v in value]
            return value
        return project(data)
