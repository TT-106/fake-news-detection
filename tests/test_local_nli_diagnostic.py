"""文本诊断模型的可选真实前向检查，不等价于正式 VLM 实验验收。"""

import os
import pytest


@pytest.mark.local_model
@pytest.mark.skipif(os.environ.get("RUN_LOCAL_NLI_SMOKE") != "1", reason="需显式设置 RUN_LOCAL_NLI_SMOKE=1；不会下载模型")
def test_local_nli_forward_and_cache():
    from src.model_inference import NLIStanceModel
    nli = NLIStanceModel()
    premise, hypothesis = "The Eiffel Tower is in Paris, France.", "The Eiffel Tower is in Paris."
    before = nli.forward_calls
    first = nli.stance(premise, hypothesis)
    assert first[0] > first[1]
    assert nli.forward_calls == before+1
    assert nli.stance(premise, hypothesis) == first
    assert nli.forward_calls == before+1
