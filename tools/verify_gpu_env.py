"""验证独立研究环境；仅测试小张量/4bit小层，不加载研究模型。"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/iterations/repair-20261117-01/gpu_environment.json"


def main():
    report = {
        "date": "2026-11-17",
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "environment": str(Path(sys.prefix).resolve()),
        "isolated_copy": True,
        "formal_vlm_inference_executed": False,
        "model_loaded": False,
        "source_environment_modified": False,
        "tests": {},
    }
    try:
        assert Path(sys.prefix).resolve() == ROOT / ".venv-research"
        packages = ("torch", "torchvision", "transformers", "bitsandbytes", "accelerate", "qwen-vl-utils",
                    "numpy", "scipy", "scikit-learn", "Pillow", "pytest", "av")
        report["packages"] = {name: importlib.metadata.version(name) for name in packages}
        assert report["packages"]["torch"] == "2.14.1+cu130"
        assert report["packages"]["torchvision"] == "0.29.1+cu130"
        assert report["packages"]["transformers"] == "4.57.6"
        assert report["packages"]["bitsandbytes"] == "0.50.2"
        import torch
        import torchvision
        import bitsandbytes as bnb
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        import qwen_vl_utils

        report["cuda_available"] = torch.cuda.is_available()
        assert report["cuda_available"]
        report["torch_cuda_runtime"] = torch.version.cuda
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["compute_capability"] = list(torch.cuda.get_device_capability(0))
        report["bf16_supported"] = torch.cuda.is_bf16_supported()
        report["transformers_qwen_class_import"] = Qwen2_5_VLForConditionalGeneration.__name__
        report["tests"]["cuda_tensor"] = float((torch.tensor([2.], device="cuda") * 2).item())
        assert report["tests"]["cuda_tensor"] == 4.

        boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., .9, .9]], device="cuda")
        scores = torch.tensor([.9, .8], device="cuda")
        keep = torchvision.ops.nms(boxes, scores, .3).cpu().tolist()
        assert keep == [0]
        report["tests"]["torchvision_cuda_nms"] = {"passed": True, "kept_indices": keep}

        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        layer = bnb.nn.Linear4bit(64, 32, bias=False, compute_dtype=torch.bfloat16,
                                 quant_type="nf4", compress_statistics=True).to("cuda")
        values = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = layer(values)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        assert output.shape == (2, 32) and bool(torch.isfinite(output).all())
        assert layer.weight.data.dtype == torch.uint8
        quant_state = layer.weight.quant_state
        assert quant_state is not None and quant_state.quant_type == "nf4"
        report["tests"]["bitsandbytes_cuda_nf4"] = {
            "passed": True,
            "input_shape": list(values.shape),
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype),
            "weight_storage_dtype": str(layer.weight.data.dtype),
            "quantization": quant_state.quant_type,
            "double_quantization": quant_state.nested,
            "forward_elapsed_ms": elapsed,
            "all_finite": True,
        }
        report["peak_allocated_bytes_for_smoke"] = torch.cuda.max_memory_allocated()
        report["memory_free_bytes"], report["memory_total_bytes"] = torch.cuda.mem_get_info()
        try:
            driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True, check=True)
            report["nvidia_driver"] = driver.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            report["nvidia_driver"] = None
        copy_path = OUTPUT.with_name("gpu_copy_manifest.json")
        manifest = json.loads(copy_path.read_text(encoding="utf-8"))
        report["source_environment"] = str(Path(manifest["source_site_packages"]).parents[1])
        report["copy_provenance"] = {key: manifest[key] for key in
                                     ("abi", "copy_method", "packages", "file_count", "total_bytes", "aggregate_sha256")}
        lock = ROOT / "requirements-gpu.lock.txt"
        report["requirements_lock"] = {"path": str(lock), "sha256": hashlib.sha256(lock.read_bytes()).hexdigest()}
        report["status"] = "GPU_TENSOR_TORCHVISION_AND_NF4_SMOKE_PASSED"
        return_code = 0
    except Exception as exc:
        report["status"] = "FAILED"
        report["error"] = type(exc).__name__ + ": " + str(exc)
        report["traceback"] = traceback.format_exc()
        return_code = 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
