param(
    [string]$SourceEnv = 'S:\coding_relative\ai-images\.runtime\venv',
    [string]$TargetEnv = '.venv-research'
)
$ErrorActionPreference = 'Stop'
$researchRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$sourceEnvPath = (Resolve-Path -LiteralPath $SourceEnv).Path
$targetEnvPath = [System.IO.Path]::GetFullPath((Join-Path $researchRoot $TargetEnv))
if (-not $targetEnvPath.StartsWith($researchRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw '目标虚拟环境必须位于本研究项目目录内。'
}
if ($targetEnvPath -eq $sourceEnvPath) { throw '目标环境不能与来源环境相同。' }
$sourcePythonPath = Join-Path $sourceEnvPath 'Scripts\python.exe'
& $sourcePythonPath -B -c "import sys,sysconfig; assert sys.version_info[:2]==(3,13); assert sysconfig.get_config_var('EXT_SUFFIX')=='.cp313-win_amd64.pyd'; print(sys.version)"
if ($LASTEXITCODE -ne 0) { throw '来源环境不是 Windows x64 CPython 3.13，拒绝复制原生扩展。' }
if (-not (Test-Path -LiteralPath $targetEnvPath)) {
    & uv venv --python 3.13 $targetEnvPath
    if ($LASTEXITCODE -ne 0) { throw '创建独立研究环境失败。' }
}
$targetPythonPath = Join-Path $targetEnvPath 'Scripts\python.exe'
& $targetPythonPath -B -c "import sys,sysconfig; assert sys.version_info[:2]==(3,13); assert sysconfig.get_config_var('EXT_SUFFIX')=='.cp313-win_amd64.pyd'"
if ($LASTEXITCODE -ne 0) { throw '目标环境不是对应的 CPython 3.13，拒绝复制。' }
$copyCode = @'
import hashlib,json,os,shutil,sys,time
from pathlib import Path
source=Path(sys.argv[1]).resolve()/"Lib/site-packages"
target=Path(sys.argv[2]).resolve()/"Lib/site-packages"
output=Path(sys.argv[3]).resolve()/"outputs/iterations/repair-20261117-01"
output.mkdir(parents=True,exist_ok=True)
names=["torch","functorch","torchgen","torch-2.14.1+cu130.dist-info","torchvision","torchvision-0.29.1+cu130.dist-info"]
for optional in ["torch.libs","torchvision.libs"]:
    if (source/optional).exists(): names.append(optional)
records=[]
started=time.perf_counter()
for name in names:
    origin=source/name
    destination=target/name
    if not origin.is_dir():raise RuntimeError("来源模块目录缺失: "+str(origin))
    if not destination.exists():
        shutil.copytree(origin,destination,copy_function=shutil.copy2,symlinks=False)
    for source_file in sorted(p for p in origin.rglob("*") if p.is_file()):
        relative=source_file.relative_to(source)
        target_file=target/relative
        if not target_file.is_file():raise RuntimeError("目标复制不完整: "+str(relative))
        if os.path.samefile(source_file,target_file):raise RuntimeError("禁止共享硬链接/同一文件: "+str(relative))
        def digest(p):
            with p.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()
        source_sha,target_sha=digest(source_file),digest(target_file)
        if source_sha!=target_sha:raise RuntimeError("复制SHA256不一致: "+str(relative))
        records.append({"path":relative.as_posix(),"bytes":source_file.stat().st_size,"sha256":source_sha})
    print("Copied and independently verified "+name,flush=True)
serialized=json.dumps(records,ensure_ascii=False,sort_keys=True,separators=(",",":"))
manifest={"source_site_packages":str(source),"target_site_packages":str(target),"abi":"cp313-win_amd64",
          "copy_method":"shutil.copytree(copy_function=copy2,symlinks=False); no pth, symlink or hardlink sharing",
          "source_environment_modified":False,"packages":names,"files":records,"file_count":len(records),
          "total_bytes":sum(r["bytes"] for r in records),"aggregate_sha256":hashlib.sha256(serialized.encode()).hexdigest(),
          "copy_and_hash_seconds":time.perf_counter()-started}
(output/"gpu_copy_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in manifest.items() if k!="files"},ensure_ascii=False),flush=True)
'@
$copyCode | & $targetPythonPath -B -X utf8 - $sourceEnvPath $targetEnvPath $researchRoot
if ($LASTEXITCODE -ne 0) { throw '复制或哈希验证失败；未继续安装。' }
$constraintsPath = Join-Path $researchRoot 'tools\constraints-gpu.txt'
@'
torch==2.14.1+cu130
torchvision==0.29.1+cu130
'@ | Set-Content -LiteralPath $constraintsPath -Encoding utf8
$lockPath = Join-Path $researchRoot 'requirements-gpu.lock.txt'
if (Test-Path -LiteralPath $lockPath) {
    & uv pip install --python $targetPythonPath --constraint $constraintsPath --link-mode copy --requirement $lockPath
} else {
    & uv pip install --python $targetPythonPath --constraint $constraintsPath --link-mode copy 'transformers==4.57.6' 'bitsandbytes==0.50.2' 'accelerate>=1.11,<2' 'qwen-vl-utils' 'numpy==2.4.4' 'scipy==1.17.1' 'scikit-learn==1.8.0' 'Pillow==12.2.0' 'pytest==9.0.3' 'requests' 'psutil' 'av' 'setuptools>=77.0.3'
}
if ($LASTEXITCODE -ne 0) { throw '依赖安装失败；禁止降级/重新下载复制的 Torch 包。' }
& uv pip freeze --python $targetPythonPath | Set-Content -LiteralPath $lockPath -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw '环境锁定失败。' }
