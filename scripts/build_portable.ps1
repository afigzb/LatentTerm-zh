<#
.SYNOPSIS
    一键打包 LatentTerm 为可分发的便携版（Embeddable Python 方案）。

.DESCRIPTION
    产物目录 dist\LatentTerm\ 结构：
        python\            下载自 python.org 的嵌入版解释器
        libs\              依赖包（pip --target 安装到此）
        core\ app.py ...   项目源码
        .streamlit\        Streamlit 配置
        启动.bat           双击即用
        使用说明.txt
        LICENSE / README.md

    用户把整个目录压缩为 zip 上传到 GitHub Release / 网盘即可。

.PARAMETER PythonVersion
    嵌入版 Python 版本，默认 3.11.9。切换版本需同时修改 launcher 的 _pth 逻辑。

.PARAMETER SkipDownload
    若传入，则跳过下载步骤（假设 dist\.cache 下已有缓存）。

.PARAMETER Zip
    若传入，打包完成后自动压缩为 zip。

.EXAMPLE
    pwsh -File scripts\build_portable.ps1
    pwsh -File scripts\build_portable.ps1 -Zip
#>

[CmdletBinding()]
param(
    [string]$PythonVersion = '3.11.9',
    [switch]$SkipDownload,
    [switch]$Zip
)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'  # 显著加速 Invoke-WebRequest

# ── 路径 ──────────────────────────────────────────────
$RepoRoot  = Split-Path -Parent $PSScriptRoot
$DistRoot  = Join-Path $RepoRoot 'dist'
$CacheDir  = Join-Path $DistRoot '.cache'
$AppName   = 'LatentTerm'
$BuildDir  = Join-Path $DistRoot $AppName
$PyDir     = Join-Path $BuildDir 'python'
$LibsDir   = Join-Path $BuildDir 'libs'
# Streamlit 用 __file__ 路径判定是否 "开发模式"：路径必须以 site-packages 结尾，
# 否则会开启 global.developmentMode 并拒绝自定义端口等配置。
$SitePkg   = Join-Path $LibsDir 'site-packages'

$PyMajorMinor = ($PythonVersion -split '\.')[0..1] -join ''  # 3.11.9 → 311
$PthFileName  = "python$PyMajorMinor._pth"

$PyZipUrl    = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
$PyZipFile   = Join-Path $CacheDir "python-$PythonVersion-embed-amd64.zip"
$GetPipUrl   = 'https://bootstrap.pypa.io/get-pip.py'
$GetPipFile  = Join-Path $CacheDir 'get-pip.py'

function Write-Step([string]$msg) {
    Write-Host ''
    Write-Host "━━ $msg " -ForegroundColor Cyan
}

function Require-File([string]$path, [string]$hint) {
    if (-not (Test-Path $path)) {
        throw "缺少文件：$path`n$hint"
    }
}

# ── 0. 清理旧产物 ─────────────────────────────────────
Write-Step '清理旧产物'
if (Test-Path $BuildDir) {
    Remove-Item -Recurse -Force $BuildDir
    Write-Host "已删除旧目录：$BuildDir"
}
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

# ── 1. 下载 embeddable Python 和 get-pip.py ────────────
Write-Step "准备 Python $PythonVersion embeddable"
if (-not $SkipDownload -or -not (Test-Path $PyZipFile)) {
    if (-not (Test-Path $PyZipFile)) {
        Write-Host "下载 $PyZipUrl"
        Invoke-WebRequest -Uri $PyZipUrl -OutFile $PyZipFile -UseBasicParsing
    } else {
        Write-Host "已存在缓存：$PyZipFile"
    }
}
Require-File $PyZipFile '请检查网络后重试，或手动下载后放入 dist\.cache\'

if (-not $SkipDownload -or -not (Test-Path $GetPipFile)) {
    if (-not (Test-Path $GetPipFile)) {
        Write-Host "下载 $GetPipUrl"
        Invoke-WebRequest -Uri $GetPipUrl -OutFile $GetPipFile -UseBasicParsing
    } else {
        Write-Host "已存在缓存：$GetPipFile"
    }
}
Require-File $GetPipFile 'get-pip.py 下载失败'

# ── 2. 解压 Python ───────────────────────────────────
Write-Step '解压 Python 到 python/'
Expand-Archive -Path $PyZipFile -DestinationPath $PyDir -Force

$PyExe = Join-Path $PyDir 'python.exe'
Require-File $PyExe '嵌入版 zip 结构异常'

# ── 3. 修改 ._pth 允许 site + 注入 libs 路径 ─────────
Write-Step "配置 $PthFileName"
$PthFile = Join-Path $PyDir $PthFileName
# 路径均相对于 python.exe 所在目录
@"
python$PyMajorMinor.zip
.
Lib\site-packages
..\libs\site-packages
..
import site
"@ | Set-Content -Path $PthFile -Encoding ASCII -NoNewline
Write-Host "已写入 $PthFile"

# ── 4. 安装 pip（到 python\Lib\site-packages） ────────
Write-Step '引导 pip'
& $PyExe $GetPipFile --no-warn-script-location --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'get-pip.py 执行失败' }

# ── 5. 安装项目依赖到 libs\ ──────────────────────────
Write-Step '安装依赖到 libs/（首次约 2~5 分钟）'
$ReqFile = Join-Path $RepoRoot 'requirements.txt'
Require-File $ReqFile 'requirements.txt 不存在'

# 使用官方源；如国内网络慢可改用清华镜像：
# -i https://pypi.tuna.tsinghua.edu.cn/simple
New-Item -ItemType Directory -Force -Path $SitePkg | Out-Null
& $PyExe -m pip install `
    --target=$SitePkg `
    --no-warn-script-location `
    --disable-pip-version-check `
    --no-compile `
    -r $ReqFile
if ($LASTEXITCODE -ne 0) { throw '依赖安装失败' }

# ── 6. 拷贝项目源码 ──────────────────────────────────
Write-Step '拷贝源码'
$SrcItems = @(
    'core',
    'app.py',
    'requirements.txt',
    'README.md',
    'LICENSE',
    '.streamlit'
)
foreach ($item in $SrcItems) {
    $src = Join-Path $RepoRoot $item
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination $BuildDir -Recurse -Force
        Write-Host "  ✓ $item"
    } else {
        Write-Host "  · 跳过（不存在）：$item" -ForegroundColor Yellow
    }
}

# ── 7. 生成 启动.bat ─────────────────────────────────
Write-Step '生成 启动.bat'
$LauncherSrc = Join-Path $PSScriptRoot 'launcher.bat'
Require-File $LauncherSrc 'scripts\launcher.bat 缺失'
Copy-Item -Path $LauncherSrc -Destination (Join-Path $BuildDir '启动.bat') -Force

# ── 8. 生成 使用说明.txt ─────────────────────────────
$ReadmeUser = @"
LatentTerm-zh · 中文小说术语挖掘
================================================

【怎么用】
  1. 双击本目录下的「启动.bat」
  2. 等待 5~10 秒，浏览器会自动打开
  3. 上传一个 txt 小说文件，输入关键词即可
  4. 用完关掉命令行黑窗口就退出

【遇到问题】
  · 如果浏览器没自动打开，手动访问：http://localhost:8965
  · 如果提示端口被占用，关掉其他用 8965 端口的程序后重试
  · 如果遇到了Welcome to Streamlit!，要求输入邮箱，直接回车即可
  · 如果被 Windows Defender 拦截，点「更多信息 → 仍要运行」即可
    （本程序不联网，全部源码公开在 GitHub）

【目录说明】
  python\     Python 运行时，请勿删除
  libs\       依赖包，请勿删除
  core\       核心算法源码
  app.py      主程序
  启动.bat    启动入口

【项目地址】
  https://github.com/afigzb/LatentTerm-zh/releases/tag/v1.0.0

构建日期：$(Get-Date -Format 'yyyy-MM-dd')
Python 版本：$PythonVersion
"@
$ReadmeUser | Set-Content -Path (Join-Path $BuildDir '使用说明.txt') -Encoding UTF8

# ── 9. 体积统计 ──────────────────────────────────────
Write-Step '构建完成'
$size = (Get-ChildItem -Recurse $BuildDir | Measure-Object -Property Length -Sum).Sum
$sizeMB = [math]::Round($size / 1MB, 1)
Write-Host "产物目录：$BuildDir"
Write-Host "总大小：  $sizeMB MB"

# ── 10. 可选压缩 ─────────────────────────────────────
if ($Zip) {
    Write-Step '压缩为 zip'
    $ZipFile = Join-Path $DistRoot "$AppName$(Get-Date -Format 'yyyyMMdd').zip"
    if (Test-Path $ZipFile) { Remove-Item $ZipFile -Force }
    Compress-Archive -Path $BuildDir -DestinationPath $ZipFile -CompressionLevel Optimal
    $zipMB = [math]::Round((Get-Item $ZipFile).Length / 1MB, 1)
    Write-Host "已生成：$ZipFile"
    Write-Host "压缩后：$zipMB MB"
}

Write-Host ''
Write-Host '==> 下一步：进入 dist\LatentTerm\ 双击「启动.bat」测试一下' -ForegroundColor Green
