[CmdletBinding()]
param([string]$RepoRoot)

$ErrorActionPreference = "Stop"
# This read-only audit needs only repository-local config. The desktop sandbox
# may not be allowed to read the user's global core.excludesFile.
$env:GIT_CONFIG_GLOBAL = "NUL"

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
}

$RepoRoot = (Resolve-Path $RepoRoot).Path
$KernelPath = Join-Path $RepoRoot "sgl-kernel-npu"
$SglangRoot = Join-Path $RepoRoot "sglang"
$SglangMain = Join-Path $SglangRoot "qwen3.5_dense_w8a8"

if (-not (Test-Path $KernelPath)) {
    throw "Missing sgl-kernel-npu checkout: $KernelPath"
}

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $output = @(& git -c "safe.directory=$WorkingDirectory" -c "core.excludesFile=" -C $WorkingDirectory @Arguments 2>&1)
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "git -C '$WorkingDirectory' $($Arguments -join ' ') failed: $($output -join [Environment]::NewLine)"
    }
    return ($output -join [Environment]::NewLine).Trim()
}

$kernelStatus = Invoke-Git $KernelPath @("status", "--short")
$kernelHead = Invoke-Git $KernelPath @("rev-parse", "HEAD")
$parentGitlinkLine = Invoke-Git $RepoRoot @("ls-tree", "HEAD", "sgl-kernel-npu")
$parentGitlink = if ($parentGitlinkLine -match "^160000 commit ([0-9a-f]{40})") { $Matches[1] } else { $null }

$worktreePaths = @()
if (Test-Path $SglangMain) {
    $worktreeOutput = Invoke-Git $SglangMain @("worktree", "list", "--porcelain")
    $worktreePaths = @($worktreeOutput -split "`r?`n" | Where-Object { $_ -like "worktree *" } | ForEach-Object {
        $_.Substring("worktree ".Length)
    })
}

$normalizedSglangRoot = $SglangRoot.Replace("\", "/").TrimEnd("/") + "/"
$externalWorktrees = @($worktreePaths | Where-Object {
    -not $_.Replace("\", "/").StartsWith($normalizedSglangRoot, [System.StringComparison]::OrdinalIgnoreCase)
})

$origin = Invoke-Git $KernelPath @("remote", "get-url", "origin")
$upstream = Invoke-Git $KernelPath @("remote", "get-url", "upstream")

[ordered]@{
    repo_root = $RepoRoot
    kernel = [ordered]@{
        path = $KernelPath
        branch = Invoke-Git $KernelPath @("branch", "--show-current")
        head = $kernelHead
        tracking = Invoke-Git $KernelPath @("rev-parse", "--abbrev-ref", "@{u}")
        clean = [string]::IsNullOrWhiteSpace($kernelStatus)
        origin = $origin
        upstream = $upstream
        origin_is_expected = $origin -eq "https://github.com/TallMessiWu/sgl-kernel-npu.git"
        upstream_is_expected = $upstream -eq "https://github.com/sgl-project/sgl-kernel-npu.git"
    }
    parent = [ordered]@{
        gitlink = $parentGitlink
        gitlink_matches_checkout = $parentGitlink -eq $kernelHead
    }
    sglang = [ordered]@{
        main_clone = $SglangMain
        registered_worktrees = $worktreePaths
        external_worktrees = $externalWorktrees
    }
} | ConvertTo-Json -Depth 6
