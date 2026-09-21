#!/usr/bin/env bash
set -euo pipefail

# 从 Ascend plog 里找出 device fault 对应的 kernel 名。
#
# stderr 里那段 EZ9999 / "vector core exception" 只说"有个 kernel 访问了非法地址"，
# 不说是哪个；runtime 会把出错任务的 kernel 名另外写进 plog（形如
# "fault kernel_name=..., func_name=..."）。这一步不用重跑服务，崩溃过一次就能查。
#
#   ./npu_fault_kernel.sh 2324816     # 报错里的 PID：EZ9999[PID: 2324816]，是 scheduler 子进程
#   ./npu_fault_kernel.sh             # 不给 PID：取最近修改的 5 个 plog
#
# 日志根目录默认 ~/ascend/log，设过 ASCEND_PROCESS_LOG_PATH 的话以它为准。

log_root="${ASCEND_PROCESS_LOG_PATH:-$HOME/ascend/log}"
pid="${1:-}"

mapfile -t files < <(
    if [[ -n "$pid" ]]; then
        find "$log_root" -type f \( -name "plog-${pid}_*.log" -o -name "device-${pid}_*.log" \) 2>/dev/null
    else
        find "$log_root" -type f -name 'plog-*.log' -printf '%T@ %p\n' 2>/dev/null |
            sort -rn | head -5 | cut -d' ' -f2-
    fi
)

if [[ ${#files[@]} -eq 0 ]]; then
    echo "在 $log_root 下没找到${pid:+ PID $pid 的} plog；确认 PID 是 scheduler 子进程的，或检查 ASCEND_PROCESS_LOG_PATH" >&2
    exit 1
fi

printf '扫描 %d 个日志：\n' "${#files[@]}"
printf '  %s\n' "${files[@]}"
echo

echo '== 出错任务的 kernel 名（最关键）=='
grep -hEi 'fault kernel|kernel_name|func_name|kernelName' "${files[@]}" | sort | uniq -c | sort -rn | head -20 || true
echo
echo '== 出错上下文（stream / task / 错误码）=='
grep -hEi 'vector core|aivec|aicore exception|kernel task|PrintDavidCoreInfo|retCode=0x31|errcode:\(95\)' "${files[@]}" | head -20 || true
