#!/bin/bash
set -euo pipefail

usage() {
    echo "用法: $0 [-f|--force] <NPU_ID_1> [NPU_ID_2] ... | all"
    echo "仅清理自己的进程时直接执行；涉及其他用户时，Enter 确认，Esc 取消。"
    echo "-f / --force: 跳过其他用户进程的确认（不改变系统权限）。"
}

force=false
targets=()
for arg in "$@"; do
    case "$arg" in
        -f|--force) force=true ;;
        -h|--help) usage; exit 0 ;;
        all) targets+=("$arg") ;;
        *)
            if [[ "$arg" =~ ^[0-9]+$ ]]; then
                targets+=("$arg")
            else
                echo "错误：无效参数 $arg" >&2
                usage >&2
                exit 2
            fi
            ;;
    esac
done
if [[ ${#targets[@]} -eq 0 ]]; then
    usage >&2
    exit 2
fi
TARGETS="${targets[*]}"

if ! smi_output=$(npu-smi info); then
    echo "错误：无法查询 NPU 进程，取消清理。" >&2
    exit 1
fi

# 第2列为卡号、第3列为 PID；双数字检查排除设备信息行，禁止 PID 0。
unique_pids=$(awk -F'|' -v targets="$TARGETS" '
    BEGIN {
        split(targets, t_arr, " ");
        for (i in t_arr) { target_map[t_arr[i]] = 1; }
    }
    {
        c2 = $2; gsub(/^[ \t]+|[ \t]+$/, "", c2);
        c3 = $3; gsub(/^[ \t]+|[ \t]+$/, "", c3);
        if (c2 ~ /^[0-9]+$/ && c3 ~ /^[0-9]+$/ && c3 + 0 > 0) {
            if ("all" in target_map || c2 in target_map) { print c3; }
        }
    }
' <<< "$smi_output" | sort -u)

if [[ -z "$unique_pids" ]]; then
    echo "NPU ($TARGETS) 上暂无运行进程。"
    exit 0
fi

# sudo 启动时仍以实际发起用户为准，而不是把 root 的进程都当成自己的。
current_uid=${SUDO_UID:-$(id -u)}
pids_to_kill=()
other_pids=()
declare -A owners=()
while IFS= read -r pid; do
    owner_uid=$(ps -p "$pid" -o uid=) || owner_uid=""
    owner_uid=${owner_uid//[[:space:]]/}
    # 查询后已退出的进程无需清理；仍存在却无法确认归属时拒绝继续。
    if [[ -z "$owner_uid" && ! -d "/proc/$pid" ]]; then
        continue
    fi
    if [[ ! "$owner_uid" =~ ^[0-9]+$ ]]; then
        echo "错误：无法确认 PID $pid 的所属用户，取消清理。" >&2
        exit 1
    fi
    pids_to_kill+=("$pid")
    owners[$pid]=$owner_uid
    if [[ "$owner_uid" != "$current_uid" ]]; then
        other_pids+=("$pid")
    fi
done <<< "$unique_pids"

if [[ ${#pids_to_kill[@]} -eq 0 ]]; then
    echo "NPU ($TARGETS) 上的进程已退出，无需清理。"
    exit 0
fi

if [[ ${#other_pids[@]} -gt 0 ]]; then
    echo "⚠️ 以下进程属于其他用户（当前用户 UID=$current_uid）："
    other_pid_list=$(IFS=,; echo "${other_pids[*]}")
    ps -p "$other_pid_list" -o pid,user,args || true
    if "$force"; then
        echo "已指定 -f/--force，跳过确认。"
    else
        # 使用控制终端，避免重定向 stdin 中的换行意外授权杀进程。
        if ! { exec 3<>/dev/tty; } 2>/dev/null; then
            echo "没有可用的交互终端，取消清理；如确定要强制清理，请显式传入 -f。" >&2
            exit 1
        fi
        printf '将强制终止 NPU (%s) 上的全部目标进程。Enter 确认，Esc 取消：' "$TARGETS" >&3
        while true; do
            if ! IFS= read -r -s -n 1 -u 3 key; then
                echo "取消清理。" >&2
                exit 130
            fi
            case "$key" in
                ''|$'\r') printf '\n' >&3; break ;;
                $'\e') printf '\n取消清理，未终止任何进程。\n' >&3; exit 130 ;;
                *) printf '\n请按 Enter 确认，或 Esc 取消：' >&3 ;;
            esac
        done
        exec 3>&-
    fi
fi

# 确认期间进程可能退出或 PID 被复用；归属变化时重新运行，而不是扩大授权范围。
confirmed_pids=()
for pid in "${pids_to_kill[@]}"; do
    owner_uid=$(ps -p "$pid" -o uid=) || owner_uid=""
    owner_uid=${owner_uid//[[:space:]]/}
    if [[ -z "$owner_uid" && ! -d "/proc/$pid" ]]; then
        continue
    fi
    if [[ "$owner_uid" != "${owners[$pid]}" ]]; then
        echo "错误：PID $pid 的所属用户已变化或无法确认，请重新运行清理。" >&2
        exit 1
    fi
    confirmed_pids+=("$pid")
done

if [[ ${#confirmed_pids[@]} -gt 0 ]]; then
    echo "锁定进程: ${confirmed_pids[*]}"
    if ! kill -9 -- "${confirmed_pids[@]}"; then
        echo "清理失败，取消后续启动；-f 不会授予终止其他用户进程的权限。" >&2
        exit 1
    fi
fi
echo "清理完成！"
