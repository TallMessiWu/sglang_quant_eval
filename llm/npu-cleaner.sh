#!/bin/bash
set -euo pipefail

usage() {
    echo "用法: $0 [-f|--force] <NPU_ID_1> [NPU_ID_2] ... | all"
    echo "目标卡上只要有 PID，就需 Enter 确认或 Esc 取消；不判断进程所属用户。"
    echo "-f / --force: 跳过已有进程的确认（不改变系统权限）。"
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

mapfile -t pids_to_kill <<< "$unique_pids"
echo "⚠️ NPU ($TARGETS) 上已有进程: ${pids_to_kill[*]}"
# ps 仅补充展示信息；其成功与否、UID 或 sudo 身份均不能豁免确认。
pid_list=$(IFS=,; echo "${pids_to_kill[*]}")
ps -p "$pid_list" -o pid,user,args || true
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

echo "锁定进程: ${pids_to_kill[*]}"
if ! kill -9 -- "${pids_to_kill[@]}"; then
    echo "清理失败，取消后续启动；-f 不会授予终止进程的权限。" >&2
    exit 1
fi
echo "清理完成！"
