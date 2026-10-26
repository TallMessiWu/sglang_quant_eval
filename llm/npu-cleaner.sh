#!/bin/bash
set -euo pipefail

usage() {
    echo "用法: $0 [-f|--force] <NPU_ID_1> [NPU_ID_2] ... | all"
    echo "目标卡上有 PID 时：Enter 确认清理，P 保留进程并继续，Esc 取消；不判断进程所属用户。"
    echo "-f / --force: 跳过已有进程的确认（不改变系统权限）。"
}

# 启动器内部协议：stdout 只返回最终卡号，提示仍实时显示；独立清理不提供换卡。
select_devices=false
if [[ ${1:-} == --select-devices ]]; then
    select_devices=true
    shift
    exec 4>&1 1>&2
fi

report_devices() {
    if "$select_devices"; then
        (IFS=,; printf '%s\n' "${targets[*]}") >&4
    fi
}

# 第2列为卡号、第3列为 PID；双数字检查排除设备信息行，禁止 PID 0。
process_rows() {
    awk -F'|' '
        {
            c2 = $2; gsub(/^[ \t]+|[ \t]+$/, "", c2);
            c3 = $3; gsub(/^[ \t]+|[ \t]+$/, "", c3);
            if (c2 ~ /^[0-9]+$/ && c3 ~ /^[0-9]+$/ && c3 + 0 > 0) {
                print c2, c3;
            }
        }
    '
}

try_reassign_devices() {
    local count snapshot id pid next=0
    local -a free_devices=() candidates=() replacement=()
    local -A busy=() selected=()
    if ! count=$(env -u ASCEND_RT_VISIBLE_DEVICES python3 -c 'import torch_npu; print(torch_npu.npu.device_count())') \
        || [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then
        echo "无法换卡：无法获取当前可访问的 NPU 数量。" >&3
        return 1
    fi
    for id in "${targets[@]}"; do
        if ((id >= count)); then
            echo "无法换卡：卡号 $id 不在当前可访问范围 0-$((count - 1))。" >&3
            return 1
        fi
        selected[$id]=1
    done
    # 按键后重新查询，不使用等待用户输入之前的占用快照。
    if ! snapshot=$(npu-smi info); then
        echo "无法换卡：无法查询 NPU 进程。" >&3
        return 1
    fi
    while read -r id pid; do
        [[ -n "$id" ]] && busy[$id]=1
    done < <(process_rows <<< "$snapshot")
    for ((id=0; id<count; id++)); do
        if [[ -z ${busy[$id]:-} ]]; then
            free_devices+=("$id")
            if [[ -z ${selected[$id]:-} ]]; then candidates+=("$id"); fi
        fi
    done
    if [[ ${#free_devices[@]} -lt ${#targets[@]} ]]; then
        printf '无法换卡：需要 %s 张卡，当前仅 %s 张空闲（空闲卡：%s）。未改变卡号或终止进程。\n' \
            "${#targets[@]}" "${#free_devices[@]}" "${free_devices[*]:-无}" >&3
        return 1
    fi
    # 先预留原选卡中的空闲卡，再按升序候选替换占用位置，保持卡数和顺序。
    for id in "${targets[@]}"; do
        if [[ -n ${busy[$id]:-} ]]; then
            replacement+=("${candidates[$next]}")
            next=$((next + 1))
        else
            replacement+=("$id")
        fi
    done
    if [[ "${replacement[*]}" == "${targets[*]}" ]]; then
        echo "原选卡当前均已空闲，继续启动，未终止任何进程。" >&3
    else
        printf '已换卡：%s -> %s；卡数不变，未终止任何进程。\n' \
            "${targets[*]}" "${replacement[*]}" >&3
    fi
    targets=("${replacement[@]}")
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
if "$select_devices"; then
    # 调用方须先展开 all 并验证卡号；内部协议不接受重复或非规范整数。
    seen=" "
    for id in "${targets[@]}"; do
        if [[ ! "$id" =~ ^(0|[1-9][0-9]*)$ || "$seen" == *" $id "* ]]; then
            echo "错误：选卡模式需要已解析的、不重复的 NPU 卡号。" >&2
            exit 2
        fi
        seen+="$id "
    done
fi

if ! smi_output=$(npu-smi info); then
    echo "错误：无法查询 NPU 进程，取消清理。" >&2
    exit 1
fi

unique_pids=$(process_rows <<< "$smi_output" | awk -v targets="$TARGETS" '
    BEGIN {
        split(targets, t_arr, " ");
        for (i in t_arr) { target_map[t_arr[i]] = 1; }
    }
    "all" in target_map || $1 in target_map { print $2; }
' | sort -u)

if [[ -z "$unique_pids" ]]; then
    echo "NPU ($TARGETS) 上暂无运行进程。"
    report_devices
    exit 0
fi

mapfile -t pids_to_kill <<< "$unique_pids"
echo "⚠️ NPU ($TARGETS) 上已有进程："
# ps/pwdx 只补充展示信息；逐个 PID 查询，避免进程退出时把路径配到别的行。
display_commands=()
display_dirs=()
display_start_times=()
command_width=7
for pid in "${pids_to_kill[@]}"; do
    if ! process_command=$(ps -p "$pid" -o args= -ww 2>/dev/null); then
        process_command="?"
    fi
    process_command=${process_command:-?}
    if ! process_start=$(LC_ALL=C ps -p "$pid" -o lstart= 2>/dev/null) ||
        ! process_start=$(LC_ALL=C date -d "$process_start" '+%Y-%m-%d %H:%M:%S' 2>/dev/null); then
        process_start="?"
    fi
    if ! command -v pwdx >/dev/null 2>&1; then
        work_dir="不可获取（未安装 pwdx）"
    elif cwd_info=$(pwdx "$pid" 2>/dev/null); then
        work_dir=${cwd_info#"$pid: "}
    else
        work_dir="不可获取"
    fi
    display_commands+=("$process_command")
    display_dirs+=("$work_dir")
    display_start_times+=("$process_start")
    if [[ ${#process_command} -gt $command_width ]]; then command_width=${#process_command}; fi
done

printf '%-10s %-19s %-*s %s\n' "PID" "START_TIME" "$command_width" "COMMAND" "工作目录"
for i in "${!pids_to_kill[@]}"; do
    printf '%-10s %-19s %-*s %s\n' "${pids_to_kill[$i]}" "${display_start_times[$i]}" \
        "$command_width" "${display_commands[$i]}" "${display_dirs[$i]}"
done

if "$force"; then
    echo "已指定 -f/--force，跳过确认。"
else
    # 使用控制终端，避免重定向 stdin 中的换行意外授权杀进程。
    if ! { exec 3<>/dev/tty; } 2>/dev/null; then
        echo "没有可用的交互终端，取消清理；如确定要强制清理，请显式传入 -f。" >&2
        exit 1
    fi
    swap_option=""
    if "$select_devices"; then swap_option="R 换到空闲卡，"; fi
    printf 'NPU (%s) 上有运行进程。Enter 确认强制终止，P 保留进程并继续，%sEsc 取消：' "$TARGETS" "$swap_option" >&3
    while true; do
        if ! IFS= read -r -s -n 1 -u 3 key; then
            echo "取消清理。" >&2
            exit 130
        fi
        case "$key" in
            ''|$'\r') printf '\n' >&3; break ;;
            p|P)
                printf '\n已跳过清理，保留全部已有进程，继续后续启动。\n' >&3
                report_devices
                exec 3>&-
                exit 0
                ;;
            r|R)
                if "$select_devices"; then
                    printf '\n正在检查可用于换卡的空闲 NPU...\n' >&3
                    if try_reassign_devices; then
                        report_devices
                        exec 3>&-
                        exit 0
                    fi
                fi
                printf '\n请按 Enter 确认强制终止，P 保留进程并继续，%s或 Esc 取消：' "$swap_option" >&3
                ;;
            $'\e') printf '\n取消清理，未终止任何进程。\n' >&3; exit 130 ;;
            *) printf '\n请按 Enter 确认强制终止，P 保留进程并继续，%s或 Esc 取消：' "$swap_option" >&3 ;;
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
report_devices
