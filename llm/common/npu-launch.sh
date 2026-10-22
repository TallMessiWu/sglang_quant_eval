#!/bin/bash

# 仅解析设备和 TP，不清理进程；首参为调用脚本自己的默认 TP。
npu_resolve_devices() {
    local default_tp=$1 count id seen=" "
    shift
    local -a devices=("$@")

    if [[ $# -eq 0 || ( $# -eq 1 && $1 == all ) ]]; then
        if ! count=$(env -u ASCEND_RT_VISIBLE_DEVICES python3 -c 'import torch_npu; print(torch_npu.npu.device_count())') \
            || [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then
            echo "错误：无法获取可用 NPU 数量或没有可用 NPU。" >&2
            return 1
        fi
        devices=()
        for ((id=0; id<count; id++)); do devices+=("$id"); done
    else
        for id in "${devices[@]}"; do
            if [[ ! "$id" =~ ^(0|[1-9][0-9]*)$ || "$seen" == *" $id "* ]]; then
                echo "错误：无效或重复卡号 $id；请传 NPU_ID ... 或单独传 all。" >&2
                return 2
            fi
            seen+="$id "
        done
    fi

    NPU_TP_SIZE=$default_tp
    if [[ $# -gt 0 ]]; then NPU_TP_SIZE=${#devices[@]}; fi
    export ASCEND_RT_VISIBLE_DEVICES
    ASCEND_RT_VISIBLE_DEVICES=$(IFS=,; echo "${devices[*]}")
    echo "🚀 [启动任务] TP=$NPU_TP_SIZE，ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
}

# 由模型启动脚本 source；参数为 DEFAULT_TP [-f|--force] [NPU_ID ... | all]。
# 所有清理失败（含 Esc 取消）必须由调用方用 `|| exit $?` 中止启动。
npu_prepare_launch() {
    local default_tp=$1 arg cleaner_path status
    shift
    local -a devices=() cleaner_args=()
    for arg in "$@"; do
        case "$arg" in
            -f|--force) cleaner_args+=("$arg") ;;
            *) devices+=("$arg") ;;
        esac
    done

    npu_resolve_devices "$default_tp" "${devices[@]}" || return $?
    if [[ ${#devices[@]} -eq 0 ]]; then
        # 不指定卡号（包括仅传 -f）时，保留不清理进程的行为。
        echo "⚠️ 未传入 NPU 卡号，保留脚本默认 TP，跳过清理。"
        return 0
    fi

    cleaner_path="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../npu-cleaner.sh"
    if [[ -x "$cleaner_path" ]]; then
        echo "🧹 [前置任务] 准备释放 NPU (${devices[*]}) 的资源..."
        if "$cleaner_path" "${cleaner_args[@]}" "${devices[@]}"; then
            sleep 1
            echo "✅ [前置任务] 进程检查完成，继续启动！"
        else
            status=$?
            echo "取消模型启动：NPU 清理未完成。" >&2
            return "$status"
        fi
    else
        echo "⚠️ 警告：未找到可执行的清理脚本 $cleaner_path，将跳过清理直接启动。"
    fi
}
