#!/bin/bash

# 由模型启动脚本 source；参数为 [-f|--force] [NPU_ID ... | all]。
# 所有清理失败（含 Esc 取消）必须由调用方用 `|| exit $?` 中止启动。
npu_prepare_launch() {
    local arg cleaner_path status
    local -a devices=() cleaner_args=()
    for arg in "$@"; do
        case "$arg" in
            -f|--force) cleaner_args+=("$arg") ;;
            all) devices+=("$arg") ;;
            *)
                if [[ "$arg" =~ ^[0-9]+$ ]]; then
                    devices+=("$arg")
                else
                    echo "错误：无效参数 $arg；用法: $0 [-f|--force] [NPU_ID ... | all]" >&2
                    return 2
                fi
                ;;
        esac
    done

    if [[ ${#devices[@]} -eq 0 ]]; then
        # 不指定卡号（包括仅传 -f）时，保留不清理进程的行为。
        unset ASCEND_RT_VISIBLE_DEVICES
        echo "⚠️ 未传入 NPU 卡号，跳过清理并 unset ASCEND_RT_VISIBLE_DEVICES"
        return 0
    fi

    cleaner_path="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../npu-cleaner.sh"
    if [[ -x "$cleaner_path" ]]; then
        echo "🧹 [前置任务] 准备释放 NPU (${devices[*]}) 的资源..."
        if "$cleaner_path" "${cleaner_args[@]}" "${devices[@]}"; then
            sleep 1
            echo "✅ [前置任务] 清理完毕！"
        else
            status=$?
            echo "取消模型启动：NPU 清理未完成。" >&2
            return "$status"
        fi
    else
        echo "⚠️ 警告：未找到可执行的清理脚本 $cleaner_path，将跳过清理直接启动。"
    fi

    if [[ " ${devices[*]} " == *" all "* ]]; then
        unset ASCEND_RT_VISIBLE_DEVICES
    else
        # -f/--force 仅透传给清理器，绝不能混进设备可见性变量。
        export ASCEND_RT_VISIBLE_DEVICES
        ASCEND_RT_VISIBLE_DEVICES=$(IFS=,; echo "${devices[*]}")
    fi
    echo "🚀 [启动任务] 当前设备可见性: ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
}
