"""DeepCAVE-X 停滞期检测模块。"""

from typing import List, Tuple


def find_stagnation_segments(
    accuracies: List[float],
    patience: int = 5,
    min_delta: float = 0.0001,
) -> List[Tuple[int, int]]:
    """
    找出连续多次没有明显改进的试验区间。

    参数
    ----------
    accuracies:
        每次试验的准确率，例如 [0.90, 0.91, 0.905, ...]。

    patience:
        连续多少次没有明显提高，才认定为停滞。
        默认是 5 次。

    min_delta:
        最少提高多少才算“真正改善”。
        0.0001 表示提高至少 0.01 个百分点。

    返回
    ----------
    停滞区间列表，例如：
    [(3, 7), (10, 15)]

    试验编号从 1 开始。
    """
    if patience < 1:
        raise ValueError("patience 必须大于或等于 1")

    if min_delta < 0:
        raise ValueError("min_delta 不能小于 0")

    if not accuracies:
        return []

    best_score = float(accuracies[0])
    no_improvement_count = 0
    stagnation_start = None
    segments = []

    for trial_number, score in enumerate(accuracies[1:], start=2):
        score = float(score)

        # 新成绩至少提高 min_delta，视为有效改善
        if score > best_score + min_delta:
            best_score = score
            no_improvement_count = 0

            # 如果之前处于停滞期，到上一次试验为止
            if stagnation_start is not None:
                segments.append(
                    (stagnation_start, trial_number - 1)
                )
                stagnation_start = None

        else:
            no_improvement_count += 1

            # 连续 patience 次没有提高时，标记停滞起点
            if (
                no_improvement_count == patience
                and stagnation_start is None
            ):
                stagnation_start = trial_number - patience + 1

    # 如果运行结束时仍处于停滞，记录到最后一次试验
    if stagnation_start is not None:
        segments.append(
            (stagnation_start, len(accuracies))
        )

    return segments


def make_diagnosis(
    segments: List[Tuple[int, int]],
    total_trials: int,
) -> str:
    """根据停滞区间生成容易理解的诊断文字。"""
    if total_trials <= 0:
        return "没有可分析的试验。"

    if not segments:
        return (
            "没有检测到持续停滞。优化过程中仍能定期找到"
            "更好的超参数组合。"
        )

    longest_segment = max(
        segments,
        key=lambda segment: segment[1] - segment[0] + 1,
    )

    longest_length = (
        longest_segment[1] - longest_segment[0] + 1
    )

    stagnation_trials = sum(
        end - start + 1
        for start, end in segments
    )

    stagnation_ratio = stagnation_trials / total_trials

    if stagnation_ratio >= 0.6:
        suggestion = (
            "停滞占比较高。建议扩大超参数范围，"
            "增加试验次数，或更换优化策略。"
        )
    elif stagnation_ratio >= 0.3:
        suggestion = (
            "存在较明显的停滞。建议检查重要参数的"
            "搜索范围，并适当增加试验次数。"
        )
    else:
        suggestion = (
            "仅出现短期停滞，当前优化过程总体正常。"
        )

    return (
        f"检测到 {len(segments)} 个停滞区间；"
        f"最长区间为第 {longest_segment[0]} 至"
        f"第 {longest_segment[1]} 次试验，"
        f"持续 {longest_length} 次。"
        f"停滞试验约占 {stagnation_ratio:.1%}。"
        f"{suggestion}"
    )



