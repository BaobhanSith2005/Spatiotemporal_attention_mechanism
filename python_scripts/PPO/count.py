import json
import re
from pathlib import Path


# 既可以填写单个 JSON 文件路径，也可以填写包含多个 JSON 的文件夹路径
INPUT_PATH = Path(
    r"D:\\奖励值收敛project_Spatiotemporal_attention_mechanism\\project_Spatiotemporal_attention_mechanism\\python_scripts\\PPO\\log\\catch_log\\catch_log_22.json"
)

# 统计结果输出路径
OUTPUT_TXT = Path(
    r"D:\\奖励值收敛project_Spatiotemporal_attention_mechanism\\project_Spatiotemporal_attention_mechanism\\python_scripts\\PPO\\test_goal_statistics.txt"
)

# 前 N 个元素统计
PREFIX_LENGTHS = [30, 40, 50, 60, 70, 100]

# 固定区间统计
INTERVALS = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 400), (400, 500)]

# 匹配 test_goal、test_goal_199、test_goal_399 这类字段
TEST_GOAL_PATTERN = re.compile(r"^test_goal(?:_(\d+))?$")


def format_probability(count_1, total_count):
    if total_count == 0:
        return "0.0000 (0.00%)"
    probability = count_1 / total_count
    return f"{probability:.4f} ({probability * 100:.2f}%)"


def analyze_test_goal(test_goal, name):
    lines = []
    total_length = len(test_goal)

    lines.append(f"数组名称: {name}")
    lines.append(f"数组总长度: {total_length}")
    lines.append("")
    lines.append("[前 N 个元素统计结果]")

    for length in PREFIX_LENGTHS:
        if length > total_length:
            lines.append(
                f"前 {length} 个元素: 超出数组实际长度 ({total_length})，无法统计"
            )
            continue

        sub_array = test_goal[:length]
        count_1 = sub_array.count(1)
        probability_text = format_probability(count_1, length)
        lines.append(
            f"前 {length} 个元素: 1 的个数 = {count_1}, 出现概率 = {probability_text}"
        )

    lines.append("")
    lines.append("[区间元素统计结果]")

    for start, end in INTERVALS:
        if start >= total_length:
            lines.append(
                f"{start}-{end} 区间: 起始位置 {start} 超出数组长度 {total_length}，无元素可统计"
            )
            continue

        actual_end = min(end, total_length)
        sub_array = test_goal[start:actual_end]
        interval_count = len(sub_array)

        if interval_count == 0:
            lines.append(f"{start}-{end} 区间: 无元素可统计")
            continue

        count_1 = sub_array.count(1)
        probability_text = format_probability(count_1, interval_count)
        lines.append(
            f"{start}-{end} 区间 (实际统计 {start}-{actual_end}): "
            f"元素总数 = {interval_count}, 1 的个数 = {count_1}, 出现概率 = {probability_text}"
        )

    return lines


def extract_test_goal_items(json_path):
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    matched_items = []
    for key, value in data.items():
        match = TEST_GOAL_PATTERN.match(key)
        if not match:
            continue

        if not isinstance(value, list):
            continue

        if not all(item in (0, 1) for item in value):
            continue

        suffix = match.group(1)
        sort_key = int(suffix) if suffix is not None else -1
        matched_items.append((sort_key, key, value))

    matched_items.sort(key=lambda item: item[0])
    return matched_items


def resolve_json_files(input_path):
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入路径: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError(f"输入文件不是 JSON 文件: {input_path}")
        return [input_path]

    if input_path.is_dir():
        json_files = sorted(input_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"在文件夹中没有找到 JSON 文件: {input_path}")
        return json_files

    raise ValueError(f"输入路径既不是文件也不是文件夹: {input_path}")


def main():
    json_files = resolve_json_files(INPUT_PATH)

    output_lines = []
    total_arrays = 0

    for json_file in json_files:
        test_goal_items = extract_test_goal_items(json_file)
        if not test_goal_items:
            continue

        output_lines.append("=" * 100)
        output_lines.append(f"文件: {json_file}")
        output_lines.append(f"提取到的 test_goal 数组数量: {len(test_goal_items)}")
        output_lines.append("=" * 100)
        output_lines.append("")

        for _, key, test_goal in test_goal_items:
            output_lines.extend(analyze_test_goal(test_goal, key))
            output_lines.append("")
            output_lines.append("-" * 100)
            output_lines.append("")
            total_arrays += 1

    if total_arrays == 0:
        output_lines.append("未在指定输入中找到 test_goal 或 test_goal_xxx 数组。")

    OUTPUT_TXT.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"统计完成，共处理 {total_arrays} 个数组。")
    print(f"结果已输出到: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
