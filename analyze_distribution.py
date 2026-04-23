from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - 运行时依赖检查
    plt = None


DEFAULT_BINS = 50
DEFAULT_MAX_PERCENTILE = 0.99
CHART_WIDTH = 1200
CHART_HEIGHT = 700


def percentile_value(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("无法对空数据计算分位数")

    if percentile <= 0:
        return sorted_values[0]

    if percentile >= 1:
        return sorted_values[-1]

    position = math.ceil(percentile * len(sorted_values)) - 1
    position = max(0, min(position, len(sorted_values) - 1))
    return sorted_values[position]


def load_numeric_column(
    csv_path: Path,
    column_name: str,
    max_percentile: float,
) -> tuple[list[float], int, int, int, float | None]:
    values: list[float] = []
    skipped = 0
    skipped_zero = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or column_name not in reader.fieldnames:
            available = ", ".join(reader.fieldnames or [])
            raise ValueError(f"未找到列 {column_name}，当前列为: {available}")

        for row in reader:
            raw_value = (row.get(column_name) or "").strip()
            if not raw_value:
                continue

            try:
                numeric_value = float(raw_value)
            except ValueError:
                skipped += 1
                continue

            if math.isclose(numeric_value, 0.0):
                skipped_zero += 1
                continue

            values.append(numeric_value)

    if not values:
        return values, skipped, skipped_zero, 0, None

    sorted_values = sorted(values)
    upper_bound = percentile_value(sorted_values, max_percentile)
    filtered_values = [value for value in values if value <= upper_bound]
    skipped_high = len(values) - len(filtered_values)
    return filtered_values, skipped, skipped_zero, skipped_high, upper_bound


def build_histogram(values: list[float], bins: int) -> list[tuple[float, float, int]]:
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)

    if math.isclose(minimum, maximum):
        return [(minimum, maximum, len(values))]

    bin_width = (maximum - minimum) / bins
    counts = [0] * bins

    for value in values:
        index = min(int((value - minimum) / bin_width), bins - 1)
        counts[index] += 1

    histogram: list[tuple[float, float, int]] = []
    for index, count in enumerate(counts):
        left = minimum + index * bin_width
        right = maximum if index == bins - 1 else minimum + (index + 1) * bin_width
        histogram.append((left, right, count))

    return histogram


def write_histogram_csv(histogram: list[tuple[float, float, int]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["bin_left", "bin_right", "frequency"])
        for left, right, frequency in histogram:
            writer.writerow([f"{left:.6f}", f"{right:.6f}", frequency])


def summarize(
    values: list[float],
    skipped: int,
    skipped_zero: int,
    skipped_high: int,
    upper_bound: float | None,
) -> list[tuple[str, str]]:
    if not values:
        summary_rows = [
            ("count", "0"),
            ("skipped_invalid", str(skipped)),
            ("skipped_zero", str(skipped_zero)),
            ("skipped_high", str(skipped_high)),
        ]
        if upper_bound is not None:
            summary_rows.append(("upper_bound", f"{upper_bound:.6f}"))
        return summary_rows

    sorted_values = sorted(values)
    count = len(sorted_values)
    mid = count // 2
    if count % 2 == 0:
        median = (sorted_values[mid - 1] + sorted_values[mid]) / 2
    else:
        median = sorted_values[mid]

    average = sum(sorted_values) / count
    return [
        ("count", str(count)),
        ("min", f"{sorted_values[0]:.6f}"),
        ("max", f"{sorted_values[-1]:.6f}"),
        ("mean", f"{average:.6f}"),
        ("median", f"{median:.6f}"),
        ("skipped_invalid", str(skipped)),
        ("skipped_zero", str(skipped_zero)),
        ("skipped_high", str(skipped_high)),
    ]
    if upper_bound is not None:
        summary_rows.append(("upper_bound", f"{upper_bound:.6f}"))
    return summary_rows


def write_summary_csv(summary_rows: list[tuple[str, str]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        writer.writerows(summary_rows)


def format_number(value: float) -> str:
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.2f}"


def render_histogram_png(
    histogram: list[tuple[float, float, int]],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    if plt is None:
        raise RuntimeError("未安装 matplotlib，无法输出 PNG。请先执行: pip install matplotlib")

    fig, ax = plt.subplots(figsize=(CHART_WIDTH / 100, CHART_HEIGHT / 100), dpi=100)

    if not histogram:
        ax.text(0.5, 0.5, f"{title}: 无可用数据", ha="center", va="center", fontsize=20, color="#333333")
        ax.axis("off")
    else:
        centers = [(left + right) / 2 for left, right, _ in histogram]
        widths = [max(right - left, 1e-9) for left, right, _ in histogram]
        frequencies = [frequency for _, _, frequency in histogram]

        ax.bar(centers, frequencies, width=widths, color="#2563eb", alpha=0.85, edgecolor="white", linewidth=0.6)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        tick_count = min(6, len(histogram))
        if tick_count == 1:
            tick_positions = [0]
        else:
            tick_positions = sorted({round(index * (len(histogram) - 1) / (tick_count - 1)) for index in range(tick_count)})

        tick_values = [centers[position] for position in tick_positions]
        tick_labels = [
            f"{format_number(histogram[position][0])}-{format_number(histogram[position][1])}"
            for position in tick_positions
        ]
        ax.set_xticks(tick_values, tick_labels, rotation=35, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, format="png")
    plt.close(fig)


def analyze_column(
    csv_path: Path,
    column_name: str,
    bins: int,
    output_dir: Path,
    max_percentile: float,
) -> None:
    values, skipped, skipped_zero, skipped_high, upper_bound = load_numeric_column(
        csv_path,
        column_name,
        max_percentile,
    )
    histogram = build_histogram(values, bins)

    write_histogram_csv(histogram, output_dir / f"{column_name}_distribution.csv")
    write_summary_csv(
        summarize(values, skipped, skipped_zero, skipped_high, upper_bound),
        output_dir / f"{column_name}_summary.csv",
    )
    render_histogram_png(
        histogram,
        title=f"{column_name} 分布图",
        x_label="值",
        y_label="频次",
        output_path=output_dir / f"{column_name}_distribution.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 CSV 中 time_value 和 dist_value 的分布并生成图表。")
    parser.add_argument(
        "--input",
        default="distribution/20230918.csv",
        help="输入 CSV 文件路径，默认读取 distribution/20230918.csv",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="直方图区间数量，默认 50",
    )
    parser.add_argument(
        "--output-dir",
        default="distribution_output",
        help="输出目录，默认是当前目录下的 distribution_output",
    )
    parser.add_argument(
        "--max-percentile",
        type=float,
        default=DEFAULT_MAX_PERCENTILE,
        help="高端极值过滤分位点，默认 0.99，表示去掉最高 1%% 的值",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bins <= 0:
        raise ValueError("--bins 必须大于 0")
    if not 0 < args.max_percentile <= 1:
        raise ValueError("--max-percentile 必须在 0 到 1 之间")

    script_dir = Path(__file__).resolve().parent
    input_path = Path(args.input)

    if input_path.is_absolute():
        csv_path = input_path
    else:
        candidates = [
            Path.cwd() / input_path,
            script_dir / input_path,
            Path.cwd() / "distribution" / input_path,
            script_dir / "distribution" / input_path,
        ]
        csv_path = next((path for path in candidates if path.exists()), candidates[0])

    csv_path = csv_path.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(
            "未找到输入文件: "
            f"{csv_path}。请检查 --input 参数，或使用默认路径 distribution/20230918.csv。"
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    analyze_column(csv_path, "time_value", args.bins, output_dir, args.max_percentile)
    analyze_column(csv_path, "dist_value", args.bins, output_dir, args.max_percentile)

    print(f"已完成统计，输出目录: {output_dir}")


if __name__ == "__main__":
    main()