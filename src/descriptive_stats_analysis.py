from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANALYSIS_START = "2022-01-01"
ANALYSIS_END = "2025-12-31"

STOCKS = [
    "600519.SS",
    "000858.SZ",
    "601318.SS",
    "600036.SS",
    "600900.SS",
    "600276.SS",
    "601012.SS",
    "300750.SZ",
    "600887.SS",
    "002415.SZ",
]

TICKER_NAMES = {
    "600519.SS": "贵州茅台",
    "000858.SZ": "五粮液",
    "601318.SS": "中国平安",
    "600036.SS": "招商银行",
    "600900.SS": "长江电力",
    "600276.SS": "恒瑞医药",
    "601012.SS": "隆基绿能",
    "300750.SZ": "宁德时代",
    "600887.SS": "伊利股份",
    "002415.SZ": "海康威视",
}

INPUT_DIR = Path("output")
OUTPUT_DIR = INPUT_DIR / "descriptive_stats"

CLOSE_PRICE_PATH = INPUT_DIR / "close_price_2022_2025.csv"
STOCK_LOG_RETURN_PATH = INPUT_DIR / "stock_log_returns_2022_2025.csv"

DESCRIPTIVE_STATS_PATH = OUTPUT_DIR / "stock_return_descriptive_statistics.csv"
COVARIANCE_MATRIX_PATH = OUTPUT_DIR / "stock_return_covariance_matrix.csv"
CORRELATION_MATRIX_PATH = OUTPUT_DIR / "stock_return_correlation_matrix.csv"
EXCEL_OUTPUT_PATH = OUTPUT_DIR / "descriptive_stats_correlation_analysis_2022_2025.xlsx"
INTERPRETATION_PATH = OUTPUT_DIR / "descriptive_correlation_interpretation.txt"

CLOSE_PRICE_FIG_PATH = OUTPUT_DIR / "stock_close_price_lines.png"
RETURN_BOXPLOT_PATH = OUTPUT_DIR / "stock_return_boxplot.png"
CORRELATION_HEATMAP_PATH = OUTPUT_DIR / "stock_return_correlation_heatmap.png"


def display(obj):
    print(obj)


def configure_plot_style():
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["font.size"] = 10


def read_indexed_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件：{path}")
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df


def load_close_price():
    close_price = read_indexed_csv(CLOSE_PRICE_PATH)
    missing = [ticker for ticker in STOCKS if ticker not in close_price.columns]
    if missing:
        raise ValueError(f"收盘价数据缺少股票列：{missing}")
    close_price = close_price.loc[ANALYSIS_START:ANALYSIS_END, STOCKS]
    return close_price.apply(pd.to_numeric, errors="coerce").ffill()


def load_stock_returns(close_price):
    if STOCK_LOG_RETURN_PATH.exists():
        returns = read_indexed_csv(STOCK_LOG_RETURN_PATH)
        missing = [ticker for ticker in STOCKS if ticker not in returns.columns]
        if not missing:
            returns = returns.loc[ANALYSIS_START:ANALYSIS_END, STOCKS]
            return returns.apply(pd.to_numeric, errors="coerce").dropna(how="all")

    returns = np.log(close_price) - np.log(close_price.shift(1))
    return returns.dropna(how="all")


def calculate_descriptive_statistics(returns):
    stats = pd.DataFrame({
        "Ticker": STOCKS,
        "Name": [TICKER_NAMES[ticker] for ticker in STOCKS],
        "Count": returns[STOCKS].count().values,
        "Mean": returns[STOCKS].mean().values,
        "Std": returns[STOCKS].std().values,
        "Variance": returns[STOCKS].var().values,
        "Min": returns[STOCKS].min().values,
        "25%": returns[STOCKS].quantile(0.25).values,
        "Median": returns[STOCKS].median().values,
        "75%": returns[STOCKS].quantile(0.75).values,
        "Max": returns[STOCKS].max().values,
        "Skewness": returns[STOCKS].skew().values,
        "Kurtosis": returns[STOCKS].kurt().values,
    })
    stats["Annualized Mean"] = stats["Mean"] * 252
    stats["Annualized Volatility"] = stats["Std"] * np.sqrt(252)
    return stats


def plot_close_prices(close_price):
    fig, axes = plt.subplots(5, 2, figsize=(14, 15), sharex=True)
    axes = axes.ravel()
    for ax, ticker in zip(axes, STOCKS):
        ax.plot(close_price.index, close_price[ticker], linewidth=1.25, color="#2563eb")
        ax.set_title(ticker, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Daily Close Prices of 10 Stocks", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(CLOSE_PRICE_FIG_PATH, bbox_inches="tight")
    plt.close(fig)


def plot_return_boxplot(returns):
    labels = [ticker.replace(".SS", "").replace(".SZ", "") for ticker in STOCKS]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.boxplot(
        [returns[ticker].dropna() * 100 for ticker in STOCKS],
        tick_labels=labels,
        showfliers=True,
        patch_artist=True,
        medianprops={"color": "#111827", "linewidth": 1.2},
        boxprops={"facecolor": "#dbeafe", "edgecolor": "#1d4ed8", "linewidth": 1.0},
        whiskerprops={"color": "#1d4ed8", "linewidth": 0.9},
        capprops={"color": "#1d4ed8", "linewidth": 0.9},
        flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": "#ef4444", "markeredgewidth": 0},
    )
    ax.axhline(0, color="#374151", linewidth=0.9, alpha=0.8)
    ax.set_title("Daily Log Return Boxplot")
    ax.set_xlabel("Stock")
    ax.set_ylabel("Daily log return (%)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RETURN_BOXPLOT_PATH, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(corr_matrix):
    labels = [ticker.replace(".SS", "").replace(".SZ", "") for ticker in STOCKS]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr_matrix.iloc[i, j]
            text_color = "white" if abs(value) > 0.55 else "#111827"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    ax.set_title("Correlation Heatmap of Daily Log Returns")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")
    fig.tight_layout()
    fig.savefig(CORRELATION_HEATMAP_PATH, bbox_inches="tight")
    plt.close(fig)


def describe_distribution(stats):
    highest_vol = stats.sort_values("Std", ascending=False).iloc[0]
    lowest_vol = stats.sort_values("Std", ascending=True).iloc[0]
    highest_mean = stats.sort_values("Mean", ascending=False).iloc[0]
    lowest_mean = stats.sort_values("Mean", ascending=True).iloc[0]
    right_skewed = stats[stats["Skewness"] > 0]["Name"].tolist()
    left_skewed = stats[stats["Skewness"] < 0]["Name"].tolist()
    high_kurtosis = stats.sort_values("Kurtosis", ascending=False).iloc[0]

    right_text = "、".join(right_skewed) if right_skewed else "无"
    left_text = "、".join(left_skewed) if left_skewed else "无"

    return [
        (
            f"日收益率均值最高的是 {highest_mean['Name']}（{highest_mean['Ticker']}），"
            f"最低的是 {lowest_mean['Name']}（{lowest_mean['Ticker']}）。"
        ),
        (
            f"波动性方面，标准差最高的是 {highest_vol['Name']}（{highest_vol['Ticker']}，"
            f"日标准差 {highest_vol['Std']:.4%}），说明该股票日收益波动最强；"
            f"标准差最低的是 {lowest_vol['Name']}（{lowest_vol['Ticker']}，"
            f"日标准差 {lowest_vol['Std']:.4%}），相对更平稳。"
        ),
        (
            f"偏度方面，正偏股票包括：{right_text}；负偏股票包括：{left_text}。"
            "正偏表示右尾更长，负偏表示左尾更长，反映极端上涨或极端下跌日的相对影响。"
        ),
        (
            f"峰度最高的是 {high_kurtosis['Name']}（{high_kurtosis['Ticker']}，"
            f"峰度 {high_kurtosis['Kurtosis']:.2f}），说明其收益率分布尾部更厚，"
            "出现极端收益的概率相对更高。"
        ),
    ]


def describe_correlation(corr_matrix):
    corr_pairs = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack()
    highest_pair = corr_pairs.sort_values(ascending=False).index[0]
    lowest_pair = corr_pairs.sort_values(ascending=True).index[0]
    highest_corr = corr_pairs.loc[highest_pair]
    lowest_corr = corr_pairs.loc[lowest_pair]
    average_corr = corr_pairs.mean()

    high_a, high_b = highest_pair
    low_a, low_b = lowest_pair
    return [
        (
            f"10 支股票两两平均相关系数为 {average_corr:.3f}。"
            "相关系数整体越低，组合分散化带来的风险下降空间越大。"
        ),
        (
            f"相关性最高的一组是 {TICKER_NAMES[high_a]}（{high_a}）与 "
            f"{TICKER_NAMES[high_b]}（{high_b}），相关系数为 {highest_corr:.3f}，"
            "两者同涨同跌特征更明显，放在同一组合中分散化效果较弱。"
        ),
        (
            f"相关性最低的一组是 {TICKER_NAMES[low_a]}（{low_a}）与 "
            f"{TICKER_NAMES[low_b]}（{low_b}），相关系数为 {lowest_corr:.3f}，"
            "组合配置时更可能提供分散化收益。"
        ),
        (
            "从相关性热图可以观察到，多数股票仍存在正相关，说明共同市场因素会同时影响这些股票；"
            "但相关系数没有全部接近 1，因此跨行业配置仍能在一定程度上降低非系统性风险。"
        ),
    ]


def build_interpretation(stats, corr_matrix):
    lines = [
        "描述性统计与相关性分析：",
        "",
        "一、收益率分布与波动性",
        *describe_distribution(stats),
        "",
        "二、相关性与分散化效应",
        *describe_correlation(corr_matrix),
        "",
        "三、图表说明",
        f"收盘价折线图：{CLOSE_PRICE_FIG_PATH}",
        f"收益率箱线图：{RETURN_BOXPLOT_PATH}",
        f"相关性热图：{CORRELATION_HEATMAP_PATH}",
    ]
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    close_price = load_close_price()
    returns = load_stock_returns(close_price)

    descriptive_stats = calculate_descriptive_statistics(returns)
    covariance_matrix = returns[STOCKS].cov()
    correlation_matrix = returns[STOCKS].corr()
    interpretation = build_interpretation(descriptive_stats, correlation_matrix)

    descriptive_stats.to_csv(DESCRIPTIVE_STATS_PATH, index=False)
    covariance_matrix.to_csv(COVARIANCE_MATRIX_PATH)
    correlation_matrix.to_csv(CORRELATION_MATRIX_PATH)
    INTERPRETATION_PATH.write_text(interpretation, encoding="utf-8")

    with pd.ExcelWriter(EXCEL_OUTPUT_PATH) as writer:
        descriptive_stats.to_excel(writer, sheet_name="Descriptive Stats", index=False)
        covariance_matrix.to_excel(writer, sheet_name="Covariance Matrix")
        correlation_matrix.to_excel(writer, sheet_name="Correlation Matrix")

    plot_close_prices(close_price)
    plot_return_boxplot(returns)
    plot_correlation_heatmap(correlation_matrix)

    print("描述性统计结果预览：")
    display(descriptive_stats)
    print("\n协方差矩阵预览：")
    display(covariance_matrix)
    print("\n相关系数矩阵预览：")
    display(correlation_matrix)
    print("\n" + interpretation)
    print("\n文件已保存到：")
    print(f"- {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
