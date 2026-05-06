from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from matplotlib.ticker import PercentFormatter


TRAIN_START = "2022-01-01"
TRAIN_END = "2024-12-31"
TRADING_DAYS = 252
RISK_FREE_RATE = 0.015

INPUT_DIR = Path("output")
THREE_FACTOR_DIR = INPUT_DIR / "three_factor_selection"
OUTPUT_DIR = INPUT_DIR / "markowitz_optimization"

SELECTED_STOCKS_PATH = THREE_FACTOR_DIR / "selected_stocks_by_three_factor_alpha.csv"
STOCK_RETURN_PATH = INPUT_DIR / "stock_log_returns_2022_2025.csv"
INDEX_RETURN_PATH = INPUT_DIR / "index_log_returns_2022_2025.csv"

ASSET_STATS_PATH = OUTPUT_DIR / "markowitz_asset_inputs.csv"
PORTFOLIO_WEIGHTS_PATH = OUTPUT_DIR / "markowitz_portfolio_weights.csv"
PORTFOLIO_SUMMARY_PATH = OUTPUT_DIR / "markowitz_portfolio_summary.csv"
EFFICIENT_FRONTIER_PATH = OUTPUT_DIR / "efficient_frontier_points.csv"
REPORT_PATH = OUTPUT_DIR / "markowitz_optimization_report.txt"
EXCEL_OUTPUT_PATH = OUTPUT_DIR / "markowitz_optimization_2022_2024.xlsx"

EFFICIENT_FRONTIER_FIG_PATH = OUTPUT_DIR / "efficient_frontier.png"
WEIGHTS_FIG_PATH = OUTPUT_DIR / "portfolio_weights_comparison.png"


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
    return df.set_index("Date").sort_index()


def load_selected_stocks():
    if not SELECTED_STOCKS_PATH.exists():
        raise FileNotFoundError(
            f"找不到任务（三）的选股结果：{SELECTED_STOCKS_PATH}。请先运行 src/three_factor_stock_selection.py。"
        )
    selected = pd.read_csv(SELECTED_STOCKS_PATH).head(5)
    if len(selected) < 5:
        raise ValueError("任务（三）的选股结果少于 5 支股票，无法执行 Markowitz 优化。")
    return selected[["Ticker", "Name"]].copy()


def load_training_returns(selected):
    tickers = selected["Ticker"].tolist()
    returns = read_indexed_csv(STOCK_RETURN_PATH)
    missing = [ticker for ticker in tickers if ticker not in returns.columns]
    if missing:
        raise ValueError(f"股票日收益率文件缺少列：{missing}")

    stock_returns = returns.loc[TRAIN_START:TRAIN_END, tickers].apply(pd.to_numeric, errors="coerce")
    stock_returns = stock_returns.dropna(how="all").dropna(axis=0)

    # 同期指数收益率只用于保持任务口径一致，并在报告中记录样本区间。
    index_returns = read_indexed_csv(INDEX_RETURN_PATH) if INDEX_RETURN_PATH.exists() else pd.DataFrame()
    if not index_returns.empty and "CSI300" in index_returns.columns:
        index_returns = index_returns.loc[TRAIN_START:TRAIN_END, ["CSI300"]].apply(pd.to_numeric, errors="coerce")

    return stock_returns, index_returns


def annualize_inputs(stock_returns):
    expected_returns = stock_returns.mean() * TRADING_DAYS
    covariance = stock_returns.cov() * TRADING_DAYS
    return expected_returns, covariance


def portfolio_return(weights, expected_returns):
    return float(np.dot(weights, expected_returns))


def portfolio_variance(weights, covariance):
    return float(weights @ covariance @ weights)


def portfolio_volatility(weights, covariance):
    variance = portfolio_variance(weights, covariance)
    return float(np.sqrt(max(variance, 0.0)))


def sharpe_ratio(weights, expected_returns, covariance):
    vol = portfolio_volatility(weights, covariance)
    if vol <= 0:
        return np.nan
    return (portfolio_return(weights, expected_returns) - RISK_FREE_RATE) / vol


def optimize_min_variance(expected_returns, covariance):
    n = len(expected_returns)
    initial = np.repeat(1 / n, n)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n

    result = minimize(
        lambda w: portfolio_variance(w, covariance.values),
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"最小方差组合求解失败：{result.message}")
    return np.clip(result.x, 0, 1)


def optimize_max_sharpe(expected_returns, covariance):
    n = len(expected_returns)
    initial = np.repeat(1 / n, n)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n

    def objective(weights):
        ratio = sharpe_ratio(weights, expected_returns.values, covariance.values)
        if np.isnan(ratio):
            return 1e6
        return -ratio

    best_result = None
    starts = [initial]
    starts.extend(np.eye(n))
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if result.success and (best_result is None or result.fun < best_result.fun):
            best_result = result

    if best_result is None:
        raise RuntimeError("最大夏普比率组合求解失败。")
    return np.clip(best_result.x, 0, 1)


def optimize_for_target_return(target_return, expected_returns, covariance):
    n = len(expected_returns)
    initial = np.repeat(1 / n, n)
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w, target=target_return: np.dot(w, expected_returns.values) - target},
    ]
    bounds = [(0, 1)] * n
    result = minimize(
        lambda w: portfolio_variance(w, covariance.values),
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        return None
    weights = np.clip(result.x, 0, 1)
    return weights / weights.sum()


def calculate_portfolio_metrics(name, weights, expected_returns, covariance):
    ret = portfolio_return(weights, expected_returns.values)
    vol = portfolio_volatility(weights, covariance.values)
    sharpe = (ret - RISK_FREE_RATE) / vol if vol > 0 else np.nan
    return {
        "Portfolio": name,
        "Expected Annual Return": ret,
        "Annual Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Risk Free Rate": RISK_FREE_RATE,
    }


def build_weights_table(selected, portfolio_weights):
    rows = []
    for portfolio_name, weights in portfolio_weights.items():
        for ticker, stock_name, weight in zip(selected["Ticker"], selected["Name"], weights):
            rows.append({
                "Portfolio": portfolio_name,
                "Ticker": ticker,
                "Name": stock_name,
                "Weight": weight,
            })
    return pd.DataFrame(rows)


def build_asset_stats(selected, stock_returns, expected_returns, covariance):
    rows = []
    for ticker, stock_name in zip(selected["Ticker"], selected["Name"]):
        rows.append({
            "Ticker": ticker,
            "Name": stock_name,
            "Trading Days": int(stock_returns[ticker].count()),
            "Expected Annual Return": expected_returns[ticker],
            "Annual Volatility": float(np.sqrt(covariance.loc[ticker, ticker])),
        })
    return pd.DataFrame(rows)


def build_efficient_frontier(expected_returns, covariance, min_variance_weights):
    min_variance_return = portfolio_return(min_variance_weights, expected_returns.values)
    max_asset_return = float(expected_returns.max())
    target_returns = np.linspace(min_variance_return, max_asset_return, 120)

    rows = []
    for target in target_returns:
        weights = optimize_for_target_return(target, expected_returns, covariance)
        if weights is None:
            continue
        ret = portfolio_return(weights, expected_returns.values)
        vol = portfolio_volatility(weights, covariance.values)
        sharpe = (ret - RISK_FREE_RATE) / vol if vol > 0 else np.nan
        row = {
            "Target Return": target,
            "Expected Annual Return": ret,
            "Annual Volatility": vol,
            "Sharpe Ratio": sharpe,
        }
        for ticker, weight in zip(expected_returns.index, weights):
            row[f"Weight_{ticker}"] = weight
        rows.append(row)

    return pd.DataFrame(rows)


def plot_efficient_frontier(frontier, asset_stats, portfolio_summary):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(
        frontier["Annual Volatility"] * 100,
        frontier["Expected Annual Return"] * 100,
        color="#2563eb",
        linewidth=2,
        label="Efficient frontier",
    )

    ax.scatter(
        asset_stats["Annual Volatility"] * 100,
        asset_stats["Expected Annual Return"] * 100,
        color="#64748b",
        s=44,
        label="Individual stocks",
        zorder=3,
    )
    for _, row in asset_stats.iterrows():
        ax.annotate(row["Ticker"], (row["Annual Volatility"] * 100, row["Expected Annual Return"] * 100), fontsize=8, xytext=(4, 4), textcoords="offset points")

    marker_styles = {
        "Minimum Variance": ("#16a34a", "o"),
        "Maximum Sharpe": ("#dc2626", "*"),
        "Equal Weight": ("#f59e0b", "D"),
    }
    for _, row in portfolio_summary.iterrows():
        color, marker = marker_styles[row["Portfolio"]]
        ax.scatter(
            row["Annual Volatility"] * 100,
            row["Expected Annual Return"] * 100,
            color=color,
            marker=marker,
            s=130 if marker == "*" else 80,
            label=row["Portfolio"],
            edgecolors="#111827",
            linewidths=0.6,
            zorder=4,
        )
        ax.annotate(
            row["Portfolio"],
            (row["Annual Volatility"] * 100, row["Expected Annual Return"] * 100),
            fontsize=9,
            xytext=(6, 6),
            textcoords="offset points",
        )

    ax.set_title("Markowitz Efficient Frontier (No Short Selling, 2022-2024)")
    ax.set_xlabel("Annual volatility (%)")
    ax.set_ylabel("Expected annual return (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(EFFICIENT_FRONTIER_FIG_PATH, bbox_inches="tight")
    plt.close(fig)


def plot_weights(weights_table):
    pivot = weights_table.pivot(index="Ticker", columns="Portfolio", values="Weight")
    order = ["Minimum Variance", "Maximum Sharpe", "Equal Weight"]
    pivot = pivot[order]

    fig, ax = plt.subplots(figsize=(10, 6.2))
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title("Portfolio Weights Comparison")
    ax.set_xlabel("Stock")
    ax.set_ylabel("Weight (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Portfolio")
    fig.tight_layout()
    fig.savefig(WEIGHTS_FIG_PATH, bbox_inches="tight")
    plt.close(fig)


def build_report(selected, asset_stats, portfolio_summary, weights_table):
    max_sharpe = portfolio_summary.set_index("Portfolio").loc["Maximum Sharpe"]
    min_var = portfolio_summary.set_index("Portfolio").loc["Minimum Variance"]
    equal = portfolio_summary.set_index("Portfolio").loc["Equal Weight"]

    max_sharpe_weights = (
        weights_table[weights_table["Portfolio"] == "Maximum Sharpe"]
        .sort_values("Weight", ascending=False)
    )
    min_var_weights = (
        weights_table[weights_table["Portfolio"] == "Minimum Variance"]
        .sort_values("Weight", ascending=False)
    )

    lines = [
        "均值-方差投资组合优化（Markowitz 模型）报告",
        "",
        f"训练期：{TRAIN_START} 至 {TRAIN_END}",
        f"无风险利率：{RISK_FREE_RATE:.2%}（年化）",
        f"卖空约束：不允许卖空，即所有权重 >= 0 且权重之和为 1。",
        f"备选股票：{'、'.join(selected['Ticker'].tolist())}",
        "",
        "一、资产输入",
        "预期收益率采用训练期日对数收益率均值乘以 252，协方差矩阵采用日收益率协方差乘以 252。",
    ]

    for _, row in asset_stats.sort_values("Expected Annual Return", ascending=False).iterrows():
        lines.append(
            f"- {row['Ticker']} {row['Name']}：预期年化收益率 {row['Expected Annual Return']:.2%}，"
            f"年化波动率 {row['Annual Volatility']:.2%}。"
        )

    lines.extend([
        "",
        "二、组合结果",
        (
            f"最小方差组合：预期收益率 {min_var['Expected Annual Return']:.2%}，"
            f"年化波动率 {min_var['Annual Volatility']:.2%}，夏普比率 {min_var['Sharpe Ratio']:.4f}。"
        ),
        (
            f"最大夏普比率组合：预期收益率 {max_sharpe['Expected Annual Return']:.2%}，"
            f"年化波动率 {max_sharpe['Annual Volatility']:.2%}，夏普比率 {max_sharpe['Sharpe Ratio']:.4f}。"
        ),
        (
            f"等权重组合：预期收益率 {equal['Expected Annual Return']:.2%}，"
            f"年化波动率 {equal['Annual Volatility']:.2%}，夏普比率 {equal['Sharpe Ratio']:.4f}。"
        ),
        "",
        "三、权重解释",
        "最大夏普组合主要配置在单位风险补偿更高的股票上；最小方差组合则更偏向低波动、低相关性的股票，以降低组合整体风险。",
    ])

    lines.append("最大夏普组合权重：")
    for _, row in max_sharpe_weights.iterrows():
        if row["Weight"] > 1e-6:
            lines.append(f"- {row['Ticker']} {row['Name']}：{row['Weight']:.2%}")

    lines.append("最小方差组合权重：")
    for _, row in min_var_weights.iterrows():
        if row["Weight"] > 1e-6:
            lines.append(f"- {row['Ticker']} {row['Name']}：{row['Weight']:.2%}")

    lines.extend([
        "",
        "四、图表输出",
        f"有效前沿图：{EFFICIENT_FRONTIER_FIG_PATH}",
        f"组合权重对比图：{WEIGHTS_FIG_PATH}",
    ])
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    selected = load_selected_stocks()
    stock_returns, index_returns = load_training_returns(selected)
    expected_returns, covariance = annualize_inputs(stock_returns)

    min_var_weights = optimize_min_variance(expected_returns, covariance)
    max_sharpe_weights = optimize_max_sharpe(expected_returns, covariance)
    equal_weights = np.repeat(1 / len(selected), len(selected))

    portfolio_weights = {
        "Minimum Variance": min_var_weights,
        "Maximum Sharpe": max_sharpe_weights,
        "Equal Weight": equal_weights,
    }

    asset_stats = build_asset_stats(selected, stock_returns, expected_returns, covariance)
    weights_table = build_weights_table(selected, portfolio_weights)
    portfolio_summary = pd.DataFrame([
        calculate_portfolio_metrics(name, weights, expected_returns, covariance)
        for name, weights in portfolio_weights.items()
    ])
    frontier = build_efficient_frontier(expected_returns, covariance, min_var_weights)
    report = build_report(selected, asset_stats, portfolio_summary, weights_table)

    asset_stats.to_csv(ASSET_STATS_PATH, index=False)
    weights_table.to_csv(PORTFOLIO_WEIGHTS_PATH, index=False)
    portfolio_summary.to_csv(PORTFOLIO_SUMMARY_PATH, index=False)
    frontier.to_csv(EFFICIENT_FRONTIER_PATH, index=False)
    REPORT_PATH.write_text(report, encoding="utf-8")

    with pd.ExcelWriter(EXCEL_OUTPUT_PATH) as writer:
        asset_stats.to_excel(writer, sheet_name="Asset Inputs", index=False)
        weights_table.to_excel(writer, sheet_name="Portfolio Weights", index=False)
        portfolio_summary.to_excel(writer, sheet_name="Portfolio Summary", index=False)
        covariance.to_excel(writer, sheet_name="Annual Covariance")
        frontier.to_excel(writer, sheet_name="Efficient Frontier", index=False)
        if not index_returns.empty:
            index_returns.to_excel(writer, sheet_name="Index Returns")

    plot_efficient_frontier(frontier, asset_stats, portfolio_summary)
    plot_weights(weights_table)

    print("资产输入：")
    display(asset_stats)
    print("\n组合权重：")
    display(weights_table)
    print("\n组合绩效：")
    display(portfolio_summary)
    print("\n" + report)
    print("\n文件已保存到：")
    print(f"- {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
