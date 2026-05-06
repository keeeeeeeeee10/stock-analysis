from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


BACKTEST_START = "2025-01-01"
BACKTEST_END = "2025-12-31"
INITIAL_DATE_CUTOFF = "2024-12-31"
TRADING_DAYS = 252
RISK_FREE_RATE = 0.015

INPUT_DIR = Path("output")
ILP_DIR = INPUT_DIR / "portfolio_ilp_allocation"
MARKOWITZ_DIR = INPUT_DIR / "markowitz_optimization"
DESCRIPTIVE_DIR = INPUT_DIR / "descriptive_stats"
OUTPUT_DIR = INPUT_DIR / "backtest_analysis"

CLOSE_PRICE_PATH = INPUT_DIR / "close_price_2022_2025.csv"
ILP_ALLOCATION_PATH = ILP_DIR / "portfolio_ilp_allocation_result.csv"
MARKOWITZ_WEIGHTS_PATH = MARKOWITZ_DIR / "markowitz_portfolio_weights.csv"
SINGLE_STOCK_STATS_PATH = DESCRIPTIVE_DIR / "stock_return_descriptive_statistics.csv"

DAILY_RETURNS_PATH = OUTPUT_DIR / "portfolio_backtest_daily_returns_2025.csv"
NAV_PATH = OUTPUT_DIR / "portfolio_backtest_net_value_2025.csv"
DRAWDOWN_PATH = OUTPUT_DIR / "portfolio_backtest_drawdowns_2025.csv"
PERFORMANCE_METRICS_PATH = OUTPUT_DIR / "portfolio_backtest_performance_metrics.csv"
PNL_METRICS_PATH = OUTPUT_DIR / "portfolio_backtest_profit_loss_metrics.csv"
SKEW_KURTOSIS_PATH = OUTPUT_DIR / "portfolio_backtest_skew_kurtosis_comparison.csv"
PORTFOLIO_WEIGHTS_PATH = OUTPUT_DIR / "portfolio_backtest_weights.csv"
REPORT_PATH = OUTPUT_DIR / "portfolio_backtest_report.txt"
EXCEL_OUTPUT_PATH = OUTPUT_DIR / "portfolio_backtest_analysis_2025.xlsx"

NAV_FIG_PATH = OUTPUT_DIR / "portfolio_cumulative_net_value.png"
DRAWDOWN_FIG_PATH = OUTPUT_DIR / "portfolio_dynamic_drawdown.png"
HIST_KDE_FIG_PATH = OUTPUT_DIR / "portfolio_return_hist_kde.png"
MONTHLY_BOXPLOT_FIG_PATH = OUTPUT_DIR / "portfolio_monthly_return_boxplot.png"
SKEW_KURT_FIG_PATH = OUTPUT_DIR / "portfolio_vs_single_stock_skew_kurtosis.png"

PORTFOLIO_NAME_MAP = {
    "Minimum Variance": "Minimum Variance",
    "Maximum Sharpe": "Maximum Sharpe",
    "Equal Weight": "Equal Weight",
}


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


def load_inputs():
    close_price = read_indexed_csv(CLOSE_PRICE_PATH)

    if not ILP_ALLOCATION_PATH.exists():
        raise FileNotFoundError(f"找不到任务（四）整数规划结果：{ILP_ALLOCATION_PATH}")
    if not MARKOWITZ_WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"找不到任务（五）Markowitz 权重结果：{MARKOWITZ_WEIGHTS_PATH}")

    ilp_allocation = pd.read_csv(ILP_ALLOCATION_PATH)
    markowitz_weights = pd.read_csv(MARKOWITZ_WEIGHTS_PATH)
    single_stock_stats = (
        pd.read_csv(SINGLE_STOCK_STATS_PATH)
        if SINGLE_STOCK_STATS_PATH.exists()
        else pd.DataFrame()
    )
    return close_price, ilp_allocation, markowitz_weights, single_stock_stats


def get_backtest_price_window(close_price, tickers):
    prices = close_price[tickers].apply(pd.to_numeric, errors="coerce").ffill()
    initial_candidates = prices.loc[:INITIAL_DATE_CUTOFF]
    if initial_candidates.empty:
        raise ValueError(f"找不到 {INITIAL_DATE_CUTOFF} 或之前的初始建仓价格。")
    initial_date = initial_candidates.index.max()

    backtest_prices = prices.loc[BACKTEST_START:BACKTEST_END]
    if backtest_prices.empty:
        raise ValueError(f"找不到 {BACKTEST_START} 至 {BACKTEST_END} 的回测价格。")

    price_window = pd.concat([prices.loc[[initial_date]], backtest_prices])
    price_window = price_window[~price_window.index.duplicated(keep="first")]
    return price_window, initial_date


def build_portfolio_weights(ilp_allocation, markowitz_weights):
    tickers = ilp_allocation["Ticker"].tolist()
    rows = []

    ilp_positive = ilp_allocation.copy()
    total_investment = ilp_positive["Investment"].sum()
    for _, row in ilp_positive.iterrows():
        weight = row["Investment"] / total_investment if total_investment > 0 else 0.0
        rows.append({
            "Portfolio": "Integer ILP",
            "Ticker": row["Ticker"],
            "Name": row["Name"],
            "Weight": weight,
            "Shares": int(row["Shares"]),
        })

    for portfolio_name, group in markowitz_weights.groupby("Portfolio", sort=False):
        for _, row in group.iterrows():
            rows.append({
                "Portfolio": PORTFOLIO_NAME_MAP.get(portfolio_name, portfolio_name),
                "Ticker": row["Ticker"],
                "Name": row["Name"],
                "Weight": float(row["Weight"]),
                "Shares": np.nan,
            })

    weights = pd.DataFrame(rows)
    weights = weights[weights["Ticker"].isin(tickers)].copy()
    return weights


def calculate_backtest_series(price_window, ilp_allocation, weights_table):
    tickers = ilp_allocation["Ticker"].tolist()
    simple_returns = price_window[tickers].pct_change().dropna()
    simple_returns = simple_returns.loc[BACKTEST_START:BACKTEST_END]

    daily_returns = pd.DataFrame(index=simple_returns.index)
    net_values = pd.DataFrame(index=price_window.index)

    shares = ilp_allocation.set_index("Ticker")["Shares"].reindex(tickers).fillna(0)
    ilp_value = price_window[tickers].mul(shares, axis=1).sum(axis=1)
    if ilp_value.iloc[0] <= 0:
        raise ValueError("整数规划组合初始市值为 0，无法回测。")
    daily_returns["Integer ILP"] = ilp_value.pct_change().loc[simple_returns.index]
    net_values["Integer ILP"] = ilp_value / ilp_value.iloc[0]

    for portfolio in ["Maximum Sharpe", "Minimum Variance", "Equal Weight"]:
        weights = (
            weights_table[weights_table["Portfolio"] == portfolio]
            .set_index("Ticker")["Weight"]
            .reindex(tickers)
            .fillna(0)
        )
        if not np.isclose(weights.sum(), 1.0):
            total = weights.sum()
            if total <= 0:
                raise ValueError(f"{portfolio} 权重之和为 0，无法回测。")
            weights = weights / total

        daily_returns[portfolio] = simple_returns.mul(weights, axis=1).sum(axis=1)
        nav = (1 + daily_returns[portfolio]).cumprod()
        net_values[portfolio] = pd.concat([
            pd.Series([1.0], index=[price_window.index[0]]),
            nav,
        ]).sort_index()

    daily_returns = daily_returns.dropna(how="all")
    net_values = net_values.ffill()
    drawdowns = net_values / net_values.cummax() - 1
    return daily_returns, net_values, drawdowns


def calculate_performance_metrics(daily_returns, net_values, drawdowns):
    rows = []
    n_days = len(daily_returns)
    for portfolio in daily_returns.columns:
        final_nav = net_values[portfolio].iloc[-1]
        annual_return = final_nav ** (TRADING_DAYS / n_days) - 1
        annual_volatility = daily_returns[portfolio].std(ddof=1) * np.sqrt(TRADING_DAYS)
        sharpe = (
            (annual_return - RISK_FREE_RATE) / annual_volatility
            if annual_volatility > 0
            else np.nan
        )
        max_drawdown = -drawdowns[portfolio].min()
        calmar = annual_return / max_drawdown if max_drawdown > 0 else np.nan
        rows.append({
            "Portfolio": portfolio,
            "Trading Days": n_days,
            "Final Net Value": final_nav,
            "Annual Return": annual_return,
            "Annual Volatility": annual_volatility,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_drawdown,
            "Calmar Ratio": calmar,
        })
    return pd.DataFrame(rows).sort_values("Final Net Value", ascending=False)


def calculate_profit_loss_metrics(daily_returns):
    rows = []
    for portfolio in daily_returns.columns:
        returns = daily_returns[portfolio].dropna()
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        win_ratio = len(wins) / len(returns) if len(returns) else np.nan
        avg_win = wins.mean() if len(wins) else np.nan
        avg_loss = abs(losses.mean()) if len(losses) else np.nan
        win_loss_ratio = avg_win / avg_loss if avg_loss and avg_loss > 0 else np.nan
        rows.append({
            "Portfolio": portfolio,
            "Total Days": len(returns),
            "Profit Days": len(wins),
            "Loss Days": len(losses),
            "Profit Day Ratio": win_ratio,
            "Average Profit": avg_win,
            "Average Loss Abs": avg_loss,
            "Profit/Loss Ratio": win_loss_ratio,
        })
    return pd.DataFrame(rows).sort_values("Profit/Loss Ratio", ascending=False)


def calculate_skew_kurtosis_comparison(daily_returns, single_stock_stats):
    portfolio_rows = []
    for portfolio in daily_returns.columns:
        returns = daily_returns[portfolio].dropna()
        portfolio_rows.append({
            "Type": "Portfolio",
            "Name": portfolio,
            "Skewness": returns.skew(),
            "Kurtosis": returns.kurt(),
        })

    rows = portfolio_rows
    if not single_stock_stats.empty and {"Ticker", "Name", "Skewness", "Kurtosis"}.issubset(single_stock_stats.columns):
        for _, row in single_stock_stats.iterrows():
            rows.append({
                "Type": "Single Stock",
                "Name": f"{row['Ticker']} {row['Name']}",
                "Skewness": row["Skewness"],
                "Kurtosis": row["Kurtosis"],
            })
        rows.append({
            "Type": "Single Stock Benchmark",
            "Name": "Single Stock Average",
            "Skewness": single_stock_stats["Skewness"].mean(),
            "Kurtosis": single_stock_stats["Kurtosis"].mean(),
        })
        rows.append({
            "Type": "Single Stock Benchmark",
            "Name": "Single Stock Median",
            "Skewness": single_stock_stats["Skewness"].median(),
            "Kurtosis": single_stock_stats["Kurtosis"].median(),
        })

    return pd.DataFrame(rows)


def plot_net_values(net_values):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for column in net_values.columns:
        ax.plot(net_values.index, net_values[column], linewidth=1.8, label=column)
    ax.set_title("Portfolio Cumulative Net Value in 2025")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(NAV_FIG_PATH, bbox_inches="tight")
    plt.close(fig)


def plot_drawdowns(drawdowns):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for column in drawdowns.columns:
        ax.plot(drawdowns.index, drawdowns[column] * 100, linewidth=1.6, label=column)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title("Portfolio Dynamic Drawdown in 2025")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(DRAWDOWN_FIG_PATH, bbox_inches="tight")
    plt.close(fig)


def plot_histograms_with_kde(daily_returns):
    portfolios = list(daily_returns.columns)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes = axes.ravel()

    for ax, portfolio in zip(axes, portfolios):
        values = daily_returns[portfolio].dropna().values
        ax.hist(values * 100, bins=28, density=True, alpha=0.65, color="#93c5fd", edgecolor="#1d4ed8")
        if len(np.unique(values)) > 1:
            kde = gaussian_kde(values)
            xs = np.linspace(values.min(), values.max(), 240)
            ax.plot(xs * 100, kde(xs) / 100, color="#dc2626", linewidth=1.8, label="KDE")
        ax.axvline(0, color="#111827", linewidth=0.8)
        ax.set_title(portfolio)
        ax.set_xlabel("Daily return (%)")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.22)

    fig.suptitle("Daily Return Distribution with KDE", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HIST_KDE_FIG_PATH, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_boxplots(daily_returns):
    returns = daily_returns.copy()
    returns["Month"] = returns.index.to_period("M").astype(str)
    months = sorted(returns["Month"].unique())
    portfolios = [column for column in daily_returns.columns]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()
    for ax, portfolio in zip(axes, portfolios):
        data = [returns.loc[returns["Month"] == month, portfolio].dropna() * 100 for month in months]
        ax.boxplot(data, tick_labels=[month[-2:] for month in months], showfliers=True, patch_artist=True)
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.set_title(portfolio)
        ax.set_xlabel("Month in 2025")
        ax.set_ylabel("Daily return (%)")
        ax.grid(True, axis="y", alpha=0.22)

    fig.suptitle("Monthly Grouped Daily Return Boxplots", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(MONTHLY_BOXPLOT_FIG_PATH, bbox_inches="tight")
    plt.close(fig)


def plot_skew_kurtosis_comparison(comparison):
    portfolios = comparison[comparison["Type"] == "Portfolio"].copy()
    benchmark = comparison[comparison["Name"].isin(["Single Stock Average", "Single Stock Median"])].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    axes[0].bar(portfolios["Name"], portfolios["Skewness"], color="#2563eb", alpha=0.8)
    for _, row in benchmark.iterrows():
        axes[0].axhline(row["Skewness"], linestyle="--", linewidth=1.1, label=row["Name"])
    axes[0].set_title("Skewness: Portfolios vs Single Stocks")
    axes[0].tick_params(axis="x", rotation=28)
    axes[0].grid(True, axis="y", alpha=0.22)
    axes[0].legend()

    axes[1].bar(portfolios["Name"], portfolios["Kurtosis"], color="#16a34a", alpha=0.8)
    for _, row in benchmark.iterrows():
        axes[1].axhline(row["Kurtosis"], linestyle="--", linewidth=1.1, label=row["Name"])
    axes[1].set_title("Kurtosis: Portfolios vs Single Stocks")
    axes[1].tick_params(axis="x", rotation=28)
    axes[1].grid(True, axis="y", alpha=0.22)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(SKEW_KURT_FIG_PATH, bbox_inches="tight")
    plt.close(fig)


def build_report(performance, pnl_metrics, skew_kurtosis, weights_table):
    best_return = performance.sort_values("Annual Return", ascending=False).iloc[0]
    best_sharpe = performance.sort_values("Sharpe Ratio", ascending=False).iloc[0]
    best_drawdown = performance.sort_values("Max Drawdown", ascending=True).iloc[0]
    best_pnl = pnl_metrics.sort_values("Profit/Loss Ratio", ascending=False).iloc[0]

    portfolio_skew = skew_kurtosis[skew_kurtosis["Type"] == "Portfolio"]
    stock_average = skew_kurtosis[skew_kurtosis["Name"] == "Single Stock Average"]

    lines = [
        "组合回测与盈亏分析报告",
        "",
        f"回测期：{BACKTEST_START} 至 {BACKTEST_END}",
        f"无风险利率：{RISK_FREE_RATE:.2%}（年化，用于夏普比率）",
        "",
        "一、组合构造",
        "整数规划组合来自任务（四）的实际股数，按 2024-12-31 收盘价作为初始建仓价计算每日市值。",
        "最大夏普、最小方差和等权重组合来自任务（五）的权重，按固定权重计算每日组合收益率。",
    ]

    for portfolio, group in weights_table.groupby("Portfolio", sort=False):
        nonzero = group[group["Weight"] > 1e-8].sort_values("Weight", ascending=False)
        weight_text = "、".join(f"{row['Ticker']} {row['Weight']:.2%}" for _, row in nonzero.iterrows())
        lines.append(f"- {portfolio}：{weight_text}")

    lines.extend([
        "",
        "二、收益与风险指标",
        (
            f"年化收益率最高的是 {best_return['Portfolio']}，"
            f"年化收益率为 {best_return['Annual Return']:.2%}。"
        ),
        (
            f"夏普比率最高的是 {best_sharpe['Portfolio']}，"
            f"夏普比率为 {best_sharpe['Sharpe Ratio']:.4f}。"
        ),
        (
            f"最大回撤最小的是 {best_drawdown['Portfolio']}，"
            f"最大回撤为 {best_drawdown['Max Drawdown']:.2%}。"
        ),
        (
            "若某组合集中持有同一只股票，则其收益和回撤会高度接近该股票在 2025 年的表现；"
            "分散化组合通常会牺牲部分上行收益，但可能改善回撤和波动。"
        ),
        "",
        "三、盈亏结构",
        (
            f"盈亏比最高的是 {best_pnl['Portfolio']}，盈亏比为 {best_pnl['Profit/Loss Ratio']:.4f}，"
            f"盈利天数占比为 {best_pnl['Profit Day Ratio']:.2%}。"
        ),
        "盈利天数占比反映上涨日频率，盈亏比反映平均上涨幅度与平均下跌幅度的相对关系，两者需要结合观察。",
        "",
        "四、偏度与峰度",
    ])

    if not stock_average.empty:
        avg = stock_average.iloc[0]
        lines.append(
            f"任务（二）中单股票平均偏度为 {avg['Skewness']:.4f}，平均峰度为 {avg['Kurtosis']:.4f}。"
        )
    for _, row in portfolio_skew.iterrows():
        lines.append(
            f"- {row['Name']}：偏度 {row['Skewness']:.4f}，峰度 {row['Kurtosis']:.4f}。"
        )

    lines.extend([
        "偏度为正表示右尾更长，可能存在较大的单日正收益；峰度越高表示尾部越厚，极端收益出现概率越高。",
        "",
        "五、输出图表",
        f"累计净值曲线：{NAV_FIG_PATH}",
        f"动态回撤曲线：{DRAWDOWN_FIG_PATH}",
        f"收益率直方图与核密度曲线：{HIST_KDE_FIG_PATH}",
        f"按月分组箱线图：{MONTHLY_BOXPLOT_FIG_PATH}",
        f"偏度峰度对比图：{SKEW_KURT_FIG_PATH}",
    ])
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    close_price, ilp_allocation, markowitz_weights, single_stock_stats = load_inputs()
    tickers = ilp_allocation["Ticker"].tolist()
    price_window, initial_date = get_backtest_price_window(close_price, tickers)
    weights_table = build_portfolio_weights(ilp_allocation, markowitz_weights)
    daily_returns, net_values, drawdowns = calculate_backtest_series(price_window, ilp_allocation, weights_table)

    performance = calculate_performance_metrics(daily_returns, net_values, drawdowns)
    pnl_metrics = calculate_profit_loss_metrics(daily_returns)
    skew_kurtosis = calculate_skew_kurtosis_comparison(daily_returns, single_stock_stats)
    report = build_report(performance, pnl_metrics, skew_kurtosis, weights_table)

    daily_returns.to_csv(DAILY_RETURNS_PATH)
    net_values.to_csv(NAV_PATH)
    drawdowns.to_csv(DRAWDOWN_PATH)
    performance.to_csv(PERFORMANCE_METRICS_PATH, index=False)
    pnl_metrics.to_csv(PNL_METRICS_PATH, index=False)
    skew_kurtosis.to_csv(SKEW_KURTOSIS_PATH, index=False)
    weights_table.to_csv(PORTFOLIO_WEIGHTS_PATH, index=False)
    REPORT_PATH.write_text(report, encoding="utf-8")

    with pd.ExcelWriter(EXCEL_OUTPUT_PATH) as writer:
        daily_returns.to_excel(writer, sheet_name="Daily Returns")
        net_values.to_excel(writer, sheet_name="Net Value")
        drawdowns.to_excel(writer, sheet_name="Drawdowns")
        performance.to_excel(writer, sheet_name="Performance Metrics", index=False)
        pnl_metrics.to_excel(writer, sheet_name="Profit Loss Metrics", index=False)
        skew_kurtosis.to_excel(writer, sheet_name="Skew Kurtosis", index=False)
        weights_table.to_excel(writer, sheet_name="Portfolio Weights", index=False)

    plot_net_values(net_values)
    plot_drawdowns(drawdowns)
    plot_histograms_with_kde(daily_returns)
    plot_monthly_boxplots(daily_returns)
    plot_skew_kurtosis_comparison(skew_kurtosis)

    print(f"初始建仓日期：{initial_date.date()}")
    print("组合权重：")
    display(weights_table)
    print("\n收益风险指标：")
    display(performance)
    print("\n盈亏指标：")
    display(pnl_metrics)
    print("\n偏度峰度对比：")
    display(skew_kurtosis.head(20))
    print("\n" + report)
    print("\n文件已保存到：")
    print(f"- {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
