from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp


TRAIN_START = "2022-01-01"
TRAIN_END = "2024-12-31"

BUDGET = 2_000_000.0
LOT_SIZE = 100
MIN_LOTS = 20
BETA_LIMIT = 1.0

INPUT_DIR = Path("output")
THREE_FACTOR_DIR = INPUT_DIR / "three_factor_selection"
OUTPUT_DIR = INPUT_DIR / "portfolio_ilp_allocation"

SELECTED_STOCKS_PATH = THREE_FACTOR_DIR / "selected_stocks_by_three_factor_alpha.csv"
CAPM_RESULTS_PATH = THREE_FACTOR_DIR / "capm_regression_results.csv"
CLOSE_PRICE_PATH = INPUT_DIR / "close_price_2022_2025.csv"
STOCK_RETURN_PATH = INPUT_DIR / "stock_log_returns_2022_2025.csv"

STOCK_INPUTS_PATH = OUTPUT_DIR / "portfolio_ilp_stock_inputs.csv"
STRICT_DIAGNOSTICS_PATH = OUTPUT_DIR / "strict_constraint_diagnostics.csv"
ALLOCATION_RESULT_PATH = OUTPUT_DIR / "portfolio_ilp_allocation_result.csv"
PORTFOLIO_SUMMARY_PATH = OUTPUT_DIR / "portfolio_ilp_summary.csv"
REPORT_PATH = OUTPUT_DIR / "portfolio_ilp_report.txt"
EXCEL_OUTPUT_PATH = OUTPUT_DIR / "portfolio_ilp_allocation_2022_2024.xlsx"

MIN_INVESTMENT_FIG_PATH = OUTPUT_DIR / "minimum_investment_vs_budget.png"
ALLOCATION_BAR_FIG_PATH = OUTPUT_DIR / "portfolio_allocation_bar.png"
WEIGHT_BETA_FIG_PATH = OUTPUT_DIR / "portfolio_weight_beta.png"


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


def safe_name(ticker):
    return ticker.replace(".", "_").replace("-", "_")


def load_selected_tickers():
    if not SELECTED_STOCKS_PATH.exists():
        raise FileNotFoundError(
            f"找不到任务（三）的选股结果：{SELECTED_STOCKS_PATH}。请先运行 src/three_factor_stock_selection.py。"
        )
    selected = pd.read_csv(SELECTED_STOCKS_PATH).head(5)
    if len(selected) < 5:
        raise ValueError("任务（三）的选股结果少于 5 支股票，无法执行本任务。")
    return selected[["Ticker", "Name"]].copy()


def load_inputs():
    selected = load_selected_tickers()
    tickers = selected["Ticker"].tolist()

    close_price = read_indexed_csv(CLOSE_PRICE_PATH)
    stock_returns = read_indexed_csv(STOCK_RETURN_PATH)
    capm = pd.read_csv(CAPM_RESULTS_PATH)

    price_data = close_price.loc[:TRAIN_END, tickers].dropna(how="all")
    price_date = price_data.index.max()
    prices = price_data.loc[price_date, tickers]

    return_data = stock_returns.loc[TRAIN_START:TRAIN_END, tickers].apply(pd.to_numeric, errors="coerce")
    annual_returns = return_data.mean(skipna=True) * 252
    trading_days = return_data.count()

    beta_map = capm.set_index("Ticker")["Beta Market"]

    rows = []
    for _, row in selected.iterrows():
        ticker = row["Ticker"]
        price = float(prices[ticker])
        beta = float(beta_map.loc[ticker])
        annual_return = float(annual_returns[ticker])
        lot_cost = price * LOT_SIZE
        min_cost = lot_cost * MIN_LOTS
        rows.append({
            "Ticker": ticker,
            "Name": row["Name"],
            "Price Date": price_date.date().isoformat(),
            "Close Price": price,
            "Trading Days": int(trading_days[ticker]),
            "Expected Annual Log Return": annual_return,
            "CAPM Beta": beta,
            "Lot Size": LOT_SIZE,
            "Minimum Lots": MIN_LOTS,
            "Lot Cost": lot_cost,
            "Minimum Investment": min_cost,
            "Expected Profit per Lot": lot_cost * annual_return,
        })

    return pd.DataFrame(rows)


def solve_ilp(stock_inputs, require_all=True):
    model_name = "strict_all_stock_ilp" if require_all else "relaxed_subset_ilp"
    model = pulp.LpProblem(model_name, pulp.LpMaximize)

    lots = {}
    include = {}
    for _, row in stock_inputs.iterrows():
        ticker = row["Ticker"]
        max_lots = int(BUDGET // row["Lot Cost"])
        if require_all:
            lots[ticker] = pulp.LpVariable(f"lots_{safe_name(ticker)}", lowBound=MIN_LOTS, cat="Integer")
        else:
            lots[ticker] = pulp.LpVariable(f"lots_{safe_name(ticker)}", lowBound=0, cat="Integer")
            include[ticker] = pulp.LpVariable(f"include_{safe_name(ticker)}", lowBound=0, upBound=1, cat="Binary")
            model += lots[ticker] >= MIN_LOTS * include[ticker], f"min_lots_if_included_{safe_name(ticker)}"
            model += lots[ticker] <= max_lots * include[ticker], f"max_lots_if_included_{safe_name(ticker)}"

    if not require_all:
        model += pulp.lpSum(include.values()) >= 1, "at_least_one_stock"

    lot_cost = stock_inputs.set_index("Ticker")["Lot Cost"].to_dict()
    annual_return = stock_inputs.set_index("Ticker")["Expected Annual Log Return"].to_dict()
    beta = stock_inputs.set_index("Ticker")["CAPM Beta"].to_dict()

    total_investment = pulp.lpSum(lots[ticker] * lot_cost[ticker] for ticker in lots)
    expected_profit = pulp.lpSum(lots[ticker] * lot_cost[ticker] * annual_return[ticker] for ticker in lots)
    beta_linear = pulp.lpSum(lots[ticker] * lot_cost[ticker] * (beta[ticker] - BETA_LIMIT) for ticker in lots)

    model += expected_profit, "maximize_expected_annual_profit"
    model += total_investment <= BUDGET, "budget_limit"
    model += beta_linear <= 0, "portfolio_beta_limit"

    status_code = model.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[status_code]
    return model, lots, status


def extract_solution(stock_inputs, lots, status, scenario):
    rows = []
    for _, row in stock_inputs.iterrows():
        ticker = row["Ticker"]
        lot_value = lots[ticker].value() if ticker in lots else 0
        lot_value = 0 if lot_value is None else int(round(lot_value))
        shares = lot_value * LOT_SIZE
        investment = shares * row["Close Price"]
        expected_profit = investment * row["Expected Annual Log Return"]
        rows.append({
            "Scenario": scenario,
            "Solver Status": status,
            "Ticker": ticker,
            "Name": row["Name"],
            "Lots": lot_value,
            "Shares": shares,
            "Close Price": row["Close Price"],
            "Investment": investment,
            "Weight": np.nan,
            "Expected Annual Log Return": row["Expected Annual Log Return"],
            "Expected Annual Profit": expected_profit,
            "CAPM Beta": row["CAPM Beta"],
            "Weighted Beta Contribution": np.nan,
        })

    allocation = pd.DataFrame(rows)
    total_investment = allocation["Investment"].sum()
    if total_investment > 0:
        allocation["Weight"] = allocation["Investment"] / total_investment
        allocation["Weighted Beta Contribution"] = allocation["Weight"] * allocation["CAPM Beta"]
    return allocation


def build_summary(allocation, scenario_note):
    total_investment = allocation["Investment"].sum()
    expected_profit = allocation["Expected Annual Profit"].sum()
    expected_return = expected_profit / total_investment if total_investment > 0 else np.nan
    portfolio_beta = allocation["Weighted Beta Contribution"].sum() if total_investment > 0 else np.nan
    return pd.DataFrame([{
        "Scenario": allocation["Scenario"].iloc[0],
        "Scenario Note": scenario_note,
        "Budget": BUDGET,
        "Total Investment": total_investment,
        "Unused Cash": BUDGET - total_investment,
        "Expected Annual Profit": expected_profit,
        "Expected Annual Portfolio Return": expected_return,
        "Portfolio Beta": portfolio_beta,
        "Beta Limit": BETA_LIMIT,
    }])


def diagnose_strict_feasibility(stock_inputs):
    total_min_investment = stock_inputs["Minimum Investment"].sum()
    min_beta = (
        (stock_inputs["Minimum Investment"] / total_min_investment * stock_inputs["CAPM Beta"]).sum()
        if total_min_investment > 0
        else np.nan
    )
    diagnostics = stock_inputs[[
        "Ticker", "Name", "Close Price", "CAPM Beta", "Minimum Lots", "Minimum Investment",
    ]].copy()
    diagnostics["Budget"] = BUDGET
    diagnostics["Total Minimum Investment All 5"] = total_min_investment
    diagnostics["Minimum-Lot Portfolio Beta"] = min_beta
    diagnostics["Budget Feasible Under Strict Rule"] = total_min_investment <= BUDGET
    diagnostics["Beta Feasible At Minimum Lots"] = min_beta <= BETA_LIMIT
    return diagnostics


def choose_solution(stock_inputs):
    strict_model, strict_lots, strict_status = solve_ilp(stock_inputs, require_all=True)
    strict_allocation = extract_solution(stock_inputs, strict_lots, strict_status, "Strict all-stock ILP")

    if strict_status == "Optimal":
        note = "原始约束可行：5 支股票均至少买入 20 手，预算和组合 beta 约束同时满足。"
        summary = build_summary(strict_allocation, note)
        return strict_allocation, summary, strict_status, None

    relaxed_model, relaxed_lots, relaxed_status = solve_ilp(stock_inputs, require_all=False)
    relaxed_allocation = extract_solution(stock_inputs, relaxed_lots, relaxed_status, "Relaxed subset ILP")
    note = (
        "原始约束不可行；参考方案允许某些股票不买入，但一旦买入仍不少于 20 手，"
        "并继续满足预算和组合 beta 约束。"
    )
    summary = build_summary(relaxed_allocation, note)
    return relaxed_allocation, summary, strict_status, relaxed_status


def plot_minimum_investment(stock_inputs):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(stock_inputs["Ticker"], stock_inputs["Minimum Investment"] / 10_000, color="#2563eb")
    ax.axhline(BUDGET / 10_000, color="#dc2626", linewidth=1.4, label="Budget")
    ax.set_title("Minimum Investment for 20 Lots per Stock")
    ax.set_xlabel("Stock")
    ax.set_ylabel("Amount (10,000 RMB)")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(MIN_INVESTMENT_FIG_PATH, bbox_inches="tight")
    plt.close(fig)


def plot_allocation_bar(allocation):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    colors = ["#16a34a" if value > 0 else "#94a3b8" for value in allocation["Investment"]]
    bars = ax.bar(allocation["Ticker"], allocation["Investment"] / 10_000, color=colors)
    ax.set_title("Optimal Investment Amount by Stock")
    ax.set_xlabel("Stock")
    ax.set_ylabel("Investment (10,000 RMB)")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(ALLOCATION_BAR_FIG_PATH, bbox_inches="tight")
    plt.close(fig)


def plot_weight_beta(allocation):
    fig, ax1 = plt.subplots(figsize=(10, 5.8))
    ax2 = ax1.twinx()
    ax1.bar(allocation["Ticker"], allocation["Weight"].fillna(0) * 100, color="#2563eb", alpha=0.78, label="Weight")
    ax2.plot(allocation["Ticker"], allocation["CAPM Beta"], color="#dc2626", marker="o", linewidth=1.6, label="Beta")
    ax1.set_title("Portfolio Weight and CAPM Beta")
    ax1.set_xlabel("Stock")
    ax1.set_ylabel("Weight (%)")
    ax2.set_ylabel("CAPM Beta")
    ax1.tick_params(axis="x", rotation=35)
    ax1.grid(True, axis="y", alpha=0.25)
    ax2.axhline(BETA_LIMIT, color="#111827", linewidth=0.9, linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(WEIGHT_BETA_FIG_PATH, bbox_inches="tight")
    plt.close(fig)


def build_report(stock_inputs, diagnostics, allocation, summary, strict_status, relaxed_status):
    selected_codes = "、".join(stock_inputs["Ticker"].tolist())
    total_min_investment = diagnostics["Total Minimum Investment All 5"].iloc[0]
    min_beta = diagnostics["Minimum-Lot Portfolio Beta"].iloc[0]

    lines = [
        "基于整数线性规划的股票投资组合配置报告",
        "",
        "一、模型设定",
        f"备选股票来自任务（三）：{selected_codes}。",
        f"训练期：{TRAIN_START} 至 {TRAIN_END}。",
        f"投资预算：{BUDGET:,.2f} 元；每手 {LOT_SIZE} 股；原始约束要求每支股票至少 {MIN_LOTS} 手。",
        "预期收益率采用训练期日对数收益率均值乘以 252，个股 beta 来自任务（三）的 CAPM 回归。",
        "目标函数为最大化组合预期年化收益金额：sum(手数_i * 每手成本_i * 年化收益率_i)。",
        "组合 beta 约束 beta <= 1 通过线性形式 sum(投资金额_i * (beta_i - 1)) <= 0 实现。",
        "",
        "二、原始约束可行性",
        f"5 支股票全部买入至少 {MIN_LOTS} 手所需最低资金为 {total_min_investment:,.2f} 元。",
        f"最低手数组合的 beta 为 {min_beta:.4f}。",
    ]

    if strict_status == "Optimal":
        lines.append("原始整数规划约束可行，并已求得最优解。")
    else:
        lines.append(
            f"原始整数规划状态为 {strict_status}，不可得到满足所有原始约束的最优解。"
        )
        lines.append(
            "主要原因通常是最低买入手数对应资金超过预算，或最低手数组合 beta 已超过上限。"
        )
        if relaxed_status:
            lines.append(
                f"因此脚本额外求解参考方案：允许某些股票不买入，但买入时仍不少于 {MIN_LOTS} 手；"
                f"该参考模型状态为 {relaxed_status}。"
            )

    portfolio = summary.iloc[0]
    lines.extend([
        "",
        "三、配置结果",
        f"采用场景：{portfolio['Scenario']}。",
        f"总投入金额：{portfolio['Total Investment']:,.2f} 元；剩余现金：{portfolio['Unused Cash']:,.2f} 元。",
        f"预期年化收益金额：{portfolio['Expected Annual Profit']:,.2f} 元。",
        f"预期年化组合收益率：{portfolio['Expected Annual Portfolio Return']:.2%}。",
        f"组合 beta：{portfolio['Portfolio Beta']:.4f}，约束上限为 {BETA_LIMIT:.2f}。",
        "",
        "四、经济意义解释",
    ])

    invested = allocation[allocation["Investment"] > 0].sort_values("Weight", ascending=False)
    if invested.empty:
        lines.append("当前模型没有产生正投资金额，需调整输入约束。")
    else:
        top = invested.iloc[0]
        lines.append(
            f"权重最高的是 {top['Name']}（{top['Ticker']}），权重为 {top['Weight']:.2%}。"
            "在整数规划中，高权重通常意味着该股票在预期收益、价格离散性和 beta 消耗之间更有优势。"
        )
        for _, row in invested.iterrows():
            lines.append(
                f"- {row['Ticker']} {row['Name']}：买入 {int(row['Lots'])} 手（{int(row['Shares'])} 股），"
                f"投入 {row['Investment']:,.2f} 元，占组合 {row['Weight']:.2%}，"
                f"年化预期收益率 {row['Expected Annual Log Return']:.2%}，"
                f"beta={row['CAPM Beta']:.4f}。"
            )
        zero_tickers = allocation[allocation["Investment"] == 0]["Ticker"].tolist()
        if zero_tickers:
            lines.append(
                f"未买入股票：{'、'.join(zero_tickers)}。在参考方案中，这些股票没有被配置，"
                "原因是原始最低手数约束不可行，且它们在预算、预期收益或 beta 约束下不利于目标函数。"
            )

    lines.extend([
        "",
        "五、输出图表",
        f"最低 20 手资金需求图：{MIN_INVESTMENT_FIG_PATH}",
        f"最优投入金额图：{ALLOCATION_BAR_FIG_PATH}",
        f"组合权重与 beta 图：{WEIGHT_BETA_FIG_PATH}",
    ])
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    stock_inputs = load_inputs()
    diagnostics = diagnose_strict_feasibility(stock_inputs)
    allocation, summary, strict_status, relaxed_status = choose_solution(stock_inputs)
    report = build_report(stock_inputs, diagnostics, allocation, summary, strict_status, relaxed_status)

    stock_inputs.to_csv(STOCK_INPUTS_PATH, index=False)
    diagnostics.to_csv(STRICT_DIAGNOSTICS_PATH, index=False)
    allocation.to_csv(ALLOCATION_RESULT_PATH, index=False)
    summary.to_csv(PORTFOLIO_SUMMARY_PATH, index=False)
    REPORT_PATH.write_text(report, encoding="utf-8")

    with pd.ExcelWriter(EXCEL_OUTPUT_PATH) as writer:
        stock_inputs.to_excel(writer, sheet_name="Stock Inputs", index=False)
        diagnostics.to_excel(writer, sheet_name="Strict Diagnostics", index=False)
        allocation.to_excel(writer, sheet_name="Allocation Result", index=False)
        summary.to_excel(writer, sheet_name="Portfolio Summary", index=False)

    plot_minimum_investment(stock_inputs)
    plot_allocation_bar(allocation)
    plot_weight_beta(allocation)

    print("整数规划输入数据：")
    display(stock_inputs)
    print("\n原始约束可行性诊断：")
    display(diagnostics)
    print("\n配置结果：")
    display(allocation)
    print("\n组合汇总：")
    display(summary)
    print("\n" + report)
    print("\n文件已保存到：")
    print(f"- {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
