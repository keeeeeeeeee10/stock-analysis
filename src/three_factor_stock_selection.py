import io
import re
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm


TRAIN_START = "2022-01-01"
TRAIN_END = "2024-12-31"
DATA_START = "2022-01-01"
DATA_END = "2025-12-31"
SIGNIFICANCE_LEVEL = 0.10
SELECT_COUNT = 5

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
OUTPUT_DIR = INPUT_DIR / "three_factor_selection"
CAPM_FIG_DIR = OUTPUT_DIR / "capm_scatter"

STOCK_RETURN_PATH = INPUT_DIR / "stock_log_returns_2022_2025.csv"
INDEX_RETURN_PATH = INPUT_DIR / "index_log_returns_2022_2025.csv"
FF3_DAILY_CACHE_PATH = OUTPUT_DIR / "fama_french_factors_daily_2022_2025.csv"

CAPM_RESULTS_PATH = OUTPUT_DIR / "capm_regression_results.csv"
THREE_FACTOR_RESULTS_PATH = OUTPUT_DIR / "three_factor_regression_results.csv"
SELECTED_STOCKS_PATH = OUTPUT_DIR / "selected_stocks_by_three_factor_alpha.csv"
REGRESSION_SUMMARY_PATH = OUTPUT_DIR / "three_factor_regression_summaries.txt"
REPORT_PATH = OUTPUT_DIR / "three_factor_stock_selection_report.txt"
EXCEL_OUTPUT_PATH = OUTPUT_DIR / "factor_regression_stock_selection_2022_2024.xlsx"

CAPM_GRID_FIG_PATH = OUTPUT_DIR / "capm_scatter_regression_grid.png"
ALPHA_BAR_FIG_PATH = OUTPUT_DIR / "three_factor_alpha_ranking.png"


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


def parse_date_series(series):
    text = series.astype(str).str.strip()
    if text.str.fullmatch(r"\d{8}").all():
        return pd.to_datetime(text, format="%Y%m%d")
    if text.str.fullmatch(r"\d{6}").all():
        return pd.to_datetime(text, format="%Y%m") + pd.offsets.MonthEnd(0)
    return pd.to_datetime(series)


def normalize_factor_columns(df):
    rename_map = {}
    for column in df.columns:
        key = str(column).strip().lower()
        key = re.sub(r"[\s_\-()/]+", "", key)
        if key in {"mktrf", "mkt", "marketriskpremium", "rmrf"} or "市场" in key:
            rename_map[column] = "MKT"
        elif key == "smb" or "规模" in key:
            rename_map[column] = "SMB"
        elif key == "hml" or "价值" in key or "账面" in key:
            rename_map[column] = "HML"
        elif key == "rf" or "无风险" in key:
            rename_map[column] = "RF"

    df = df.rename(columns=rename_map)
    required = ["MKT", "SMB", "HML"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"三因子数据缺少列：{missing}")

    if "RF" not in df.columns:
        print("三因子数据缺少 RF 列，已按 RF=0 处理。")
        df["RF"] = 0.0

    factors = df[["MKT", "SMB", "HML", "RF"]].apply(pd.to_numeric, errors="coerce")
    if factors.abs().quantile(0.95).max() > 1:
        factors = factors / 100
    return factors


def load_local_ff3_daily():
    local_candidates = [
        FF3_DAILY_CACHE_PATH,
        INPUT_DIR / "fama_french_factors_daily_2022_2025.csv",
        Path("data/fama_french_factors_daily_2022_2025.csv"),
        Path("data/fama_french_factors_daily.csv"),
        Path("fama_french_factors_daily_2022_2025.csv"),
        Path("fama_french_factors_daily.csv"),
        Path("ff3_daily.csv"),
        Path("data/ff3_daily.csv"),
    ]

    for path in local_candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = parse_date_series(df["Date"])
        df = df.set_index("Date").sort_index()
        factors = normalize_factor_columns(df).loc[DATA_START:DATA_END]
        if not factors.empty:
            if path == FF3_DAILY_CACHE_PATH:
                return factors, f"本地缓存 {path}（若未手动替换，首次缓存来自 Ken French 在线日度数据）"
            return factors, f"本地文件 {path}"

    return pd.DataFrame(), None


def download_ken_french_daily_factors():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        file_name = z.namelist()[0]
        with z.open(file_name) as f:
            raw = pd.read_csv(f, skiprows=3)

    raw = raw.rename(columns={raw.columns[0]: "Date"})
    raw = raw[raw["Date"].astype(str).str.fullmatch(r"\d{8}")]
    raw["Date"] = pd.to_datetime(raw["Date"], format="%Y%m%d")
    raw = raw.set_index("Date").sort_index()
    factors = normalize_factor_columns(raw).loc[DATA_START:DATA_END]
    return factors, "Ken French 在线日度数据"


def load_ff3_daily_factors():
    factors, source = load_local_ff3_daily()
    if factors.empty:
        factors, source = download_ken_french_daily_factors()
    factors.to_csv(FF3_DAILY_CACHE_PATH)
    return factors, source


def load_returns():
    stock_returns = read_indexed_csv(STOCK_RETURN_PATH)
    index_returns = read_indexed_csv(INDEX_RETURN_PATH)

    missing = [ticker for ticker in STOCKS if ticker not in stock_returns.columns]
    if missing:
        raise ValueError(f"股票收益率数据缺少列：{missing}")
    if "CSI300" not in index_returns.columns:
        raise ValueError("指数收益率数据缺少 CSI300 列。")

    stock_returns = stock_returns.loc[DATA_START:DATA_END, STOCKS]
    index_returns = index_returns.loc[DATA_START:DATA_END, ["CSI300"]]
    stock_returns = stock_returns.apply(pd.to_numeric, errors="coerce")
    index_returns = index_returns.apply(pd.to_numeric, errors="coerce")
    return stock_returns, index_returns


def significance_stars(p_value):
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def fit_ols(y, x):
    x_const = sm.add_constant(x, has_constant="add")
    model = sm.OLS(y, x_const, missing="drop").fit()
    return model


def run_capm(stock_returns, index_returns, factors):
    data = stock_returns.join(index_returns, how="inner").join(factors[["RF"]], how="inner")
    data = data.loc[TRAIN_START:TRAIN_END].dropna()
    data["Market_Excess"] = data["CSI300"] - data["RF"]

    rows = []
    models = {}
    for ticker in STOCKS:
        regression_data = data[[ticker, "RF", "Market_Excess"]].dropna().copy()
        regression_data["Stock_Excess"] = regression_data[ticker] - regression_data["RF"]
        model = fit_ols(regression_data["Stock_Excess"], regression_data[["Market_Excess"]])
        models[ticker] = (model, regression_data)
        rows.append({
            "Ticker": ticker,
            "Name": TICKER_NAMES[ticker],
            "N": int(model.nobs),
            "Alpha": model.params["const"],
            "Alpha t": model.tvalues["const"],
            "Alpha p": model.pvalues["const"],
            "Alpha Sig": significance_stars(model.pvalues["const"]),
            "Beta Market": model.params["Market_Excess"],
            "Beta Market t": model.tvalues["Market_Excess"],
            "Beta Market p": model.pvalues["Market_Excess"],
            "Beta Market Sig": significance_stars(model.pvalues["Market_Excess"]),
            "R2": model.rsquared,
            "Adj R2": model.rsquared_adj,
        })

    return pd.DataFrame(rows).sort_values("Alpha", ascending=False), models


def run_three_factor(stock_returns, factors):
    data = stock_returns.join(factors, how="inner")
    data = data.loc[TRAIN_START:TRAIN_END].dropna()

    rows = []
    models = {}
    for ticker in STOCKS:
        regression_data = data[[ticker, "RF", "MKT", "SMB", "HML"]].dropna().copy()
        regression_data["Stock_Excess"] = regression_data[ticker] - regression_data["RF"]
        model = fit_ols(regression_data["Stock_Excess"], regression_data[["MKT", "SMB", "HML"]])
        models[ticker] = model
        rows.append({
            "Ticker": ticker,
            "Name": TICKER_NAMES[ticker],
            "N": int(model.nobs),
            "Alpha": model.params["const"],
            "Alpha Annualized": model.params["const"] * 252,
            "Alpha t": model.tvalues["const"],
            "Alpha p": model.pvalues["const"],
            "Alpha Sig": significance_stars(model.pvalues["const"]),
            "Beta_MKT": model.params["MKT"],
            "Beta_MKT p": model.pvalues["MKT"],
            "Beta_SMB": model.params["SMB"],
            "Beta_SMB p": model.pvalues["SMB"],
            "Beta_HML": model.params["HML"],
            "Beta_HML p": model.pvalues["HML"],
            "R2": model.rsquared,
            "Adjusted R2": model.rsquared_adj,
        })

    results = pd.DataFrame(rows).sort_values("Alpha", ascending=False)
    return results, models


def select_stocks(three_factor_results):
    results = three_factor_results.copy()
    results["Meets Strict Rule"] = (
        (results["Alpha"] > 0)
        & (results["Alpha p"] < SIGNIFICANCE_LEVEL)
    )
    qualified = results[results["Meets Strict Rule"]].sort_values("Alpha", ascending=False)
    selected = qualified.head(SELECT_COUNT).copy()
    selected["Selection Rule"] = (
        f"Strict: positive Alpha and Alpha p < {SIGNIFICANCE_LEVEL:.2f}; "
        "rank by Alpha descending"
    )

    if len(selected) < SELECT_COUNT:
        selected_codes = set(selected["Ticker"])
        fallback = results[
            ~results["Ticker"].isin(selected_codes)
        ].sort_values("Alpha", ascending=False).head(SELECT_COUNT - len(selected)).copy()
        fallback["Selection Rule"] = (
            f"Fallback: fewer than {SELECT_COUNT} stocks have positive significant alpha; "
            "filled by next highest Alpha and flagged as not strictly qualified"
        )
        selected = pd.concat([selected, fallback], ignore_index=True)

    return selected.head(SELECT_COUNT)


def plot_capm_grid(capm_models):
    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    axes = axes.ravel()
    for ax, ticker in zip(axes, STOCKS):
        model, data = capm_models[ticker]
        x = data["Market_Excess"]
        y = data["Stock_Excess"]
        ax.scatter(x * 100, y * 100, s=10, alpha=0.35, color="#2563eb", edgecolors="none")

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = model.params["const"] + model.params["Market_Excess"] * x_line
        ax.plot(x_line * 100, y_line * 100, color="#dc2626", linewidth=1.4)

        ax.set_title(f"{ticker}: alpha={model.params['const']:.4%}, beta={model.params['Market_Excess']:.2f}", fontsize=9)
        ax.set_xlabel("Market excess return (%)")
        ax.set_ylabel("Stock excess return (%)")
        ax.grid(True, alpha=0.25)

    fig.suptitle("CAPM Scatter Plots and Regression Lines (2022-2024)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(CAPM_GRID_FIG_PATH, bbox_inches="tight")
    plt.close(fig)


def plot_capm_individual(capm_models):
    CAPM_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in STOCKS:
        model, data = capm_models[ticker]
        x = data["Market_Excess"]
        y = data["Stock_Excess"]

        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        ax.scatter(x * 100, y * 100, s=12, alpha=0.35, color="#2563eb", edgecolors="none")
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = model.params["const"] + model.params["Market_Excess"] * x_line
        ax.plot(x_line * 100, y_line * 100, color="#dc2626", linewidth=1.6)
        ax.set_title(f"CAPM: {ticker}")
        ax.set_xlabel("Market excess return (%)")
        ax.set_ylabel("Stock excess return (%)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(CAPM_FIG_DIR / f"{ticker.replace('.', '_')}_capm_scatter.png", bbox_inches="tight")
        plt.close(fig)


def plot_alpha_ranking(three_factor_results, selected):
    ordered = three_factor_results.sort_values("Alpha", ascending=True)
    selected_set = set(selected["Ticker"])
    colors = ["#16a34a" if ticker in selected_set else "#64748b" for ticker in ordered["Ticker"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(ordered["Ticker"], ordered["Alpha"] * 10000, color=colors)
    ax.axvline(0, color="#111827", linewidth=0.9)
    ax.set_title("Three-Factor Alpha Ranking (Daily Alpha, bps)")
    ax.set_xlabel("Daily alpha (basis points)")
    ax.set_ylabel("Stock")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ALPHA_BAR_FIG_PATH, bbox_inches="tight")
    plt.close(fig)


def save_regression_summaries(three_factor_models, factor_source):
    lines = [
        "Fama-French 三因子回归摘要",
        f"训练期：{TRAIN_START} 至 {TRAIN_END}",
        f"三因子数据来源：{factor_source}",
        "=" * 90,
    ]
    for ticker in STOCKS:
        lines.append(f"\n{ticker} {TICKER_NAMES[ticker]}")
        lines.append(str(three_factor_models[ticker].summary()))
        lines.append("=" * 90)
    REGRESSION_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def build_report(capm_results, three_factor_results, selected, factor_source):
    significant_count = int((three_factor_results["Alpha p"] < SIGNIFICANCE_LEVEL).sum())
    positive_significant_count = int(
        ((three_factor_results["Alpha"] > 0) & (three_factor_results["Alpha p"] < SIGNIFICANCE_LEVEL)).sum()
    )
    selected_codes = "、".join(selected["Ticker"].tolist())
    best_alpha = three_factor_results.iloc[0]
    best_adj_r2 = three_factor_results.sort_values("Adjusted R2", ascending=False).iloc[0]

    lines = [
        "基于三因子模型的选股报告",
        "",
        f"训练期：{TRAIN_START} 至 {TRAIN_END}",
        f"三因子数据来源：{factor_source}",
        "",
        "一、CAPM 基准回归",
        (
            "CAPM 使用沪深300日对数收益率作为市场收益率，"
            "并用三因子数据中的 RF 构造股票和市场的超额收益。"
        ),
        (
            "CAPM 回归表已输出每只股票的 alpha、beta、t 值、p 值和显著性星号；"
            f"散点图与回归线见 {CAPM_GRID_FIG_PATH} 以及 {CAPM_FIG_DIR}/。"
        ),
        "",
        "二、Fama-French 三因子回归",
        (
            "三因子回归以股票超额收益为因变量，以 MKT、SMB、HML 为解释变量，"
            "记录 alpha、三个因子暴露、R2、调整 R2 和 alpha p 值。"
        ),
        (
            f"alpha 最大的是 {best_alpha['Name']}（{best_alpha['Ticker']}），"
            f"日 alpha 为 {best_alpha['Alpha']:.4%}，p 值为 {best_alpha['Alpha p']:.4f}。"
        ),
        (
            f"调整 R2 最高的是 {best_adj_r2['Name']}（{best_adj_r2['Ticker']}），"
            f"调整 R2 为 {best_adj_r2['Adjusted R2']:.4f}，说明三因子对该股票收益波动的解释力相对更强。"
        ),
        "",
        "三、选股依据与结果",
        (
            f"选股规则为：先筛选三因子回归中 alpha > 0 且 alpha p < {SIGNIFICANCE_LEVEL:.2f} 的股票，"
            f"再按 alpha 从大到小选取前 {SELECT_COUNT} 支。"
        ),
        f"训练期内 alpha 显著的股票数量为 {significant_count} 支，其中正 alpha 且显著的股票数量为 {positive_significant_count} 支。",
        f"最终选出的备选资产为：{selected_codes}。",
        "",
        "入选理由：",
    ]

    if positive_significant_count < SELECT_COUNT:
        lines.append(
            f"说明：当前数据下严格满足“正 alpha 且 p < {SIGNIFICANCE_LEVEL:.2f}”的股票不足 {SELECT_COUNT} 支，"
            "因此结果表中补入了 alpha 排名靠前的股票，并通过 Selection Rule / Meets Strict Rule 标记其是否严格满足规则。"
        )

    for _, row in selected.iterrows():
        if row.get("Meets Strict Rule", False):
            reason = (
                f"- {row['Ticker']} {row['Name']}：三因子日 alpha={row['Alpha']:.4%}"
                f"（年化约 {row['Alpha Annualized']:.2%}），p={row['Alpha p']:.4f}，"
                f"调整 R2={row['Adjusted R2']:.4f}，alpha 为正且在统计上显著。"
            )
        else:
            reason = (
                f"- {row['Ticker']} {row['Name']}：严格满足条件的股票不足 {SELECT_COUNT} 支，"
                f"按 alpha 排名补入；日 alpha={row['Alpha']:.4%}，p={row['Alpha p']:.4f}，"
                f"调整 R2={row['Adjusted R2']:.4f}。"
            )
        lines.append(reason)

    lines.extend([
        "",
        "注意：若你从锐思数据库下载了中国市场三因子日度数据，可将文件放在 "
        "output/fama_french_factors_daily_2022_2025.csv 或 data/fama_french_factors_daily.csv，"
        "脚本会优先使用本地文件。当前结果的数据来源见本报告开头。",
    ])
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CAPM_FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    stock_returns, index_returns = load_returns()
    factors, factor_source = load_ff3_daily_factors()

    capm_results, capm_models = run_capm(stock_returns, index_returns, factors)
    three_factor_results, three_factor_models = run_three_factor(stock_returns, factors)
    selected = select_stocks(three_factor_results)
    report = build_report(capm_results, three_factor_results, selected, factor_source)

    capm_results.to_csv(CAPM_RESULTS_PATH, index=False)
    three_factor_results.to_csv(THREE_FACTOR_RESULTS_PATH, index=False)
    selected.to_csv(SELECTED_STOCKS_PATH, index=False)
    REPORT_PATH.write_text(report, encoding="utf-8")
    save_regression_summaries(three_factor_models, factor_source)

    with pd.ExcelWriter(EXCEL_OUTPUT_PATH) as writer:
        capm_results.to_excel(writer, sheet_name="CAPM Results", index=False)
        three_factor_results.to_excel(writer, sheet_name="Three Factor Results", index=False)
        selected.to_excel(writer, sheet_name="Selected Stocks", index=False)
        factors.to_excel(writer, sheet_name="FF3 Daily Factors")

    plot_capm_grid(capm_models)
    plot_capm_individual(capm_models)
    plot_alpha_ranking(three_factor_results, selected)

    print("CAPM 回归结果：")
    display(capm_results)
    print("\n三因子回归结果：")
    display(three_factor_results)
    print("\n最终选出的 5 支股票：")
    display(selected[[
        "Ticker", "Name", "Alpha", "Alpha p", "Meets Strict Rule",
        "Beta_MKT", "Beta_SMB", "Beta_HML", "Adjusted R2",
    ]])
    print("\n" + report)
    print("\n文件已保存到：")
    print(f"- {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
