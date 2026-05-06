import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)


# =========================
# 1. 参数设置
# =========================

ANALYSIS_START = "2022-01-01"
ANALYSIS_END = "2025-12-31"

# 10 支候选股票（A 股）+ 沪深 300 指数
STOCKS = [
    "600519.SS",  # 贵州茅台
    "000858.SZ",  # 五粮液
    "601318.SS",  # 中国平安
    "600036.SS",  # 招商银行
    "600900.SS",  # 长江电力
    "600276.SS",  # 恒瑞医药
    "601012.SS",  # 隆基绿能
    "300750.SZ",  # 宁德时代
    "600887.SS",  # 伊利股份
    "002415.SZ",  # 海康威视
]
INDEX_TICKER = "000300.SS"
INDEX_COLUMN = "CSI300"

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
    INDEX_COLUMN: "沪深300",
}

PRICE_TICKERS = {ticker: ticker for ticker in STOCKS}
PRICE_TICKERS[INDEX_COLUMN] = INDEX_TICKER

OUTPUT_DIR = Path("output")
PRICE_CACHE_PATH = OUTPUT_DIR / "close_price_2022_2025.csv"
EXCEL_OUTPUT_PATH = OUTPUT_DIR / "stock_ff3_analysis_2022_2025.xlsx"
FF3_CACHE_PATH = OUTPUT_DIR / "fama_french_factors_daily_2022_2025.csv"
MERGED_OUTPUT_PATH = OUTPUT_DIR / "merged_daily_price_ff3_2022_2025.csv"
STOCK_LOG_RETURN_PATH = OUTPUT_DIR / "stock_log_returns_2022_2025.csv"
INDEX_LOG_RETURN_PATH = OUTPUT_DIR / "index_log_returns_2022_2025.csv"
STOCK_CUMULATIVE_RETURN_PATH = OUTPUT_DIR / "stock_cumulative_return_2022_2025.csv"
INDEX_CUMULATIVE_RETURN_PATH = OUTPUT_DIR / "index_cumulative_return_2022_2025.csv"
FINAL_RANKING_PATH = OUTPUT_DIR / "final_stock_cumulative_return_ranking.csv"
INTERPRETATION_PATH = OUTPUT_DIR / "cumulative_return_interpretation.txt"


def _read_table(path):
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _parse_date_series(series):
    text = series.astype(str).str.strip()
    if text.str.fullmatch(r"\d{6}").all():
        return pd.to_datetime(text, format="%Y%m") + pd.offsets.MonthEnd(0)
    if text.str.fullmatch(r"\d{8}").all():
        return pd.to_datetime(text, format="%Y%m%d")
    return pd.to_datetime(series)


def _normalize_price_frame(df):
    df = df.copy()
    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.rename(columns={INDEX_TICKER: INDEX_COLUMN})
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_local_close_prices(start, end):
    """读取本地收盘价缓存或用户导出的价格文件。"""
    local_candidates = [
        PRICE_CACHE_PATH,
        Path("close_price_2022_2025.csv"),
        Path("data/close_price_2022_2025.csv"),
        Path("price_daily_close.csv"),
        Path("data/price_daily_close.csv"),
        Path("close_price_2022_2025.xlsx"),
        Path("data/close_price_2022_2025.xlsx"),
    ]
    for local_path in local_candidates:
        if local_path.exists():
            df = _normalize_price_frame(_read_table(local_path))
            return df.loc[start:end], str(local_path)
    return pd.DataFrame(), None


def eastmoney_secid(ticker):
    """将 yfinance 风格的 A 股代码转换为东方财富 secid。"""
    code, suffix = ticker.split(".")
    if suffix == "SS":
        return f"1.{code}"
    if suffix == "SZ":
        return f"0.{code}"
    raise ValueError(f"暂不支持的交易所后缀：{ticker}")


def download_eastmoney_close(ticker, start, end, max_retry=3, sleep_seconds=0.6):
    """从东方财富逐只下载未复权日收盘价，避免 yfinance 全量限流。"""
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": eastmoney_secid(ticker),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "0",
        "beg": pd.Timestamp(start).strftime("%Y%m%d"),
        "end": pd.Timestamp(end).strftime("%Y%m%d"),
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    last_error = None
    for retry_idx in range(max_retry):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=20)
            response.raise_for_status()
            payload = response.json()
            klines = payload.get("data", {}).get("klines", [])
            if not klines:
                raise ValueError("东方财富返回为空")

            rows = [line.split(",") for line in klines]
            df = pd.DataFrame(rows)
            one = pd.DataFrame({
                "Date": pd.to_datetime(df.iloc[:, 0]),
                ticker: pd.to_numeric(df.iloc[:, 2], errors="coerce"),
            })
            one = one.dropna(subset=[ticker]).set_index("Date").sort_index()
            return one.loc[start:end], "东方财富"
        except Exception as exc:
            last_error = exc
            time.sleep(sleep_seconds * (retry_idx + 1))

    raise RuntimeError(f"{ticker} 东方财富下载失败：{last_error}")


def _extract_yfinance_close(raw, ticker):
    if raw.empty:
        return pd.Series(dtype=float, name=ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        elif "Close" in raw.columns.get_level_values(-1):
            close = raw.xs("Close", axis=1, level=-1)
        else:
            raise ValueError("yfinance 返回结果缺少 Close 列")
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        if "Close" not in raw.columns:
            raise ValueError("yfinance 返回结果缺少 Close 列")
        close = raw["Close"]
    close.name = ticker
    return close


def download_yfinance_close(ticker, start, end, max_retry=2, sleep_seconds=2.0):
    """yfinance 只作为备用源；end 参数为开区间，所以需要加一天。"""
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    last_error = None
    for retry_idx in range(max_retry):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end_exclusive,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            close = _extract_yfinance_close(raw, ticker)
            if close.empty:
                raise ValueError("yfinance 返回为空")
            one = close.to_frame()
            one.index = pd.to_datetime(one.index)
            return one.loc[start:end], "yfinance"
        except Exception as exc:
            last_error = exc
            time.sleep(sleep_seconds * (retry_idx + 1))
    raise RuntimeError(f"{ticker} yfinance 下载失败：{last_error}")


def collect_close_prices(ticker_by_column, start, end):
    """优先使用本地缓存；缺失标的再逐只下载，降低被限流的概率。"""
    expected_columns = list(ticker_by_column)
    close_price, local_source = load_local_close_prices(start, end)
    source_notes = []
    failed = []

    if not close_price.empty:
        source_notes.append(f"本地缓存：{local_source}")
        close_price = close_price.reindex(columns=expected_columns)

    for column, source_ticker in ticker_by_column.items():
        has_cached_data = (
            column in close_price.columns
            and close_price[column].loc[start:end].notna().any()
        )
        if has_cached_data:
            continue

        downloaded = None
        errors = []
        for downloader in (download_eastmoney_close, download_yfinance_close):
            try:
                downloaded, source_name = downloader(source_ticker, start, end)
                downloaded = downloaded.rename(columns={source_ticker: column})
                source_notes.append(f"{column}：{source_name}")
                break
            except Exception as exc:
                errors.append(str(exc))

        if downloaded is None or downloaded.empty:
            failed.append((column, "；".join(errors)))
            continue

        if close_price.empty:
            close_price = downloaded[[column]]
        elif column in close_price.columns:
            new_column = f"{column}__new"
            close_price = close_price.join(downloaded[[column]], how="outer", rsuffix="__new")
            close_price[column] = close_price[column].combine_first(close_price[new_column])
            close_price = close_price.drop(columns=[new_column])
        else:
            close_price = close_price.join(downloaded[[column]], how="outer")

        time.sleep(0.3)

    close_price = close_price.reindex(columns=expected_columns).sort_index()
    close_price = close_price.loc[start:end]
    close_price = close_price.dropna(how="all").ffill()

    missing_columns = [col for col in expected_columns if close_price[col].isna().all()]
    if missing_columns:
        details = "\n".join(f"- {ticker}: {reason}" for ticker, reason in failed)
        raise SystemExit(f"以下标的没有取得有效收盘价：{missing_columns}\n{details}")

    first_complete_date = close_price.dropna().index.min()
    if pd.notna(first_complete_date) and first_complete_date != close_price.index.min():
        print(f"存在起始缺失值，累计涨跌幅从首个完整交易日 {first_complete_date.date()} 开始计算。")
        close_price = close_price.loc[first_complete_date:].ffill()

    return close_price, source_notes, failed


def _normalize_ff3_columns(df):
    rename_map = {}
    for column in df.columns:
        key = str(column).strip().lower().replace("_", "").replace(" ", "")
        if key in {"mkt-rf", "mktrf", "mkt", "marketriskpremium", "市场风险溢价", "市场超额收益"}:
            rename_map[column] = "MKT"
        elif key == "smb" or "规模" in key:
            rename_map[column] = "SMB"
        elif key == "hml" or "价值" in key:
            rename_map[column] = "HML"
        elif key == "rf" or "无风险" in key:
            rename_map[column] = "RF"
    df = df.rename(columns=rename_map)
    required = ["MKT", "SMB", "HML"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Fama-French 文件缺少列：{missing}")
    if "RF" not in df.columns:
        df["RF"] = 0.0
    keep_columns = [col for col in ["MKT", "SMB", "HML", "RF"] if col in df.columns]
    return df[keep_columns]


def load_local_ff3_factors_daily(start, end):
    local_candidates = [
        FF3_CACHE_PATH,
        Path("output/three_factor_selection/fama_french_factors_daily_2022_2025.csv"),
        Path("fama_french_factors_daily_2022_2025.csv"),
        Path("fama_french_factors_daily.csv"),
        Path("data/fama_french_factors_daily_2022_2025.csv"),
        Path("data/fama_french_factors_daily.csv"),
        Path("ff3_daily.csv"),
        Path("data/ff3_daily.csv"),
    ]

    for local_path in local_candidates:
        if not local_path.exists():
            continue
        ff_raw = _read_table(local_path)
        if "Date" not in ff_raw.columns:
            ff_raw = ff_raw.rename(columns={ff_raw.columns[0]: "Date"})
        ff_raw["Date"] = _parse_date_series(ff_raw["Date"])
        ff_raw = ff_raw.set_index("Date").sort_index()
        ff = _normalize_ff3_columns(ff_raw).apply(pd.to_numeric, errors="coerce")
        if ff.abs().quantile(0.95).max() > 1:
            ff = ff / 100
        return ff.loc[start:end], f"本地文件 {local_path}"

    return pd.DataFrame(), None


def load_ff3_factors_daily(start, end):
    """读取日度 Fama-French 三因子；本地/锐思导出优先，在线 Ken French 兜底。"""
    ff, source = load_local_ff3_factors_daily(start, end)
    if not ff.empty:
        return ff, source

    ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    response = requests.get(ff_url, timeout=30)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        file_name = z.namelist()[0]
        with z.open(file_name) as f:
            ff_raw = pd.read_csv(f, skiprows=3)

    ff_raw = ff_raw.rename(columns={ff_raw.columns[0]: "Date"})
    ff_raw = ff_raw[ff_raw["Date"].astype(str).str.fullmatch(r"\d{8}")]
    ff_raw["Date"] = pd.to_datetime(ff_raw["Date"], format="%Y%m%d")
    ff_raw = ff_raw.set_index("Date").sort_index()
    ff = _normalize_ff3_columns(ff_raw).astype(float) / 100
    return ff.loc[start:end], "Ken French 在线日度数据"


def merge_daily_price_with_daily_factors(close_price, ff_factors):
    """按日度交易日期合并收盘价和日度三因子，保留日度索引。"""
    return close_price.join(ff_factors, how="left").ffill()


def build_interpretation(final_stock_return_df, final_index_return):
    positive_count = int((final_stock_return_df["Cumulative Return"] > 0).sum())
    outperform = final_stock_return_df[
        final_stock_return_df["Cumulative Return"] > final_index_return
    ]
    best = final_stock_return_df.iloc[0]
    worst = final_stock_return_df.iloc[-1]
    average_return = final_stock_return_df["Cumulative Return"].mean()

    outperform_names = "、".join(outperform["Name"].tolist()) if not outperform.empty else "无"
    lines = [
        "初步解释：",
        (
            f"2022-01-01 至 2025-12-31 区间内，10 支股票中有 {positive_count} 支最终累计涨跌幅为正，"
            f"平均累计涨跌幅为 {average_return:.2%}。"
        ),
        (
            f"表现最好的是 {best['Name']}（{best['Ticker']}），最终累计涨跌幅为 {best['Cumulative Return']:.2%}；"
            f"表现最弱的是 {worst['Name']}（{worst['Ticker']}），最终累计涨跌幅为 {worst['Cumulative Return']:.2%}。"
        ),
        (
            f"同期沪深300累计涨跌幅为 {final_index_return:.2%}，跑赢沪深300的股票有 {len(outperform)} 支："
            f"{outperform_names}。"
        ),
        "以上结果仅基于未复权收盘价计算，尚未考虑分红、交易成本、停牌流动性和风险调整收益。后续可结合三因子回归进一步解释超额收益来源。",
    ]
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================
    # 2. 下载股票和指数日收盘价数据
    # =========================
    close_price, source_notes, failed_tickers = collect_close_prices(
        ticker_by_column=PRICE_TICKERS,
        start=ANALYSIS_START,
        end=ANALYSIS_END,
    )

    print("价格数据来源：")
    for note in source_notes:
        print(f"- {note}")
    if failed_tickers:
        print("以下标的下载失败，但缓存中已有数据则不影响运行：")
        for ticker, reason in failed_tickers:
            print(f"- {ticker}: {reason}")

    print("收盘价数据预览：")
    display(close_price.head())
    display(close_price.tail())

    # =========================
    # 3. 读取 Fama-French 三因子日度数据
    # =========================
    ff_factors, ff_source = load_ff3_factors_daily(ANALYSIS_START, ANALYSIS_END)
    print(f"Fama-French 三因子来源：{ff_source}")
    print("Fama-French 日度三因子数据预览：")
    display(ff_factors.head())
    display(ff_factors.tail())

    # =========================
    # 4. 生成数据框、日期索引、升序排序和缺失值前向填充
    # =========================
    close_price.index = pd.to_datetime(close_price.index)
    close_price = close_price.sort_index().ffill()
    ff_factors.index = pd.to_datetime(ff_factors.index)
    ff_factors = ff_factors.sort_index().ffill()
    merged_data = merge_daily_price_with_daily_factors(close_price, ff_factors)

    print("合并后的日度价格 + 日度因子数据预览：")
    display(merged_data.head())
    display(merged_data.tail())

    # =========================
    # 5. 计算每日对数收益率
    # Rt = ln(Pt) - ln(Pt-1)
    # =========================
    log_returns = np.log(close_price) - np.log(close_price.shift(1))
    log_returns = log_returns.dropna(how="all")
    stock_log_returns = log_returns[STOCKS]
    index_log_returns = log_returns[[INDEX_COLUMN]]

    print("股票每日对数收益率预览：")
    display(stock_log_returns.head())
    print("指数每日对数收益率预览：")
    display(index_log_returns.head())

    # =========================
    # 6. 计算 10 支股票累计涨跌幅序列
    # 累计涨跌幅 = Pt / P0 - 1
    # =========================
    stock_cumulative_return = close_price[STOCKS] / close_price[STOCKS].iloc[0] - 1
    index_cumulative_return = close_price[[INDEX_COLUMN]] / close_price[[INDEX_COLUMN]].iloc[0] - 1

    print("10 支股票累计涨跌幅序列预览：")
    display(stock_cumulative_return.head())
    display(stock_cumulative_return.tail())

    # =========================
    # 7. 统计最终累计涨跌幅并生成初步解释
    # =========================
    final_stock_return = stock_cumulative_return.iloc[-1].sort_values(ascending=False)
    final_stock_return_df = pd.DataFrame({
        "Ticker": final_stock_return.index,
        "Name": [TICKER_NAMES.get(ticker, ticker) for ticker in final_stock_return.index],
        "Cumulative Return": final_stock_return.values,
        "Cumulative Return (%)": final_stock_return.values * 100,
    })
    final_index_return = float(index_cumulative_return.iloc[-1, 0])
    interpretation = build_interpretation(final_stock_return_df, final_index_return)

    print("2022-2025 年 10 支股票最终累计涨跌幅排名：")
    display(final_stock_return_df)
    print(interpretation)

    # =========================
    # 8. 保存为 Excel 和 CSV 文件
    # =========================
    with pd.ExcelWriter(EXCEL_OUTPUT_PATH) as writer:
        close_price.to_excel(writer, sheet_name="Close Price")
        ff_factors.to_excel(writer, sheet_name="FF3 Daily Factors")
        merged_data.to_excel(writer, sheet_name="Merged Daily Data")
        stock_log_returns.to_excel(writer, sheet_name="Stock Log Returns")
        index_log_returns.to_excel(writer, sheet_name="Index Log Returns")
        stock_cumulative_return.to_excel(writer, sheet_name="Stock Cumulative")
        index_cumulative_return.to_excel(writer, sheet_name="Index Cumulative")
        final_stock_return_df.to_excel(writer, sheet_name="Final Ranking", index=False)

    close_price.to_csv(PRICE_CACHE_PATH)
    ff_factors.to_csv(FF3_CACHE_PATH)
    merged_data.to_csv(MERGED_OUTPUT_PATH)
    stock_log_returns.to_csv(STOCK_LOG_RETURN_PATH)
    index_log_returns.to_csv(INDEX_LOG_RETURN_PATH)
    stock_cumulative_return.to_csv(STOCK_CUMULATIVE_RETURN_PATH)
    index_cumulative_return.to_csv(INDEX_CUMULATIVE_RETURN_PATH)
    final_stock_return_df.to_csv(FINAL_RANKING_PATH, index=False)
    INTERPRETATION_PATH.write_text(interpretation, encoding="utf-8")

    print("文件已保存：")
    print(f"1. {EXCEL_OUTPUT_PATH}")
    print(f"2. {PRICE_CACHE_PATH}")
    print(f"3. {FF3_CACHE_PATH}")
    print(f"4. {MERGED_OUTPUT_PATH}")
    print(f"5. {STOCK_LOG_RETURN_PATH}")
    print(f"6. {INDEX_LOG_RETURN_PATH}")
    print(f"7. {STOCK_CUMULATIVE_RETURN_PATH}")
    print(f"8. {INDEX_CUMULATIVE_RETURN_PATH}")
    print(f"9. {FINAL_RANKING_PATH}")
    print(f"10. {INTERPRETATION_PATH}")


if __name__ == "__main__":
    main()
