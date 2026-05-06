# 股票投资组合优化与回测分析

本项目使用 Python 对 A 股候选股票进行数据采集、预处理、描述性统计、三因子回归选股、整数线性规划配置、Markowitz 均值-方差优化，以及 2025 年组合回测与盈亏分析。项目最终输出可直接用于实验报告的表格、图片和文字分析。

## 实验目标

通过 Python 对股票投资组合优化配置和盈亏情况进行数据分析，建立资产组合的市场模型，并将主要结果可视化。实验覆盖以下内容：

1. 数据采集与预处理
2. 描述性统计与相关性分析
3. 基于 CAPM 和 Fama-French 三因子模型的选股
4. 基于整数线性规划的股票投资组合配置
5. Markowitz 均值-方差组合优化
6. 2025 年组合回测与盈亏分析
7. 总结与思考

## 项目结构

```text
financial_analyze/
├── README.md
├── ans_qa.txt
├── src/
│   ├── data_preprocess.py
│   ├── descriptive_stats_analysis.py
│   ├── three_factor_stock_selection.py
│   ├── portfolio_ilp_allocation.py
│   ├── markowitz_portfolio_optimization.py
│   └── portfolio_backtest_analysis.py
├── output/
│   ├── close_price_2022_2025.csv
│   ├── stock_log_returns_2022_2025.csv
│   ├── index_log_returns_2022_2025.csv
│   ├── descriptive_stats/
│   ├── three_factor_selection/
│   ├── portfolio_ilp_allocation/
│   ├── markowitz_optimization/
│   └── backtest_analysis/
└── ref_txt_and_fig/
    ├── txt/
    └── fig/
```

## 环境说明

建议使用已有 conda 环境：

```bash
conda activate stock_analysis
```

项目中使用的主要库：

```text
pandas
numpy
requests
yfinance
matplotlib
openpyxl
statsmodels
scipy
patsy
pulp
```

其中 `statsmodels/scipy/patsy/pulp` 已在开发过程中安装到 `stock_analysis` 环境中。若环境缺失，可执行：

```bash
python -m pip install statsmodels scipy patsy pulp
```

也可以直接使用项目根目录的 `environment.yml` 创建环境：

```bash
conda env create -f environment.yml
conda activate stock_analysis
```

如果本机已经存在同名环境，可用下面的命令更新：

```bash
conda env update -f environment.yml --prune
conda activate stock_analysis
```

## 运行顺序

建议按以下顺序运行脚本，因为后续任务依赖前面生成的数据和结果：

```bash
conda activate stock_analysis

python src/data_preprocess.py
python src/descriptive_stats_analysis.py
python src/three_factor_stock_selection.py
python src/portfolio_ilp_allocation.py
python src/markowitz_portfolio_optimization.py
python src/portfolio_backtest_analysis.py
```

运行完成后，所有结果都会保存在 `output/` 下。

## 各脚本功能

### 1. `src/data_preprocess.py`

对应任务（一）：数据采集与预处理。

主要功能：

- 收集 10 支候选股票和沪深 300 指数在 2022-01-01 至 2025-12-31 期间的日收盘价。
- 优先读取本地缓存，缺失时使用东方财富接口逐只下载，`yfinance` 作为备用源，降低限流风险。
- 读取 Fama-French 三因子日度数据，优先使用本地/锐思导出文件；若没有，则使用 Ken French 日度数据作为兜底。
- 将 Date 转为 `datetime` 并设为索引，按日期升序排序，对缺失值前向填充。
- 计算股票和指数每日对数收益率。
- 计算 10 支股票累计涨跌幅和最终排名。

主要输出：

```text
output/close_price_2022_2025.csv
output/fama_french_factors_daily_2022_2025.csv
output/merged_daily_price_ff3_2022_2025.csv
output/stock_log_returns_2022_2025.csv
output/index_log_returns_2022_2025.csv
output/stock_cumulative_return_2022_2025.csv
output/final_stock_cumulative_return_ranking.csv
output/cumulative_return_interpretation.txt
output/stock_ff3_analysis_2022_2025.xlsx
```

### 2. `src/descriptive_stats_analysis.py`

对应任务（二）：描述性统计与相关性分析。

主要功能：

- 基于 2022-2025 年股票日收益率计算均值、标准差、方差、偏度、峰度、年化收益率和年化波动率。
- 计算 10 支股票协方差矩阵和相关系数矩阵。
- 绘制收盘价折线图、收益率箱线图和相关性热图。
- 生成关于波动性、收益分布和分散化效果的文字解释。

主要输出：

```text
output/descriptive_stats/stock_return_descriptive_statistics.csv
output/descriptive_stats/stock_return_covariance_matrix.csv
output/descriptive_stats/stock_return_correlation_matrix.csv
output/descriptive_stats/descriptive_correlation_interpretation.txt
output/descriptive_stats/stock_close_price_lines.png
output/descriptive_stats/stock_return_boxplot.png
output/descriptive_stats/stock_return_correlation_heatmap.png
```

### 3. `src/three_factor_stock_selection.py`

对应任务（三）：基于三因子模型的选股。

主要功能：

- 使用 2022-2024 年数据对 10 支股票进行 CAPM 回归。
- 输出每只股票 CAPM 的 alpha、beta、t 值、p 值、显著性和 R2。
- 绘制 CAPM 散点图与回归线。
- 使用 Fama-French 三因子进行多因子回归，输出 alpha、beta_MKT、beta_SMB、beta_HML、R2、调整 R2、alpha p 值。
- 按选股规则筛选 5 支股票作为后续组合优化备选资产。

当前结果选出的 5 支股票：

```text
600900.SS 长江电力
601318.SS 中国平安
600519.SS 贵州茅台
600887.SS 伊利股份
600036.SS 招商银行
```

主要输出：

```text
output/three_factor_selection/capm_regression_results.csv
output/three_factor_selection/three_factor_regression_results.csv
output/three_factor_selection/selected_stocks_by_three_factor_alpha.csv
output/three_factor_selection/three_factor_regression_summaries.txt
output/three_factor_selection/three_factor_stock_selection_report.txt
output/three_factor_selection/capm_scatter_regression_grid.png
output/three_factor_selection/capm_scatter/
output/three_factor_selection/three_factor_alpha_ranking.png
```

### 4. `src/portfolio_ilp_allocation.py`

对应任务（四）：基于整数线性规划的投资组合配置。

主要功能：

- 使用任务（三）选出的 5 支股票。
- 以 2022-2024 年年化对数收益率作为预期收益率。
- 使用任务（三）CAPM beta 作为个股 beta。
- 设定投资预算 200 万元、每手 100 股、组合 beta <= 1。
- 原题要求每支股票不少于 20 手，脚本会先检查该严格约束是否可行。
- 若严格约束不可行，额外给出参考方案：允许某些股票不买入，但一旦买入仍不少于 20 手。

当前关键结果：

```text
严格约束不可行：5 支股票每支至少 20 手需要 3,351,360 元，超过 200 万预算。
参考方案：买入长江电力 676 手，即 67,600 股。
投入金额：1,997,580 元。
剩余现金：2,420 元。
组合 beta：0.2471。
```

主要输出：

```text
output/portfolio_ilp_allocation/portfolio_ilp_stock_inputs.csv
output/portfolio_ilp_allocation/strict_constraint_diagnostics.csv
output/portfolio_ilp_allocation/portfolio_ilp_allocation_result.csv
output/portfolio_ilp_allocation/portfolio_ilp_summary.csv
output/portfolio_ilp_allocation/portfolio_ilp_report.txt
output/portfolio_ilp_allocation/minimum_investment_vs_budget.png
output/portfolio_ilp_allocation/portfolio_allocation_bar.png
output/portfolio_ilp_allocation/portfolio_weight_beta.png
```

### 5. `src/markowitz_portfolio_optimization.py`

对应任务（五）：均值-方差投资组合优化。

主要功能：

- 使用任务（三）选出的 5 支股票。
- 基于 2022-2024 年日收益率估计年化收益率和年化协方差矩阵。
- 在不允许卖空条件下求解最小方差组合和最大夏普比率组合。
- 无风险利率设为 1.5% 年化。
- 绘制有效前沿，并标出最小方差组合、最大夏普组合和等权重组合。

当前关键结果：

```text
最小方差组合：
长江电力 62.47%，贵州茅台 15.44%，伊利股份 17.32%，招商银行 4.77%，中国平安 0.00%。

最大夏普组合：
长江电力 100.00%。

等权重组合：
5 支股票各 20.00%。
```

主要输出：

```text
output/markowitz_optimization/markowitz_asset_inputs.csv
output/markowitz_optimization/markowitz_portfolio_weights.csv
output/markowitz_optimization/markowitz_portfolio_summary.csv
output/markowitz_optimization/efficient_frontier_points.csv
output/markowitz_optimization/markowitz_optimization_report.txt
output/markowitz_optimization/efficient_frontier.png
output/markowitz_optimization/portfolio_weights_comparison.png
```

### 6. `src/portfolio_backtest_analysis.py`

对应任务（六）：组合回测与盈亏分析。

主要功能：

- 使用 2025-01-01 至 2025-12-31 的实际价格数据回测 4 个组合：
  - 整数规划组合
  - 最大夏普比率组合
  - 最小方差组合
  - 等权重组合
- 计算日收益率序列和累计净值曲线。
- 计算年化收益率、年化波动率、夏普比率、最大回撤、Calmar 比率。
- 计算盈利天数占比、平均盈利、平均亏损、盈亏比。
- 计算组合偏度和峰度，并与任务（二）的单股票结果对比。
- 绘制累计净值、动态回撤、收益率直方图/KDE、月度箱线图、偏度峰度对比图。

当前关键结果：

```text
2025 年回测表现最好的是等权重组合：
年化收益率 2.79%，年化波动率 12.66%，夏普比率 0.1020，最大回撤 7.82%。

整数规划组合和最大夏普组合完全一致，因为二者均为 100% 长江电力。
```

主要输出：

```text
output/backtest_analysis/portfolio_backtest_daily_returns_2025.csv
output/backtest_analysis/portfolio_backtest_net_value_2025.csv
output/backtest_analysis/portfolio_backtest_drawdowns_2025.csv
output/backtest_analysis/portfolio_backtest_performance_metrics.csv
output/backtest_analysis/portfolio_backtest_profit_loss_metrics.csv
output/backtest_analysis/portfolio_backtest_skew_kurtosis_comparison.csv
output/backtest_analysis/portfolio_backtest_report.txt
output/backtest_analysis/portfolio_cumulative_net_value.png
output/backtest_analysis/portfolio_dynamic_drawdown.png
output/backtest_analysis/portfolio_return_hist_kde.png
output/backtest_analysis/portfolio_monthly_return_boxplot.png
output/backtest_analysis/portfolio_vs_single_stock_skew_kurtosis.png
```

## 报告引用材料

为了方便写实验报告，项目已经整理出一个引用材料目录：

```text
ref_txt_and_fig/
├── txt/
└── fig/
```

### `ref_txt_and_fig/txt`

这里存放重要文字说明和关键结果表，包括：

- 数据预处理累计涨跌幅解释
- 描述性统计与相关性解释
- 三因子选股报告
- 整数规划报告
- Markowitz 优化报告
- 回测分析报告
- 总结与思考 `ans_qa`
- CAPM 和三因子回归结果表
- 投资组合配置和回测指标表

### `ref_txt_and_fig/fig`

这里存放最适合放入报告的图片，包括：

- 收盘价折线图
- 收益率箱线图
- 相关性热图
- CAPM 散点回归图
- 三因子 alpha 排名图
- 整数规划配置图
- Markowitz 有效前沿图
- 回测累计净值曲线
- 动态回撤曲线
- 收益率分布图
- 月度箱线图
- 偏度峰度对比图

## 重要结果摘要

### 数据预处理

2022-2025 年期间，10 支股票中有 3 支最终累计涨跌幅为正。表现最好的是中国平安，最终累计涨跌幅为 34.12%；表现最弱的是隆基绿能，最终累计涨跌幅为 -78.55%。同期沪深 300 累计涨跌幅为 -5.85%。

### 三因子选股

由于当前三因子数据采用 Ken French 日度因子缓存作为兜底，三因子对 A 股解释力较弱。严格满足“正 alpha 且 p < 0.10”的股票不足 5 支，因此按 alpha 排名补足 5 支，最终选出长江电力、中国平安、贵州茅台、伊利股份和招商银行。

### 整数规划

原始约束“5 支股票每支至少 20 手、总资金 200 万”不可行。原因是最低买入金额为 3,351,360 元，超过预算。放松参考方案最终全部配置到长江电力，买入 676 手。

### Markowitz 优化

最大夏普组合为 100% 长江电力；最小方差组合主要配置长江电力，同时分散到伊利股份、贵州茅台和招商银行。

### 回测分析

2025 年实际回测中，等权重组合表现最好，年化收益率 2.79%，夏普比率 0.1020，最大回撤 7.82%。整数规划组合和最大夏普组合均为 100% 长江电力，在 2025 年表现弱于等权重组合，说明样本内最优不一定能延续到样本外。

## 数据源说明

价格数据：

- 优先读取本地缓存 `output/close_price_2022_2025.csv`
- 缺失时使用东方财富接口逐只下载
- `yfinance` 仅作为备用源，以降低限流影响

三因子数据：

- 优先读取本地日度三因子文件
- 如果没有提供锐思中国市场三因子数据，则使用 Ken French 日度三因子作为兜底
- 当前结果中三因子报告明确说明了该数据源限制

如需使用锐思数据，可将日度三因子文件放在以下任一位置后重新运行脚本：

```text
output/fama_french_factors_daily_2022_2025.csv
data/fama_french_factors_daily.csv
data/fama_french_factors_daily_2022_2025.csv
```

文件至少应包含：

```text
Date, MKT, SMB, HML, RF
```

其中 `RF` 可选；如果缺失，脚本会按 0 处理。

## 注意事项

1. 当前项目不是 Git 仓库，因此没有提交记录。
2. `output/` 是脚本运行生成的完整结果。
3. `ref_txt_and_fig/` 是从 `output/` 中挑选出的报告引用材料。
4. 如果重新运行三因子选股，后续整数规划、Markowitz 和回测也建议重新运行，以保持结果一致。
5. 由于实验期价格和因子数据来自不同市场口径，三因子回归结果需要谨慎解释。

## 推荐报告写作顺序

1. 先引用 `ref_txt_and_fig/fig/01_stock_close_price_lines.png`、`02_stock_return_boxplot.png`、`03_stock_return_correlation_heatmap.png` 说明数据特征。
2. 再引用 `03_three_factor_stock_selection_report.txt` 和 `05_three_factor_alpha_ranking.png` 说明选股依据。
3. 然后引用 `04_portfolio_ilp_report.txt` 和整数规划三张图说明交易约束和不可行诊断。
4. 接着引用 `05_markowitz_optimization_report.txt`、`09_markowitz_efficient_frontier.png`、`10_markowitz_portfolio_weights_comparison.png` 说明均值-方差优化。
5. 最后引用 `06_backtest_analysis_report.txt` 和回测图表说明样本外表现。
6. 总结部分可直接参考 `ans_qa.txt` 或 `ref_txt_and_fig/txt/07_summary_and_thoughts_ans_qa.txt`。
