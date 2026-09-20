# Predicted beta guard

推荐从 [单次快照处理与 context 接入](SNAPSHOT_INTEGRATION.md) 开始。数据加载并对齐后，处理一次运行当天最新的 X、F、specific variance 和市场权重/原 beta，将结果放进 context；未来所有交易日复用这份结果。

```python
from barra_guard import adjust_barra

adjusted = adjust_barra(X, F, d, market_weights=w, predicted_beta=beta, symbols=symbols)
# context 的字段由你的工程定义：保存 adjusted.predicted_beta 和 adjusted.risk。
# 若只交易一个子集：保存 adjusted.for_symbols(traded_symbols)。
```

核心仅依赖 NumPy，没有日期循环、表格 loader 或 planner bridge。完整矩阵可按需通过 `to_dense()` 生成；需要 CVXPY 时安装可选依赖：

```bash
uv sync --locked --extra optimization
uv run --locked --extra optimization python examples/standalone_barra.py
```

两套独立的参考实现，分别保存在 `stock_covariance/` 与 `factor_covariance/`。每个目录都有实现、示例、测试、独立 uv 环境和依赖锁文件，不互相导入。

| 目录 / 文档 | 内容 |
|---|---|
| [stock_covariance](stock_covariance/README.md) | 二分法 + 股票协方差更新；支持结构化风险和按需导出，调整本身只依赖 NumPy。 |
| [factor_covariance](factor_covariance/README.md) | CVXPY 二次规划 + 因子协方差更新，默认使用 CLARABEL。 |
| [数学说明](BETA_ADJUSTMENT_METHODS.md) | 两种方法的变量定义、公式与计算顺序。 |
| [市场权重恢复](MARKET_RECOVERY.md) | 缺少市场权重时，从 X、F、对角 D 和原 beta 求出隐含权重与市场方差。 |
| [结构化优化集成说明](STRUCTURED_INTEGRATION.md) | 给下游 Agent 的完整接入说明：避免 N×N 展开、CVXPY 风险项、tracking error、子集、矩阵导出与性能实测。 |
| [快照预处理与接入](SNAPSHOT_INTEGRATION.md) | 单次函数调用、context 中保存什么、未来各日复用、迁移文件清单。 |
| [验证记录](VALIDATION.md) | 本地测试结果、复现命令和端到端组合优化检查。 |

以股票协方差方案为例：

```bash
cd stock_covariance
uv sync --locked
uv run --locked python example.py
uv run --locked pytest -q
```

因子协方差方案在 `factor_covariance/` 中执行同样的命令。两个环境均已在本机建立，可直接使用各自的 `.venv/bin/python`，无需每次重新安装。

在项目根目录运行两套结果进入后续 CVXPY 组合优化的演示：

```bash
factor_covariance/.venv/bin/python verify_optimizer.py
```

根目录新增的 `market_recovery.py` 只依赖 NumPy，使用现有股票协方差环境运行，无需新建环境：

```bash
stock_covariance/.venv/bin/python recover_market_example.py
stock_covariance/.venv/bin/python -m pytest tests/test_market_recovery.py -q
```

所有验证都使用小规模合成数据。当前没有接入真实 Barra 数据，也没有宣称风险预测有所改善。

准备接入大规模组合优化时，先读 [STRUCTURED_INTEGRATION.md](STRUCTURED_INTEGRATION.md)。新增接口示例（从仓库根目录导入）：

```python
from stock_covariance import adjust_from_factors

model = adjust_from_factors(X, F, specific_variances, market_weights)
risk = model.cvxpy_risk(holdings)  # holdings 是 CVXPY 仿射表达式
# 将 risk.variance 放入目标，必须同时加入全部 risk.constraints。
# 完整协方差只在需要时生成：model.to_dense()
```

CVXPY 为可选依赖，在 `stock_covariance/` 内执行 `uv sync --locked --extra optimization` 后，可运行 `uv run --locked --extra optimization python structured_example.py`。该增强保持原股票协方差方案的数学结果；原 dense 接口仍可使用。
