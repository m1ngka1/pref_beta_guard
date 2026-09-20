# Predicted beta guard

推荐下游集成从 [Standalone 模块与 Trade Planner 接入说明](TRADE_PLANNER_INTEGRATION.md) 开始。根目录已提供可安装的 `barra-beta-guard` 包：完整 Barra 数据输入，输出带股票/日期标签的调整结果，支持已有或恢复的市场权重、精确股票子集风险、CVXPY 风险块和完整矩阵导出。

```bash
uv sync --locked --extra tables --extra optimization
uv run --locked --extra tables --extra optimization python examples/standalone_barra.py
```

数组入口为 `from barra_guard import BarraSnapshot, adjust_barra`；Trade Planner 风格表格入口为 `from barra_guard.tabular import adjust_factor_risk_data`。该包不依赖 Trade Planner；其风险适配器位于 `barra_guard/planner.py`。本地参考仓库仅用于只读对照验证，未修改。

两套独立的参考实现，分别保存在 `stock_covariance/` 与 `factor_covariance/`。每个目录都有实现、示例、测试、独立 uv 环境和依赖锁文件，不互相导入。

| 目录 / 文档 | 内容 |
|---|---|
| [stock_covariance](stock_covariance/README.md) | 二分法 + 股票协方差更新；支持结构化风险和按需导出，调整本身只依赖 NumPy。 |
| [factor_covariance](factor_covariance/README.md) | CVXPY 二次规划 + 因子协方差更新，默认使用 CLARABEL。 |
| [数学说明](BETA_ADJUSTMENT_METHODS.md) | 两种方法的变量定义、公式与计算顺序。 |
| [市场权重恢复](MARKET_RECOVERY.md) | 缺少市场权重时，从 X、F、对角 D 和原 beta 求出隐含权重与市场方差。 |
| [结构化优化集成说明](STRUCTURED_INTEGRATION.md) | 给下游 Agent 的完整接入说明：避免 N×N 展开、CVXPY 风险项、tracking error、子集、矩阵导出与性能实测。 |
| [Standalone / Trade Planner 集成](TRADE_PLANNER_INTEGRATION.md) | 标签及日期对齐、完整市场到交易子集、provider/context/risk 插件、目标缩放与报告一致性。 |
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
