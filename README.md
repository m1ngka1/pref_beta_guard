# Predicted beta guard

两套独立的参考实现，分别保存在 `stock_covariance/` 与 `factor_covariance/`。每个目录都有实现、示例、测试、独立 uv 环境和依赖锁文件，不互相导入。

| 目录 / 文档 | 内容 |
|---|---|
| [stock_covariance](stock_covariance/README.md) | 二分法 + 股票协方差更新，只依赖 NumPy。 |
| [factor_covariance](factor_covariance/README.md) | CVXPY 二次规划 + 因子协方差更新，默认使用 CLARABEL。 |
| [数学说明](BETA_ADJUSTMENT_METHODS.md) | 两种方法的变量定义、公式与计算顺序。 |
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

所有验证都使用小规模合成数据。当前没有接入真实 Barra 数据，也没有宣称风险预测有所改善。
