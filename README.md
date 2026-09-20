# Barra beta guard

对运行当天最新的一份 Barra 风险数据处理一次，结果放入现有 context，未来交易日复用。核心仅依赖 NumPy；不包含数据 loader、日期管线或 planner bridge。

```python
from barra_guard import adjust_barra, risk_arrays, portfolio_variance

adjusted = adjust_barra(X, F, d, market_weights=w, predicted_beta=beta, symbols=symbols)
risk_data = risk_arrays(adjusted.risk)  # 只有 NumPy 数组和数值，可直接存入 context
beta_star = adjusted.predicted_beta
variance = portfolio_variance(risk_data, portfolio_weights)  # 下游数值计算
# 只交易子集时：risk_data = risk_arrays(adjusted.for_symbols(traded_symbols))
```

缺少 w 时传 `market_weights=None`，用原 beta 恢复；已有 w 时 beta 可省略。X、F、d 及向量必须已按同一股票/因子顺序对齐，d 是方差。完整用法见 [接入说明](SNAPSHOT_INTEGRATION.md)，数学依据见 [推导](BETA_ADJUSTMENT_METHODS.md)。

已有或反推的 w 使用统一的 `weight_tolerance=1e-8`：合计误差与负权重总量均在容差内时截负、归一化，超出则报错。修正幅度记录在 `adjusted.diagnostics`；提供原 beta 时仍检查修正后的模型一致性。

输出不是 CVXPY 对象，不要求下游使用特定 solver。需要完整 Σ* 时调用 `covariance_matrix(risk_data)`。独立的 `barra_guard/cvxpy_adapter.py` 仅是可选的表达式构建函数；不用 CVXPY 就无需使用或迁移该文件。

## 文件结构

| 路径 | 内容 |
|---|---|
| `barra_guard/` | 单快照入口、输入检查、股票子集风险接口。 |
| `stock_covariance/` | 推荐的股票协方差修正；二分法、结构化风险与 dense 参考实现。 |
| `factor_covariance/` | 保持 X、D 不变，只修改 F 的备选方法；需要 CVXPY，可能无解。 |
| `market_recovery.py` | 从原 beta 和风险模型恢复市场权重。 |
| `market_weights.py` | 共用权重容差、检查与微小误差修正。 |
| `tests/` | 所有数学、输入边界和下游优化测试。 |
| `examples/standalone_barra.py` | 纯 NumPy 示例：预处理、保存数据、下游风险计算。 |

## 安装、运行、迁移

Python >= 3.11。只保留根目录这一套配置和锁文件：

```bash
uv sync --locked
uv run --locked python examples/standalone_barra.py
# 完整测试包括因子协方差方法和可选 CVXPY 接口：
uv run --locked --extra optimization pytest -q
```

集成到下游环境可安装本仓库或 `uv build` 生成的 wheel，也可按接入说明复制核心文件。仅预处理和数值风险不需要 `optimization` extra。测试使用合成数据；验证了数学一致性和接口，不代表已验证生产数据或预测改善。

因子协方差备选方法可直接调用：

```python
from factor_covariance import adjust_factor_covariance

result = adjust_factor_covariance(X, F, d, w, preserve_order=True)
F_star = result.factor_covariance
beta_star = result.beta_after
```

该方法默认不强制排序；`preserve_order=True` 保留弱排序，`pin_negative=True` 额外固定原负 beta 为下限。不可行时抛出 `InfeasibleAdjustmentError`，求解或重建失败抛出 `SolverFailureError`；不会自动换方法。股票协方差方案的原 dense 接口仍为 `stock_covariance.adjust_covariance(Sigma, w)`。
