# Barra beta guard

对运行当天最新的一份 Barra 风险数据处理一次，结果放入现有 context，未来交易日复用。核心仅依赖 NumPy；不包含数据 loader、日期管线或 planner bridge。

```python
from barra_guard import adjust_barra

adjusted = adjust_barra(X, F, d, market_weights=w, predicted_beta=beta, symbols=symbols)
risk = adjusted.risk                          # 完整模型的结构化 Σ*
beta_star = adjusted.predicted_beta
# 只交易部分股票时，准备一次：risk = adjusted.for_symbols(traded_symbols)
```

缺少 w 时传 `market_weights=None`，用原 beta 恢复；已有 w 时 beta 可省略。X、F、d 及向量必须已按同一股票/因子顺序对齐，d 是方差。完整用法见 [接入说明](SNAPSHOT_INTEGRATION.md)，数学依据见 [推导](BETA_ADJUSTMENT_METHODS.md)。

## 文件结构

| 路径 | 内容 |
|---|---|
| `barra_guard/` | 单快照入口、输入检查、股票子集风险接口。 |
| `stock_covariance/` | 推荐的股票协方差修正；二分法、结构化风险与 dense 参考实现。 |
| `factor_covariance/` | 保持 X、D 不变，只修改 F 的备选方法；需要 CVXPY，可能无解。 |
| `market_recovery.py` | 从原 beta 和风险模型恢复市场权重。 |
| `tests/` | 所有数学、输入边界和下游优化测试。 |
| `examples/standalone_barra.py` | 唯一的端到端使用示例。 |

## 安装、运行、迁移

Python >= 3.11。只保留根目录这一套配置和锁文件：

```bash
uv sync --locked --extra optimization
uv run --locked --extra optimization pytest -q
uv run --locked --extra optimization python examples/standalone_barra.py
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
