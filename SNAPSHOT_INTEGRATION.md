# 最新 Barra 快照：加载后处理一次，再放进 context

本功能不管理日期。每次运行使用当天最新的一份风险信息，未来交易日共用这个处理结果。Trade Planner 的 context 即使有 T 维，也可以只是重复同一份静态风险数据；这不意味着存在 T 份不同的 Barra 快照。

## 1. 单次调用

在现有 loader 完成股票、因子、日期口径和单位对齐后，在构建 context 之前调用：

```python
from barra_guard import adjust_barra

adjusted = adjust_barra(
    X, F, d,
    market_weights=w,        # 已有权重；缺失时传 None
    predicted_beta=beta,    # 没有 w 时必须提供；有 w 时可以省略
    symbols=full_symbols,   # 可选；后面按股票名称选取子集时需要
    lower_bound=0.15,
)
```

X 为 N×K，F 为 K×K，d 为 N 个 specific variance（也接受 N×1 或 N×N 对角矩阵）。w、beta 为 N 或 N×1。**没有时间维度**；对角元素必须是方差，不是波动率。若 loader 返回 DataFrame/Series，在原数据层按一致顺序 `.reindex(...)` 后转成 NumPy；此模块不再重复实现表格对齐。

有 w 就直接使用；缺 w 时从 X/F/d/beta 恢复。两者都有时会检查原 beta 与模型是否一致。权重必须非负且合计为 1，不自动截负或归一化；恢复时要求 d 严格为正，必须包含参考市场全部成分股；重现 beta 本身不能证明恢复的是供应商实际市场。诊断容差为 1e-8，但调整入口要求 w 严格非负、合计误差不超过 1e-12，所以微小负权重也可能被拒绝；不会静默修正。推导见 [数学说明](BETA_ADJUSTMENT_METHODS.md)。输入错误或无法满足数值容差会明确报错，不返回部分结果。

## 2. context 保存什么

```python
# 字段名是示意，由现有工程定义；不需要替换你现有的 context 类。
context_fields["predicted_beta"] = adjusted.predicted_beta
context_fields["adjusted_risk"] = adjusted.risk
```

`adjusted.risk` 是完整股票集合的结构化 Σ*。未来每天从同一个字段取用，不再调用调整函数，也不用复制 T 份 covariance。日期、币种、模型名等现有元数据继续由原工程保存。

如果只交易部分股票，先在完整模型上校准，再按下游股票顺序取一次视图：

```python
risk = adjusted.for_symbols(traded_symbols)
context_fields["predicted_beta"] = risk.predicted_beta
context_fields["adjusted_risk"] = risk
```

这个视图精确表示完整 Σ* 的对应主子矩阵；求解器规模按交易股票数增长。不能只用当前订单股票反推市场权重，也不能对子集重新归一化市场权重。集合外若有非零固定持仓或基准暴露，应把这些股票也包括进视图。

股票协方差方案需要这个 extra risk field：**原始 X、F、d 三个字段不能单独表示修正后的风险。** 原 X 继续用于行业、风格等经济暴露约束；实际组合风险使用 `adjusted_risk`。不要只替换 beta 后继续按旧 X/F/d 计算风险。

因子协方差方案的原接口仍在 `factor_covariance/` 中，返回修改后的 F；它可沿用原来的因子风险公式。这里的简化入口针对股票协方差方案，不会自动切换两种方法。

## 3. 下游使用

```python
risk = context_fields["adjusted_risk"]

# 数值风险；p 为与 risk 股票顺序一致的权重或金额暴露
variance = risk.variance(p)
Sigma_times_p = risk.matvec(p)
gradient = risk.gradient(p)
asset_variances = risk.diagonal()
Sigma_star = risk.to_dense()  # 仅在需要完整矩阵时调用
```

结构化 CVXPY 风险支持每天的不同持仓，但使用同一份 covariance：

```python
for t in range(number_of_trading_days):
    dollars = cp.multiply(prices[t], position_shares[t])
    block = risk.cvxpy_risk(dollars)
    objective_terms.append(risk_weight * block.variance)
    constraints.extend(block.constraints)  # 必须加入，否则风险与持仓脱离
```

辅助等式能避免 CVXPY 将 `w * (delta @ p)` 直接展开成 N×N 系数；不要内联重写这些平方项。

这里的未来持仓/价格属于下游规划，**不是每日重新加载 Barra**。份额到金额的转换及已有数值缩放由下游掌握；这个模块只接收最终暴露，不再包一层 planner adapter。跟踪误差使用 `p - benchmark`。

若只接受完整矩阵，可先将 `risk.to_dense()` 存入 context，再沿用 `cp.quad_form`。这条路径简单，但有矩阵存储成本。若下游在求解前给持仓赋值以计算参考目标，子集风险块支持 `block.set_reference_values()` 同步辅助变量；更简单的方式是用 `risk.variance(reference_exposure)` 数值计算风险。优化与报告必须使用同一份调整后风险。

## 4. 代码分工与迁移

| 文件 | 职责 |
|---|---|
| `barra_guard/prepare.py` | 唯一预处理入口、权重来源选择、beta 一致性检查及返回结果。 |
| `barra_guard/risk.py` | 可选的股票子集风险、完整矩阵导出和凸优化表达式。 |
| `barra_guard/_validation.py` | 共用的数组和标签检查。 |
| `stock_covariance/beta_guard.py`、`structured.py` | 二分校准和结构化协方差数学核心。 |
| `market_recovery.py` | 缺少 w 时的求解。 |

直接迁移时一起带走 `barra_guard/`、`stock_covariance/` 的 Python 文件及 `market_recovery.py`；保留包内相对导入，也可安装根目录包。没有 pandas、Trade Planner 或日期管线依赖；仅计算调整和数值风险只需 NumPy，调用 CVXPY 接口时才需要 CVXPY。

可选的因子协方差方案单独位于 `factor_covariance/`，只有选择该方法时才需迁移；根目录安装包已包含两种方案。所有测试在 `tests/`，只维护根目录一份配置和锁文件。

```bash
uv sync --locked --extra optimization
uv run --locked --extra optimization python examples/standalone_barra.py
uv run --locked --extra optimization pytest -q
```

结构化表示避免 N×N 风险矩阵的存储和展开，但不保证每个求解器或规模都更快；小规模 dense 可能更快。实际接入仍应对比风险、持仓约束、编译及求解耗时。推导见 [BETA_ADJUSTMENT_METHODS.md](BETA_ADJUSTMENT_METHODS.md)。
