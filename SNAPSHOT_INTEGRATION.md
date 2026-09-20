# 最新 Barra 快照：加载后处理一次，再放进 context

本功能不管理日期。每次运行使用当天最新的一份风险信息，未来交易日共用这个处理结果。Trade Planner 的 context 即使有 T 维，也可以只是重复同一份静态风险数据；这不意味着存在 T 份不同的 Barra 快照。

## 1. 单次调用

在现有 loader 完成股票、因子、日期口径和单位对齐后，在构建 context 之前调用：

```python
from barra_guard import adjust_barra, risk_arrays

adjusted = adjust_barra(
    X, F, d,
    market_weights=w,        # 已有权重；缺失时传 None
    predicted_beta=beta,    # 没有 w 时必须提供；有 w 时可以省略
    symbols=full_symbols,   # 可选；后面按股票名称选取子集时需要
    lower_bound=0.15,
)
```

X 为 N×K，F 为 K×K，d 为 N 个 specific variance（也接受 N×1 或 N×N 对角矩阵）。w、beta 为 N 或 N×1。**没有时间维度**；对角元素必须是方差，不是波动率。若 loader 返回 DataFrame/Series，在原数据层按一致顺序 `.reindex(...)` 后转成 NumPy；此模块不再重复实现表格对齐。

有 w 就使用它；缺 w 时从 X/F/d/beta 恢复。两条路径以及独立的 stock/factor adjustment 入口共用 `weight_tolerance=1e-8`。先检查原始权重的合计误差 `abs(sum(w)-1)` 和**负权重绝对值的总和**，两者都不超过容差才将负值设为零、再除以修正后合计；超出就报错，不会修复实质性的多空组合或不完整市场。输入数组不被修改；结果的 `market_weights` 是计算实际采用的权重。

修正记录在 `adjusted.diagnostics` 的 `input_weight_sum`、`input_full_investment_error`、`input_minimum_weight`、`input_negative_weight_mass`、`weight_correction_l1` 和 `weight_correction_max`。只要提供了原 beta（包括反推路径），修正后的权重仍须满足 `max(abs(beta_recomputed-beta)) <= beta_consistency_tolerance * max(1, max(abs(beta)))`，默认 beta 容差为 `1e-8`；失败则报错，误差通过时记录为 `input_beta_consistency_max_error`。权重变化小不保证 beta 变化小，所以这项复核独立保留。没有提供 beta 时，直接由模型和修正后的权重计算 beta。市场方差不变等性质均相对于这组修正后的权重定义。

`weight_tolerance` 可在入口显式设置，并传入 recovery；它与二分求解的 `tolerance=1e-12` 分开，后者不变。独立调用 `market_recovery` 仍返回原始权重，不修正，诊断参数为 `consistency_tolerance`，默认同为 `1e-8`。`adjusted.recovery_diagnostics` 保留这些原始诊断。

恢复时要求 d 严格为正，必须包含参考市场全部成分股；重现 beta 本身不能证明恢复的是供应商实际市场。推导见 [数学说明](BETA_ADJUSTMENT_METHODS.md)。输入错误或无法满足数值容差会明确报错，不返回部分结果。

## 2. context 保存什么

```python
# 字段名是示意，由现有工程定义；不需要替换你现有的 context 类。
context_fields["predicted_beta"] = adjusted.predicted_beta
context_fields["adjusted_risk"] = risk_arrays(adjusted.risk)
```

`risk_arrays(...)` 导出只含 NumPy 数组和浮点数的字典，表示完整股票集合的 Σ*。结果不含任何 CVXPY 对象，也不依赖原 X/F/d 对象继续存在。未来每天复用这个字典，不重新调整，也不用复制 T 份 covariance。股票顺序、日期、币种等元数据继续由原工程保存；不要拆开或单独修改这些相互关联的风险系数。

如果只交易部分股票，先在完整模型上校准，再按下游股票顺序取一次视图：

```python
risk = adjusted.for_symbols(traded_symbols)
context_fields["predicted_beta"] = risk.predicted_beta
context_fields["adjusted_risk"] = risk_arrays(risk)
```

这个视图精确表示完整 Σ* 的对应主子矩阵；求解器规模按交易股票数增长。不能只用当前订单股票反推市场权重，也不能对子集重新归一化市场权重。集合外若有非零固定持仓或基准暴露，应把这些股票也包括进视图。

股票协方差方案需要这个 extra risk field：**原始 X、F、d 三个字段不能单独表示修正后的风险。** 原 X 继续用于行业、风格等经济暴露约束；实际组合风险使用 `adjusted_risk`。不要只替换 beta 后继续按旧 X/F/d 计算风险。

因子协方差方案的原接口仍在 `factor_covariance/` 中，返回修改后的 F；它可沿用原来的因子风险公式。这里的简化入口针对股票协方差方案，不会自动切换两种方法。

## 3. 下游使用：不依赖 solver

```python
from barra_guard import portfolio_variance, covariance_matrix

data = context_fields["adjusted_risk"]
variance = portfolio_variance(data, p)   # 已知持仓的风险，纯 NumPy
Sigma_star = covariance_matrix(data)    # 按需生成矩阵，交给任意 solver
```

若 context 只想保存完整矩阵，预处理时直接存 `adjusted.to_dense()` 或子集 `risk.to_dense()` 即可；之后用 `p.T @ Sigma_star @ p` 计算风险。矩阵导出需要平方级存储。

不生成矩阵时，字典的六个字段足以在自己的求解器中重现同一个风险：

| 字段 | 含义 |
|---|---|
| `U` | 当前股票的风险平方根暴露，来自 F=CCᵀ、U=XC；不是原始 X。 |
| `d` | 当前股票原 specific variance。 |
| `w` | 当前股票对应的完整市场权重，不在子集内重新归一化。 |
| `delta` | 当前股票的 beta 改变量 β*−β。 |
| `h` | 完整模型 Uᵀw，长度为因子数，子集也保留这个完整市场信息。 |
| `v_out` | 未包含股票的 Σ dᵢwᵢ²；完整股票集合时为 0。 |

对当前集合的暴露 p（权重或金额），令

$$
t=\delta^\top p,\qquad z=p+wt,\qquad y=U^\top p+ht.
$$

则 **修正后方差** 是 $y^\top y+\sum_i d_i z_i^2+v_{out}t^2$。六个字段共同表示新风险；仅 U 和 d 仍不是完整的调整结果。推导见 [数学说明](BETA_ADJUSTMENT_METHODS.md)。

`portfolio_variance` 处理已知数值持仓，不接收优化器变量；自己的 solver 可以按上述公式构建目标。为保留规模优势，使用显式的 t、z、y 及对应线性等式，不要把 rank-one 变换展开成 N×N 系数矩阵。优化和风险报告必须使用同一份数据。

### 可选：只有下游使用 CVXPY 时

```python
from barra_guard.cvxpy_adapter import risk_expression

variance_expr, risk_constraints = risk_expression(data, position_dollars)
objective_terms.append(risk_weight * variance_expr)
constraints.extend(risk_constraints)  # 必须全部加入
```

这是单独的表达式构建函数，不做求解、不设置 solver，也不修改 data；返回普通的 `(表达式, 约束列表)`，不再创建自定义 CVXPY 风险类。**不要把返回的表达式放回 context。** 未使用 CVXPY 的下游可完全忽略这个文件。

`volatility=True` 返回波动率表达式，用于支持二阶锥的求解器。若要在求解前评估参考目标，直接用 `portfolio_variance(data, reference_exposure)`，不依赖辅助变量赋值。股数到金额的转换和每天的持仓变量由你的 solver 管理。

## 4. 代码分工与迁移

| 文件 | 职责 |
|---|---|
| `barra_guard/prepare.py` | 唯一预处理入口、权重来源选择、beta 一致性检查及返回结果。 |
| `barra_guard/risk.py` | 股票子集的 NumPy 风险计算及矩阵导出。 |
| `barra_guard/arrays.py` | 纯数值数据导出、函数式风险计算和矩阵构造。 |
| `barra_guard/cvxpy_adapter.py` | 可选的 CVXPY 表达式构建，预处理不会导入它。 |
| `barra_guard/_validation.py` | 共用的数组和标签检查。 |
| `stock_covariance/beta_guard.py`、`structured.py` | 二分校准和结构化协方差数学核心。 |
| `market_recovery.py` | 缺少 w 时的求解。 |
| `market_weights.py` | recovery 与两种 adjustment 共用的权重检查、容差和微小修正。 |

直接迁移时一起带走 `barra_guard/`（不用 CVXPY 可排除 `cvxpy_adapter.py`）、`stock_covariance/` 的 Python 文件及 `market_recovery.py`、`market_weights.py`；保留包内相对导入，也可安装根目录包。没有 pandas、Trade Planner 或日期管线依赖；仅计算调整和数值风险只需 NumPy，调用 CVXPY 接口时才需要 CVXPY。

可选的因子协方差方案单独位于 `factor_covariance/`，只有选择该方法时才需迁移；根目录安装包已包含两种方案。所有测试在 `tests/`，只维护根目录一份配置和锁文件。

```bash
uv sync --locked
uv run --locked python examples/standalone_barra.py
uv run --locked --extra optimization pytest -q
```

结构化表示避免 N×N 风险矩阵的存储和展开，但不保证每个求解器或规模都更快；小规模 dense 可能更快。实际接入仍应对比风险、持仓约束、编译及求解耗时。推导见 [BETA_ADJUSTMENT_METHODS.md](BETA_ADJUSTMENT_METHODS.md)。

从旧接口迁移：`.cvxpy_risk()` 和自定义风险块已移除；需要 CVXPY 时用独立的 `risk_expression(data, exposure)`。预处理数据类和原 NumPy 方法继续可用，它们本身不是 solver 对象。
