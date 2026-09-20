# Standalone Barra 模块：数据入口和 Trade Planner 接入

现在可以把本仓库作为独立 Python 包安装。推荐数据流为：

```text
完整模型的 Barra 数据 + 市场权重/原 beta
    → adjust_barra / adjust_factor_risk_data
    → AdjustedBarra / AdjustedBarraPanel
    → for_symbols(交易股票顺序)
    → 优化风险块、beta、风险报告，或按需导出协方差
```

**股票协方差方案的输出不能只装进原有的 X、F、对角 D 三个字段。** 修正改变了 specific risk 在股票之间的相关结构；如果直接将原三字段发给旧的 Barra 风险公式，下游会算回未调整的风险。因此输出是明确的 `AdjustedBarra` 对象：它保留原始输入供业务约束使用，并携带完整的修正后风险表示。原始 factor loading 仍表示原来的行业、风格等含义，不用合成风险暴露去替换它。

## 1. 安装和文件边界

Python >= 3.11；核心仅依赖 NumPy。pandas 表格接口和 CVXPY 接口分别通过可选依赖启用。在下游项目自己的环境中安装：

```bash
uv pip install '/Users/mingkaijia/Desktop/Projects/pred_beta_guard[tables,optimization]'
```

也可以在下游 uv 项目执行 `uv add '/Users/mingkaijia/Desktop/Projects/pred_beta_guard[tables,optimization]'`，或者在此仓库 `uv build` 后安装 wheel。正式集成建议锁定实际安装的版本/提交。根目录 `pyproject.toml` 已提供可安装包；原来两个方法子目录里的独立 uv 项目仍保留。

| 文件 | 职责 |
|---|---|
| `barra_guard/core.py` | 带标签的单日输入、权重来源选择、调整结果、精确子集视图。 |
| `barra_guard/tabular.py` | 对齐 Trade Planner 风格的 pandas 数据，按日期返回结果。 |
| `barra_guard/planner.py` | 股数转金额风险、结构化风险块、旧 objective 接口兼容适配。 |
| `stock_covariance/structured.py` | 原结构化数学核心，保留已有接口。 |
| `market_recovery.py` | 没有市场权重时的恢复，保留已有检查。 |

包不依赖 `trade_planner`，不访问数据库、不自动加载市场文件、不修改 provider 或 context。模型选择、币种转换、期限转换和生产缓存由现有数据层管理。

## 2. 数组入口：一个模型快照进，一个调整结果出

```python
from barra_guard import BarraSnapshot, adjust_barra

source = BarraSnapshot(
    as_of="2026-06-19", model_id="YOUR_MODEL", currency="JPY", horizon="daily",
    symbols=tuple(full_model_symbols), factor_names=tuple(factor_names),
    factor_exposure=X,       # N×K，行/列顺序与上述标签一致
    factor_covariance=F,     # K×K，decimal return covariance
    specific_variance=d,    # N 或 N×1，方差，不是 volatility，也不是 N×N D
    market_weights=w,       # N 或 N×1；缺失时传 None
    predicted_beta=beta,    # 可选；w 缺失时必须提供
)
adjusted = adjust_barra(source, lower_bound=0.15)
```

`as_of` 是 ISO 日期；`model_id/currency/horizon` 是必填口径标签，不会触发任何换算。所有数组必须有限，股票代码和因子名必须是不重复的字符串。输入数组保存为独立只读副本。

有 w 时使用 w；只有 beta 时调用已有恢复函数。有 w 和 beta 时，额外检查 beta 与 X/F/d/w 一致；默认最大偏差容差是 `1e-8 * max(1, max(abs(beta)))`，可通过 `beta_consistency_tolerance` 配置。输入 w 不合法时直接报错，不会偷偷改走恢复路径。

恢复出的权重必须通过恢复诊断和调整接口的严格检查，不自动截负或归一化。恢复检查容差与调整入口不同，约 1e-15 的负权重也可能被入口拒绝；处理原则见 [MARKET_RECOVERY.md](MARKET_RECOVERY.md)。恢复需要 strictly positive d；直接用已知 w 调整允许 d 中有零值。

输出字段和方法：

| 接口 | 含义 |
|---|---|
| `adjusted.source` | 原始模型快照，包括原始 X/F/d 和全部标签。 |
| `adjusted.predicted_beta` | 从修正后协方差重算的 beta，完整股票顺序。 |
| `adjusted.market_weights` | 实际使用的市场权重；两种来源统一接口。 |
| `adjusted.weights_source` | `provided` 或 `recovered`。 |
| `adjusted.diagnostics`、`recovery_diagnostics` | 调整和恢复的数值诊断。 |
| `adjusted.risk` | 完整模型的结构化 Σ*，可算风险、梯度等。 |
| `adjusted.for_symbols(symbols)` | 按指定股票顺序准备精确子集视图。 |
| `adjusted.to_dense()` | 按需生成完整 N×N Σ*。 |

后续风险计算必须使用结果对象，不能只取 `adjusted.predicted_beta` 而继续使用旧 covariance。

## 3. 表格入口：直接接 FactorRiskData

读取的参考仓库是 `/Users/mingkaijia/Documents/Codex/2026-06-19/thi/outputs/trade_planner`，HEAD 为 `604da0c`。这里只读取并调用了参考仓库，未修改它的文件。

该仓库 `trade_planner/data.py` 的 `FactorRiskData` 包含 `factor_exposure/factor_covariance/specific_variance`。下面的适配器按属性读取，可以直接接受那个 dataclass，也接受具有同名属性的自定义对象：

```python
from barra_guard.tabular import adjust_factor_risk_data

# full_model_symbols 必须来自模型股票全集，而不是 orders.index。
full_data = provider.load_factor_risk_data(full_model_symbols, planning_dates)
adjusted_panel = adjust_factor_risk_data(
    full_data, dates=planning_dates,
    model_id="YOUR_MODEL", currency="JPY", horizon="daily",
    market_weights=market_weights_table,  # 没有时传 None
    predicted_beta=predicted_beta_table,  # 有权重时可以省略
    lower_bound=0.15,
)
day = adjusted_panel.at(planning_dates[0])
view = day.for_symbols(ctx.symbols)
beta_by_date = adjusted_panel.beta_frame(ctx.symbols)
```

支持的输入形式与 Trade Planner 的主要形式一致：

- X：股票索引的静态 DataFrame，或 `(date, symbol)` MultiIndex DataFrame，列是全部因子。
- F：因子索引/列的静态 DataFrame、`(date, factor)` MultiIndex DataFrame、日期到矩阵的 mapping，或 K×K / T×K×K 数组。
- d、w、beta：股票索引的静态 Series、日期×股票 DataFrame、`(date, symbol)` 的 Series/DataFrame，或 N / N×1 / T×N 数组。MultiIndex DataFrame 的数值列分别名为 `specific_variance`、`market_weights`、`predicted_beta`。

带标签的数据会按 X 对齐，F 的行列标签必须和 X 的全部因子匹配。数组必须事先与 X 的标签及调用时的 dates 顺序对齐。日期规范化为无时区的模型日；重复日期、重复标签、缺失值、缺失日期都会报错，不 forward fill。

一个 panel 要求同一模型、币种、期限、因子和股票全集。股票顺序由首个请求日期的 X 确定，后续日期按标签重排；股票全集发生变化时分开准备 panel。静态字段可重复应用到请求日期，但这只是调用者明确提供的静态数据，不代表未来日期风险数据已知。生产中必须由数据层保证 point-in-time 可用性。

**当前 Trade Planner 的 `build_context_from_provider` 仅用订单股票查询风险数据。** 不能直接用那个筛选后的 context 反推完整市场权重。集成方应在数据层增加/复用完整模型股票列表的来源，单独准备 full_data；交易 context 仍可保持 M 只股票。当前参考 provider 没有“获取模型全集”的方法，因此这个列表必须来自真正的 Barra 数据源，不能在模块里猜测。

## 4. 股票子集也保留精确风险，但不扩大优化变量到完整市场

设交易集合为 J，大小 M；完整模型有 N 只股票。使用已有记号 δ=β*−β、F=CCᵀ、U=XC，预先计算：

\[
h=U^\top w,\qquad v_{out}=\sum_{i\notin J}d_i w_i^2.
\]

对于 M 维组合金额或权重 p，令：

\[
t=\delta_J^\top p,\qquad z=p+w_Jt,\qquad y=U_J^\top p+ht.
\]

则完整市场 Σ* 的 J 主子矩阵风险恰好为：

\[
\boxed{p^\top\Sigma^*_{JJ}p=\|y\|^2+\sum_{i\in J}d_i z_i^2+v_{out}t^2.}
\]

最后一项不能漏掉：虽然未交易股票的原持仓为零，变换后它们仍通过市场方向贡献 specific risk。实现直接求和计算 v_out，避免用两个相近的大数相减。

`for_symbols` 准备视图时仍需一次完整模型 O(NK+N) 运算；把视图缓存到模型日期和交易集合上。之后数值风险和风险块系数是 O(MK+M)，CVXPY 辅助变量只按 M、K 增长，不会引入 N 维持仓。原始 full model 仍保留在内存中，所以它节省的是下游求解规模，并不使数据层存储与 N 无关。

```python
view = adjusted_panel.at(date).for_symbols(traded_symbols)
block = view.cvxpy_risk(position_dollars)
# 将 block.variance 加入目标，并将 *block.constraints 加入 cp.Problem。
beta_dollars = view.predicted_beta @ position_dollars

variance = view.variance(dollar_values)
sigma_times_p = view.matvec(dollar_values)
gradient = view.gradient(dollar_values)
diagonal = view.diagonal()
Sigma_traded = view.to_dense()  # 仅在需要时生成 M×M 主子矩阵
```

子集 market weights 不重新归一化，beta 也不重新校准。若有集合外的固定持仓或跟踪基准，视图集合应覆盖这些非零暴露，不能直接省略它们；将不可交易持仓作为常量放进该视图的表达式即可。

## 5. Trade Planner 的两个接入方式

### 小规模兼容方式：保留 objective 协议

```python
from dataclasses import replace
from barra_guard.planner import DensePlannerRiskAdapter

config = replace(config, risk_model=DensePlannerRiskAdapter(adjusted_panel))
result = TradePlanner(config).solve(ctx)
```

这条路径按日期缓存交易子集的 M×M Σ*，符合现有 `RiskModel.objective(position_shares, ctx, date_index)` 协议。适合交易集合小、希望先验证行为的场景。它不是 matrix-free，不会自动按规模切换，也不自动调整求解器。

### 结构化方式：风险表达式和辅助约束一起接入

```python
from barra_guard.planner import PlannerRiskAdapter

risk_model = PlannerRiskAdapter(adjusted_panel)
block = risk_model.risk_block(position_shares, objective_ctx, date_index)
objective_terms.append(risk_weight * block.variance)
constraints.extend(block.constraints)
risk_blocks.append(block)  # 当前这一个 Problem 的局部列表
```

`PlannerRiskAdapter.objective()` 会明确报错，防止把它直接塞入旧协议后丢失辅助约束。不要在适配器上攒一个跨 solve 共享的可变约束列表。

维护参考 Trade Planner 的 Agent 应按下面的具体顺序修改 `planner.py`，原业务项保持原样：

1. `_objective_terms` 在现有 objective terms 外，额外返回 `risk_constraints`、`risk_blocks` 两个局部列表。对 inventory risk 和 residual risk，若模型提供 `risk_block`，追加其加权 variance、全部 constraints 及 block；否则沿用旧模型的 `objective` 调用。
2. `solve` 解包三个返回值，将 risk_constraints 合并进创建 `cp.Problem` 的同一 constraints 列表。原 cost、alpha、terminal penalty、path risk、持仓约束及约束诊断保持原逻辑；辅助等式属于模型定义，不应作为可以自动放松的业务约束。
3. `_objective_multiplier` 增加 `risk_blocks=()` 参数。在现有 `decision_variable.value = reference / share_scale[None, :]` **之后**，读取 `total_objective.value` **之前**，执行 `for block in risk_blocks: block.set_reference_values()`。`solve` 传入本次列表。否则辅助变量没有值，原代码会把参考目标读成 None，退回倍率 1，可能损害求解稳定性。
4. `set_reference_values()` 只设置用于参考目标或 warm start 的初始值，不能替代辅助等式。每次修改持仓参考值后都要重新调用。它也支持仿射主动暴露。
5. 使用 `_objective_context` 给出的价格，与 `_objective_terms` 现有的累计/剩余持仓变量配对。`per_name` 模式下，该价格已乘 share_scale，变量已除以 share_scale；适配器只做一次 `price * position_shares`，不要额外乘缩放、NAV 或价格。
6. 沿用当前求解器、可行性证书与错误处理；首次接入先对照同一风险模型的 dense 解，而不是假定求解时间不会变化。

参考仓库的实际调用和完整对照脚本见 [examples/trade_planner_bridge.py](examples/trade_planner_bridge.py)。其中 dense 路径直接运行未修改的 `TradePlanner.solve`；structured 路径使用真实 context/state/constraints/cost/scaling helpers 组装同一个小问题，演示上面的接入点。这个脚本是验证 harness，不是可覆盖生产 `solve()` 的完整实现，也没有直接修改参考仓库。

## 6. 优化外的风险消费方也要更新

参考仓库 `trade_planner/calibration.py::_security_covariance` 会重新从原 X/F/d 拼出完整风险矩阵。接入时必须同时替换它的风险数据来源：需要矩阵时调用 `risk_model.covariance_for_date(ctx, t)`；只需要组合风险时调用 `risk_model.variance(shares, ctx, t)`，只需要个股方差时调用 `view.diagonal()`。检查该文件里默认构造 `BarraFactorRiskModel()` 的路径，让优化和校准/报告使用同一个 adjusted panel，避免目标和报告口径不同。

行业、风格、国家等业务暴露限制仍读取原始 X；beta 限制读取调整后 beta。`U` 是风险平方根暴露，不能当成原始 factor names 对应的经济暴露。

参考 `BarraFactorRiskModel` 还支持 include/exclude factors、specific overlays 和 specific variance multiplier。这些选项没有被适配器静默继承：先明确你希望 beta floor 针对哪一个完整风险模型。推荐顺序是先完成因子选择和 d 的 overlay/multiplier，再校准该模型的 beta。若 w 要恢复，先用未修改且与供应商 beta 同口径的原始模型恢复 w，然后把 w 传给修改后的模型，重新计算 beta；不要用旧供应商 beta 去反推修改后模型的 w。

如果业务坚持在调整后继续叠加独立风险，应在组合目标中明确添加，并说明总模型的 beta floor/市场方差保证可能因此失效。不要把这个额外风险伪装成同一个校准结果。

## 7. 验证、运行与范围

2026-09-20 本地验证共 **214 项测试通过**：根目录 86 项（含新增 26 项独立模块测试）、股票协方差子目录 104 项、因子协方差子目录 24 项。覆盖标签乱序、多日期输入、恢复与提供权重两条路径、缺失/重复数据、子集风险和全矩阵等价、参考目标赋值，以及子集求解规模不随完整市场大小增长。

实际参考仓库对接使用 30 股票完整模型、3 只交易股票、3 个日期的合成数据，同时开启 inventory/residual risk。普通单位和 per_name 模式均 optimal；dense/structured 最大交易差异约 6.7e-8 股，最大相对目标差异 6.7e-16，硬约束证书通过。每次有 6 个风险块；实际 QP 为 75 变量、333 个等式非零项。结果见 [trade_planner_bridge_results.json](trade_planner_bridge_results.json)。

已构建 wheel，并在仓库之外只安装 wheel+NumPy 的新环境验证调整可运行，未安装 pandas、CVXPY 或 Trade Planner。核心模块可独立使用。这里尚未接入生产 Barra 数据，也没有修改用户实际生产优化器；性能仍需用真实模型和约束测量。

从本仓库根目录复现：

```bash
uv sync --locked --extra tables --extra optimization
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest -q
.venv/bin/python examples/standalone_barra.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python examples/trade_planner_bridge.py \
  --trade-planner-root /Users/mingkaijia/Documents/Codex/2026-06-19/thi/outputs/trade_planner \
  --output trade_planner_bridge_results.json
```

交接顺序：先阅读本文、运行上述对照，再处理数据层全集查询和优化器的三个返回值；最后统一校准/报告的风险入口。通用推导、beta floor 保证及之前的完整模型性能对照见 [STRUCTURED_INTEGRATION.md](STRUCTURED_INTEGRATION.md)。
