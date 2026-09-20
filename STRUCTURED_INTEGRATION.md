# 股票协方差修正：结构化实现与集成说明

本文是交给下游优化代码维护者的集成说明。实现已经提供：优先调用 `stock_covariance.adjust_from_factors`，通过返回对象的 `cvxpy_risk` 把修正后的风险接入优化器；仅在需要导出时调用 `to_dense`。不需要重新实现 beta 调整算法。

若从带标签/日期的 Barra 数据接入现有应用，优先使用新增 `barra_guard` 外层模块，见 [TRADE_PLANNER_INTEGRATION.md](TRADE_PLANNER_INTEGRATION.md)。它复用本文核心，并提供更紧凑的交易子集风险公式；下文稀疏 scatter 仍数学正确，但生产子集接入优先使用 `for_symbols`，避免完整 N 维辅助变量。

这次增强保持股票协方差方案的数学结果，只改变存储和计算方式。`factor_covariance/` 方案及原来的 `adjust_covariance(Sigma, w)` 接口保持不变。

## 1. 什么时候需要，为什么做

当输入本来就是因子风险模型、股票数 N 明显大于因子数 K，且后续要反复求组合风险或做凸优化时，优先使用本接口。它避免先构建 N×N 的股票协方差，再让优化器处理完整二次项。

例如，10,000 只股票的 float64 完整矩阵，仅一个数组就约 800 MB；100 个因子的结构化结果约 8.48 MB。这是数组存储量估算，不是峰值内存：原始输入、中间数组、CVXPY 和求解器还有额外开销。

本接口要求 specific covariance 为对角阵。股票数很少、下游接口强制接收完整矩阵，或者 specific covariance 含有真实的非对角项时，可以保留原 dense 路径。不要为了套用本接口而丢掉相关 specific risk。

结构化表示避免了风险模型本身的 N² 存储，**不保证任何求解器、任何约束下都更快**。辅助变量会改变数值求解过程；集成时仍要比较实际组合问题的总耗时和解误差。第 8 节给出了本地结果及复现方式。

## 2. 从原模型到新的 Sigma Star

原 predicted beta 的定义是：

\[
\Sigma=XFX^\top+D,\qquad s=w^\top\Sigma w,\qquad \beta=\Sigma w/s.
\]

这里 X 为 N×K 的股票因子暴露，F 为 K×K 因子协方差；D=diag(d)，d 是 N 个 **specific variance**，不是波动率。w 是这个 beta 所参考的市场组合权重，长度 N，非负且合计为 1。所有数组的股票顺序必须完全一致。

继续使用已有规则，令下限 L=0.15，通过二分法求平移量 λ：

\[
\beta_i^*=\max(L,\beta_i-\lambda),\qquad w^\top\beta^*=1.
\]

这会处理所有低于下限的 beta，并允许它们在下限处并列；不只是把负值改成 0.15。定义改变量 δ=β*−β，则 wᵀδ=0，原有更新为：

\[
\Sigma^*=\Sigma+s(\beta^*\beta^{*\top}-\beta\beta^\top).
\]

关键等价变形是令 A=I+δwᵀ，然后利用 Σw=sβ 得到：

\[
\boxed{\Sigma^*=A\Sigma A^\top.}
\]

这个恒等式不需要显式生成 A。它也说明新协方差仍半正定；当原 Σ 正定时，wᵀδ=0 使 A 可逆，因此 Σ* 也正定。市场方差、beta 弱排序、剔除市场后的剩余协方差保持不变，具体以浮点容差为准。没有增加新的回归或优化问题。

为了利用因子结构，将 F 分解为 CCᵀ，再计算 U=XC。实现用 K×K 特征值分解，允许 F 奇异。**U 是吸收了因子风险的暴露，不是原始 X。**

对于任意组合暴露 p（可以是权重、金额，也可以是相对基准的主动权重），定义：

\[
z=A^\top p=p+w(\delta^\top p).
\]

于是完整的修正后风险可以写为：

\[
\boxed{p^\top\Sigma^*p=\|U^\top z\|_2^2+\sum_i d_i z_i^2.}
\]

这正是完整 Σ* 的风险，没有做低秩截断或丢弃 specific risk。分解 F 时只会将相对尺度 1e-12 容差内的负特征值舍入为零；更大的负值直接报错。`factor_square_root_relative_error` 会报告分解重建误差，因此与原始浮点 F 的等价性也以该误差为准。

## 3. 数据准备与 Python 接口

实现位于 [stock_covariance/structured.py](stock_covariance/structured.py)，原 dense 参考实现位于 [stock_covariance/beta_guard.py](stock_covariance/beta_guard.py)。本仓库根目录可直接导入 `stock_covariance`；新增的根目录 `pyproject.toml` 支持将 `barra_guard`、此核心和市场恢复模块一起安装到下游环境。注意 `stock_covariance/pyproject.toml` 仍是独立 uv 应用配置（`package=false`）；安装时指向仓库根目录，具体命令见新集成说明。

```python
from stock_covariance import adjust_from_factors

# X: (N,K); F: (K,K); d: (N,)；w: (N,) 或 (N,1)
model = adjust_from_factors(X, F, d, w, lower_bound=0.15)
beta_star = model.beta_after
print(model.diagnostics)
```

输入检查和口径要求：

- F 半正定；d 非负；原市场方差 s 严格为正。构造 Σ* 不要求每个 d 都大于零。
- 也接受 D 的 N×N 对角矩阵，但建议直接传 d，避免输入存储和检查的 N² 成本；F 的维度是 K×K。
- w 严格非负，合计误差不超过 1e-12；代码不自动截负、不归一化。
- 模型日期、币种、收益频率和风险期限必须一致。若供应商给的是 specific volatility，先按相同口径平方得到 d。
- 在完整风险模型股票集合上先做权重恢复和校准，再映射交易股票。至少必须覆盖参考市场的全部成分股；不要先筛成当前持仓集合再重归一化 w。
- 这里的 w 是 beta 的参考市场组合，不一定是任意一个纯因子的 factor-mimicking portfolio。跨国家模型若参考市场不同，应分别处理，不能直接拼接 local beta。

没有 w 时，可先接入已有恢复模块：

```python
from market_recovery import recover_from_factors

recovered = recover_from_factors(X, F, d, predicted_beta)
if not recovered.market_consistent:
    raise ValueError(f"市场权重恢复与模型不相容: {recovered.diagnostics}")
model = adjust_from_factors(X, F, d, recovered.weights, lower_bound=0.15)
```

恢复路径额外要求 d 严格为正。它的 `market_consistent` 默认容差为 1e-8，比调整接口的输入检查宽松，因此即使诊断通过，最后一行仍可能拒绝微小负权重或预算偏差。这种情况应保留诊断并停止接入；本次增强没有引入自动清洗规则。不要静默裁剪或归一化后继续，也不要只凭能重现 beta 就认定恢复的是供应商实际市场。详细说明见 [MARKET_RECOVERY.md](MARKET_RECOVERY.md)。

返回对象拥有独立、只读的数组副本。`beta_target` 是校准目标，`beta_after` 是由表示的 Σ* 重算的结果；下游 beta 约束推荐使用后者。`market_variance` 和 `updated_market_variance` 分别是修正前后值。数据不变时只构造一次；X、F、d、w 或下限变化时重新构造对象和风险表达式。

## 4. 如何在凸优化里使用 Sigma Star

使用显式辅助变量表示 z，避免把它展开成 N×N 系数：

\[
t=\delta^\top p,\qquad z=p+wt,\qquad y=U^\top z.
\]

将原来的风险项 `quad_form(p, Sigma_star)` 替换为 `sum_squares(y) + sum(d * square(z))`，同时加入这三个等式。二次项的 Hessian 为对角结构，等式的非零系数数量约 O(NK+N)。新变量是求解器内部辅助变量，组合本身仍是 p。

**务必加入 `risk.constraints` 的全部约束。** 单独使用 `risk.variance` 会让辅助变量与持仓脱离，得到错误的风险乃至错误的最优解。

完整接入写法如下；X、F、d、w、alpha_value 由现有数据层提供：

```python
import cvxpy as cp
import numpy as np
from stock_covariance import adjust_from_factors

model = adjust_from_factors(X, F, d, w, lower_bound=0.15)
N = model.n_assets
p = cp.Variable(N, name="holdings")
alpha = cp.Parameter(N, value=np.asarray(alpha_value).reshape(N))
risk = model.cvxpy_risk(p)
gamma = 3.0  # 示例：保留现有策略的风险系数及是否含 1/2 的约定

constraints = [
    cp.sum(p) == 1, p >= 0,
    model.beta_after @ p >= 0.4,
    model.beta_after @ p <= 1.2,
    *risk.constraints,
]
problem = cp.Problem(cp.Minimize(gamma * risk.variance - alpha @ p), constraints)
problem.solve(solver="OSQP", eps_abs=1e-8, eps_rel=1e-8, max_iter=30000)
if problem.status != cp.OPTIMAL:
    raise RuntimeError(f"组合求解未达到要求: {problem.status}")

weights = p.value
actual_variance = model.variance(weights)
np.testing.assert_allclose(risk.variance.value, actual_variance, atol=1e-7, rtol=1e-6)
```

可直接运行的完整合成数据例子见 [structured_example.py](stock_covariance/structured_example.py)。风险模型计算仅依赖 NumPy，调用 `cvxpy_risk` 时才导入 CVXPY：

```bash
cd stock_covariance
uv sync --locked --extra optimization
uv run --locked --extra optimization python structured_example.py
```

对已有优化器，保留原有目标中收益、成本、换手的定义，以及预算、持仓上下限、行业和风格约束；这些仍作用于 p。**不要把持仓约束或收益项改到 z 上，也不要将 z 当作最终持仓返回。** 风险惩罚的系数和原来的 `1/2` 约定也保持不变。

不要用以下写法代替 helper：

```python
# 不推荐：CVXPY 可能把 rank-one 乘积展开成 N×N 系数矩阵
z_inline = p + w * (model.delta @ p)
# 然后直接将 z_inline 放进所有平方项，会失去预期的规模优势。
```

也不要构造完整 A，或把风险写成“原风险 + 一个平方 − 另一个平方”；后者虽然代数正确，CVXPY 通常无法按 DCP 规则识别其凸性。本 helper 使用非负平方和，无需 `psd_wrap` 或 `assume_PSD`。

这是通用的因子模型辅助变量建模方式，参见 [OSQP 官方 portfolio 示例](https://osqp.org/docs/examples/portfolio.html)。本仓库额外加入 t、z，使 beta 调整也保持该结构。

### Tracking error、风险上限和重复优化

若 b 是投资组合的跟踪基准，使用 `risk = model.cvxpy_risk(p - b)` 即得到 `(p-b)ᵀΣ*(p-b)`。b 与参考市场 w 可以不同；p 和 b 必须采用同一股票顺序和单位。

若要限制波动率不超过 `vol_limit`，加入 `risk.volatility <= vol_limit`，并使用支持二阶锥的求解器，例如 CLARABEL。也可限制 `risk.variance <= vol_limit**2`；这已经不是 OSQP 支持的纯线性约束 QP。`variance` 是方差，`volatility` 才是标准差；金额风险与权重风险也不能混用。其他新增约束的求解器兼容性由集成方检查。

固定模型反复改变 alpha、基准或上一期持仓时，尽可能用 `cp.Parameter` 更新数值并复用同一个 `Problem`，按需启用 warm start；用 `problem.is_dpp()` 检查整体模型能否复用编译结果，参见 [CVXPY 官方说明](https://www.cvxpy.org/tutorial/advanced/index.html)。本 helper 接收仿射暴露表达式，但模型本身的 U、d、δ、w 是 NumPy 常量；它没有实现风险数据变化时的参数更新接口。

### 优化股票只是模型的一个子集

仍然先在完整 N 只股票上构造 model，再用稀疏映射 E 将 M 只可交易股票的持仓嵌入全模型：

```python
from scipy import sparse

# indices: M 个不重复的股票位置，顺序就是交易变量的顺序
M = len(indices)
E = sparse.csc_matrix((np.ones(M), (indices, np.arange(M))), shape=(model.n_assets, M))
p = cp.Variable(M)
risk = model.cvxpy_risk(E @ p)
# 跟踪全模型基准时：model.cvxpy_risk(E @ p - benchmark_full)
# 组合 beta：model.beta_after[indices] @ p
# 预算、交易、持仓限制仍作用于 M 维 p，并加入 *risk.constraints。
```

全模型集合中无法交易的既有持仓可以作为常量加到 `E @ p` 上。股票映射、基准和持仓的对齐属于集成方的数据层责任；当前数组 API 不携带股票代码，不会替你排序。

## 5. 后续需要完整矩阵或其他风险量时

```python
Sigma_star = model.to_dense()            # N×N，顺序同 X
Sigma_subset = model.to_dense(indices)   # 指定股票的主子矩阵，顺序同 indices

variance = model.variance(p_value)       # pᵀΣ*p
sigma_times_p = model.matvec(p_value)    # Σ*p
gradient = model.gradient(p_value)       # 2Σ*p
asset_variances = model.diagonal()      # diag(Σ*)
```

完整矩阵按下面的公式按需生成：

\[
\Sigma^*=UU^\top+\operatorname{diag}(d)
+s\bigl(\beta\delta^\top+\delta\beta^\top+\delta\delta^\top\bigr).
\]

只需要 M 只股票的矩阵时，取 U、d、β、δ 的对应行，使用同一个 s；不能在子集上重新定义或归一化市场。`to_dense(indices)` 就实现了这一点，结果等于完整 Σ* 的对应主子矩阵。

结构化预处理为 O(NK²+K³+N×二分迭代数)，结果存储为 O(NK+N)，预处理中还有 K² 工作区。单次风险、矩阵向量乘、梯度和所有个股方差计算为 O(NK)。导出完整矩阵为 O(N²K) 计算和 O(N²) 输出；导出 M 只股票则为 O(M²K) 和 O(M²)。这不包括求解器分解和迭代成本。

## 6. 误差与失败处理

预处理会检查 beta 重建误差、市场方差相对变化、下限违反量和 F 分解误差。二分法默认容差为 1e-12，默认最大迭代 128；输出检查默认使用约 1e-9 的容差（beta 重建按 beta 尺度调整）。不应对浮点 beta 写精确的 `min >= 0.15` 断言。

非法输入抛 `ValueError`；校准或重建失败抛 `NumericalError`（位于 `stock_covariance.beta_guard`）。集成方应记录诊断并沿用现有失败处理路径，不要捕获所有异常后静默使用不合格结果。

模型输入满足假设时的 beta 校准可行，并不意味着下游持仓约束一定可行。`infeasible`、`unbounded`、迭代超限及 `optimal_inaccurate` 都应按生产系统现有规则处理；示例仅接受 `optimal`。即使状态为 optimal，也检查最终原始持仓约束及 `model.variance(p)`；不要只检查辅助变量。

## 7. 集成执行顺序与验收

1. 对齐完整风险模型的股票顺序、日期、币种、期限和单位；确认 d 为方差及 w 的市场定义。缺 w 时使用已有恢复模块并保留诊断。
2. 在现有模型加载处调用 `adjust_from_factors`，记录 diagnostics；每个模型快照只做一次。
3. 在现有优化器里替换风险表达式并加入全部辅助约束。保留原持仓变量、业务约束和目标系数，按实际策略选择总风险或主动风险。
4. 小规模同一输入下，与 `adjust_covariance(X @ F @ X.T + diag(d), w)` 比较导出矩阵、市场方差、beta、随机组合风险和最优目标值。D 奇异导致解不唯一时，以目标、风险和约束可行性为主，不要求持仓逐项相同。
5. 用生产规模与真实约束对比原因子模型、修正后的 dense 和 structured 三条路径。dense 与 structured 才是同一个优化问题；原因子模型只作为性能参照。分别记录预处理、CVXPY 编译、求解耗时、迭代数、约束残差及内存。
6. 检查 `problem.get_problem_data(cp.OSQP)` 的 `P/A/F` 稀疏矩阵规模。仅含本风险块和简单持仓约束时，P 应为对角，约束非零数应按 NK 而非 N² 增长；其他交易成本或业务约束可能额外增加非零项。本检查使用当前 CVXPY 的 OSQP 数据字段，升级版本时重新验证。
7. 通过生产对照后切换默认入口；保留 dense 接口用于导出、小规模比对或已测得更快的场景。本库不提供未经生产测量的自动规模阈值，不自动切换求解器。

没有必要让集成 Agent 重写另一个校准优化器、再做历史收益回归，或去构造新的 Barra F*。本接口表示的是股票协方差方案的 Σ*；原模型的 X、F、d 只是它的输入，不能单独把原 X/F/d 当作调整后的风险模型交给下游。

## 8. 已完成的验证与速度对照

合成数据验证覆盖 dense 等价、风险和梯度、奇异 F 和奇异 Σ、零权重、尺度变化、子集导出、主动风险、组合约束、SOCP 风险上限，以及 CVXPY 实际生成的稀疏结构。实现和测试不依赖真实 Barra 文件。

本机 Python 3.14.4、NumPy 2.5.3、CVXPY 1.9.3、OSQP 1.1.3，单 BLAS 线程，40 个因子；每组 3 次新建问题，报告编译加求解的中位数，单位毫秒：

| 股票数 | 原因子风险 | 修正后 dense | 修正后 structured |
|---:|---:|---:|---:|
| 100 | 23.7 | 8.2 | 23.5 |
| 300 | 32.1 | 25.3 | 30.4 |
| 1,000 | 179.6 | 678.9 | 381.3 |

1,000 股票时，结构化比修正后的 dense 约快 1.8 倍，但比原因子模型约慢 2.1 倍；100、300 股票时 dense 更快。**这项增强避免了矩阵展开，但不能承诺相对于原优化器没有额外耗时。** 本次测量未覆盖真实模型、更大规模或生产业务约束，不能直接当作上线延迟承诺。

1,000 股票时，结构化结果数组约 0.368 MB，完整矩阵 8 MB；交给 OSQP 的 P/A/F 稀疏数组合计约 0.850 MB 对 16.104 MB。数组字节数不含求解器工作区、CVXPY 中间表示和进程峰值内存。dense 路径基于已知 PSD 的构造使用 `psd_wrap`，避免把额外 N×N PSD 检查开销算进去。

同一修正模型下，dense 和 structured 的持仓最大差异约 1.2e-11。原始测量、各次编译/求解耗时、稀疏矩阵规模和约束误差保存在 [structured_benchmark_results.json](structured_benchmark_results.json)。复现（在仓库根目录）：

```bash
# 根目录全部测试现在还包含 pandas 表格适配测试，使用根目录环境
uv sync --locked --extra tables --extra optimization
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m pytest tests -q
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  stock_covariance/.venv/bin/python benchmark_structured.py
# 子目录的 NumPy 调整测试
(cd stock_covariance && .venv/bin/python -m pytest -q)
```

`benchmark_structured.py` 可通过 `--sizes`、`--factors`、`--repeats` 和 `--output` 改变测量范围及保存路径；默认仅测 100、300、1,000 股票和 40 因子，避免不必要的大规模计算。
