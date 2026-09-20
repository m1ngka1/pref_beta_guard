# 本地实现与验证记录

验证日期：2026-09-20。使用小规模合成数据，未接入真实 Barra 文件。两套实现分别放在独立目录，各有自己的 uv 项目、锁文件和 `.venv`。

## 环境与运行方式

本机实际版本：Python 3.14.4、NumPy 2.5.3、pytest 9.1.1；凸优化验证使用 CVXPY 1.9.3、CLARABEL 0.11.1 和 OSQP 1.1.3。股票协方差调整本身不需要 CVXPY；本次结构化优化验证已在该目录环境启用可选的 `optimization` extra。

每个子目录中运行：

```bash
uv sync --locked
uv run --locked pytest -q
uv run --locked python example.py
```

环境已经建立，也可在对应目录直接执行 `.venv/bin/python -m pytest -q`。安装使用当前项目内的虚拟环境，没有修改全局 Python 包。

## 两套代码的独立测试

| 实现 | 最终结果 | 主要检查 |
|---|---|---|
| 股票协方差方案 | **51 passed** | 已知四股票解、20 组随机协方差、beta 下限、排序、市场方差、回归剩余协方差、PSD、零权重股票、奇异矩阵、单位缩放及错误路径。 |
| 因子协方差方案 | **24 passed** | 可解析双因子最优解、全因子空间与逐股票解的一致性、排序和精确值约束、原 beta 并列、相关 specific covariance、共线 loading、奇异 F、单位缩放及无解/求解器失败。 |

股票协方差方案还用 5 组小规模问题，穷举约束 active set 并解 KKT 线性系统，与二分法结果对照，验证其加权 beta 平方改动目标；此验证不使用凸优化器。

两个示例脚本均已实际运行。股票协方差方案复现：

```text
原 beta：[-0.10, 0.10, 1.80, 2.20]
新 beta：[ 0.15, 0.15, 1.65, 2.05]
共同平移量：约 0.15
二分迭代数：39
```

因子协方差方案示例开启排序和负 beta 精确固定选项，得到 `optimal`；同一脚本随后运行不可行的单因子例子，并正确捕获 `InfeasibleAdjustmentError`。

## 接入后续组合优化

根目录的 `verify_optimizer.py` 在同一个四股票、双因子模型上运行两种修正，再分别把输出协方差直接传入 CVXPY 的最小方差组合优化。没有使用 `psd_wrap` 或 `assume_PSD` 跳过矩阵识别。

组合约束为：权重非负、合计为 1、单只股票权重不超过 0.7、组合 beta 在 0.3 至 1.1 之间。两次优化均返回 **optimal**，脚本也重新检查了持仓约束。

| 该例子的检查量 | 股票协方差方案 | 因子协方差方案 |
|---|---:|---:|
| 新协方差最小特征值 | 0.03983778 | 0.04000000 |
| 市场方差相对误差 | 8.80e-13 | 0 |
| 目标 beta 与矩阵重算 beta 的最大误差 | 9.02e-13 | 2.22e-16 |
| 重算 beta 的下限违反量 | 6.61e-14 | 0 |
| 后续组合优化状态 | optimal | optimal |

两套方案修改后的 covariance 不同，因此最小方差持仓和目标值不同；这不是哪套方案预测更准确的证据。表中的误差是该端到端例子的实际值，不是全部输入下的误差上界。

## 同一输入下，因子协方差方案无解而股票协方差方案可解

另使用一个两股票、单因子模型：loading 为 `[1, -1]`，因子方差为 1，两只股票的 specific variance 均为 0.01，市场权重为 `[0.75, 0.25]`。

保持原 loading、specific risk 和市场方差时，单个因子的方差没有调整自由度，因此因子协方差方案返回明确的不可行异常。股票协方差方案正常返回：

```text
新 beta：[1.28333333, 0.15]
```

该结果仍满足原市场权重加权平均 beta 为 1，且更新矩阵通过 PSD 检查。

从根目录复现端到端验证：

```bash
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  factor_covariance/.venv/bin/python verify_optimizer.py
```

已保存的完整输出见 [validation_results.json](validation_results.json)。

## 保证的范围

股票协方差方案的可行性结论依赖合法 PSD 协方差、正的市场方差、非负且合计为 1 的市场权重，以及 0 和 1 之间的下限。数学上的有解不等于任意浮点输入都能达到任意严格容差；代码在未收敛或重建检查失败时明确报错。

为保持常规调用速度，原 dense 股票协方差方案默认不执行完整矩阵特征值检查；验证中使用 `check_psd=True`。因子协方差方案始终检查因子 F，额外完整矩阵检查也可开启。原接口没有自动裁剪特征值、调整权重或放宽 beta 约束。新增结构化路径为求 F 的 PSD 平方根，允许将相对 1e-12 容差内的负特征值舍入为零，并报告分解重建误差；不接受实质不定的 F。

这些测试验证公式实现、约束处理和优化器对接，不验证真实市场上的预测效果，也不保证任意后续持仓约束都可行。

## 增补：缺少市场权重时的恢复验证

`market_recovery.py` 新增 dense 求解与因子结构求解两条路径。输入支持 X 为 N×K、F 为 K×K dense covariance、D 为 N×N 对角矩阵，以及 beta 为 N×1 列向量；也支持 D 对角线和 beta 的一维数组形式。

`tests/test_market_recovery.py` 实际运行结果为 **43 passed**。其中包含 20 组随机模型的已知权重恢复、两条求解路径对照、奇异 F、零权重股票、单位缩放，以及输入不完整或 beta 经过变换后的兼容性诊断。另验证恢复出的正权重可接入现有股票协方差调整函数。

六股票示例的真实市场方差为 0.0559085，因子路径恢复值为 0.05590849999999985，权重最大误差为 7.22e-16；与 dense 路径的权重差异为 6.94e-16。实际输出见 [recovery_results.json](recovery_results.json)，运行方法与解释见 [MARKET_RECOVERY.md](MARKET_RECOVERY.md)。

这是合成数据上的代数验证。当前用户确认没有本地实际模型数据，因此未声称完成真实 Barra 权重恢复。

## 增补：结构化股票协方差与凸优化接口

`stock_covariance/structured.py` 提供不生成 N×N 矩阵的预处理、风险、梯度、矩阵向量乘、对角线和按需导出；CVXPY helper 使用显式辅助变量避免把 rank-one 变换展开成 N×N 系数。完整集成要求见 [STRUCTURED_INTEGRATION.md](STRUCTURED_INTEGRATION.md)。

结构化核心完成时的阶段回归验证共 **188 passed**（下文还有 standalone 模块增补）：

| 测试范围 | 数量 | 新增验证 |
|---|---:|---|
| `stock_covariance/tests` | 104 | 原 51 项及新增 53 项：与 dense API 对照、36 组随机/尺度组合、风险、梯度及有限差分、子集导出、奇异模型、只读快照、非法输入和无 N×N 预处理。 |
| `factor_covariance/tests` | 24 | 原方案回归通过，实现未修改。 |
| 根目录 `tests` | 60 | 原市场恢复 43 项及新增 17 项：dense/structured 最优解、主动风险、换手和 beta 约束、SOC 风险上限、子集映射、Parameter 复用及 OSQP 稀疏系数结构。 |

结构检查还对照了直接内联 `p + w * (delta @ p)` 的写法，验证它确实可能生成 N² 规模的系数；helper 的等式非零项保持 O(NK+N)，二次项为对角。`structured_example.py` 已实际运行并得到 optimal，其风险与导出矩阵重算结果一致。

本地单线程 OSQP 测量（40 因子，每组 3 次，编译加求解中位数）结果如下，单位为毫秒：

| 股票数 | 原因子风险 | 修正后 dense | 修正后 structured |
|---:|---:|---:|---:|
| 100 | 23.7 | 8.2 | 23.5 |
| 300 | 32.1 | 25.3 | 30.4 |
| 1,000 | 179.6 | 678.9 | 381.3 |

结构化并非在每个规模都更快，也没有消除相对原因子模型的求解开销。1,000 股票时，存储结果数组约 0.368 MB 对 dense 的 8 MB；OSQP 的 P/A/F 稀疏数组合计约 0.850 MB 对 16.104 MB。这里没有测量进程峰值内存。dense 测速使用 `psd_wrap` 跳过已知 PSD 构造的额外验证；小规模正确性测试不跳过该识别。

两种表示对应的最优持仓最大差异约 1.2e-11。完整逐次结果及求解矩阵大小见 [structured_benchmark_results.json](structured_benchmark_results.json)。生产速度还取决于实际模型条件数、因子数、求解器和组合约束，当前未有真实数据验证。

已在股票协方差环境离线安装锁定的 optimization extra；从仓库根目录复现：

```bash
(cd stock_covariance && OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest -q)
(cd factor_covariance && OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest -q)
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 stock_covariance/.venv/bin/python -m pytest tests/test_market_recovery.py tests/test_structured_optimization.py -q
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 stock_covariance/.venv/bin/python stock_covariance/structured_example.py
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 stock_covariance/.venv/bin/python benchmark_structured.py
```

## 当前入口：单次最新快照

按实际使用方式收敛：数据加载后调用 `adjust_barra(X, F, d, ...)` 一次，结果保存进 context，未来各日复用同一份风险模型。移除了多日期表格管线、Trade Planner bridge、强制 snapshot 元数据及 pandas 依赖。原始数学核心与两种方法保持不变。

旧 bridge 的合成验证属于提交 `76caad0` 的历史结果，相关脚本和结果现已移除。当前版本不对 Trade Planner 做任何自动接入；接入说明见 [SNAPSHOT_INTEGRATION.md](SNAPSHOT_INTEGRATION.md)。

新测试直接检查一个准备结果被多个未来持仓表达式复用，验证风险与同一 Σ* 一致、输入后续变化不会污染已准备结果；也保留权重恢复、子集等价和求解规模检查。

当前回归结果为 **208 passed**：根目录 80（其中单快照入口 20）、股票协方差 104、因子协方差 24。测试数量变化是移除了已删除多日期/bridge 的测试，并增加单次准备、多日复用及输入隔离检查。独立例子运行 optimal，0.2.0 wheel 在仓库外仅有 NumPy 的环境里通过预处理和子集风险验证；无 pandas、CVXPY 或 planner 模块依赖。
