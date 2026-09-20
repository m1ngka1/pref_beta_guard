# 因子协方差方案：因子协方差调整

独立的 NumPy + CVXPY 实现。保持 loading、specific covariance 和市场方差不变，先解关于 h 的二次规划，再闭式更新因子协方差。没有额外历史回归。

## 运行

在本目录执行：

```bash
uv sync --locked
uv run --locked python example.py
uv run --locked pytest -q
```

本目录有独立的 `pyproject.toml`、`uv.lock` 和 `.venv`。默认求解器是 CLARABEL；可通过 `solver` 和 `solver_options` 更改，参数遵循 [CVXPY 求解器接口](https://www.cvxpy.org/tutorial/solvers/index.html)。

## 调用

```python
from beta_guard import adjust_factor_covariance

result = adjust_factor_covariance(
    X, F, specific_variances, market_weights,
    lower_bound=0.15,
    preserve_order=True,
    pin_negative=False,
)
F_new = result.factor_covariance
Sigma_new = result.covariance
beta_new = result.beta_after
```

- `X`：N×K loading 矩阵。
- `F`：K×K 因子协方差，必须半正定。
- `specific_variances`：N 维 specific **variance** 数组，不是 volatility；也可以传入完整 N×N specific covariance。
- `market_weights`：N 维市场权重，非负且合计为 1。股票和因子顺序、币种、预测期限必须匹配。
- `objective_weights`：目标函数 W 的正对角元素，长度为 N，默认全 1；不是市场权重。
- `preserve_order`：默认 `False`，开启后加入排序约束，原 beta 精确相同的股票也保持并列。
- `pin_negative`：默认 `False`，开启后要求原负 beta 恰好等于下限。
- `feasibility_tolerance`：输出约束检查容差，默认 `1e-7`。
- `check_psd`：默认 `False`。F 始终进行 PSD 检查；开启后额外检查完整 D 与输出股票协方差，可能需要 O(N³) 运算。关闭检查时，完整 D 仍须由调用者保证半正定。

返回 `h`、新 F、新股票协方差、原 beta、目标 beta、从输出矩阵重算的 `beta_after`，以及 `status`、目标函数值和数值诊断。原数据不会被修改。如果无需调整，直接返回 `status="unchanged"`，不调用求解器。

## 计算顺序

1. `q = X.T @ w`，`c = F @ q`，`t = q @ c`。
2. `s = t + w @ D @ w`，`beta = (X @ c + D @ w) / s`。要求 t 和 s 都严格为正。
3. 求 h，最小化 `0.5 * sum(objective_weights * (X @ h)**2)`，约束 `beta + X @ h >= L`、`q @ h = 0`，以及选定的排序或精确值约束。
4. 计算 `F_perp = F - outer(c, c) / t`，`c_new = c + s * h`。
5. `F_new = F_perp + outer(c_new, c_new) / t`，`Sigma_new = X @ F_new @ X.T + D`。
6. 从新矩阵重算 beta 和市场方差，并检查全部目标约束。

求解后会移除 h 在 q 方向上微小的浮点残差，然后重新验证约束。这不是放宽目标约束。分子中的调整量是 s×h，分母是 t，不能混用两种市场方差。

## 无解与失败

本方法不保证可行。排序约束和负值精确目标可能进一步缩小可行域。

- `InfeasibleAdjustmentError`：求解器确认约束无解。
- `SolverFailureError`：求解器失败、返回 inaccurate 等非 optimal 状态，或结果没有通过数值检查。
- `ValueError`：输入非法，包括 t=0、specific variance 为负等。

这些情况不会伪造成功结果，不会自动放宽约束或切换到股票协方差方案。示例脚本包含一个明确无解的单因子模型，展示如何处理异常。

## 验证

测试包含可解析的双因子例子、可选排序和精确值约束、全因子空间与逐股票解的一致性、相关 specific risk、零权重股票、共线 loading、奇异 F、单位缩放、无解和求解器失败路径。

执行结果和两种方案进入后续组合优化的检查见上级目录的 `VALIDATION.md`。验证使用合成数据，不代表真实 Barra 数据上的预测表现。
