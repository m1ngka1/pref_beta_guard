# 股票协方差方案：股票协方差调整

独立的 NumPy 调整实现，不需要凸优化器或历史回归。二分法计算统一下限与平移量，再闭式更新协方差。新增结构化接口直接接收 X、F、specific variance 和市场权重，避免生成完整股票协方差，并提供可选的 CVXPY 风险接口。

**下游 Agent 集成请先读 [完整集成说明](../STRUCTURED_INTEGRATION.md)**，其中包含数学推导、输入约定、可运行例子、辅助约束、tracking error、股票子集、矩阵导出、速度结果和验收顺序。

## 推荐的结构化调用

在本目录中：

```python
from structured import adjust_from_factors

model = adjust_from_factors(X, F, d, market_weights, lower_bound=0.15)
variance = model.variance(portfolio_weights)
beta_star = model.beta_after
Sigma_star = model.to_dense()  # 按需调用；优化时通常不需要
```

在仓库根目录或作为模块集成时使用 `from stock_covariance import adjust_from_factors`。d 是方差向量；也接受对角 D，不能是一般相关 specific covariance。返回对象还提供 `matvec`、`gradient`、`diagonal`、`to_dense(indices)` 和 `cvxpy_risk`。

可选的凸优化演示：

```bash
uv sync --locked --extra optimization
uv run --locked --extra optimization python structured_example.py
```

`risk = model.cvxpy_risk(p)` 返回 `risk.variance`、`risk.volatility` 和 **必须加入问题的 `risk.constraints`**。持仓约束和 alpha 仍作用于 p，不作用于内部辅助变量。不要自行将线性变换展开成 N×N 矩阵。

## 运行

在本目录执行：

```bash
uv sync --locked
uv run --locked python example.py
uv run --locked pytest -q
```

依赖与版本固定在本目录的 `pyproject.toml` 和 `uv.lock`；虚拟环境是本目录的 `.venv`。测试同样不依赖凸优化器。

## 原完整矩阵接口（保持兼容）

```python
from beta_guard import adjust_covariance

# Sigma = X @ F @ X.T + D
result = adjust_covariance(Sigma, market_weights, lower_bound=0.15)
Sigma_new = result.covariance
beta_new = result.beta_after
```

- `Sigma`：完整股票协方差矩阵，必须半正定，市场方差必须大于零。
- `market_weights`：与矩阵股票顺序一致的市场权重，一维数组，非负、合计为 1，允许部分权重为零。
- `lower_bound`：统一下限，必须在 0 和 1 之间；不仅调整负 beta，也处理低于下限的正值。
- `tolerance`：二分法的加权平均误差容差，默认 `1e-12`。
- `max_iterations`：二分法上限，默认 128；超限或浮点精度不足时抛出 `NumericalError`。
- `check_psd=True`：额外检查输入、输出矩阵的半正定性。需要特征值分解，因此默认关闭，适用于已确认合法的风险模型输入；关闭检查并不放宽输入必须半正定的要求。

返回值中，`beta_before` 是从输入矩阵重算的原 beta，`beta_target` 是二分法生成的目标，`beta_after` 是从输出矩阵重算的 beta。`shift`、`iterations` 和 `diagnostics` 用于检查收敛与重建误差。

也可以独立调用 `calibrate_betas(beta, market_weights)` 获取目标 beta；它会检查原 beta 的市场加权平均为 1。

## 算法和保证

1. 计算 `s = w @ Sigma @ w`，`beta = Sigma @ w / s`。
2. 二分法求 `w @ maximum(L, beta - shift) = 1`。
3. 得到 `beta_target = maximum(L, beta - shift)`。
4. 更新 `Sigma_new = Sigma + s * (outer(beta_target, beta_target) - outer(beta, beta))`。代码使用等价的增量展开，减少小改动时的相减误差。
5. 重算 beta 与市场方差，检查输出后再返回。

上述输入条件下，标量方程有唯一非负解；更新保持市场方差、剔除市场后的剩余协方差和 beta 的弱排序。低 beta 可以并列。实现中的保证以数值容差为准，不会把未收敛结果当作成功。

原 dense 接口中，二分法每轮为 O(N)，生成完整输出矩阵为 O(N²)；启用完整 PSD 检查为 O(N³)。不修改输入、不自动归一化权重、不裁剪特征值。结构化接口需要对 K×K 的 F 求平方根，仅允许将 1e-12 相对容差内的负特征值舍入为零，并报告重建误差；没有实质性的 PSD 修复。`ValueError` 表示非法输入，`NumericalError` 表示收敛或重建检查失败。

## 验证

测试覆盖聊天中的四股票例子、无需调整、零权重股票、奇异协方差、方差单位缩放、20 组随机矩阵、非法输入和未收敛路径。另用小规模穷举 active set 并求解 KKT 线性系统，独立验证二分法确实得到市值加权 beta 改动最小的解。

与另一方案的端到端比较、实际执行结果见上级目录的 `VALIDATION.md`。所有示例均为合成数据，不代表已验证真实 Barra 预测效果。
