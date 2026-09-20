# 从 predicted beta 恢复市场权重和方差

当数据没有市场组合权重，但有同口径的 predicted beta 和风险模型时，可以尝试恢复这些权重。这里恢复的是 **beta 所参考的市场组合**，不一定是某个纯因子的 factor-mimicking portfolio。

## 输入应该取哪些股票？

推荐先取该模型覆盖的全量股票，而不是只取当前持仓。严格来说，至少需要覆盖参考市场组合的全部成分股；额外的非成分股可以保留，它们的真实市场权重应为零。

设有 N 只股票、K 个因子，则输入为：

- $X$：N×K 的 factor loading，每行对应一只股票，每列对应一个因子。
- $F$：K×K 的 dense factor covariance，必须半正定；其大小由因子数决定。
- $D$：N×N 对角 specific covariance，对角元素是 specific variance，不是 volatility。
- $\beta$：N×1 的原 predicted beta，股票顺序与 X、D 一致。

实现也接受一维 beta，以及只保存 D 对角线的一维 specific variance 数组。所有 beta 必须相对同一个参考市场，且数据日期、币种和预测期限一致；不能把不同国家相对各自市场的 local beta 混成一个共同市场的 beta 向量。

股票协方差为：

$$
\Sigma=XFX^\top+D.
$$

只要 F 半正定且 D 的每个对角元素严格为正，Sigma 就一定正定，即使 X 有共线列也不影响这个结论。若存在零 specific variance，Sigma 仍可能正定，但需要单独检查，不能仅由上述条件保证。

## 求解顺序

令 $w$ 为未知市场权重，$s$ 为未知市场总方差。原定义为：

$$
\beta=\frac{\Sigma w}{s},
\qquad s=w^\top\Sigma w.
$$

先解一个线性方程，得到中间向量 $v$：

$$
\boxed{\Sigma v=\beta.}
$$

不需要显式计算矩阵逆。由 $\Sigma w=s\beta$ 得到 $w=sv$；代回市场方差定义，则：

$$
s=s^2\beta^\top v.
$$

所以，在 Sigma 正定、beta 非零时，唯一的隐含权重与方差为：

$$
\boxed{s=\frac{1}{\beta^\top v},\qquad w=\frac{v}{\beta^\top v}.}
$$

这里的分母是 beta 与 v 的内积，**不是 v 的元素之和**。不要直接归一化 v，因为那样可能改变重算出来的 beta。

## 计算时不必生成完整 Sigma

[market_recovery.py](market_recovery.py) 提供两条路径：

```python
from market_recovery import recover_from_factors, recover_from_covariance

# 推荐：输入 X、dense F、对角 D 和 beta 列向量。
result = recover_from_factors(X, F, D, beta)
w = result.weights               # 返回一维数组，便于接入现有代码
s = result.market_variance
market_vol = result.market_volatility

# 如果已经有完整 Sigma，也可以直接求解。
reference = recover_from_covariance(Sigma, beta)
```

第一条路径利用 Woodbury 恒等式，将求解转为 K×K 的系统；不会构建完整 N×N 的 Sigma，也不要求 F 可逆。计算量约为 O(NK²+K³)。如果 D 已按完整矩阵传入，还会检查它确实为对角矩阵；直接传对角线数组可省去这部分 N² 检查和输入存储。

第二条路径是 dense 参考实现，先以 Cholesky 检查 Sigma 正定，再求解，计算量为 O(N³)。对于存在相关 specific covariance 的完整 D，应使用这一条路径，不可擅自丢弃 D 的非对角元素。

## 得到权重后，先看诊断

两个函数都返回原始隐含权重，不截负值、不归一化，也不自动修改现有 beta 调整模型。

```python
print(result.market_consistent)
print(result.diagnostics)
```

主要诊断包括权重之和、负权重总量、线性方程残差，以及从恢复权重重算 beta 的误差。`market_consistent` 表示权重在默认 1e-8 的容差内，与满仓 long-only 市场相容；不代表已经证明它就是供应商的实际市场组合。

尤其要注意：对于任意非零 beta，正定 Sigma 都能产生一组隐含权重，并重现这个 beta。因此，**重算 beta 很准确并不能单独证明恢复正确**。权重合计、非负性和参考市场定义仍需检查。

如果只取股票子集、beta 字段经过四舍五入，或输入口径不匹配，恢复结果可能出现净权重偏离 1、负权重等问题。此时代码保留结果并标记不一致；不会把它“修好”后假装恢复成功。线性求解或重建数值误差过大时，则抛出 `RecoveryNumericalError`。

浮点舍入也可能让真实为零的权重出现约 1e-15 的负值。诊断会允许这种数值误差，但原始权重仍不改变；现有 beta 调整接口严格要求非负权重，后续若要处理这类误差，需要明确处理规则并重新检查 beta，不能直接无条件裁剪和归一化。

## 本地验证

目前没有实际模型数据。示例按上述矩阵形状生成一个已知市场，再只把 X、F、D、beta 交给恢复函数，最后与隐藏的真实权重和方差比较。

从项目根目录运行：

```bash
stock_covariance/.venv/bin/python recover_market_example.py
stock_covariance/.venv/bin/python -m pytest tests/test_market_recovery.py -q
```

本地结果为 **43 项测试通过**，覆盖已知解、20 组随机模型、奇异 F、单位缩放、零权重成分、列向量和对角矩阵输入，以及缺失市场成分、四舍五入 beta、long-short 权重等诊断场景。

六股票示例恢复权重的最大误差为约 7.22e-16，恢复市场方差为 0.0559085，与生成数据时的真值一致到浮点误差。完整示例输出保存在 [recovery_results.json](recovery_results.json)。
