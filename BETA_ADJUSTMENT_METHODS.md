# Predicted beta 调整：两种协方差调整方法的计算顺序

我们希望把负的 predicted beta 调成正值，并让调整后的协方差与新 beta 一致。下面采用统一下限 $L=0.15$：低于下限的正 beta 也一起处理，避免把原来的排序反转。上标 $*$ 表示调整后的量。

## 1. 先从 predicted beta 的定义开始

一只股票的 predicted beta，是它与市场收益的预测协方差，除以市场的预测方差：

$$
\beta_i=\frac{\operatorname{Cov}(r_i,r_m)}{\operatorname{Var}(r_m)}.
$$

这里 $r_i$ 是第 $i$ 只股票的收益，$r_m$ 是计算 beta 所用市场组合的收益。

在 Barra 模型中，股票协方差矩阵为：

$$
\Sigma=XFX^\top+D.
$$

其中，$X$ 是 factor loading 矩阵，每一行对应一只股票；$F$ 是 factor covariance；$D$ 是 specific covariance，通常为 specific variance 构成的对角矩阵。如果输入是 specific volatility，需要先平方。上标 $\top$ 表示转置。

令 $w$ 为这个市场组合的股票权重列向量，$s$ 为市场总方差，则所有股票的 beta 可以一起算出：

$$
s=w^\top\Sigma w,
\qquad
\boxed{\beta=\frac{\Sigma w}{s}.}
$$

本文假设 $\Sigma$ 半正定、$s>0$，市场为 long-only 满仓组合，即 $w_i\ge0$、$\sum_iw_i=1$。股票集合要覆盖市场的全部持仓，数据使用相同币种和预测期限。

由定义必然有：

$$
w^\top\beta=1.
$$

所以，抬高市场成分股的负 beta 时，通常也需要调整其他股票，才能让市场自身的 beta 仍为 1。两种方法都选择保持市场方差 $s$ 不变。

## 2. 股票协方差方案：二分法求 beta，再更新股票协方差

### 第一步：求一个共同平移量

设 $\lambda$ 为所有股票共用的平移量。我们希望新 beta 为：

$$
\beta_i^*=\max(L,\beta_i-\lambda),
\qquad L=0.15.
$$

也就是先统一向下平移，再将低于下限的值设为下限。为了让市场加权平均仍等于 1，只需求解一个标量方程：

$$
\boxed{\sum_iw_i\max(L,\beta_i-\lambda)=1.}
$$

用二分法即可：

1. 如果 $\lambda=0$ 已满足等式，直接使用它。
2. 否则在 $[0,\max_i\beta_i-L]$ 内二分。
3. 每次计算中点处的加权平均；若大于 1，就增大 $\lambda$，否则减小它。
4. 当加权平均与 1 的误差达到设定容差时停止，例如 $10^{-12}$。

为什么有解？左端点处的加权平均至少为 1，右端点处等于 $L<1$；该函数连续，并且在经过 1 的位置严格递减。因此，在前述假设和 $0<L<1$ 下，存在唯一的非负解。这里不需要凸优化器。

### 第二步：生成新 beta

将求出的 $\lambda$ 代回：

$$
\boxed{\beta_i^*=\max(L,\beta_i-\lambda).}
$$

这样所有 beta 都不低于 $L$，市场加权平均仍为 1，原来的排序不会反转，但低 beta 股票可能并列。未落到下限的股票之间，beta 差距保持不变。非市场成分股也使用同一个映射，但不参与加权校准。

例如，四只等权股票的 beta 从 $(-0.1,0.1,1.8,2.2)$ 变为 $(0.15,0.15,1.65,2.05)$，调整前后的平均值都为 1。

对于权重为正的市场成分股，这个解也恰好最小化满足下限和加权平均约束时的 $\sum_iw_i(\beta_i^*-\beta_i)^2$，因此有“尽量少改 beta”的明确含义。

### 第三步：直接更新股票协方差

把原协方差中与市场相关的部分拆出来，记剩余协方差为 $R$：

$$
R=\Sigma-s\beta\beta^\top.
$$

$R$ 表示股票收益剔除市场收益后的回归剩余协方差。它仍包含市场之外的行业、风格等共同风险，不等于 Barra 的 specific covariance $D$。

保持 $R$ 和市场方差 $s$ 不变，只替换 beta：

$$
\boxed{
\Sigma^*=R+s\beta^*\beta^{*\top}
=\Sigma+s\left(\beta^*\beta^{*\top}-\beta\beta^\top\right).
}
$$

这是直接的矩阵计算，不需要另做历史回归。由于 $R\succeq0$、$Rw=0$，且 $w^\top\beta^*=1$，可以保证：

$$
\Sigma^*\succeq0,
\qquad w^\top\Sigma^*w=s,
\qquad \frac{\Sigma^*w}{s}=\beta^*.
$$

即新矩阵合法、市场方差不变，从新矩阵重算出的 beta 就是目标值。原矩阵若正定，新矩阵也正定，但条件数不保证不变。

输出为 $\beta^*$ 和 $\Sigma^*$。整个流程是“一次标量二分法＋闭式更新”，协方差变化至多为 rank-2；需要时再生成完整的股票协方差矩阵。

## 3. 因子协方差方案：二次规划求调整量，再更新因子协方差

因子协方差方案保持原 $X,D$ 不变，只修改 $F$。这会限制哪些 beta 调整可以实现。

### 第一步：算出市场因子暴露及相关协方差

先算市场的因子暴露向量 $q$：

$$
q=X^\top w.
$$

再算向量 $c$ 和标量 $t$：

$$
\boxed{c=Fq,\qquad t=q^\top Fq.}
$$

$c$ 是每个因子收益与“市场收益的共同因子部分”的协方差；$t$ 是这一市场共同因子部分的方差。注意它与市场总方差 $s$ 的关系是：

$$
s=t+w^\top Dw,
\qquad \beta=\frac{Xc+Dw}{s}.
$$

下面的因子协方差更新要求 $t>0$。

### 第二步：解二次规划，得到 h

令 $h$ 为因子空间中的 beta 调整向量，它的长度等于因子数量。新 beta 被限制为：

$$
\beta^*=\beta+Xh.
$$

为了满足下限，同时尽量少改 beta，求解：

$$
\begin{aligned}
\min_h\quad &\frac12(Xh)^\top W(Xh)\\
\text{s.t.}\quad &(\beta+Xh)_i\ge L,\quad\text{对每只股票 }i,\\
&q^\top h=0.
\end{aligned}
$$

这里大写 $W$ 是衡量各股票 beta 改动的正对角权重矩阵，默认可取单位矩阵；它与市场权重向量 $w$ 不同。约束 $q^\top h=0$ 保证市场加权平均 beta 仍为 1。

这个基础问题不自动保留排序。如果需要，就按原 beta 排序，对相邻股票加入“新 beta 不下降”的线性约束；如果要求原负 beta 精确等于 $L$，则对这些股票再加等式约束。

由于所有调整必须通过 $Xh$ 表达，这一步可能无解。只有得到有效可行解后，才能继续更新 $F$。

### 第三步：拆出因子协方差的剩余部分

用第一步算出的 $c,t$，直接计算：

$$
F_\perp=F-\frac{cc^\top}{t}.
$$

这里 $F_\perp$ 是因子收益剔除市场共同因子收益后的回归剩余协方差。

具体来说，若 $f$ 是因子收益向量，市场共同因子收益就是 $m_f=q^\top f$。因此：

$$
\operatorname{Cov}(f,m_f)=Fq=c,
\qquad \operatorname{Var}(m_f)=t.
$$

因子对 $m_f$ 的回归斜率就是 $c/t$。这些量都已经包含在预测协方差 $F$ 里，**不需要取历史数据另外跑回归**。

### 第四步：直接更新因子协方差

先将因子与市场共同因子收益的协方差更新为 $c^*=c+sh$，再保留 $F_\perp$、替换市场相关部分：

$$
\boxed{
F^*=F_\perp+\frac{c^*c^{*\top}}{t}
=F-\frac{cc^\top}{t}
+\frac{(c+sh)(c+sh)^\top}{t}.
}
$$

注意：调整量是 $sh$，分母是 $t$，两种市场方差不能混用。由 $q^\top h=0$ 可得：

$$
F^*\succeq0,
\qquad F^*q=c+sh,
\qquad q^\top F^*q=t.
$$

最后生成新股票协方差：

$$
\boxed{\Sigma^*=XF^*X^\top+D.}
$$

它的市场方差仍为 $s$，重算 beta 得到：

$$
\frac{\Sigma^*w}{s}
=\frac{X(c+sh)+Dw}{s}
=\beta+Xh
=\beta^*.
$$

输出为 $\beta^*,F^*$，以及按需生成的 $\Sigma^*$。整个流程是“一次二次规划＋闭式更新”，没有第二个优化问题，也没有额外的历史回归。已有 $X,F,D,w$ 时，不需要另外提供 factor portfolio 权重。

## 4. 当前选择与检查

当前优先使用股票协方差方案：允许改变原来的因子表示，换取下限与弱排序的可行性保证，以及更简单的计算。因子协方差方案适合必须保留原 $X,D$ 的场景，但可能在二次规划阶段无解。

两种方法完成后，都应在数值容差内检查 beta 下限、$w^\top\beta^*=1$、市场方差不变，以及从新协方差重算的 beta 是否一致。排序则按所选方法的要求检查。

这些调整不保证个股波动率不变，也不保证所有组合风险上升。它们保证的是模型内部的一致性，不是预测一定更准确；后续组合优化的持仓约束是否兼容，仍是另一个问题。

## 5. 股票协方差方案的结构化计算

令 $\delta=\beta^*-\beta$、$A=I+\delta w^\top$，可将同一个更新写为 $\Sigma^*=A\Sigma A^\top$。不需要真的生成 A 或完整的 $\Sigma^*$。

将 $F=CC^\top$、$U=XC$。对组合暴露 p，令 $z=p+w(\delta^\top p)$，则修正后方差为：

$$
p^\top\Sigma^*p=\|U^\top z\|_2^2+\sum_i d_i z_i^2.
$$

其中 d 是对角 specific variance。风险计算仍然利用因子结构，并与完整矩阵结果一致。凸优化中需要显式辅助变量来避免展开成大矩阵；新增 `stock_covariance.adjust_from_factors` 和 `cvxpy_risk` 已实现该接口。完整推导、导出矩阵及下游集成顺序见 [STRUCTURED_INTEGRATION.md](STRUCTURED_INTEGRATION.md)。
