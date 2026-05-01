# lmfit 置信区间计算和不确定性评估实现分析

## 重要澄清：关于"自举法 (Bootstrap)"

**在阅读本报告之前，请务必理解以下关键事实：**

### lmfit 中**不存在** Bootstrap (自举法) 实现

经过系统性搜索和代码核查，确认 lmfit 代码库中**完全没有** Bootstrap 自举法的实现。

#### 缺席依据

| 核查维度 | 结果 | 依据 |
|----------|------|------|
| **关键词搜索** | 无匹配 | `bootstrap`、`boot.?strap`、`resample`、`jackknife` 全代码库搜索均无结果 |
| **公开 API** | 无相关函数 | `lmfit/__init__.py` 只导出: `conf_interval`, `conf_interval2d`, `Minimizer.emcee()` |
| **官方文档** | 无相关描述 | `doc/confidence.rst` 和 `doc/fitting.rst` 只描述轮廓法和 MCMC |
| **示例文件** | 无相关示例 | `examples/` 目录下无任何 bootstrap 相关代码 |
| **测试文件** | 无相关测试 | `tests/` 目录下无任何 bootstrap 相关测试 |

#### 概念澄清：MCMC vs Bootstrap

之前的分析可能存在概念混淆，在此明确区分：

| 特性 | MCMC (emcee) | Bootstrap (自举法) |
|------|-------------|-------------------|
| **方法类别** | 贝叶斯方法 | 频率主义方法 |
| **核心思想** | 从后验分布采样 | 对数据有放回重采样 |
| **数据操作** | 不修改原始数据 | 重采样生成多组伪数据 |
| **参数操作** | 探索参数空间 | 对每组重采样数据重新拟合 |
| **结果解释** | 可信区间 (Credible Interval) | 置信区间 (Confidence Interval) |
| **lmfit 中是否存在** | ✅ 存在 (`minimizer.emcee()`) | ❌ 不存在 |

**注意**: lmfit 使用的 `emcee` 是 **MCMC (马尔可夫链蒙特卡洛)** 方法，属于贝叶斯推断框架，与 Bootstrap 自举法有本质区别。

---

## 概述

lmfit 提供了**三种**置信区间计算和不确定性评估方法：

| 方法 | 实现模块 | 方法类别 | 核心原理 |
|------|----------|----------|----------|
| **1. 协方差矩阵估计** | `minimizer.py` | 频率主义 | 从雅可比矩阵计算 `covar = inv(J.T @ J)`，`stderr = sqrt(covar[i,i])` |
| **2. 轮廓似然法** | `confidence.py` | 频率主义 | 固定一个参数，重新优化其他参数，用 F-test 比较 chi-square 变化 |
| **3. MCMC 贝叶斯采样** | `minimizer.py` (emcee) | 贝叶斯 | 马尔可夫链蒙特卡洛采样，从后验分布分位数估计可信区间 |

本文档详细分析这三种方法的实现路径、模块间的协作机制，以及结果如何与参数对象和拟合报告集成。

---

## 1. 模块架构

### 1.1 核心模块

| 模块 | 路径 | 主要职责 |
|------|------|----------|
| **confidence.py** | `lmfit/confidence.py` | 轮廓法置信区间计算的核心实现 |
| **minimizer.py** | `lmfit/minimizer.py` | 最小化器，包含协方差估计和 MCMC (emcee) 方法 |
| **model.py** | `lmfit/model.py` | 模型接口，集成置信区间和报告功能 |
| **parameter.py** | `lmfit/parameter.py` | 参数类，存储值、不确定性和相关性 |
| **printfuncs.py** | `lmfit/printfuncs.py` | 报告生成函数 |

### 1.2 模块依赖关系

```
printfuncs.py (报告生成)
    ↑
model.py (Model/ModelResult 接口层)
    ↑
    ├── confidence.py (轮廓法)
    └── minimizer.py (协方差估计 + MCMC/emcee)
              ↑
        parameter.py (参数存储)
```

---

## 2. 方法一：协方差矩阵估计

这是最常用、最快的不确定性评估方法，在最小二乘拟合后自动执行。

### 2.1 核心原理

对于非线性最小二乘问题，参数的协方差矩阵通过雅可比矩阵 (Jacobian) 估计：

```
covar = reduced_chi_square * inv(J.T @ J)
```

其中：
- `J` 是雅可比矩阵 (残差对参数的偏导数)
- `reduced_chi_square = chi_square / (n_data - n_params)`
- 标准误差: `stderr_i = sqrt(covar[i, i])`
- 相关系数: `correl[i,j] = covar[i,j] / (stderr_i * stderr_j)`

### 2.2 实现位置

**关键函数**: `_int2ext_cov_x()` - `minimizer.py:766`

```python
def _int2ext_cov_x(self, internal_cov, x):
    """Transform covariance matrix to external parameter space."""
```

### 2.3 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│              最小二乘拟合 (leastsq / least_squares)          │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  1. 雅可比矩阵获取                                             │
│     - leastsq: 从 pcov 获取                                  │
│     - least_squares: 从 ret.jac 计算 hess = J.T @ J         │
│     - 其他方法: 使用 numdifftools 数值差分估计                │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 协方差矩阵计算                                             │
│     covar = inv(hess)                                         │
│     if scale_covar:                                            │
│         covar *= redchi  # 按约化 chi-square 缩放            │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  3. 更新 Parameter 对象                                        │
│     for each parameter:                                        │
│         par.stderr = sqrt(covar[i, i])                       │
│         par.correl = {other_name: correl[i,j], ...}          │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 创建不确定性 ufloats (可选)                               │
│     result.uvars = params.create_uvars(covar=result.covar)  │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 关键代码位置

| 功能 | 文件位置 | 行号 |
|------|----------|------|
| leastsq 后协方差处理 | `lmfit/minimizer.py` | ~800 |
| `_int2ext_cov_x()` 变换 | `lmfit/minimizer.py` | 766 |
| `scale_covar` 参数 | `lmfit/minimizer.py` | 364 |
| `calc_covar` 参数 | `lmfit/minimizer.py` | 394 |

---

## 3. 方法二：轮廓似然法 (Profile Likelihood)

当协方差矩阵的正态假设不成立时（如参数接近边界、强非线性模型），使用轮廓法获得更可靠的置信区间。

### 3.1 核心原理

轮廓法的基本思想是：对于每个参数，固定该参数的值，然后重新优化其他所有参数，观察 chi-square 的变化。通过 F-test 比较最佳拟合和新拟合的 chi-square 来计算概率。

**数学原理**:

要找到参数 `p` 在置信水平 `sigma` 下的置信区间，需要找到满足以下条件的 `p_val`：

```
F-test_probability(best_fit, fit_with_p=p_val) = sigma_probability
```

其中：
- `F-test` 比较两个嵌套模型的 chi-square 差异
- `sigma_probability = erf(sigma/√2)`

### 3.2 入口函数

**函数定义**: `confidence.py:57`

```python
def conf_interval(minimizer, result, p_names=None, sigmas=None, trace=False,
                  maxiter=200, verbose=False, prob_func=None,
                  min_rel_change=1e-5):
```

**参数说明**:
- `minimizer`: Minimizer 对象，持有目标函数
- `result`: 最小化结果 (`MinimizerResult`)
- `p_names`: 要计算置信区间的参数名列表（默认所有可变参数）
- `sigmas`: sigma 水平列表（默认 `[1, 2, 3]`，对应 68.27%, 95.45%, 99.73% 置信度）
- `trace`: 是否保存追踪信息用于绘制轮廓轨迹
- `prob_func`: 概率计算函数（默认使用 F-test）

### 3.3 核心类: ConfidenceInterval

**类定义**: `confidence.py:165`

```python
class ConfidenceInterval:
    """Class used to calculate the confidence interval."""
```

#### 初始化阶段 (`__init__`): `confidence.py:168`

初始化时进行以下验证：
1. 检查所有可变参数是否有合理的 `stderr`
2. 检查至少有 2 个可变参数
3. 将 sigma 水平转换为概率值：
   - sigma ≥ 1: `prob = erf(sigma/√2)`
   - sigma < 1: 直接作为概率值

#### 主要方法

| 方法 | 位置 | 功能 |
|------|------|------|
| `calc_all_ci()` | `confidence.py:252` | 计算所有参数的置信区间 |
| `calc_ci(para, direction)` | `confidence.py:265` | 计算单个参数在指定方向的置信区间 |
| `find_limit(para, direction)` | `confidence.py:315` | 找到参数边界使得概率超过目标 sigma |
| `calc_prob(para, val, ...)` | `confidence.py:376` | 计算给定参数值的概率 |

### 3.4 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                    conf_interval() 入口                       │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              ConfidenceInterval 实例化                        │
│  - 验证参数 (≥2 可变参数, 有合理 stderr)                      │
│  - 转换 sigma 为概率值                                        │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    calc_all_ci()                             │
│         对每个参数调用 calc_ci() 计算正负方向                 │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    calc_ci(para, direction)                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 1. 固定目标参数 (para.vary = False)                  │    │
│  │ 2. find_limit() 找到边界使得 prob > max(sigmas)      │    │
│  │ 3. 使用 root_scalar() 找精确解                        │    │
│  │    - 方法: toms748 (Tomlin 748 算法)                │    │
│  │    - 目标: calc_prob(val) = target_prob              │    │
│  └─────────────────────────────────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    calc_prob(para, val)                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 1. 设置参数值: para.value = val                      │    │
│  │ 2. 准备拟合: minimizer.prepare_fit(params)           │    │
│  │ 3. 重新优化: minimizer.leastsq()                     │    │
│  │ 4. 计算概率: prob_func(best_fit, new_fit)           │    │
│  │    - 默认: f_compare() 使用 F-test                   │    │
│  └─────────────────────────────────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  返回: 字典 {参数名: [(prob1, val1), (prob2, val2), ...]}  │
└─────────────────────────────────────────────────────────────┘
```

### 3.5 F-test 概率计算

**函数**: `f_compare()` - `confidence.py:17`

```python
def f_compare(best_fit, new_fit):
    """Return the probability calculated using the F-test."""
    nfree = best_fit.nfree
    nfix = best_fit.nvarys - new_fit.nvarys
    dchi = new_fit.chisqr / best_fit.chisqr - 1.0
    return f.cdf(dchi * nfree / nfix, nfix, nfree)
```

**原理**:
- 使用 F 分布的累积分布函数 (CDF)
- 比较最佳拟合和固定参数后的拟合的 chi-square 比值
- `nfix`: 被固定的参数数量
- `nfree`: 自由度

### 3.6 二维置信区间

**函数**: `conf_interval2d()` - `confidence.py:394`

用于计算两个参数同时固定时的置信区域：

```python
def conf_interval2d(minimizer, result, x_name, y_name, nx=10, ny=10,
                    limits=None, prob_func=None, nsigma=5, chi2_out=False):
```

**工作原理**:
1. 在二维网格上固定两个参数
2. 对每个网格点重新优化其他参数
3. 计算 chi-square 或 sigma 值
4. 返回网格坐标和对应的概率/chi-square 值

### 3.7 与协方差方法的比较

| 特性 | 协方差矩阵 | 轮廓法 |
|------|-----------|--------|
| **计算速度** | 快 (O(1)) | 慢 (O(n_params × n_sigmas × n_fits)) |
| **分布假设** | 假设正态分布 | 无假设 (数据驱动) |
| **边界处理** | 差 (边界处正态假设失效) | 好 (检测并警告边界) |
| **非线性处理** | 差 (线性近似) | 好 (真实探索参数空间) |
| **结果形式** | 对称的 stderr | 各 sigma 水平的精确参数值 (可不对称) |
| **参数更新** | 直接更新 `par.stderr` | 结果在独立字典中，不修改 Parameter |

---

## 4. 方法三：MCMC 贝叶斯采样 (emcee)

使用马尔可夫链蒙特卡洛方法探索参数的后验概率分布，从采样链的分位数估计可信区间。

### 4.1 核心原理

MCMC 方法使用贝叶斯采样来估计参数的后验概率分布：

```
后验 ∝ 似然 × 先验
```

lmfit 的 emcee 实现假设：
- **先验**: 均匀分布 (参数在边界内时先验概率为常数，边界外为 0)
- **似然**: 从残差计算的高斯对数似然

通过分析采样链的分位数来估计可信区间：

| 分位数 | 含义 | 对应 sigma (正态近似) |
|--------|------|----------------------|
| 15.87% | 下边界 | -1σ |
| 50% | 中位数 (最佳估计) | 0σ |
| 84.13% | 上边界 | +1σ |

**注意**: 这是**可信区间 (Credible Interval)**，贝叶斯意义下的"参数落在这个区间的概率为 68%"，与频率主义的**置信区间 (Confidence Interval)** 概念不同。

### 4.2 入口方法

**方法定义**: `minimizer.py:1125`

```python
def emcee(self, params=None, steps=1000, nwalkers=100, burn=0, thin=1,
          ntemps=1, pos=None, reuse_sampler=False, workers=1,
          float_behavior='posterior', is_weighted=True, seed=None,
          progress=True, run_mcmc_kwargs={}):
```

**关键参数**:
- `steps`: 每个 walker 的采样步数
- `nwalkers`: 并行 walker 数量 (建议 `>> nvarys`)
- `burn`: 丢弃的初始采样数 (burn-in，让链收敛)
- `thin`: 稀疏采样因子 (只保留每 thin 个样本，减少自相关)
- `is_weighted`: 残差是否已按数据不确定性加权
- `float_behavior`: 标量输出的含义 (`'posterior'` 或 `'chi2'`)

### 4.3 对数后验概率计算

**方法**: `_lnprob()` - `minimizer.py:1033`

```python
def _lnprob(self, theta, userfcn, params, var_names, bounds, userargs=(),
            userkws=None, float_behavior='posterior', is_weighted=True,
            nan_policy='raise'):
    """Calculate the log-posterior probability."""
```

**工作流程**:

```
┌─────────────────────────────────────────────────────────────┐
│                    _lnprob() 计算流程                         │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. 边界检查: 如果 theta 超出 bounds，返回 -np.inf            │
│    (参数值会被裁剪到边界内，必须在注入前检查)                  │
│    这实现了均匀先验: 边界内先验为常数，边界外为 -∞          │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 注入参数值: params[name].value = val                      │
│ 3. 更新约束: params.update_constraints()                      │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. 计算目标函数: out = userfcn(params, *userargs, **userkws)│
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. 计算对数似然:                                             │
│    ┌─────────────────────────────────────────────────────┐  │
│    │ 情况 A: 输出是残差向量 (out.size > 1)               │  │
│    │   - is_weighted=True:                                │  │
│    │     lnprob = -0.5 * sum(residual^2)                 │  │
│    │     (假设残差是加权的: (data-model)/sigma)          │  │
│    │                                                      │  │
│    │   - is_weighted=False:                               │  │
│    │     边际化处理 __lnsigma 参数                        │  │
│    │     lnprob = -0.5 * sum((res/sigma)^2 + ln(2πσ²)) │  │
│    │     (sigma = exp(__lnsigma) 作为 nuisance 参数)     │  │
│    └─────────────────────────────────────────────────────┘  │
│    ┌─────────────────────────────────────────────────────┐  │
│    │ 情况 B: 输出是标量                                   │  │
│    │   - float_behavior='posterior': 直接作为 lnprob    │  │
│    │   - float_behavior='chi2': lnprob = -0.5 * chi2   │  │
│    └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 采样流程

```
┌─────────────────────────────────────────────────────────────┐
│                    emcee() 主流程                            │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. 准备阶段:                                                  │
│    - prepare_fit() 准备参数                                  │
│    - 检查目标函数输出类型                                     │
│    - 如果 is_weighted=False，添加 __lnsigma 参数            │
│      (用于边际化处理数据不确定性)                             │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 初始化采样器:                                              │
│    - 创建边界数组 bounds = [(min1, max1), (min2, max2), ...]│
│    - 初始化 walker 位置: p0 = 1 + randn() * 1e-4 * var_arr │
│      (在最佳值附近小范围扰动)                                 │
│    - 创建 EnsembleSampler:                                    │
│      emcee.EnsembleSampler(nwalkers, nvarys, _lnprob, ...)  │
│      (仿射不变集总采样器)                                     │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. 运行 MCMC 采样:                                            │
│    sampler.run_mcmc(p0, steps, progress=progress, ...)       │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. 后处理:                                                    │
│    - 获取链: chain = sampler.get_chain(thin=thin, discard=burn)│
│    - 展平: flatchain = chain.reshape((-1, nvarys))          │
│    - 计算分位数: np.percentile(flatchain, [15.87, 50, 84.13])│
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. 更新参数不确定性:                                          │
│    ┌─────────────────────────────────────────────────────┐  │
│    │ for each parameter:                                  │  │
│    │   std_l, median, std_u = quantiles[:, i]           │  │
│    │   params[name].value = median                       │  │
│    │   params[name].stderr = 0.5 * (std_u - std_l)      │  │
│    │                                                      │  │
│    │ # 计算相关系数                                        │
│    │ corrcoefs = np.corrcoef(flatchain.T)                │  │
│    └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 4.5 可信区间估计

emcee 方法使用采样链的分位数来估计可信区间：

**标准误差计算**:
```python
stderr = 0.5 * (percentile_84.13 - percentile_15.87)
```

这种方法的优势：
- **不假设正态分布**: 直接从后验分布估计
- **可处理非对称不确定性**: 自然捕获分布的偏度
- **完整后验信息**: 通过 `flatchain` 可做进一步分析

### 4.6 MinimizerResult 中的额外属性

emcee 采样完成后，`MinimizerResult` 包含以下额外属性：

| 属性 | 类型 | 说明 |
|------|------|------|
| `chain` | ndarray | 采样链，形状 `((steps-burn)//thin, nwalkers, nvarys)` |
| `flatchain` | DataFrame | 展平的采样链，可通过 `result.flatchain[parname]` 访问 |
| `lnprob` | ndarray | 每个样本的对数概率 |
| `acor` | ndarray | 每个参数的自相关时间 (如果可计算，用于诊断收敛) |
| `acceptance_fraction` | ndarray | 每个 walker 的接受率 (理想: 0.2-0.5) |

### 4.7 与频率主义方法的比较

| 特性 | 协方差/轮廓法 (频率主义) | MCMC (贝叶斯) |
|------|-------------------------|---------------|
| **哲学框架** | 频率主义 | 贝叶斯 |
| **参数解释** | 参数是固定的未知常数 | 参数是随机变量 |
| **区间解释** | 置信区间: "重复实验时，95%的区间包含真实值" | 可信区间: "参数落在这个区间的概率为 95%" |
| **先验信息** | 不使用 | 使用 (lmfit 中用均匀先验) |
| **计算方式** | 基于似然函数的渐近性质 | 从后验分布采样 |
| **结果形式** | 点估计 + 标准误差 | 完整分布 (分位数摘要) |
| **收敛诊断** | 无 | 有 (自相关时间、接受率) |

---

## 5. 结果集成机制

### 5.1 Parameter 类

**位置**: `parameter.py`

Parameter 对象存储以下与不确定性相关的属性：

| 属性 | 类型 | 说明 |
|------|------|------|
| `value` | float | 参数最佳值 |
| `stderr` | float 或 None | 标准误差 |
| `correl` | dict | 与其他参数的相关系数 `{other_param: correlation}` |
| `init_value` | float | 初始值 |
| `min` / `max` | float | 边界 |
| `vary` | bool | 是否可变 |
| `expr` | str 或 None | 约束表达式 |

### 5.2 三种方法对 Parameter 的更新

```
┌─────────────────────────────────────────────────────────────┐
│              三种方法对 Parameter 对象的影响                  │
└─────────────────────────────────────────────────────────────┘

方法一: 协方差矩阵估计 (最小二乘后)
─────────────────────────────────────
✓ 更新 par.value (最佳拟合值)
✓ 更新 par.stderr = sqrt(covar[i,i])
✓ 更新 par.correl = {other: correl[i,j], ...}
✓ 存储 result.covar 矩阵
✓ 可选: result.uvars = params.create_uvars(covar=covar)

方法二: 轮廓法 (conf_interval)
─────────────────────────────
✗ 不修改 Parameter 对象
✓ 结果在独立字典中: {param_name: [(prob1, val1), (prob2, val2), ...]}
✓ 需要用 ci_report() 查看
✓ 可选: trace=True 时保存完整轨迹

方法三: MCMC (emcee)
─────────────────────
✓ 更新 par.value = 采样链中位数
✓ 更新 par.stderr = 0.5 * (84.13%分位数 - 15.87%分位数)
✓ 更新 par.correl = {other: corrcoef[i,j], ...}
✓ 添加额外属性到 MinimizerResult:
   - result.chain, result.flatchain
   - result.lnprob
   - result.acor, result.acceptance_fraction
```

### 5.3 MinimizerResult 类

**位置**: `minimizer.py:175`

拟合结果的容器，包含：

| 属性 | 说明 |
|------|------|
| `params` | Parameters 对象 |
| `covar` | 协方差矩阵 (从最小二乘估计) |
| `residual` | 残差数组 |
| `chisqr` | chi-square 值 |
| `redchi` | 约化 chi-square |
| `aic` / `bic` | 信息准则 |
| `errorbars` | 是否成功估计不确定性 |
| `method` | 使用的拟合方法 |

### 5.4 ModelResult 类

**位置**: `model.py`

`ModelResult` 继承自 `MinimizerResult`，并添加了置信区间和报告相关的方法：

#### conf_interval() 方法 - `model.py:1771`

```python
def conf_interval(self, **kwargs):
    """Calculate the confidence intervals for the variable parameters."""
    self.ci_out = conf_interval(self, self, **kwargs)
    return self.ci_out
```

**特点**:
- 直接调用 `confidence.conf_interval()`
- 结果存储在 `self.ci_out` 中
- 可以通过 `ModelResult` 实例直接调用

#### ci_report() 方法 - `model.py:1784`

```python
def ci_report(self, with_offset=True, ndigits=5, **kwargs):
    """Return a formatted text report of the confidence intervals."""
    return ci_report(self.conf_interval(**kwargs),
                     with_offset=with_offset, ndigits=ndigits)
```

#### fit_report() 方法 - `model.py:1807`

```python
def fit_report(self, modelpars=None, show_correl=True,
               min_correl=0.1, sort_pars=False, correl_mode='list'):
    """Return a printable fit report."""
```

---

## 6. 报告生成

### 6.1 fit_report() - `printfuncs.py:84`

生成完整的拟合报告，包含：

1. **拟合统计信息** (`[[Fit Statistics]]`)
   - 拟合方法
   - 函数评估次数
   - 数据点数、变量数
   - chi-square、reduced chi-square
   - AIC、BIC 信息准则

2. **变量信息** (`[[Variables]]`)
   - 参数名、值、标准误差
   - 初始值
   - 是否固定、是否有约束

3. **相关性信息** (`[[Correlations]]`)
   - 排序的相关性列表
   - 或相关系数矩阵 (表格形式)

### 6.2 ci_report() / report_ci() - `printfuncs.py:421`

生成置信区间的格式化报告：

```python
def ci_report(ci, with_offset=True, ndigits=5):
    """Return text of a report for confidence intervals."""
```

**输出格式示例**:
```
                  _BEST_  -1.00σ   -2.00σ   -3.00σ   +1.00σ   +2.00σ   +3.00σ
   param1:      0.10000  -0.001  -0.003  -0.005  +0.001  +0.003  +0.005
   param2:      2.00000  -0.020  -0.040  -0.060  +0.020  +0.040  +0.060
```

**参数说明**:
- `with_offset=True`: 显示与最佳值的偏差 (推荐)
- `with_offset=False`: 显示绝对值

### 6.3 HTML 报告

- `params_html_table()` - `printfuncs.py`
- `fitreport_html_table()` - `minimizer.py`

用于在 Jupyter notebook 等环境中显示美观的 HTML 表格。

---

## 7. 调用路径示例

### 7.1 使用协方差矩阵 (默认)

```python
from lmfit import Minimizer, report_fit

# 1. 定义残差函数
def residual(params, x, data):
    model = ...
    return data - model

# 2. 创建参数和最小化器
params = lmfit.create_params(a=0.1, b=1.0)
mini = Minimizer(residual, params, fcn_args=(x,), fcn_kws={'data': data})

# 3. 执行拟合 (自动计算协方差和 stderr)
out = mini.leastsq()

# 4. 显示报告 (包含协方差估计的不确定性)
report_fit(out)

# 访问结果
print(out.params['a'].value)   # 最佳值
print(out.params['a'].stderr)  # 标准误差 (从协方差)
print(out.covar)               # 协方差矩阵
```

### 7.2 使用轮廓法

```python
from lmfit import conf_interval, report_ci

# 1. 先做最小二乘拟合 (需要 stderr 作为起点)
out = mini.leastsq()

# 2. 计算轮廓置信区间
ci, trace = conf_interval(mini, out, sigmas=[1, 2, 3], trace=True)

# 3. 显示报告
report_ci(ci)

# 4. 访问结果
print(ci['a'])  # [(prob1, val1), (prob2, val2), ...]

# 5. trace 可用于绘制轮廓轨迹
import matplotlib.pyplot as plt
plt.scatter(trace['a']['a'], trace['a']['b'], c=trace['a']['prob'])
```

### 7.3 使用 MCMC (emcee)

```python
from lmfit import report_fit

# 1. 使用 emcee 采样
result = mini.emcee(
    steps=5000,      # 每个 walker 的步数
    burn=500,        # burn-in 步数
    thin=20,         # 稀疏采样
    nwalkers=100,    # walker 数量
    is_weighted=True
)

# 2. 显示报告
report_fit(result)

# 3. 访问结果
print(result.params['a'].value)   # 中位数
print(result.params['a'].stderr)  # 半 IQR

# 4. 访问采样链
print(result.flatchain)              # DataFrame
print(result.flatchain['a'].describe())  # 参数 a 的后验统计
print(result.acceptance_fraction)    # 接受率 (诊断收敛)
print(result.chain.shape)             # (n_samples, n_walkers, n_params)
```

### 7.4 使用 Model 接口

```python
from lmfit import Model

# 1. 创建模型
model = Model(gaussian_func)

# 2. 拟合 (自动协方差估计)
result = model.fit(data, params, x=x)

# 3. 拟合报告
print(result.fit_report())

# 4. 轮廓法置信区间
ci = result.conf_interval()
print(result.ci_report())
```

---

## 8. 模块间协作关系

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           用户代码层                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌───────────────────────────┐  │
│  │  Minimizer  │    │   Model     │    │  conf_interval(), emcee() │  │
│  │  直接使用    │    │  高层接口    │    │       直接调用            │  │
│  └──────┬──────┘    └──────┬──────┘    └─────────────┬─────────────┘  │
└─────────┼──────────────────┼──────────────────────────┼──────────────────┘
          │                  │                          │
          ▼                  ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           接口层                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                     ModelResult (model.py)                          ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    ││
│  │  │ conf_interval() │  │  ci_report()    │  │  fit_report()   │    ││
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘    ││
│  └───────────┼─────────────────────┼─────────────────────┼──────────────┘│
└──────────────┼─────────────────────┼─────────────────────┼─────────────────┘
               │                     │                     │
               ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          核心实现层                                       │
│  ┌───────────────────────┐      ┌──────────────────────────────────┐   │
│  │  confidence.py        │      │         minimizer.py              │   │
│  │  ┌─────────────────┐  │      │  ┌─────────────────────────────┐ │   │
│  │  │ ConfidenceInterval│ │      │  │    协方差矩阵估计            │ │   │
│  │  │  - calc_all_ci()  │ │      │  │  - 从雅可比矩阵计算          │ │   │
│  │  │  - calc_ci()      │ │      │  │  - 更新 par.stderr           │ │   │
│  │  │  - find_limit()   │ │      │  └─────────────────────────────┘ │   │
│  │  │  - calc_prob()    │ │      │  ┌─────────────────────────────┐ │   │
│  │  └─────────────────┘  │      │  │      Minimizer.emcee()      │ │   │
│  │  ┌─────────────────┐  │      │  │  ┌───────────────────────┐  │ │   │
│  │  │  f_compare()    │ │      │  │  │  _lnprob() 对数后验    │  │ │   │
│  │  │  (F-test 概率)  │ │      │  │  │  概率计算               │  │ │   │
│  │  └─────────────────┘  │      │  │  └───────────────────────┘  │ │   │
│  │  ┌─────────────────┐  │      │  └─────────────────────────────┘ │   │
│  │  │ conf_interval2d │ │      │  ┌─────────────────────────────┐ │   │
│  │  │  (二维置信区间)  │ │      │  │    MinimizerResult           │ │   │
│  │  └─────────────────┘  │      │  │  - params, covar, chisqr...  │ │   │
│  └───────────────────────┘      │  │  - chain, flatchain, lnprob  │ │   │
│                                 │  │    (emcee 特有)              │ │   │
│                                 │  └─────────────────────────────┘ │   │
│                                 └──────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          数据存储层                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                        parameter.py                                  ││
│  │  ┌───────────────────────────────────────────────────────────────┐  ││
│  │  │                      Parameter 类                               │  ││
│  │  │  - value: 参数值                                                │  ││
│  │  │  - stderr: 标准误差                                             │  ││
│  │  │  - correl: 相关系数字典                                          │  ││
│  │  │  - init_value, min, max, vary, expr...                         │  ││
│  │  └───────────────────────────────────────────────────────────────┘  ││
│  │  ┌───────────────────────────────────────────────────────────────┐  ││
│  │  │                      Parameters 类 (dict 子类)                  │  ││
│  │  │  - 多个 Parameter 对象的容器                                     │  ││
│  │  │  - 支持约束表达式 (asteval)                                      │  ││
│  │  │  - update_constraints(): 更新约束参数                           │  ││
│  │  │  - create_uvars(): 创建带不确定性的 ufloats                    │  ││
│  │  └───────────────────────────────────────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────────┘│
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          报告输出层                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                       printfuncs.py                                  ││
│  │  ┌───────────────────────────────────────────────────────────────┐  ││
│  │  │  fit_report()                                                   │  ││
│  │  │    - 生成完整拟合报告                                            │  ││
│  │  │    - 包含统计信息、参数值、stderr、相关性                         │  ││
│  │  └───────────────────────────────────────────────────────────────┘  ││
│  │  ┌───────────────────────────────────────────────────────────────┐  ││
│  │  │  ci_report() / report_ci()                                      │  ││
│  │  │    - 生成置信区间报告                                            │  ││
│  │  │    - 按 sigma 水平显示参数值                                     │  ││
│  │  └───────────────────────────────────────────────────────────────┘  ││
│  │  ┌───────────────────────────────────────────────────────────────┐  ││
│  │  │  params_html_table(), fitreport_html_table()                   │  ││
│  │  │    - HTML 格式报告 (用于 Jupyter)                               │  ││
│  │  └───────────────────────────────────────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. 三种方法的比较与选择

### 9.1 综合比较

| 特性 | 协方差矩阵 | 轮廓法 | MCMC (emcee) |
|------|-----------|--------|--------------|
| **方法类别** | 频率主义 | 频率主义 | 贝叶斯 |
| **模块** | `minimizer.py` | `confidence.py` | `minimizer.py` |
| **原理** | 雅可比矩阵逆 | 轮廓似然 + F-test | 后验分布采样 |
| **分布假设** | 渐近正态 | 无 (数据驱动) | 无 (数据驱动) |
| **计算速度** | 最快 (O(1)) | 中等 | 最慢 |
| **结果形式** | 对称 stderr | 各 sigma 水平值 | 完整分布 + 分位数 |
| **参数更新** | 直接更新 | 独立字典 | 直接更新 |
| **收敛诊断** | 无 | 无 | 有 (自相关、接受率) |
| **额外信息** | 协方差矩阵 | 轮廓轨迹 | 完整采样链 |

### 9.2 选择建议

| 场景 | 推荐方法 | 理由 |
|------|----------|------|
| **快速探索、初步分析** | 协方差矩阵 | 最快，自动执行 |
| **参数接近边界** | 轮廓法 | 协方差的正态假设失效 |
| **强非线性模型** | 轮廓法或 MCMC | 线性近似不可靠 |
| **需要完整后验分布** | MCMC | 可做任意分位数、相关性分析 |
| **需要贝叶斯解释** | MCMC | "参数落在区间内的概率" |
| **需要非对称不确定性** | 轮廓法或 MCMC | 协方差只能给对称 stderr |
| **模型选择、贝叶斯因子** | MCMC | 后验样本可用于高级分析 |

### 9.3 典型工作流

```
┌─────────────────────────────────────────────────────────────┐
│                    典型数据分析工作流                          │
└─────────────────────────────────────────────────────────────┘

第一步: 快速拟合与探索
─────────────────────────
result = mini.leastsq()          # 最小二乘拟合
report_fit(result)                # 查看协方差估计的不确定性
# 检查:
# - stderr 是否合理?
# - 参数是否在边界附近?
# - 相关性是否过高?

第二步: 验证与确认 (如果需要)
─────────────────────────────────
# 选项 A: 轮廓法验证 (如果怀疑正态假设)
ci = conf_interval(mini, result)
report_ci(ci)
# 比较: 协方差的 stderr 是否与轮廓法的 1σ 一致?
# 如果不一致 → 模型非线性强，需要更谨慎

# 选项 B: MCMC 完整分析 (如果需要完整后验)
result_mcmc = mini.emcee(steps=5000, burn=500)
report_fit(result_mcmc)
# 检查收敛:
# - acceptance_fraction 是否在 0.2-0.5 之间?
# - acor (自相关时间) 是否合理?
# 高级分析:
# - result_mcmc.flatchain 用于 corner plot
# - 任意分位数估计
# - 参数间的非线性相关性

第三步: 结果报告
─────────────────
# 协方差/轮廓法:
report_fit(result)        # 参数值 + stderr
report_ci(ci)             # 各 sigma 水平置信区间

# MCMC:
report_fit(result_mcmc)   # 中位数 + 半 IQR
# 可选: 用 corner 包绘制后验分布
import corner
corner.corner(result_mcmc.flatchain, labels=result_mcmc.var_names)
```

---

## 10. 关键设计特点

### 10.1 统一的 Parameter 接口

```python
# 三种方法都更新相同的属性 (轮廓法除外)
par.value      # 最佳值/中位数
par.stderr     # 标准误差
par.correl     # 相关系数

# 这意味着:
# - 用户代码不需要关心用了哪种方法
# - report_fit() 可以统一显示所有方法的结果
```

### 10.2 不确定性传播

通过 `uncertainties` 库支持：

```python
# 从协方差矩阵创建带不确定性的 ufloats
result.uvars = params.create_uvars(covar=result.covar)

# 可用于误差传播计算
# 例如: 如果 y = a * x + b，且 a, b 有不确定性
# 则 result.uvars['a'] * x + result.uvars['b'] 自动传播不确定性
```

### 10.3 边界处理

1. **协方差矩阵**:
   - 边界附近雅可比矩阵条件数差
   - 可能导致 `stderr` 估计不准

2. **轮廓法**:
   - 在 `find_limit()` 中检测参数是否达到边界
   - 如果达到边界且概率 < max(sigmas)，发出警告

3. **MCMC**:
   - 在 `_lnprob()` 开头检查边界
   - 超出边界返回 `-np.inf` (后验概率为 0)
   - 注意：不能在注入参数后检查，因为 Parameter 值会被裁剪

---

## 11. 数据流汇总

### 11.1 完整数据流图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           数据输入                                        │
│                         data, params, model                               │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           拟合/采样                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │   leastsq()      │    │  least_squares() │    │    emcee()       │  │
│  │  (Levenberg-     │    │  (Trust Region)  │    │  (MCMC 采样)     │  │
│  │   Marquardt)     │    │                  │    │                  │  │
│  └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘  │
│           │                       │                       │               │
│           ▼                       ▼                       ▼               │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    协方差矩阵估计 (自动执行)                       │  │
│  │  - 从雅可比矩阵 J 计算: covar = inv(J.T @ J)                      │  │
│  │  - 按 redchi 缩放: covar *= redchi                                │  │
│  │  - 更新: par.stderr = sqrt(covar[i,i])                            │  │
│  └──────────────────────────────────┬───────────────────────────────┘  │
│                                      │                                    │
│                                      ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                        MinimizerResult                             │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │  通用属性:                                                   │  │  │
│  │  │  - params, residual, chisqr, redchi, aic, bic               │  │  │
│  │  │  - covar, errorbars, method                                 │  │  │
│  │  │                                                              │  │  │
│  │  │  emcee 额外属性:                                             │  │  │
│  │  │  - chain, flatchain, lnprob                                 │  │  │
│  │  │  - acor, acceptance_fraction                                 │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────┬───────────────────────────────┘  │
└─────────────────────────────────────┼───────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
           ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
           │  conf_interval │ │  (无需额外)   │ │  (无需额外)   │
           │  (轮廓法计算)  │ │                │ │                │
           └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
                   │                   │                   │
                   ▼                   ▼                   ▼
           ┌────────────────────────────────────────────────────────┐
           │                    结果汇总                            │
           ├────────────────────────────────────────────────────────┤
           │                                                        │
           │  Parameter 对象 (由协方差或 MCMC 更新):               │
           │  ┌──────────────────────────────────────────────────┐ │
           │  │  par.value      # 最佳拟合值 / 中位数             │ │
           │  │  par.stderr     # sqrt(covar[i,i]) / 半 IQR      │ │
           │  │  par.correl     # 相关系数字典                     │ │
           │  └──────────────────────────────────────────────────┘ │
           │                                                        │
           │  轮廓法结果 (独立字典):                                │
           │  ci = {param_name: [(prob1, val1), (prob2, val2),...]}│
           │                                                        │
           │  MCMC 额外结果:                                        │
           │  result.flatchain    # DataFrame，完整后验样本        │
           │  result.chain        # 原始采样链                     │
           │  result.lnprob       # 对数概率值                     │
           │                                                        │
           └────────────────────────────────────────────────────────┘
                                      │
                                      ▼
           ┌────────────────────────────────────────────────────────┐
           │                    报告生成                            │
           ├────────────────────────────────────────────────────────┤
           │                                                        │
           │  report_fit(result)                                    │
           │  ├── 显示 par.value, par.stderr                       │
           │  ├── 显示 par.correl (相关性)                         │
           │  └── 显示拟合统计量 (chisqr, aic, bic, ...)           │
           │                                                        │
           │  report_ci(ci)                                         │
           │  └── 显示轮廓法各 sigma 水平的参数值                   │
           │                                                        │
           │  result.fit_report()  (ModelResult 方法)              │
           │  result.ci_report()   (ModelResult 方法)              │
           │                                                        │
           └────────────────────────────────────────────────────────┘
```

### 11.2 与参数对象的集成关系

| 方法 | par.value | par.stderr | par.correl | result.covar | 其他输出 |
|------|-----------|------------|------------|--------------|----------|
| **协方差矩阵** | ✓ (最佳拟合) | ✓ (sqrt(covar[i,i])) | ✓ | ✓ | result.uvars |
| **轮廓法** | ✗ (不修改) | ✗ (不修改) | ✗ (不修改) | ✗ (不使用) | ci 字典, trace |
| **MCMC** | ✓ (中位数) | ✓ (半 IQR) | ✓ | ✗ (不用协方差) | flatchain, chain, lnprob |

**注意**: 轮廓法不修改 Parameter 对象，这意味着：
- `report_fit()` 不会显示轮廓法的结果
- 需要用 `report_ci()` 单独查看
- 轮廓法可以与其他方法的结果对比验证

---

## 12. 参考代码位置

| 功能 | 文件位置 | 行号 |
|------|----------|------|
| **协方差矩阵估计** | | |
| 最小二乘后协方差处理 | `lmfit/minimizer.py` | ~800 |
| `_int2ext_cov_x()` 协方差变换 | `lmfit/minimizer.py` | 766 |
| `scale_covar` 参数定义 | `lmfit/minimizer.py` | 364 |
| | | |
| **轮廓法** | | |
| `conf_interval()` 入口 | `lmfit/confidence.py` | 57 |
| `ConfidenceInterval` 类 | `lmfit/confidence.py` | 165 |
| `calc_all_ci()` | `lmfit/confidence.py` | 252 |
| `calc_ci()` | `lmfit/confidence.py` | 265 |
| `find_limit()` | `lmfit/confidence.py` | 315 |
| `calc_prob()` | `lmfit/confidence.py` | 376 |
| `f_compare()` (F-test) | `lmfit/confidence.py` | 17 |
| `conf_interval2d()` | `lmfit/confidence.py` | 394 |
| | | |
| **MCMC (emcee)** | | |
| `Minimizer.emcee()` | `lmfit/minimizer.py` | 1125 |
| `_lnprob()` 对数后验 | `lmfit/minimizer.py` | 1033 |
| | | |
| **结果与报告** | | |
| `MinimizerResult` 类 | `lmfit/minimizer.py` | 175 |
| `ModelResult.conf_interval()` | `lmfit/model.py` | 1771 |
| `ModelResult.ci_report()` | `lmfit/model.py` | 1784 |
| `fit_report()` | `lmfit/printfuncs.py` | 84 |
| `ci_report()` | `lmfit/printfuncs.py` | 421 |
| `Parameter` 类 | `lmfit/parameter.py` | ~550 |
| `Parameters` 类 | `lmfit/parameter.py` | 62 |
| | | |
| **Bootstrap 自举法** | | |
| **不存在** | — | — |

---

## 13. 常见问题与澄清

### Q1: lmfit 有没有 Bootstrap 自举法？

**答: 没有。**

经过系统性搜索确认，lmfit 代码库中完全没有 Bootstrap 自举法的实现。搜索关键词包括：
- `bootstrap` — 无匹配
- `boot.?strap` (正则变体) — 无匹配
- `resample` — 无匹配
- `jackknife` — 无匹配

### Q2: 那 `emcee` 是什么？是 Bootstrap 吗？

**答: 不是。** `emcee` 是 **MCMC (马尔可夫链蒙特卡洛)** 方法，属于贝叶斯推断框架。

**MCMC vs Bootstrap 的本质区别**:

| 维度 | MCMC (emcee) | Bootstrap |
|------|-------------|-----------|
| **框架** | 贝叶斯 | 频率主义 |
| **核心操作** | 在参数空间采样，探索后验分布 | 在数据空间重采样，有放回地抽取伪数据 |
| **数据修改** | 不修改原始数据 | 重采样生成多组不同的伪数据集 |
| **拟合次数** | 一次采样过程 (但每步都计算似然) | 对每组重采样数据都要重新拟合 |
| **结果解释** | 可信区间 (参数是随机变量) | 置信区间 (参数是固定常数) |
| **lmfit 中** | ✅ 存在 (`minimizer.emcee()`) | ❌ 不存在 |

### Q3: 我需要 Bootstrap，怎么办？

**选项 A: 自己实现**

```python
# 简单的 Bootstrap 示例 (lmfit 外部)
import numpy as np

def bootstrap_fit(model, data, x, n_boot=1000):
    """手动实现 Bootstrap 不确定性估计"""
    n_data = len(data)
    boot_params = []
    
    for _ in range(n_boot):
        # 有放回重采样
        idx = np.random.choice(n_data, size=n_data, replace=True)
        boot_data = data[idx]
        boot_x = x[idx]
        
        # 用重采样数据拟合
        result = model.fit(boot_data, x=boot_x)
        boot_params.append([result.params[p].value for p in result.var_names])
    
    return np.array(boot_params)

# 使用:
# boot_samples = bootstrap_fit(model, data, x, n_boot=1000)
# stderr = np.std(boot_samples, axis=0)  # Bootstrap 标准误差
# ci = np.percentile(boot_samples, [2.5, 97.5], axis=0)  # 95% 置信区间
```

**选项 B: 使用其他库**

- `scipy.stats.bootstrap` — SciPy 1.7+ 提供的 Bootstrap 函数
- `arch.bootstrap` — 专门的 Bootstrap 库
- `bootstrapped` — 轻量级 Bootstrap 库

### Q4: 三种方法应该用哪个？

**快速决策树**:

```
开始
  │
  ├── 只是想快速看一下结果？
  │   └── 用协方差矩阵 (默认自动计算)
  │       → report_fit(result)
  │
  ├── 参数在边界附近？
  │   └── 用轮廓法
  │       → ci = conf_interval(mini, result)
  │       → report_ci(ci)
  │
  ├── 强非线性模型？
  │   ├── 先试试轮廓法
  │   └── 如果需要完整分布 → 用 MCMC
  │
  └── 需要以下任一？
      ├── 贝叶斯概率解释 ("参数落在区间的概率")
      ├── 完整后验分布
      ├── 参数间的非线性相关性
      ├── 模型选择/贝叶斯因子
      └── 非对称不确定性的精确估计
          └── 用 MCMC
              → result = mini.emcee(steps=5000, burn=500)
              → 用 corner 包可视化后验
```

### Q5: 为什么轮廓法不修改 Parameter 对象？

**设计考虑**:

1. **轮廓法的结果更丰富**: 轮廓法返回各 sigma 水平的精确值，可能不对称，而 `Parameter.stderr` 只是一个对称的数字

2. **多结果共存**: 用户可能想同时查看协方差估计和轮廓法结果，互不干扰

3. **对比验证**: 可以对比 `stderr` (协方差) 和 `1σ` 轮廓值，判断正态假设是否合理

```python
# 示例: 对比两种方法
out = mini.leastsq()
ci = conf_interval(mini, out)

# 协方差的 stderr
stderr_a = out.params['a'].stderr

# 轮廓法的 1σ 区间 (不对称)
ci_a = ci['a']
# 找到 prob ≈ 0.6827 (1σ) 的上下界

# 如果两者差异大 → 正态假设不成立 → 模型非线性强
```

---

## 附录: lmfit 不确定性评估方法速查表

| 方法 | 函数/方法 | 结果位置 | 报告函数 | 速度 |
|------|-----------|----------|----------|------|
| 协方差矩阵 | `leastsq()`, `least_squares()` 等 | `par.stderr`, `result.covar` | `report_fit()` | ⚡ 最快 |
| 轮廓法 | `conf_interval()` | `ci` 字典 | `report_ci()` | ⏱️ 中等 |
| MCMC | `emcee()` | `par.stderr`, `result.flatchain` | `report_fit()`, `corner.corner()` | 🐢 最慢 |

**Bootstrap**: ❌ 不存在，需自行实现或使用其他库
