# lmfit 置信区间计算和不确定性评估实现分析

## 概述

lmfit 提供了两种主要的置信区间计算和不确定性评估方法：

1. **轮廓法 (Profile Likelihood)** - 通过固定一个参数并重新优化其他参数来计算置信区间
2. **马尔可夫链蒙特卡洛法 (MCMC / emcee)** - 通过贝叶斯采样来估计后验分布和不确定性

本文档详细分析这两种方法的实现路径、模块间的协作机制，以及结果如何与参数对象和拟合报告集成。

---

## 1. 模块架构

### 1.1 核心模块

| 模块 | 路径 | 主要职责 |
|------|------|----------|
| **confidence.py** | `lmfit/confidence.py` | 轮廓法置信区间计算的核心实现 |
| **minimizer.py** | `lmfit/minimizer.py` | 最小化器，包含 MCMC (emcee) 方法 |
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
    └── minimizer.py (MCMC/emcee)
              ↑
        parameter.py (参数存储)
```

---

## 2. 轮廓法 (Profile Likelihood) 实现

### 2.1 核心原理

轮廓法的基本思想是：对于每个参数，固定该参数的值，然后重新优化其他所有参数，观察 chi-square 的变化。通过 F-test 比较最佳拟合和新拟合的 chi-square 来计算概率。

### 2.2 入口函数

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
- `sigmas`: sigma 水平列表（默认 [1, 2, 3]，对应 68.27%, 95.45%, 99.73% 置信度）
- `trace`: 是否保存追踪信息用于绘制轮廓轨迹
- `prob_func`: 概率计算函数（默认使用 F-test）

### 2.3 核心类: ConfidenceInterval

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

### 2.4 工作流程

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

### 2.5 F-test 概率计算

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

### 2.6 二维置信区间

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

---

## 3. MCMC (emcee) 方法实现

### 3.1 核心原理

MCMC 方法使用贝叶斯采样来估计参数的后验概率分布。通过分析采样链的分位数来估计置信区间。

### 3.2 入口方法

**方法定义**: `minimizer.py:1125`

```python
def emcee(self, params=None, steps=1000, nwalkers=100, burn=0, thin=1,
          ntemps=1, pos=None, reuse_sampler=False, workers=1,
          float_behavior='posterior', is_weighted=True, seed=None,
          progress=True, run_mcmc_kwargs={}):
```

**关键参数**:
- `steps`: 每个 walker 的采样步数
- `nwalkers`: 并行 walker 数量 (建议 >> nvarys)
- `burn`: 丢弃的初始采样数 (burn-in)
- `thin`: 稀疏采样因子 (只保留每 thin 个样本)
- `is_weighted`: 残差是否已按数据不确定性加权
- `float_behavior`: 标量输出的含义 (`'posterior'` 或 `'chi2'`)

### 3.3 对数后验概率计算

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
│    │   - is_weighted=False:                               │  │
│    │     边际化处理 __lnsigma 参数                        │  │
│    │     lnprob = -0.5 * sum((res/sigma)^2 + ln(2πσ²)) │  │
│    └─────────────────────────────────────────────────────┘  │
│    ┌─────────────────────────────────────────────────────┐  │
│    │ 情况 B: 输出是标量                                   │  │
│    │   - float_behavior='posterior': 直接作为 lnprob    │  │
│    │   - float_behavior='chi2': lnprob = -0.5 * chi2   │  │
│    └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 采样流程

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
│    - 创建 EnsembleSampler:                                    │
│      emcee.EnsembleSampler(nwalkers, nvarys, _lnprob, ...)  │
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

### 3.5 置信区间估计

emcee 方法使用采样链的分位数来估计置信区间：

| 分位数 | 含义 | 对应 sigma |
|--------|------|-----------|
| 15.87% | 下边界 | -1σ |
| 50% | 中位数 (最佳估计) | 0σ |
| 84.13% | 上边界 | +1σ |

**标准误差计算**:
```python
stderr = 0.5 * (percentile_84.13 - percentile_15.87)
```

这种方法的优势：
- 不假设正态分布
- 可以处理非对称的不确定性
- 直接从后验分布估计

### 3.6 MinimizerResult 中的额外属性

emcee 采样完成后，`MinimizerResult` 包含以下额外属性：

| 属性 | 类型 | 说明 |
|------|------|------|
| `chain` | ndarray | 采样链，形状 `((steps-burn)//thin, nwalkers, nvarys)` |
| `flatchain` | DataFrame | 展平的采样链，可通过 `result.flatchain[parname]` 访问 |
| `lnprob` | ndarray | 每个样本的对数概率 |
| `acor` | ndarray | 每个参数的自相关时间 (如果可计算) |
| `acceptance_fraction` | ndarray | 每个 walker 的接受率 |

---

## 4. 结果集成机制

### 4.1 Parameter 类

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

### 4.2 MinimizerResult 类

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

### 4.3 ModelResult 类

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

## 5. 报告生成

### 5.1 fit_report() - `printfuncs.py:84`

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

### 5.2 ci_report() / report_ci() - `printfuncs.py:421`

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

### 5.3 HTML 报告

- `params_html_table()` - `printfuncs.py`
- `fitreport_html_table()` - `minimizer.py`

用于在 Jupyter notebook 等环境中显示美观的 HTML 表格。

---

## 6. 调用路径示例

### 6.1 使用轮廓法

```python
from lmfit import Minimizer, conf_interval, report_ci, report_fit

# 1. 定义残差函数
def residual(params, x, data):
    model = ...
    return data - model

# 2. 创建参数和最小化器
params = lmfit.create_params(a=0.1, b=1.0)
mini = Minimizer(residual, params, fcn_args=(x,), fcn_kws={'data': data})

# 3. 执行拟合
out = mini.leastsq()
report_fit(out)  # 显示拟合报告

# 4. 计算置信区间
ci, trace = conf_interval(mini, out, trace=True)
report_ci(ci)  # 显示置信区间报告
```

### 6.2 使用 MCMC (emcee)

```python
from lmfit import Minimizer, report_fit

# 1. 创建最小化器
mini = Minimizer(residual, params, fcn_args=(x,), fcn_kws={'data': data})

# 2. 使用 emcee 采样
result = mini.emcee(steps=5000, burn=500, thin=20, is_weighted=True)

# 3. 显示结果
report_fit(result)

# 4. 访问采样链
print(result.flatchain)        # DataFrame
print(result.acceptance_fraction)
print(result.chain.shape)
```

### 6.3 使用 Model 接口

```python
from lmfit import Model

# 1. 创建模型
model = Model(gaussian_func)

# 2. 拟合
result = model.fit(data, params, x=x)

# 3. 拟合报告
print(result.fit_report())

# 4. 置信区间 (轮廓法)
ci = result.conf_interval()
print(result.ci_report())
```

---

## 7. 模块间协作关系

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
│  │  │ ConfidenceInterval│ │      │  │      Minimizer.emcee()      │ │   │
│  │  │  - calc_all_ci()  │ │      │  │  ┌───────────────────────┐  │ │   │
│  │  │  - calc_ci()      │ │      │  │  │  _lnprob() 对数后验    │  │ │   │
│  │  │  - find_limit()   │ │      │  │  │  概率计算               │  │ │   │
│  │  │  - calc_prob()    │ │      │  │  └───────────────────────┘  │ │   │
│  │  └─────────────────┘  │      │  └─────────────────────────────┘ │   │
│  │  ┌─────────────────┐  │      │  ┌─────────────────────────────┐ │   │
│  │  │  f_compare()    │ │      │  │    MinimizerResult           │ │   │
│  │  │  (F-test 概率)  │ │      │  │  - params, covar, chisqr...  │ │   │
│  │  └─────────────────┘  │      │  │  - chain, flatchain, lnprob  │ │   │
│  │  ┌─────────────────┐  │      │  │    (emcee 特有)              │ │   │
│  │  │ conf_interval2d │ │      │  └─────────────────────────────┘ │   │
│  │  │  (二维置信区间)  │ │      └──────────────────────────────────┘   │
│  │  └─────────────────┘  │                                             │
│  └───────────────────────┘                                             │
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

## 8. 关键设计特点

### 8.1 轮廓法的优点

1. **基于似然比**：不假设参数分布为正态
2. **处理边界**：当参数接近边界时给出警告
3. **可追踪**：`trace=True` 时保存所有中间结果，可绘制轮廓轨迹
4. **可定制**：支持自定义 `prob_func` 替代默认的 F-test

### 8.2 MCMC (emcee) 的优点

1. **贝叶斯框架**：直接估计后验分布
2. **非对称不确定性**：自然处理非正态分布
3. **相关性捕获**：通过 `flatchain` 可分析参数间的复杂相关性
4. **收敛诊断**：提供 `acor` (自相关时间) 和 `acceptance_fraction`

### 8.3 参数设计

1. **统一存储**：`Parameter.stderr` 同时用于：
   - 最小二乘的协方差矩阵估计
   - 轮廓法的置信区间
   - MCMC 的分位数估计

2. **不确定性传播**：通过 `uncertainties` 库支持：
   ```python
   params.create_uvars(covar=None)  # 创建带不确定性的 ufloats
   ```

### 8.4 边界处理

1. **轮廓法**：
   - 在 `find_limit()` 中检测参数是否达到边界
   - 如果达到边界且概率 < max(sigmas)，发出警告

2. **MCMC**：
   - 在 `_lnprob()` 开头检查边界
   - 超出边界返回 `-np.inf` (后验概率为 0)
   - 注意：不能在注入参数后检查，因为 Parameter 值会被裁剪

---

## 9. 总结

lmfit 提供了两种互补的置信区间计算方法：

| 特性 | 轮廓法 | MCMC (emcee) |
|------|--------|--------------|
| **模块** | `confidence.py` | `minimizer.py` |
| **原理** | 轮廓似然 + F-test | 贝叶斯采样 |
| **假设** | 渐近 chi-square 分布 | 无 (数据驱动) |
| **输出** | 各 sigma 水平的参数值 | 完整后验分布采样 |
| **参数更新** | 不直接更新 Parameter | 更新 value 和 stderr |
| **计算成本** | 中等 (每个 sigma 需要多次拟合) | 高 (需要大量采样) |
| **适用场景** | 大多数情况、快速评估 | 复杂分布、需要完整后验 |

### 数据流

```
拟合 (leastsq/least_squares/...)
    ↓
MinimizerResult {params, covar, chisqr, ...}
    ↓
    ├───→ 轮廓法: conf_interval() → ci 字典 → ci_report()
    │
    └───→ MCMC: emcee() → 更新 params.value/stderr
                    → 添加 chain/flatchain/lnprob 到 result
                    → report_fit() 显示结果
```

### 与参数对象的集成

- **最小二乘拟合**：从协方差矩阵 `covar` 计算 `stderr`
- **轮廓法**：结果存储在独立的字典中，不直接修改 Parameter
- **MCMC**：直接更新 `Parameter.value` (中位数) 和 `Parameter.stderr` (半 IQR)
- **所有方法**：结果都可以通过 `report_fit()` 和 `ci_report()` 格式化输出

---

## 10. 参考代码位置

| 功能 | 文件位置 | 行号 |
|------|----------|------|
| conf_interval() 入口 | `lmfit/confidence.py` | 57 |
| ConfidenceInterval 类 | `lmfit/confidence.py` | 165 |
| calc_all_ci() | `lmfit/confidence.py` | 252 |
| calc_ci() | `lmfit/confidence.py` | 265 |
| find_limit() | `lmfit/confidence.py` | 315 |
| calc_prob() | `lmfit/confidence.py` | 376 |
| f_compare() (F-test) | `lmfit/confidence.py` | 17 |
| conf_interval2d() | `lmfit/confidence.py` | 394 |
| Minimizer.emcee() | `lmfit/minimizer.py` | 1125 |
| _lnprob() | `lmfit/minimizer.py` | 1033 |
| MinimizerResult 类 | `lmfit/minimizer.py` | 175 |
| ModelResult.conf_interval() | `lmfit/model.py` | 1771 |
| ModelResult.ci_report() | `lmfit/model.py` | 1784 |
| fit_report() | `lmfit/printfuncs.py` | 84 |
| ci_report() | `lmfit/printfuncs.py` | 421 |
| Parameter 类 | `lmfit/parameter.py` | ~550 |
| Parameters 类 | `lmfit/parameter.py` | 62 |
