# lmfit 核心拟合器架构分析报告

## 目录
1. [算法分发机制与调用链](#1-算法分发机制与调用链)
2. [结果统一组装机制](#2-结果统一组装机制)
3. [序列化机制与跨算法差异](#3-序列化机制与跨算法差异)

---

## 1. 算法分发机制与调用链

### 1.1 整体架构概览

lmfit 的核心设计采用了**分层抽象**的方式，通过 `Minimizer` 类作为统一入口，将不同的优化算法封装为独立的方法，最终通过 `minimize()` 方法进行分发。

### 1.2 算法分类体系

#### 1.2.1 方法分类字典

在 `minimizer.py` 中定义了 `SCALAR_METHODS` 字典，用于将用户友好的短名称映射到 scipy 的标准方法名：

```python
SCALAR_METHODS = {
    'nelder': 'Nelder-Mead',
    'powell': 'Powell',
    'cg': 'CG',
    'bfgs': 'BFGS',
    'newton': 'Newton-CG',
    'lbfgsb': 'L-BFGS-B',
    'l-bfgsb': 'L-BFGS-B',
    'tnc': 'TNC',
    'cobyla': 'COBYLA',
    'cobyqa': 'COBYQA',
    'slsqp': 'SLSQP',
    'dogleg': 'dogleg',
    'trust-ncg': 'trust-ncg',
    'differential_evolution': 'differential_evolution',
    'trust-constr': 'trust-constr',
    'trust-exact': 'trust-exact',
    'trust-krylov': 'trust-krylov'
}
```
**位置**: `minimizer.py:84-100`

#### 1.2.2 支持的优化方法总览

| 方法类别 | 方法名称 | 底层实现 | 特殊要求 |
|---------|---------|---------|---------|
| **最小二乘** | `leastsq` | `scipy.optimize.leastsq` | 无 |
| **最小二乘(带边界)** | `least_squares` | `scipy.optimize.least_squares` | 无 |
| **标量优化** | `nelder`, `powell`, `cg`, `bfgs` 等 | `scipy.optimize.minimize` | 无 |
| **全局优化** | `differential_evolution` | `scipy.optimize.differential_evolution` | 所有参数需有限边界 |
| **全局优化** | `brute` | `scipy.optimize.brute` | 需边界或 brute_step |
| **全局优化** | `basinhopping` | `scipy.optimize.basinhopping` | 无 |
| **全局优化** | `ampgo` | 内置 AMPGO 实现 | 无 |
| **全局优化** | `shgo` | `scipy.optimize.shgo` | 需有限边界 |
| **全局优化** | `dual_annealing` | `scipy.optimize.dual_annealing` | 需有限边界 |
| **全局优化** | `direct` | `scipy.optimize.direct` | 需有限边界 |
| **贝叶斯采样** | `emcee` | `emcee` 包 | 需安装 emcee >= 3.0 |

### 1.3 核心分发机制

#### 1.3.1 `minimize()` 方法分发逻辑

`Minimizer.minimize()` 是统一的入口方法，通过字符串匹配路由到具体的优化方法：

```python
def minimize(self, method='leastsq', params=None, **kws):
    # ... 参数处理 ...
    
    user_method = method.lower()
    if user_method.startswith('leasts'):
        function = self.leastsq
    elif user_method.startswith('least_s'):
        function = self.least_squares
    elif user_method == 'brute':
        function = self.brute
    elif user_method == 'basinhopping':
        function = self.basinhopping
    elif user_method == 'ampgo':
        function = self.ampgo
    elif user_method == 'emcee':
        function = self.emcee
    elif user_method == 'shgo':
        function = self.shgo
    elif user_method == 'dual_annealing':
        function = self.dual_annealing
    elif user_method == 'direct':
        function = self.direct
    else:
        function = self.scalar_minimize
        for key, val in SCALAR_METHODS.items():
            if (key.lower().startswith(user_method) or
                    val.lower().startswith(user_method)):
                kwargs['method'] = val
    return function(**kwargs)
```
**位置**: `minimizer.py:2356-2455`

#### 1.3.2 分发策略特点

1. **前缀匹配**: 使用 `startswith()` 进行模糊匹配，如 `'leasts'` 匹配 `'leastsq'`
2. **两级分发**: 
   - 第一级: 区分专用方法（leastsq, least_squares, brute 等）
   - 第二级: 通用标量方法通过 `SCALAR_METHODS` 映射到 scipy 标准名
3. **大小写不敏感**: 用户输入统一转换为小写进行匹配

### 1.4 完整调用链分析

#### 1.4.1 调用链概览

```
用户调用
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Model.fit() / Minimizer.minimize()                      │
│  - 准备参数和配置                                          │
│  - 选择优化方法                                            │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Minimizer.{specific_method}()                           │
│  (如: leastsq(), least_squares(), scalar_minimize()等)   │
│  - 调用 prepare_fit() 初始化 MinimizerResult              │
│  - 设置特定方法的参数和选项                                 │
│  - 准备边界和初始值                                        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  scipy.optimize.{solver}                                │
│  (底层优化算法执行)                                        │
│  - 调用用户的残差函数/penalty函数                          │
│  - 执行优化迭代                                            │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  结果后处理                                               │
│  - 将 scipy 结果复制到 MinimizerResult                     │
│  - 调用 _calculate_statistics() 计算统计量                 │
│  - 计算协方差矩阵和不确定性                                 │
│  - 返回统一的 MinimizerResult 对象                         │
└─────────────────────────────────────────────────────────┘
```

#### 1.4.2 关键方法详解

##### `prepare_fit()` 方法

这是每个优化方法调用前的**统一初始化**步骤：

```python
def prepare_fit(self, params=None):
    self._abort = False
    self.result = MinimizerResult()
    result = self.result
    
    # ... 参数初始化 ...
    
    # 确定可变参数和表达式约束参数
    result.var_names = []
    result.init_vals = []
    result._init_vals_internal = []
    result.params.update_constraints()
    
    for name, par in self.result.params.items():
        if par.expr is not None:
            par.vary = False
        if par.vary:
            result.var_names.append(name)
            result._init_vals_internal.append(par.setup_bounds())
            result.init_vals.append(par.value)
    # ...
    return result
```
**位置**: `minimizer.py:618-709`

**主要职责**:
1. 创建新的 `MinimizerResult` 实例
2. 深拷贝参数对象，避免修改原始参数
3. 区分可变参数和约束参数（有 `expr` 的参数）
4. 为每个可变参数设置内部边界转换
5. 初始化结果对象的基础属性

##### 以 `leastsq()` 为例的具体方法执行流程

```python
def leastsq(self, params=None, max_nfev=None, **kws):
    # 1. 准备拟合
    result = self.prepare_fit(params=params)
    result.method = 'leastsq'
    variables = result._init_vals_internal  # 内部表示的初始值
    
    # 2. 设置方法特定参数
    lskws = dict(Dfun=None, full_output=True, col_deriv=False, ...)
    # ... 参数合并和处理 ...
    
    # 3. 调用底层 scipy 求解器
    try:
        lsout = scipy_leastsq(self.__residual, variables, **lskws)
    except AbortFitException:
        pass
    
    # 4. 处理结果
    if not result.aborted:
        _best, _cov, _infodict, errmsg, ier = lsout
    else:
        _best = result.last_internal_values
        # ...
    
    # 5. 计算残差和统计量
    result.residual = self.__residual(_best)
    result._calculate_statistics()
    
    # 6. 计算协方差和不确定性
    if result.errorbars:
        result.covar = self._int2ext_cov_x(_cov, _best)
        self._calculate_uncertainties_correlations()
    
    return result
```
**位置**: `minimizer.py:1621-1736`

#### 1.4.3 Model 接口的调用链

当使用 `Model.fit()` 时，调用链多了一层抽象：

```python
# Model.fit() 方法
def fit(self, data, params=None, weights=None, method='leastsq', ...):
    # 1. 准备参数
    if params is None:
        params = self.make_params(verbose=verbose)
    else:
        params = deepcopy(params)
    
    # 2. 创建 ModelResult（继承自 Minimizer）
    output = ModelResult(self, params, method=method, ...)
    
    # 3. 执行拟合
    output.fit(data=data, weights=weights)
    
    # 4. 后处理
    output.components = self.components
    return output
```
**位置**: `model.py:1033-1183`

`ModelResult.fit()` 内部调用：

```python
def fit(self, data=None, params=None, weights=None, method=None, ...):
    # ... 设置数据和权重 ...
    
    self.init_fit = self.model.eval(params=self.init_params, **self.userkws)
    
    # 调用 Minimizer.minimize()
    _ret = self.minimize(method=self.method, params=self.init_params)
    
    self.model.post_fit(_ret)
    _ret.params.create_uvars(covar=_ret.covar)
    
    # 将 MinimizerResult 的属性复制到 ModelResult
    for attr in dir(_ret):
        if not attr.startswith('_'):
            try:
                setattr(self, attr, getattr(_ret, attr))
            except AttributeError:
                pass
    
    # 计算 best_fit, best_values 等
    self.best_fit = self.model.eval(params=_ret.params, **self.userkws)
    # ...
```
**位置**: `model.py:1539-1597`

---

## 2. 结果统一组装机制

### 2.1 结果对象层次结构

lmfit 使用**两层结果对象**体系：

```
┌──────────────────────────────────────────────────────┐
│  MinimizerResult (minimizer.py)                      │
│  - 底层优化结果的统一封装                              │
│  - 包含参数、统计量、协方差矩阵等                       │
└──────────────────────────────────────────────────────┘
                    ▲
                    │ 继承
                    │
┌──────────────────────────────────────────────────────┐
│  ModelResult (model.py)                              │
│  - 面向 Model 接口的高级结果对象                      │
│  - 包含最佳拟合曲线、初始值、R² 等                     │
│  - 支持绘图、评估不确定性等高级功能                     │
└──────────────────────────────────────────────────────┘
```

### 2.2 MinimizerResult 核心结构

```python
class MinimizerResult:
    def __init__(self, **kws):
        for key, val in kws.items():
            setattr(self, key, val)
```
**位置**: `minimizer.py:175-256`

**设计特点**:
- 使用**动态属性**模式，通过 `setattr` 灵活设置属性
- 允许不同算法添加特有的属性
- 但通过 `_calculate_statistics()` 等方法保证核心属性的存在

### 2.3 核心属性统一填充

#### 2.3.1 不同算法的结果来源差异

| 方法 | 原始结果对象 | 协方差来源 |
|-----|-------------|-----------|
| `leastsq` | `scipy.optimize.leastsq` 返回元组 | 直接返回的 `_cov` |
| `least_squares` | `scipy.optimize.OptimizeResult` | 从 Jacobian 计算: `inv(J^T J)` |
| `scalar_minimize` | `scipy.optimize.OptimizeResult` | **numdifftools** 数值计算 Hessian |
| `emcee` | MCMC 采样链 | 从样本统计量估计 |
| 全局优化方法 | 各算法特定结果 | **numdifftools** 数值计算 |

#### 2.3.2 统一统计量计算

`_calculate_statistics()` 方法是所有算法后处理的**核心统一环节**：

```python
def _calculate_statistics(self):
    """Calculate the fitting statistics."""
    self.nvarys = len(self.init_vals)
    
    if not hasattr(self, 'residual'):
        self.residual = -np.inf
    
    # 根据残差类型计算 chi-square
    if isinstance(self.residual, np.ndarray):
        self.chisqr = (self.residual**2).sum()
        self.ndata = len(self.residual)
        self.nfree = self.ndata - self.nvarys
    else:
        self.chisqr = self.residual
        self.ndata = 1
        self.nfree = 1
    
    # 计算归一化统计量
    self.redchi = self.chisqr / max(1, self.nfree)
    
    # 计算信息准则
    self.chisqr = max(self.chisqr, 1.e-250*self.ndata)
    _neg2_log_likel = self.ndata * np.log(self.chisqr / self.ndata)
    self.aic = _neg2_log_likel + 2 * self.nvarys
    self.bic = _neg2_log_likel + np.log(self.ndata) * self.nvarys
```
**位置**: `minimizer.py:299-317`

**统一计算的统计量**:
| 统计量 | 公式 | 说明 |
|-------|------|------|
| `chisqr` | $\sum (residual^2)$ | 卡方值 |
| `redchi` | $chisqr / nfree$ | 约简卡方 |
| `ndata` | `len(residual)` | 数据点数 |
| `nfree` | $ndata - nvarys$ | 自由度 |
| `aic` | $N \ln(\chi^2/N) + 2 N_{varys}$ | AIC 信息准则 |
| `bic` | $N \ln(\chi^2/N) + \ln(N) N_{varys}$ | BIC 信息准则 |

#### 2.3.3 协方差矩阵的统一处理

不同算法的协方差计算方式不同，但最终统一到 `result.covar` 属性：

##### 方式1: 直接从求解器获取 (leastsq)

```python
# 在 leastsq() 中
_best, _cov, _infodict, errmsg, ier = lsout

if result.errorbars:
    # 内部空间 -> 外部参数空间的转换
    result.covar = self._int2ext_cov_x(_cov, _best)
    self._calculate_uncertainties_correlations()
```
**位置**: `minimizer.py:1727-1730`

##### 方式2: 从 Jacobian 计算 (least_squares)

```python
# 在 least_squares() 中
try:
    if issparse(ret.jac):
        hess = (ret.jac.T @ ret.jac).toarray()
    elif isinstance(ret.jac, LinearOperator):
        identity = np.eye(ret.jac.shape[1], dtype=ret.jac.dtype)
        hess = (ret.jac.T * ret.jac) * identity
    else:
        hess = np.matmul(ret.jac.T, ret.jac)
    result.covar = np.linalg.inv(hess)
    self._calculate_uncertainties_correlations()
except LinAlgError:
    pass
```
**位置**: `minimizer.py:1606-1617`

##### 方式3: 数值计算 Hessian (标量方法和全局方法)

```python
def _calculate_covariance_matrix(self, fvars):
    """使用 numdifftools 计算 Hessian 矩阵"""
    try:
        Hfun = ndt.Hessian(self.penalty, step=1.e-4)
        hessian_ndt = Hfun(fvars)
        cov_x = inv(hessian_ndt) * 2.0  # cov = inv(Hessian) * 2
        
        if cov_x.diagonal().min() < 0:
            cov_x = None
    except (LinAlgError, ValueError):
        cov_x = None
    return cov_x
```
**位置**: `minimizer.py:720-763`

**调用条件**:
- `not result.aborted` (拟合未中止)
- `self.calc_covar` (用户要求计算)
- `HAS_NUMDIFFTOOLS` (已安装 numdifftools)
- `len(result.residual) > len(result.var_names)` (过定问题)

#### 2.3.4 不确定性和相关性计算

```python
def _calculate_uncertainties_correlations(self):
    """Calculate parameter uncertainties and correlations."""
    self.result.errorbars = True
    
    # 可选: 根据 redchi 缩放协方差
    if self.scale_covar:
        self.result.covar *= self.result.redchi
    
    # 为每个参数计算 stderr 和相关性
    for ivar, name in enumerate(self.result.var_names):
        par = self.result.params[name]
        par.stderr = float(np.sqrt(self.result.covar[ivar, ivar]))
        par.correl = {}
        
        # 计算与其他参数的相关系数
        for jvar, varn2 in enumerate(self.result.var_names):
            if jvar != ivar:
                par.correl[varn2] = float(
                    self.result.covar[ivar, jvar] / 
                    (par.stderr * np.sqrt(self.result.covar[jvar, jvar]))
                )
    
    # 创建 uncertainties 包的 ufloats
    if self.result.errorbars:
        self.result.uvars = self.result.params.create_uvars(
            covar=self.result.covar
        )
```
**位置**: `minimizer.py:795-818`

### 2.4 算法特有属性的处理

不同算法会在 `MinimizerResult` 上添加**特有属性**，通过**属性复制**机制处理：

#### 2.4.1 scipy 结果属性的复制

```python
# 在 scalar_minimize() 中
if not result.aborted:
    if isinstance(ret, dict):
        for attr, value in ret.items():
            setattr(result, attr, value)
    else:
        for attr in dir(ret):
            if not attr.startswith('_'):
                setattr(result, attr, getattr(ret, attr))
```
**位置**: `minimizer.py:1001-1008`

这意味着 `scipy.optimize.OptimizeResult` 的所有属性（如 `x`, `fun`, `jac`, `hess`, `nit`, `nfev` 等）都会被复制到 `MinimizerResult`。

#### 2.4.2 各算法的特有属性示例

| 算法 | 特有属性前缀 | 说明 |
|-----|-------------|------|
| `brute` | `brute_*` | `brute_x0`, `brute_fval`, `brute_grid`, `brute_Jout`, `candidates` |
| `ampgo` | `ampgo_*` | `ampgo_x0`, `ampgo_fval`, `ampgo_eval`, `ampgo_msg`, `ampgo_tunnel` |
| `shgo` | `shgo_*` | `shgo_x`, `shgo_xl`, `shgo_fun`, `shgo_funl`, `shgo_nfev` 等 |
| `dual_annealing` | `da_*` | `da_x`, `da_fun`, `da_nfev`, `da_nhev` 等 |
| `direct` | `direct_*` | `direct_x`, `direct_fun` 等 |
| `least_squares` | `least_squares_*` | `least_squares_nfev` |
| `emcee` | 无前缀 | `chain`, `lnprob`, `flatchain`, `acor`, `acceptance_fraction` |

以 `shgo()` 为例:

```python
def shgo(self, params=None, max_nfev=None, **kws):
    # ...
    if not result.aborted:
        for attr, value in ret.items():
            if attr in ['success', 'message']:
                setattr(result, attr, value)
            else:
                setattr(result, f'shgo_{attr}', value)  # 添加前缀
    # ...
```
**位置**: `minimizer.py:2170-2175`

### 2.5 ModelResult 的增强

`ModelResult` 在 `MinimizerResult` 基础上添加了**模型特定**的属性：

```python
# 在 ModelResult.fit() 中
self.init_fit = self.model.eval(params=self.init_params, **self.userkws)
self.init_values = self.model._make_all_args(self.init_params)
self.best_values = self.model._make_all_args(_ret.params)
self.best_fit = self.model.eval(params=_ret.params, **self.userkws)

# 计算 R²
if (self.data is not None and len(self.data) > 1
   and isinstance(self.best_fit, np.ndarray)
   and len(self.best_fit) > 1):
    dat = coerce_arraylike(self.data)
    resid = ((dat - self.best_fit)**2).sum()
    sstot = ((dat - dat.mean())**2).sum()
    self.rsquared = 1.0 - resid/max(tiny, sstot)
```
**位置**: `model.py:1588-1597`

**ModelResult 新增属性**:
| 属性 | 说明 |
|-----|------|
| `model` | 关联的 Model 对象 |
| `data` | 拟合数据 |
| `weights` | 拟合权重 |
| `init_params` | 初始参数 |
| `init_fit` | 初始模型计算值 |
| `init_values` | 初始参数字典 |
| `best_fit` | 最佳拟合曲线 |
| `best_values` | 最佳参数字典 |
| `rsquared` | 决定系数 $R^2$ |
| `components` | 复合模型的组件列表 |

---

## 3. 序列化机制与跨算法差异

### 3.1 序列化架构概览

lmfit 使用 **JSON + 自定义编码** 的序列化方案，核心在 `jsonutils.py` 模块：

```
┌──────────────────────────────────────────────────────┐
│  用户调用: dumps() / dump()                           │
└──────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────┐
│  Model.dumps() / ModelResult.dumps()                 │
│  - 收集需要序列化的属性                                │
│  - 调用 encode4js() 处理特殊类型                       │
└──────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────┐
│  encode4js() (jsonutils.py)                          │
│  - 递归编码 Python 对象为 JSON 可序列化格式            │
│  - 特殊类型: ndarray, complex, callable, DataFrame   │
└──────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────┐
│  json.dumps()                                         │
│  - 最终序列化为 JSON 字符串                            │
└──────────────────────────────────────────────────────┘
```

### 3.2 核心编码/解码函数

#### 3.2.1 `encode4js()` 实现

```python
def encode4js(obj):
    """Prepare an object for JSON encoding."""
    
    # pandas DataFrame
    if isinstance(obj, DataFrame):
        return dict(__class__='PDataFrame', value=obj.to_json())
    
    # pandas Series
    if isinstance(obj, Series):
        return dict(__class__='PSeries', value=obj.to_json())
    
    # uncertainties 包的 ufloat
    if isinstance(obj, uncertainties.core.AffineScalarFunc):
        return dict(__class__='UFloat', val=obj.nominal_value, err=obj.std_dev)
    
    # numpy ndarray
    if isinstance(obj, np.ndarray):
        if 'complex' in obj.dtype.name:
            val = [(obj.real).tolist(), (obj.imag).tolist()]
        elif obj.dtype.name == 'object':
            val = [encode4js(item) for item in obj]
        else:
            val = obj.flatten().tolist()
        return dict(__class__='NDArray', __shape__=obj.shape,
                    __dtype__=obj.dtype.name, value=val)
    
    # 复数
    if isinstance(obj, complex):
        return dict(__class__='Complex', value=(obj.real, obj.imag))
    
    # 列表/元组
    if isinstance(obj, (tuple, list)):
        ctype = 'List' if isinstance(obj, list) else 'Tuple'
        val = [encode4js(item) for item in obj]
        return dict(__class__=ctype, value=val)
    
    # 字典
    if isinstance(obj, dict):
        out = dict(__class__='Dict')
        for key, val in obj.items():
            out[encode4js(key)] = encode4js(val)
        return out
    
    # 可调用对象 (函数) - 使用 dill 序列化
    if callable(obj):
        value = str(b64encode(dill.dumps(obj)), 'utf-8')
        return dict(__class__='Callable', __name__=obj.__name__,
                    pyversion=pyvers, value=value,
                    importer=find_importer(obj))
    
    return obj
```
**位置**: `jsonutils.py:46-97`

#### 3.2.2 特殊类型编码策略

| 类型 | 编码方式 | 解码方式 |
|-----|---------|---------|
| `numpy.ndarray` | `{'__class__': 'NDArray', '__shape__': ..., '__dtype__': ..., 'value': ...}` | `np.fromiter()` + reshape |
| `complex` | `{'__class__': 'Complex', 'value': (real, imag)}` | 直接重组 |
| `pandas.DataFrame` | `{'__class__': 'PDataFrame', 'value': df.to_json()}` | `pd.read_json()` |
| `uncertainties.UFloat` | `{'__class__': 'UFloat', 'val': ..., 'err': ...}` | `uncertainties.ufloat()` |
| `callable` (函数) | `dill.dumps()` + base64 + 模块信息 | 优先从模块导入，失败则 `dill.loads()` |

### 3.3 ModelResult 的序列化

#### 3.3.1 `ModelResult.dumps()` 实现

```python
def dumps(self, **kws):
    """Represent ModelResult as a JSON string."""
    out = {'__class__': 'lmfit.ModelResult', '__version__': '2',
           'model': encode4js(self.model._get_state())}
    
    # Parameters 使用自身的 dumps 方法
    for attr in ('params', 'init_params'):
        out[attr] = getattr(self, attr).dumps()
    
    # 其他属性批量编码
    for attr in ('aborted', 'aic', 'best_values', 'bic', 'chisqr',
                 'ci_out', 'col_deriv', 'covar', 'errorbars', 'flatchain',
                 'ier', 'init_values', 'lmdif_message', 'message',
                 'method', 'nan_policy', 'ndata', 'nfev', 'nfree',
                 'nvarys', 'redchi', 'residual', 'rsquared', 'scale_covar',
                 'calc_covar', 'success', 'userargs', 'userkws', 'values',
                 'var_names', 'weights', 'user_options'):
        try:
            val = getattr(self, attr)
        except AttributeError:
            continue
        if isinstance(val, np.bool_):
            val = bool(val)
        out[attr] = encode4js(val)
    
    return json.dumps(out, **kws)
```
**位置**: `model.py:1908-1951`

**序列化内容分析**:

| 类别 | 属性 | 说明 |
|-----|------|------|
| **元数据** | `__class__`, `__version__` | 类型标识和版本号 |
| **模型** | `model` | 通过 `_get_state()` 获取模型状态 |
| **参数** | `params`, `init_params` | Parameters 对象独立序列化 |
| **拟合状态** | `aborted`, `success`, `message`, `ier`, `lmdif_message` | 拟合成功/失败信息 |
| **统计量** | `chisqr`, `redchi`, `aic`, `bic`, `rsquared` | 拟合质量指标 |
| **维度信息** | `ndata`, `nvarys`, `nfree`, `nfev` | 数据和计算规模 |
| **数值结果** | `covar`, `residual`, `var_names`, `best_values`, `init_values` | 核心数值结果 |
| **配置** | `method`, `nan_policy`, `scale_covar`, `calc_covar`, `userargs`, `userkws` | 拟合配置 |
| **可选** | `flatchain`, `ci_out` | MCMC 链或置信区间结果 |

### 3.4 不同算法的序列化差异

#### 3.4.1 序列化内容与算法无关性

**核心设计原则**: `ModelResult.dumps()` 序列化的是**统一的核心属性**，而非算法特有属性。

查看序列化的属性列表，你会发现：
- ✅ 包含: `method` (算法名称)
- ❌ **不包含**: 算法特有属性如 `chain` (emcee), `candidates` (brute), `shgo_x` 等

#### 3.4.2 明确不序列化的属性

| 属性 | 所属算法 | 不序列化原因 |
|-----|---------|-------------|
| `chain` | emcee | MCMC 采样链可能非常大 |
| `lnprob` | emcee | 对数概率数组 |
| `flatchain` | emcee | 有条件序列化 (在列表中但可能为 None) |
| `candidates` | brute | 候选参数列表 |
| `brute_*` | brute | brute 方法特有结果 |
| `ampgo_*` | ampgo | AMPGO 特有结果 |
| `shgo_*` | shgo | SHGO 特有结果 |
| `da_*` | dual_annealing | 对偶退火特有结果 |
| `direct_*` | direct | DIRECT 特有结果 |
| `jac`, `hess` | 多种 | scipy 求解器的中间结果 |

#### 3.4.3 重要例外: `flatchain`

注意 `flatchain` **在序列化列表中**，但在 `ModelResult` 中它是一个属性访问器：

```python
@property
def flatchain(self):
    """Show flatchain view of the sampling chain from `emcee` method."""
    if not hasattr(self, 'chain'):
        return None
    # ... pandas DataFrame 构造
```
**位置**: `minimizer.py:258-272`

这意味着:
- 如果 `chain` 不存在 (非 emcee 方法)，`flatchain` 返回 `None`
- `encode4js(None)` 会被正确序列化为 `null`

### 3.5 反序列化机制

#### 3.5.1 `ModelResult.loads()` 实现

```python
def loads(self, s, funcdefs=None, **kws):
    """Load ModelResult from a JSON string."""
    modres = json.loads(s, **kws)
    if 'modelresult' not in modres['__class__'].lower():
        raise AttributeError('ModelResult.loads() needs saved ModelResult')
    
    modres = decode4js(modres)
    
    # 重建 Model
    self.model = _buildmodel(decode4js(modres['model']), funcdefs=funcdefs)
    
    # 重建 Parameters (支持版本 1 和 2)
    modres_vers = modres.get('__version__', '1')
    if modres_vers == '1':
        # 旧版本兼容
        for target in ('params', 'init_params'):
            state = {'unique_symbols': modres['unique_symbols'], 'params': []}
            # ...
    elif modres_vers == '2':
        for target in ('params', 'init_params'):
            _pars = Parameters()
            _pars.loads(modres[target])
            # ...
    
    # 恢复其他属性
    for attr in ('aborted', 'aic', 'best_fit', 'best_values', 'bic',
                 'chisqr', 'ci_out', 'col_deriv', 'covar', 'data',
                 'errorbars', 'fjac', 'flatchain', 'ier', 'init_fit',
                 'init_values', 'kws', 'lmdif_message', 'message',
                 'method', 'nan_policy', 'ndata', 'nfev', 'nfree',
                 'nvarys', 'redchi', 'residual', 'rsquared', 'scale_covar',
                 'calc_covar', 'success', 'userargs', 'userkws',
                 'var_names', 'weights', 'user_options'):
        setattr(self, attr, decode4js(modres.get(attr, None)))
    
    # 重新计算派生属性
    self.best_fit = self.model.eval(self.params, **self.userkws)
    # ...
    
    return self
```
**位置**: `model.py:1976-2064`

### 3.6 序列化限制与注意事项

#### 3.6.1 函数序列化的限制

```python
# 在 decode4js() 中处理 Callable 类型
elif classname == 'Callable':
    out = obj['__name__']
    try:
        # 优先尝试从原模块导入
        out = import_from(obj['importer'], out)
        unpacked = True
    except (ImportError, AttributeError):
        unpacked = False
    
    if not unpacked:
        # 版本不匹配警告
        spyvers = obj.get('pyversion', '?')
        if not pyvers == spyvers:
            msg = f"Could not unpack dill-encoded callable '{out}', saved with Python version {spyvers}"
            warnings.warn(msg)
        
        # 回退到 dill 反序列化
        try:
            out = dill.loads(b64decode(obj['value']))
        except RuntimeError:
            msg = f"Could not unpack dill-encoded callable '{out}`, saved with Python version {spyvers}"
            warnings.warn(msg)
```
**位置**: `jsonutils.py:135-152`

**函数序列化限制**:
1. **Python 版本兼容性**: 不同 Python 版本的 dill 序列化可能不兼容
2. **模块依赖**: 如果函数来自已安装模块，优先从模块导入（更可靠）
3. **lambda 函数**: 只能依赖 dill 反序列化，版本不匹配时可能失败
4. **闭包和环境**: dill 会尝试捕获函数的闭包环境，但复杂环境可能失败

#### 3.6.2 算法特有信息的丢失

由于序列化时**不保存算法特有属性**，反序列化后会丢失:

```python
# 假设使用 emcee 拟合并保存
result = model.fit(data, method='emcee', ...)
result.dump('result.json')

# 重新加载
loaded = load_modelresult('result.json')

# 以下属性将丢失:
loaded.chain        # AttributeError (不存在)
loaded.lnprob       # AttributeError (不存在)
loaded.acceptance_fraction  # AttributeError (不存在)
```

#### 3.6.3 版本兼容性

序列化格式有版本管理:

```python
# 序列化时
out = {'__class__': 'lmfit.ModelResult', '__version__': '2', ...}

# 反序列化时
modres_vers = modres.get('__version__', '1')
if modres_vers == '1':
    # 旧版本格式处理
elif modres_vers == '2':
    # 当前版本格式处理
```

这保证了向后兼容性，但**不保证向前兼容性**（新版本保存的文件可能无法用旧版本 lmfit 加载）。

---

## 附录: 关键代码位置速查

| 功能 | 文件 | 行号 |
|-----|------|------|
| 算法分发表 `SCALAR_METHODS` | `minimizer.py` | 84-100 |
| 核心分发方法 `minimize()` | `minimizer.py` | 2356-2455 |
| `MinimizerResult` 类定义 | `minimizer.py` | 175-324 |
| 统计量计算 `_calculate_statistics()` | `minimizer.py` | 299-317 |
| 准备拟合 `prepare_fit()` | `minimizer.py` | 618-709 |
| `ModelResult` 类定义 | `model.py` | 1479-2460 |
| `ModelResult.dumps()` 序列化 | `model.py` | 1908-1951 |
| `ModelResult.loads()` 反序列化 | `model.py` | 1976-2064 |
| JSON 编码 `encode4js()` | `jsonutils.py` | 46-97 |
| JSON 解码 `decode4js()` | `jsonutils.py` | 100-158 |

---

## 总结

### 架构设计亮点

1. **清晰的分层抽象**:
   - `Minimizer` 处理底层优化
   - `Model` 提供高级建模接口
   - 结果对象 `MinimizerResult` / `ModelResult` 统一封装

2. **灵活的算法分发**:
   - 基于字符串的模糊匹配
   - 可扩展的方法体系
   - 统一的调用入口 `minimize()`

3. **统一的结果组装**:
   - `_calculate_statistics()` 确保核心统计量一致
   - 协方差矩阵多种计算方式，但统一接口
   - 算法特有属性通过前缀命名避免冲突

4. **健壮的序列化机制**:
   - 基于 JSON 的人类可读格式
   - 特殊类型（ndarray, complex, 函数）的智能处理
   - 版本管理保证向后兼容

### 潜在限制

1. **算法特有信息序列化丢失**: 全局优化和 MCMC 方法的详细结果无法持久化
2. **函数序列化依赖 dill**: 跨 Python 版本可能存在兼容性问题
3. **大对象未处理**: MCMC 链 (`chain`) 可能非常大，但设计上选择不序列化

