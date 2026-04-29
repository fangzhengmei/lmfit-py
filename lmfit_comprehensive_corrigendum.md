# lmfit 多点纠偏综合勘误报告

> 本文档系统复核三个核心模块的实现细节，并分析它们之间的调用关联与影响。

---

## 目录

1. [模块一：算法路由匹配规则](#模块一算法路由匹配规则)
2. [模块二：结果对象关系与职责边界](#模块二结果对象关系与职责边界)
3. [模块三：采样结果持久化机制](#模块三采样结果持久化机制)
4. [跨模块关联与影响分析](#跨模块关联与影响分析)
5. [完整调用链全景图](#完整调用链全景图)

---

## 模块一：算法路由匹配规则

### 1.1 核心分发逻辑

**源码位置**：`minimizer.py:2430-2455`

```python
user_method = method.lower()

# 第一级：专用方法分发
if user_method.startswith('leasts'):
    function = self.leastsq
elif user_method.startswith('least_s'):
    function = self.least_squares
elif user_method == 'brute':
    function = self.brute
# ... 其他专用方法 ...

# 第二级：标量方法分发（回退分支）
else:
    function = self.scalar_minimize
    for key, val in SCALAR_METHODS.items():
        # ⚠️ 关键：检查 key/val 是否以 user_method 为前缀
        if (key.lower().startswith(user_method) or
                val.lower().startswith(user_method)):
            kwargs['method'] = val
```

### 1.2 关键匹配规则

| 规则 | 说明 |
|-----|------|
| **方向** | 字典的 `key` 或 `val` **以用户输入为前缀**（不是反过来） |
| **大小写** | 用户输入先转 `lower()`，然后比较也是用 `key.lower()` / `val.lower()` |
| **歧义处理** | 多个匹配时，按字典插入顺序，**最后一个匹配覆盖前面的** |
| **无匹配** | `kwargs['method']` 从未被设置，`scalar_minimize()` 使用默认值 |

### 1.3 `scalar_minimize()` 默认参数

**源码位置**：`minimizer.py:820`

```python
def scalar_minimize(self, method='Nelder-Mead', params=None, max_nfev=None, **kws):
```

**关键**：当 `kwargs` 中没有 `method` 时，使用默认值 `'Nelder-Mead'`。

### 1.4 精确匹配案例对照表

| 用户输入 | 匹配过程 | 实际方法 |
|---------|---------|---------|
| `'lbfgs'` | `'lbfgsb'.startswith('lbfgs')` → True | `'L-BFGS-B'` |
| `'l'` | `'l-bfgsb'.startswith('l')` → True（最后一个以 'l' 开头） | `'L-BFGS-B'` |
| `'trust'` | `'trust-krylov'.startswith('trust')` → True（最后一个以 'trust' 开头） | `'trust-krylov'` |
| `'powells'` | `'powell'.startswith('powells')` → False（6字符 < 7字符） | **退化为 `'Nelder-Mead'`** |
| `'unknown'` | 无任何 key/val 以 'unknown' 开头 | **退化为 `'Nelder-Mead'`** |

---

## 模块二：结果对象关系与职责边界

### 2.1 类继承关系的关键勘误

#### ❌ 之前的错误判断

```
ModelResult 继承自 MinimizerResult
```

#### ✅ 实际源码结构

**源码位置**：`model.py:1479`

```python
class ModelResult(Minimizer):  # 继承自 Minimizer，不是 MinimizerResult！
    """Result from the Model fit.

    This has many attributes and methods for viewing and working with the
    results of a fit using Model. It inherits from Minimizer, so that it
    can be used to modify and re-run the fit for the Model.
    """
```

### 2.2 完整类层次图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Minimizer (优化器类)                           │
│  位置: minimizer.py:326                                          │
│                                                                  │
│  职责: 执行优化算法                                                │
│  方法: minimize(), leastsq(), scalar_minimize(), emcee() 等      │
│  属性: self.userfcn, self.kws, self.params, self.result 等       │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ 继承
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    ModelResult                                   │
│  位置: model.py:1479                                            │
│                                                                  │
│  职责: 面向 Model 接口的高级结果对象                               │
│  继承自 Minimizer: 可调用 self.minimize() 重新拟合                │
│  新增能力: plot(), eval_uncertainty(), dump/load 等             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  MinimizerResult (独立类)                         │
│  位置: minimizer.py:175                                          │
│                                                                  │
│  职责: 单次优化的结果数据容器                                       │
│  不继承任何类                                                      │
│  属性: params, chisqr, covar, residual, chain, lnprob 等        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 数据传递机制：属性复制

**关键发现**：`ModelResult` **不是通过继承**获得 `MinimizerResult` 的数据，而是通过**运行时属性复制**。

**源码位置**：`model.py:1577-1586`

```python
def fit(self, data=None, params=None, weights=None, method=None, ...):
    # ...
    _ret = self.minimize(method=self.method, params=self.init_params)
    # _ret 是 MinimizerResult 实例
    
    # ⚠️ 核心机制：属性复制
    for attr in dir(_ret):
        if not attr.startswith('_'):
            try:
                setattr(self, attr, getattr(_ret, attr))
            except AttributeError:
                pass
```

### 2.4 `dir()` 的行为与影响

`dir(_ret)` 返回 `MinimizerResult` 实例的**所有非私有属性**，包括：

| 属性类型 | 示例 | `getattr()` 行为 |
|---------|------|-----------------|
| 实例属性 | `_ret.chain`, `_ret.lnprob` | 直接返回值 |
| `@property` | `_ret.flatchain` | **调用 getter 方法**，返回计算后的值 |
| 方法 | `_ret.show_candidates()` | 返回方法对象（函数）|

### 2.5 职责边界对照表

| 类 | 继承关系 | 核心职责 | 关键能力 |
|---|---------|---------|---------|
| `Minimizer` | 无父类 | 执行优化算法 | `minimize()`, `leastsq()`, `emcee()` 等 |
| `MinimizerResult` | 无父类 | 存储单次优化结果 | 数据容器：`params`, `chisqr`, `chain` 等 |
| `ModelResult` | 继承自 `Minimizer` | Model 接口的高级结果 | 重新拟合 + `plot()`, `dump()` 等 |

---

## 模块三：采样结果持久化机制

### 3.1 emcee 方法设置的属性

**源码位置**：`minimizer.py:1456-1466`

```python
def emcee(self, params=None, steps=1000, ...):
    # ... 采样过程 ...
    
    result.chain = np.copy(chain)                    # 原始采样链
    result.lnprob = np.copy(lnprobability)          # 对数概率
    result.errorbars = True
    result.nvarys = len(result.var_names)
    result.nfev = nwalkers*steps
    
    try:
        result.acor = self.sampler.get_autocorr_time()          # 自相关时间
    except AutocorrError as e:
        print(str(e))
    result.acceptance_fraction = self.sampler.acceptance_fraction  # 接受率
```

### 3.2 `MinimizerResult.flatchain` 的本质

**源码位置**：`minimizer.py:258-272`

```python
class MinimizerResult:
    @property
    def flatchain(self):
        """Show flatchain view of the sampling chain from `emcee` method."""
        if not hasattr(self, 'chain'):
            return None
        
        # 返回 pandas DataFrame
        if len(self.chain.shape) == 4:
            return pd.DataFrame(self.chain[0, ...].reshape((-1, self.nvarys)),
                                columns=self.var_names)
        elif len(self.chain.shape) == 3:
            return pd.DataFrame(self.chain.reshape((-1, self.nvarys)),
                                columns=self.var_names)
```

**关键发现**：`flatchain` 是一个 `@property`，依赖 `self.chain` 存在。

### 3.3 属性复制时的行为

在 `ModelResult.fit()` 中执行属性复制时：

```python
for attr in dir(_ret):
    if not attr.startswith('_'):
        setattr(self, attr, getattr(_ret, attr))
```

| `attr` | `getattr(_ret, attr)` 行为 | `setattr(self, ...)` 结果 |
|--------|---------------------------|--------------------------|
| `'chain'` | 返回 `_ret.chain` (ndarray) | `self.chain` = ndarray |
| `'lnprob'` | 返回 `_ret.lnprob` (ndarray) | `self.lnprob` = ndarray |
| `'flatchain'` | **调用 property getter**，返回 DataFrame | `self.flatchain` = DataFrame |
| `'acor'` | 返回 `_ret.acor` (ndarray 或 None) | `self.acor` = ndarray 或 None |
| `'acceptance_fraction'` | 返回数组 | `self.acceptance_fraction` = 数组 |

**⚠️ 关键**：`flatchain` 不再是依赖 `chain` 的 property，而是**独立的 DataFrame 实例属性**！

### 3.4 序列化属性列表

**源码位置**：`model.py:1932-1938`

```python
for attr in ('aborted', 'aic', 'best_values', 'bic', 'chisqr',
             'ci_out', 'col_deriv', 'covar', 'errorbars', 'flatchain',  # ✅ flatchain 在列表中
             'ier', 'init_values', 'lmdif_message', 'message',
             'method', 'nan_policy', 'ndata', 'nfev', 'nfree',
             'nvarys', 'redchi', 'residual', 'rsquared', 'scale_covar',
             'calc_covar', 'success', 'userargs', 'userkws', 'values',
             'var_names', 'weights', 'user_options'):
```

### 3.5 持久化结果对照表

| 属性 | 是否在序列化列表 | 序列化时的实际值 | 反序列化后 |
|-----|----------------|-----------------|-----------|
| `flatchain` | ✅ 是 | DataFrame（来自 property getter 求值）| DataFrame |
| `chain` | ❌ 否 | 不在列表中，不序列化 | 不存在 |
| `lnprob` | ❌ 否 | 不在列表中，不序列化 | 不存在 |
| `acor` | ❌ 否 | 不在列表中，不序列化 | 不存在 |
| `acceptance_fraction` | ❌ 否 | 不在列表中，不序列化 | 不存在 |

### 3.6 持久化的实际效果

```python
# 1. 使用 emcee 拟合
result = model.fit(data, method='emcee', steps=1000, ...)

# 此时 result 上有：
# - result.chain (原始多维数组)
# - result.flatchain (DataFrame，由 property getter 求值后复制)
# - result.lnprob, result.acor, result.acceptance_fraction

# 2. 序列化
json_str = result.dumps()

# 序列化内容包括：
# ✅ flatchain (DataFrame 被 encode4js 处理)
# ❌ chain (不在列表中)
# ❌ lnprob (不在列表中)
# ❌ acor (不在列表中)
# ❌ acceptance_fraction (不在列表中)

# 3. 反序列化
loaded = load_modelresult(json_str)

# 反序列化后：
# ✅ loaded.flatchain = DataFrame (恢复)
# ❌ loaded.chain → AttributeError (不存在)
# ❌ loaded.lnprob → AttributeError (不存在)
```

---

## 跨模块关联与影响分析

### 4.1 完整数据流

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              用户调用层                                      │
│  result = model.fit(data, method='emcee', steps=1000, ...)                │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        模块1：算法路由匹配                                    │
│                                                                              │
│  1. Model.fit() 创建 ModelResult 实例                                        │
│  2. ModelResult.fit() 调用 self.minimize(method='emcee')                   │
│  3. Minimizer.minimize() 路由到 self.emcee()                                │
│     - user_method = 'emcee'.lower() = 'emcee'                              │
│     - 匹配: elif user_method == 'emcee' → function = self.emcee           │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        模块2：结果对象数据传递                                │
│                                                                              │
│  1. Minimizer.emcee() 返回 MinimizerResult 实例 (_ret)                      │
│     - 设置: _ret.chain, _ret.lnprob, _ret.acor, _ret.acceptance_fraction   │
│     - _ret.flatchain 是 @property，依赖 _ret.chain                          │
│                                                                              │
│  2. ModelResult.fit() 执行属性复制                                            │
│     for attr in dir(_ret):                                                   │
│         setattr(self, attr, getattr(_ret, attr))                            │
│                                                                              │
│     - 'chain' → getattr 返回 ndarray → self.chain = ndarray               │
│     - 'lnprob' → getattr 返回 ndarray → self.lnprob = ndarray             │
│     - 'flatchain' → getattr 调用 property getter → self.flatchain = DataFrame │
│                                                                              │
│  ⚠️ 关键点：self.flatchain 现在是独立的 DataFrame，不再依赖 self.chain！    │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        模块3：序列化持久化                                    │
│                                                                              │
│  ModelResult.dumps() 遍历显式属性列表：                                       │
│                                                                              │
│  ✅ 序列化的属性：                                                            │
│     - 'flatchain' → encode4js(DataFrame) → JSON                            │
│                                                                              │
│  ❌ 不序列化的属性（不在列表中）：                                             │
│     - 'chain'                                                                │
│     - 'lnprob'                                                               │
│     - 'acor'                                                                 │
│     - 'acceptance_fraction'                                                  │
└────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 关键关联点分析

#### 关联点1：算法选择 → 结果属性

| 使用的方法 | MinimizerResult 上的特有属性 |
|-----------|----------------------------|
| `leastsq` | 无特有属性（标准属性） |
| `scalar_minimize` | 复制 scipy.OptimizeResult 的属性 |
| `emcee` | `chain`, `lnprob`, `acor`, `acceptance_fraction` |
| `brute` | `candidates`, `brute_*` 前缀属性 |
| `shgo` | `shgo_*` 前缀属性 |

**影响**：不同算法产生的 `MinimizerResult` 实例有不同的属性集合。

#### 关联点2：dir() + getattr() → 隐式转换

```
MinimizerResult 上的属性          getattr() 行为        ModelResult 上的结果
─────────────────────────────────────────────────────────────────────────────
chain (ndarray)              → 返回值          → self.chain = ndarray
flatchain (@property)        → 调用 getter     → self.flatchain = DataFrame  
                                          ⬆️
                                          │
                              这是关键的隐式转换！
```

**影响**：`flatchain` 从"依赖 `chain` 的动态属性"变成"独立的数据副本"。

#### 关联点3：序列化列表 → 信息丢失

**序列化列表是硬编码的**，不依赖实际存在的属性：

```python
# 不管使用什么方法，只序列化这个列表中的属性
for attr in ('aborted', 'aic', ..., 'flatchain', ...):
    try:
        val = getattr(self, attr)
    except AttributeError:
        continue  # 不存在则跳过
    out[attr] = encode4js(val)
```

**影响**：
- 如果使用 `emcee`，`chain` 存在但不在列表中 → **丢失**
- 如果使用其他方法，`flatchain` property 返回 `None` → 序列化为 `null`

### 4.3 算法路由对持久化的连锁影响

```
用户选择 method='emcee'
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ Minimizer.minimize() 路由到 emcee()                 │
│                                                      │
│ MinimizerResult 设置:                                │
│   - result.chain = np.copy(chain)                   │
│   - result.lnprob = np.copy(lnprobability)         │
│   - result.acor = ...                               │
│   - result.acceptance_fraction = ...                │
│                                                      │
│ result.flatchain 是 @property，依赖 result.chain    │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ ModelResult.fit() 执行属性复制                       │
│                                                      │
│ for attr in dir(_ret):                              │
│     setattr(self, attr, getattr(_ret, attr))       │
│                                                      │
│ 结果:                                                │
│   - self.chain = ndarray (直接复制)                 │
│   - self.flatchain = DataFrame (getter 求值结果)   │
│                                                      │
│ ⚠️ self.flatchain 不再依赖 self.chain！             │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ ModelResult.dumps() 序列化                          │
│                                                      │
│ 序列化列表检查:                                       │
│   ✅ 'flatchain' 在列表中 → 序列化 DataFrame         │
│   ❌ 'chain' 不在列表中 → 不序列化                   │
│   ❌ 'lnprob' 不在列表中 → 不序列化                 │
│   ❌ 'acor' 不在列表中 → 不序列化                   │
│   ❌ 'acceptance_fraction' 不在列表中 → 不序列化   │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ 反序列化后                                           │
│                                                      │
│ ✅ loaded.flatchain = DataFrame (恢复)              │
│ ❌ loaded.chain → AttributeError (不存在)           │
│ ❌ loaded.lnprob → AttributeError (不存在)          │
│                                                      │
│ 后果:                                                │
│ - 无法重新计算 flatchain（因为 chain 丢失）          │
│ - 无法进行后续的 MCMC 分析（如自相关分析）           │
└─────────────────────────────────────────────────────┘
```

---

## 完整调用链全景图

### 5.1 类关系图

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              类层次结构                                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐                                                           │
│   │  Minimizer  │ ←───────────────────┐                                    │
│   └──────┬──────┘                      │ 继承                               │
│          │                             ▼                                    │
│          │                    ┌──────────────┐                             │
│          │                    │ ModelResult  │                             │
│          │                    └──────────────┘                             │
│          │                             │                                    │
│          │ 产生                         │ 通过属性复制获得                   │
│          ▼                             ▼                                    │
│   ┌─────────────────┐          ┌─────────────────┐                         │
│   │ MinimizerResult │ ◄─────── │   数据属性      │                         │
│   │  (数据容器)     │   复制    │  (chain等)     │                         │
│   └─────────────────┘          └─────────────────┘                         │
│                                                                              │
└────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 完整调用时序图

```
用户                    Model                    Minimizer              MinimizerResult
 │                         │                          │                          │
 │── fit(data, method='emcee') ──▶                    │                          │
 │                         │                          │                          │
 │                         │── ModelResult() ──▶      │                          │
 │                         │                          │                          │
 │                         │── fit() ──▶               │                          │
 │                         │                          │                          │
 │                         │                          │── minimize('emcee') ──▶  │
 │                         │                          │                          │
 │                         │                          │── emcee() ──▶            │
 │                         │                          │                          │
 │                         │                          │                          │── prepare_fit()
 │                         │                          │                          │── 执行 MCMC 采样
 │                         │                          │                          │
 │                         │                          │                          │── result.chain = ...
 │                         │                          │                          │── result.lnprob = ...
 │                         │                          │                          │── result.acor = ...
 │                         │                          │                          │── result.flatchain (@property)
 │                         │                          │                          │
 │                         │                          │◀── return MinimizerResult ──
 │                         │                          │                          │
 │                         │                          │ 属性复制:                 │
 │                         │                          │ for attr in dir(_ret):   │
 │                         │                          │     setattr(self, attr,   │
 │                         │                          │             getattr(_ret, attr))
 │                         │                          │                          │
 │                         │                          │ ⚠️ 关键转换:              │
 │                         │                          │ 'flatchain' → DataFrame   │
 │                         │                          │ 不再依赖 'chain'！        │
 │                         │                          │                          │
 │                         │◀── return ModelResult ── │                          │
 │                         │                          │                          │
 │◀── return ModelResult ── │                          │                          │
 │                         │                          │                          │
 │── dumps() ──▶            │                          │                          │
 │                         │                          │                          │
 │                         │ 序列化列表检查:            │                          │
 │                         │ ✅ 'flatchain' 序列化     │                          │
 │                         │ ❌ 'chain' 不序列化       │                          │
 │                         │ ❌ 'lnprob' 不序列化      │                          │
 │                         │                          │                          │
 │◀── return JSON string ── │                          │                          │
```

---

## 关键勘误总结

| 模块 | 之前的错误 | 实际情况 | 影响 |
|-----|-----------|---------|------|
| **算法路由** | `'lbfgs'` 退化为 `'Nelder-Mead'` | `'lbfgs'` 命中 `'L-BFGS-B'` | 匹配方向是关键 |
| **类继承** | `ModelResult` 继承自 `MinimizerResult` | `ModelResult` 继承自 `Minimizer` | 数据通过属性复制传递 |
| **持久化** | `flatchain` 依赖 `chain` | 属性复制后 `flatchain` 是独立 DataFrame | 序列化的是 DataFrame，不是 property |

---

## 源码位置速查

| 内容 | 文件 | 行号 |
|-----|------|------|
| `ModelResult` 类定义 | `model.py` | 1479 |
| 属性复制逻辑 | `model.py` | 1581-1586 |
| `MinimizerResult.flatchain` property | `minimizer.py` | 258-272 |
| `emcee()` 设置结果属性 | `minimizer.py` | 1456-1466 |
| 算法分发逻辑 | `minimizer.py` | 2430-2455 |
| 序列化属性列表 | `model.py` | 1932-1938 |
