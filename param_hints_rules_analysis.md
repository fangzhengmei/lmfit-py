# lmfit param_hints 键名存储规则与表达式引用规则分析

## 目录
1. [概述](#概述)
2. [对账测试方法](#对账测试方法)
3. [规则1：提示键名存储规则](#规则1提示键名存储规则)
4. [规则2：param_hints 传播规则](#规则2param_hints-传播规则)
5. [规则3：表达式引用规则](#规则3表达式引用规则)
6. [关键对比与澄清](#关键对比与澄清)
7. [对账证据汇总](#对账证据汇总)
8. [结论与最佳实践](#结论与最佳实践)

---

## 概述

本文档通过**对账测试**明确区分 lmfit 中两个容易混淆的规则：

| 规则类型 | 说明 | 关键点 |
|---------|------|--------|
| **提示键名存储规则** | `param_hints` 字典的键名如何存储 | 单模型 vs CompositeModel 行为不同 |
| **表达式引用规则** | `expr` 表达式中如何引用参数名 | 统一使用带前缀的完整参数名 |

**重要发现**：
1. **提示键名存储规则**在单模型和 CompositeModel 中**行为不同**
2. **表达式引用规则**是**统一的**：始终使用带前缀的完整参数名
3. 这两个规则经常被混淆，导致用户困惑

---

## 对账测试方法

### 测试代码

使用 `param_hints_verification.py` 脚本进行对账测试，核心测试逻辑：

```python
# 测试1：单模型的 param_hints 键名
g1 = GaussianModel(prefix='g1_')
print(g1.param_hints)
# 实际输出：
# {'sigma': {'min': 0}, 
#  'fwhm': {'expr': '2.3548200*g1_sigma'}, 
#  'height': {'expr': '0.3989423*g1_amplitude/max(1e-15, g1_sigma)'}}

# 测试2：CompositeModel 的 param_hints 键名
model = g1 + GaussianModel(prefix='g2_')
print(model.param_hints)
# 实际输出：
# {'g1_sigma': {'min': 0}, 
#  'g1_fwhm': {'expr': '2.3548200*g1_sigma'}, 
#  'g1_height': {'expr': '...'},
#  'g2_sigma': {'min': 0}, 
#  'g2_fwhm': {'expr': '2.3548200*g2_sigma'}, ...}

# 测试3：表达式引用规则
params = model.make_params()
print(params['g1_fwhm'].expr)
# 实际输出：'2.3548200*g1_sigma'（使用带前缀的参数名）
```

### 关键观察

从测试输出中可以观察到两个核心现象：

| 观察点 | 单模型 (GaussianModel) | CompositeModel |
|--------|------------------------|-----------------|
| param_hints 键名 | `'sigma'`, `'fwhm'`, `'height'` | `'g1_sigma'`, `'g1_fwhm'`, `'g1_height'` |
| expr 中的参数名 | `'g1_sigma'`, `'g1_amplitude'` | `'g1_sigma'`, `'g1_amplitude'` |

**关键发现**：
- 键名存储规则：单模型不带前缀，CompositeModel 带前缀
- 表达式引用规则：**始终使用带前缀的参数名**（无论键名如何存储）

---

## 规则1：提示键名存储规则

### 核心代码分析

**Model.set_param_hint()** (`model.py:665-676`):

```python
def set_param_hint(self, name, **kwargs):
    """Set hints to use when creating parameters."""
    npref = len(self._prefix)
    if npref > 0 and name.startswith(self._prefix):
        name = name[npref:]  # 剥离前缀！
    
    if name not in self.param_hints:
        self.param_hints[name] = {}
    
    for key, val in kwargs.items():
        if key in self._hint_names:
            self.param_hints[name][key] = val  # 使用剥离后的名称存储
```

**关键逻辑**：
1. 如果输入的 `name` 以模型前缀开头，**会被剥离前缀**
2. 存储在 `self.param_hints` 中的键是**剥离前缀后的名称**

### 单模型的键名存储

对于 `GaussianModel(prefix='g1_')`：

```
GaussianModel._set_paramhints_prefix() 调用:
  1. self.set_param_hint('sigma', min=0)
     - 'sigma' 不以 'g1_' 开头
     - 存储键: 'sigma'
     - self.param_hints['sigma'] = {'min': 0}
  
  2. self.set_param_hint('fwhm', expr=fwhm_expr(self))
     - fwhm_expr(self) 返回 '2.3548200*g1_sigma'
     - 'fwhm' 不以 'g1_' 开头
     - 存储键: 'fwhm'
     - self.param_hints['fwhm'] = {'expr': '2.3548200*g1_sigma'}
```

**实际存储结果**：

| 输入的 name | 存储的键 | 存储的值 |
|-------------|---------|---------|
| `'sigma'` | `'sigma'` | `{'min': 0}` |
| `'fwhm'` | `'fwhm'` | `{'expr': '2.3548200*g1_sigma'}` |
| `'height'` | `'height'` | `{'expr': '0.3989423*g1_amplitude/max(1e-15, g1_sigma)'}` |

### 边界情况：传入带前缀的名称

```python
g1 = GaussianModel(prefix='g1_')

# 传入带前缀的名称
g1.set_param_hint('g1_sigma', max=10)

# 分析：
# - name = 'g1_sigma'
# - self._prefix = 'g1_'
# - 'g1_sigma'.startswith('g1_') → True
# - name = 'g1_sigma'[3:] = 'sigma'
# - 合并到现有提示: self.param_hints['sigma'] = {'min': 0, 'max': 10}
```

**对账测试输出**：
```
初始 g1.param_hints:
   'sigma': {'min': 0}
   'fwhm': {'expr': '2.3548200*g1_sigma'}
   'height': {'expr': '0.3989423*g1_amplitude/max(1e-15, g1_sigma)'}

调用 set_param_hint('g1_sigma', max=10) 后:
   'sigma': {'min': 0, 'max': 10}  ← 已合并！
```

### CompositeModel 的键名存储

**CompositeModel 初始化时的传播逻辑** (`model.py:1273-1276`):

```python
# CompositeModel.__init__ 中的传播代码
for side in (left, right):
    prefix = side.prefix
    for basename, hint in side.param_hints.items():
        self.param_hints[f"{prefix}{basename}"] = hint  # 添加前缀！
```

**传播过程**：

```
子模型 g1 (prefix='g1_') 的 param_hints:
  'sigma': {'min': 0}
  'fwhm': {'expr': '2.3548200*g1_sigma'}
  'height': {'expr': '0.3989423*g1_amplitude/max(1e-15, g1_sigma)'}

传播到 CompositeModel (model = g1 + g2):
  for basename, hint in g1.param_hints.items():
      self.param_hints[f"{g1.prefix}{basename}"] = hint

  即：
  - 'g1_' + 'sigma' = 'g1_sigma' → self.param_hints['g1_sigma'] = {'min': 0}
  - 'g1_' + 'fwhm' = 'g1_fwhm' → self.param_hints['g1_fwhm'] = {'expr': '2.3548200*g1_sigma'}
  - 'g1_' + 'height' = 'g1_height' → self.param_hints['g1_height'] = {...}

注意：hint 中的 expr 值保持不变！
```

**对账测试输出**：
```
peaks12.param_hints (第一层 CompositeModel):
   'g1_sigma': {'min': 0}
   'g1_fwhm': {'expr': '2.3548200*g1_sigma'}
   'g1_height': {'expr': '0.3989423*g1_amplitude/max(1e-15, g1_sigma)'}
   'g2_sigma': {'min': 0}
   'g2_fwhm': {'expr': '2.3548200*g2_sigma'}
   'g2_height': {'expr': '0.3989423*g2_amplitude/max(1e-15, g2_sigma)'}
```

### 规则1 总结

| 模型类型 | 输入的 name | 存储的键名 | 说明 |
|---------|------------|-----------|------|
| **单模型带前缀** | `'sigma'` | `'sigma'` | 不以 `'g1_'` 开头，不剥离 |
| **单模型带前缀** | `'g1_sigma'` | `'sigma'` | 以 `'g1_'` 开头，被剥离 |
| **单模型无前缀** | `'sigma'` | `'sigma'` | 无前缀，直接存储 |
| **CompositeModel** | - | `'g1_sigma'` | 从子模型传播，添加前缀 |

**关键公式**：
```
单模型存储键名 = 输入名称.startswith(prefix) ? 输入名称[len(prefix):] : 输入名称

CompositeModel 存储键名 = 子模型前缀 + 子模型 param_hints 键名
```

---

## 规则2：param_hints 传播规则

### 嵌套复合模型的传播

考虑嵌套模型：`((g1 + g2) + g3) + bkg`

**传播过程**：

```
第1层：g1 + g2 = peaks12
  g1.param_hints → peaks12.param_hints:
    'sigma' → 'g1_sigma'
    'fwhm' → 'g1_fwhm'
    'height' → 'g1_height'
  g2.param_hints → peaks12.param_hints:
    'sigma' → 'g2_sigma'
    'fwhm' → 'g2_fwhm'
    'height' → 'g2_height'

第2层：peaks12 + g3 = all_peaks
  peaks12.param_hints → all_peaks.param_hints:
    'g1_sigma' → 'g1_sigma' (peaks12.prefix = '')
    'g1_fwhm' → 'g1_fwhm'
    'g2_sigma' → 'g2_sigma'
    'g2_fwhm' → 'g2_fwhm'
  g3.param_hints → all_peaks.param_hints:
    'sigma' → 'g3_sigma'
    'fwhm' → 'g3_fwhm'
    'height' → 'g3_height'

第3层：all_peaks + bkg = model
  all_peaks.param_hints → model.param_hints (all_peaks.prefix = '')
  bkg.param_hints → model.param_hints (如果 LinearModel 有 hints)
```

**关键点**：
- CompositeModel 的 `prefix = ''`（空字符串）
- 所以传播时 `f"{prefix}{basename}" = basename`
- 即：键名保持不变

**对账测试输出**：
```
model.param_hints (第三层 CompositeModel):
   'g1_fwhm': {'expr': '2.3548200*g1_sigma'}
   'g1_height': {'expr': '0.3989423*g1_amplitude/max(1e-15, g1_sigma)'}
   'g2_fwhm': {'expr': '2.3548200*g2_sigma'}
   'g2_height': {'expr': '0.3989423*g2_amplitude/max(1e-15, g2_sigma)'}
   'g3_fwhm': {'expr': '2.3548200*g3_sigma'}
   'g3_height': {'expr': '0.3989423*g3_amplitude/max(1e-15, g3_sigma)'}
```

### 传播过程中的不变性

**重要发现**：在传播过程中，**hint 的值（包括 expr）保持不变**！

```
子模型 g1.param_hints['fwhm']:
  {'expr': '2.3548200*g1_sigma'}

传播到 peaks12.param_hints['g1_fwhm']:
  {'expr': '2.3548200*g1_sigma'}  ← expr 不变！

传播到 all_peaks.param_hints['g1_fwhm']:
  {'expr': '2.3548200*g1_sigma'}  ← expr 不变！

传播到最终 model.param_hints['g1_fwhm']:
  {'expr': '2.3548200*g1_sigma'}  ← expr 不变！
```

**这是理解表达式引用规则的关键**：表达式中的参数名**从不被修改**，始终保持原始值。

---

## 规则3：表达式引用规则

### 核心问题

表达式中的参数名应该如何引用？

```python
# 以下哪个是正确的？
model.set_param_hint('fwhm', expr='2.3548200*sigma')      # 不带前缀
model.set_param_hint('fwhm', expr='2.3548200*g1_sigma')   # 带前缀
```

### 答案：必须使用带前缀的完整参数名

**对账测试证据**：

```python
g1 = GaussianModel(prefix='g1_')
params = g1.make_params()

print(params['g1_fwhm'].expr)
# 输出：'2.3548200*g1_sigma'  ← 使用带前缀的参数名！

print(params['g1_height'].expr)
# 输出：'0.3989423*g1_amplitude/max(1e-15, g1_sigma)'  ← 都带前缀！
```

### 为什么必须使用带前缀的参数名？

**原因**：表达式在 `Parameters` 对象的上下文中执行，而 `Parameters` 对象的键是**带前缀的完整参数名**。

**执行流程**：

```
1. make_params() 创建 Parameters 对象
   params = {
       'g1_amplitude': Parameter(...),
       'g1_center': Parameter(...),
       'g1_sigma': Parameter(...),
       'g1_fwhm': Parameter(..., expr='2.3548200*g1_sigma'),
       'g1_height': Parameter(..., expr='0.3989423*g1_amplitude/max(1e-15, g1_sigma)')
   }

2. 表达式在 astevel 解释器中执行
   符号表包含：'g1_amplitude', 'g1_center', 'g1_sigma', ...
   
3. 计算 'g1_fwhm' 时：
   执行 '2.3548200*g1_sigma'
   → 查找 'g1_sigma' → 找到！
   → 计算成功！

4. 如果表达式是 '2.3548200*sigma'（不带前缀）：
   执行 '2.3548200*sigma'
   → 查找 'sigma' → 未找到！
   → 报错：NameError
```

### GaussianModel 中的表达式构建

**fwhm_expr 和 height_expr 函数** (`models.py:31-40`):

```python
def fwhm_expr(model):
    """Return constraint expression for fwhm."""
    fmt = "{factor:.7f}*{prefix:s}sigma"
    return fmt.format(factor=model.fwhm_factor, prefix=model.prefix)
    # 例如：'2.3548200*g1_sigma' (当 model.prefix='g1_')

def height_expr(model):
    """Return constraint expression for maximum peak height."""
    fmt = "{factor:.7f}*{prefix:s}amplitude/max({}, {prefix:s}sigma)"
    return fmt.format(tiny, factor=model.height_factor, prefix=model.prefix)
    # 例如：'0.3989423*g1_amplitude/max(1e-15, g1_sigma)'
```

**关键点**：这些函数在构建表达式时就**直接包含了前缀**！

### 用户自定义约束的表达式

当用户定义跨组件约束时，也必须使用带前缀的参数名：

```python
model = GaussianModel(prefix='g1_') + GaussianModel(prefix='g2_')

# 正确：使用带前缀的参数名
model.set_param_hint('g2_center', expr='g1_center + delta')
model.set_param_hint('g2_sigma', expr='1.5 * g1_sigma')
model.set_param_hint('g2_amplitude', expr='0.5 * g1_amplitude')

# 错误：使用不带前缀的参数名
# model.set_param_hint('g2_center', expr='center + delta')  # 'center' 不存在！
# model.set_param_hint('g2_sigma', expr='1.5 * sigma')      # 'sigma' 不存在！
```

**对账测试输出**：
```
Parameters 对象中的参数:
   g1_center: value=5.0
   g2_center: value=8.0, expr='g1_center + delta'  ← 使用带前缀的参数名
   g1_sigma: value=1.0
   g2_sigma: value=1.5
   delta: value=3.0
```

### 规则3 总结

**表达式引用规则是统一的**：

| 场景 | 正确的表达式 | 错误的表达式 |
|------|------------|------------|
| 单模型内部引用 | `'2.3548200*g1_sigma'` | `'2.3548200*sigma'` |
| 跨组件引用 | `'g2_center = g1_center + delta'` | `'g2_center = center + delta'` |
| 嵌套引用 | `'g3_center = g2_center + 3'` | `'g3_center = center + 3'` |

**核心原则**：
> 表达式中的参数名必须与 `Parameters` 对象中的键完全一致，
> 而 `Parameters` 对象的键始终是**带前缀的完整参数名**。

---

## 关键对比与澄清

### 最容易混淆的点

| 概念 | 单模型 (GaussianModel) | CompositeModel |
|------|------------------------|-----------------|
| **param_hints 键名** | `'sigma'`, `'fwhm'`, `'height'` | `'g1_sigma'`, `'g1_fwhm'`, `'g1_height'` |
| **expr 中的参数名** | `'g1_sigma'`, `'g1_amplitude'` | `'g1_sigma'`, `'g1_amplitude'` |

**重要发现**：
- **提示键名存储规则**在单模型和 CompositeModel 中**行为不同**
- **表达式引用规则**是**统一的**：始终使用带前缀的完整参数名

### 为什么会有这种差异？

**设计原因**：

1. **单模型的 param_hints 键名不带前缀**
   - 单模型内部，所有参数共享同一个前缀
   - 不带前缀的键名更简洁
   - `set_param_hint` 可以智能处理带/不带前缀的输入

2. **CompositeModel 的 param_hints 键名带前缀**
   - CompositeModel 包含多个子模型，可能有多个同名参数
   - 必须通过前缀区分
   - 传播时自动添加子模型前缀

3. **表达式引用规则统一**
   - 表达式最终在 `Parameters` 对象上下文中执行
   - `Parameters` 对象的键始终是带前缀的完整参数名
   - 所以表达式必须使用带前缀的参数名

### 直观图示

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 单模型 (GaussianModel, prefix='g1_')                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  param_hints (存储时不带前缀)                                             │
│  ┌─────────────┬─────────────────────────────────────────────────────┐ │
│  │ 键名         │ 值                                                    │ │
│  ├─────────────┼─────────────────────────────────────────────────────┤ │
│  │ 'sigma'     │ {'min': 0}                                          │ │
│  │ 'fwhm'      │ {'expr': '2.3548200*g1_sigma'}  ← 表达式带前缀！   │ │
│  │ 'height'    │ {'expr': '0.3989423*g1_amplitude/...'}            │ │
│  └─────────────┴─────────────────────────────────────────────────────┘ │
│                                                                         │
│  注意：键名不带前缀，但表达式中的参数名带前缀！                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ CompositeModel (g1 + g2)                                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  param_hints (存储时带前缀)                                               │
│  ┌─────────────┬─────────────────────────────────────────────────────┐ │
│  │ 键名         │ 值                                                    │ │
│  ├─────────────┼─────────────────────────────────────────────────────┤ │
│  │ 'g1_sigma'  │ {'min': 0}                                          │ │
│  │ 'g1_fwhm'   │ {'expr': '2.3548200*g1_sigma'}  ← 表达式不变！     │ │
│  │ 'g2_sigma'  │ {'min': 0}                                          │ │
│  │ 'g2_fwhm'   │ {'expr': '2.3548200*g2_sigma'}                     │ │
│  └─────────────┴─────────────────────────────────────────────────────┘ │
│                                                                         │
│  注意：键名带前缀，表达式也带前缀（保持不变）                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 对账证据汇总

### 证据1：单模型 param_hints 键名

**测试代码**：
```python
g1 = GaussianModel(prefix='g1_')
print(g1.param_hints)
```

**实际输出**：
```python
{
    'sigma': {'min': 0}, 
    'fwhm': {'expr': '2.3548200*g1_sigma'}, 
    'height': {'expr': '0.3989423*g1_amplitude/max(1e-15, g1_sigma)'}
}
```

**结论**：
- 键名：`'sigma'`, `'fwhm'`, `'height'`（**不带前缀**）
- 表达式：`'g1_sigma'`, `'g1_amplitude'`（**带前缀**）

### 证据2：CompositeModel param_hints 键名

**测试代码**：
```python
model = GaussianModel(prefix='g1_') + GaussianModel(prefix='g2_')
print(model.param_hints)
```

**实际输出**：
```python
{
    'g1_sigma': {'min': 0}, 
    'g1_fwhm': {'expr': '2.3548200*g1_sigma'}, 
    'g1_height': {'expr': '...'},
    'g2_sigma': {'min': 0}, 
    'g2_fwhm': {'expr': '2.3548200*g2_sigma'},
    'g2_height': {'expr': '...'}
}
```

**结论**：
- 键名：`'g1_sigma'`, `'g1_fwhm'`（**带前缀**）
- 表达式：`'g1_sigma'`, `'g2_sigma'`（**保持不变**）

### 证据3：make_params 后的表达式

**测试代码**：
```python
g1 = GaussianModel(prefix='g1_')
params = g1.make_params()
print(f"g1_fwhm.expr = {params['g1_fwhm'].expr}")
print(f"g1_height.expr = {params['g1_height'].expr}")
```

**实际输出**：
```
g1_fwhm.expr = '2.3548200*g1_sigma'
g1_height.expr = '0.3989423*g1_amplitude/max(1e-15, g1_sigma)'
```

**结论**：表达式中的参数名**带前缀**。

### 证据4：嵌套复合模型的一致性

**测试代码**：
```python
g1 = GaussianModel(prefix='g1_')
g2 = GaussianModel(prefix='g2_')
g3 = GaussianModel(prefix='g3_')

# 多层嵌套
peaks12 = g1 + g2
all_peaks = peaks12 + g3

print("第1层 (g1+g2) param_hints:")
for key in ['g1_fwhm', 'g2_fwhm']:
    print(f"  '{key}': {peaks12.param_hints[key]}")

print("\n第2层 ((g1+g2)+g3) param_hints:")
for key in ['g1_fwhm', 'g2_fwhm', 'g3_fwhm']:
    print(f"  '{key}': {all_peaks.param_hints[key]}")
```

**实际输出**：
```
第1层 (g1+g2) param_hints:
  'g1_fwhm': {'expr': '2.3548200*g1_sigma'}
  'g2_fwhm': {'expr': '2.3548200*g2_sigma'}

第2层 ((g1+g2)+g3) param_hints:
  'g1_fwhm': {'expr': '2.3548200*g1_sigma'}  ← 不变！
  'g2_fwhm': {'expr': '2.3548200*g2_sigma'}  ← 不变！
  'g3_fwhm': {'expr': '2.3548200*g3_sigma'}
```

**结论**：
- 键名：`'g1_fwhm'`, `'g2_fwhm'`（始终带前缀）
- 表达式：`'2.3548200*g1_sigma'`（**始终不变**）
- 嵌套层数不影响结果

### 证据5：传入带前缀的 name

**测试代码**：
```python
g1 = GaussianModel(prefix='g1_')
print("初始:")
print(f"  g1.param_hints['sigma'] = {g1.param_hints.get('sigma')}")

# 传入带前缀的 name
g1.set_param_hint('g1_sigma', max=10)

print("\n调用 set_param_hint('g1_sigma', max=10) 后:")
print(f"  g1.param_hints['sigma'] = {g1.param_hints.get('sigma')}")
print(f"  g1.param_hints['g1_sigma'] = {g1.param_hints.get('g1_sigma', '不存在')}")
```

**实际输出**：
```
初始:
  g1.param_hints['sigma'] = {'min': 0}

调用 set_param_hint('g1_sigma', max=10) 后:
  g1.param_hints['sigma'] = {'min': 0, 'max': 10}  ← 已合并！
  g1.param_hints['g1_sigma'] = 不存在  ← 没有这个键！
```

**结论**：
- 输入 `'g1_sigma'` 被剥离前缀为 `'sigma'`
- 存储在键 `'sigma'` 下，而不是 `'g1_sigma'`
- 与已有的 hint 合并：`{'min': 0, 'max': 10}`

---

## 结论与最佳实践

### 最终准确结论

#### 结论1：提示键名存储规则

**单模型（带前缀）**：
- `param_hints` 的键名是**不带前缀的原始参数名**
- 例如：`'sigma'`, `'fwhm'`, `'height'`
- 如果传入带前缀的名称，会被自动剥离前缀

**CompositeModel**：
- `param_hints` 的键名是**带前缀的完整参数名**
- 例如：`'g1_sigma'`, `'g1_fwhm'`, `'g2_sigma'`
- 传播时自动添加子模型的前缀

**代码证据**：
```python
# Model.set_param_hint()
if npref > 0 and name.startswith(self._prefix):
    name = name[npref:]  # 剥离前缀

# CompositeModel.__init__() 传播时
self.param_hints[f"{prefix}{basename}"] = hint  # 添加前缀
```

#### 结论2：表达式引用规则

**统一规则**：表达式中的参数名**必须使用带前缀的完整参数名**。

**原因**：
- 表达式在 `Parameters` 对象的上下文中执行
- `Parameters` 对象的键始终是带前缀的完整参数名
- 例如：`'g1_amplitude'`, `'g1_center'`, `'g1_sigma'`

**代码证据**：
```python
# fwhm_expr() 构建表达式时就包含前缀
def fwhm_expr(model):
    fmt = "{factor:.7f}*{prefix:s}sigma"
    return fmt.format(factor=model.fwhm_factor, prefix=model.prefix)
    # 例如：'2.3548200*g1_sigma'
```

#### 结论3：传播过程中的不变性

**表达式在传播过程中保持不变**：
- 从单模型传播到 CompositeModel 时，`expr` 的值不被修改
- 多层嵌套传播时，表达式始终保持不变
- 这确保了表达式的一致性

**代码证据**：
```python
# CompositeModel 传播时只修改键名，不修改值
for basename, hint in side.param_hints.items():
    self.param_hints[f"{prefix}{basename}"] = hint  # hint 不变！
```

### 最佳实践

#### 实践1：理解单模型的 set_param_hint

```python
g1 = GaussianModel(prefix='g1_')

# 以下两种方式等价（单模型内部）
g1.set_param_hint('sigma', min=0)        # 不带前缀
g1.set_param_hint('g1_sigma', min=0)     # 带前缀（会被剥离）

# 但表达式必须使用带前缀的参数名
g1.set_param_hint('fwhm', expr='2.3548200*g1_sigma')  # 正确
# g1.set_param_hint('fwhm', expr='2.3548200*sigma')   # 错误！
```

#### 实践2：CompositeModel 的 set_param_hint

```python
model = GaussianModel(prefix='g1_') + GaussianModel(prefix='g2_')

# CompositeModel 中，键名和表达式都带前缀
model.set_param_hint('g1_sigma', min=0)              # 键名带前缀
model.set_param_hint('g2_sigma', min=0)              # 键名带前缀

# 表达式也必须带前缀
model.set_param_hint('g2_center', expr='g1_center + 5')  # 正确
# model.set_param_hint('g2_center', expr='center + 5')   # 错误！
```

#### 实践3：跨组件约束

```python
g1 = GaussianModel(prefix='g1_')
g2 = GaussianModel(prefix='g2_')
g3 = GaussianModel(prefix='g3_')
model = g1 + g2 + g3

# 设置辅助参数
model.set_param_hint('spacing', value=3.0, vary=False)

# 跨组件约束（使用完整参数名）
model.set_param_hint('g2_center', expr='g1_center + spacing')
model.set_param_hint('g3_center', expr='g2_center + spacing')  # 嵌套引用

# 验证
params = model.make_params(
    g1_amplitude=100, g1_center=5, g1_sigma=1,
    g2_amplitude=50, g2_sigma=1,
    g3_amplitude=50, g3_sigma=1
)

print(params['g2_center'].expr)  # 'g1_center + spacing'
print(params['g3_center'].expr)  # 'g2_center + spacing'
```

#### 实践4：访问 param_hints 的正确方式

```python
# 单模型
g1 = GaussianModel(prefix='g1_')
# 使用不带前缀的键名访问
hint = g1.param_hints.get('sigma')     # 正确
# hint = g1.param_hints.get('g1_sigma')  # 错误！（单模型中不存在这个键）

# CompositeModel
model = g1 + GaussianModel(prefix='g2_')
# 使用带前缀的键名访问
hint = model.param_hints.get('g1_sigma')  # 正确
hint = model.param_hints.get('g2_sigma')  # 正确
# hint = model.param_hints.get('sigma')    # 错误！
```

### 常见问题解答

#### Q1：为什么单模型的 param_hints 键名不带前缀？

**A**：这是设计选择。单模型内部所有参数共享同一个前缀，不带前缀的键名更简洁。`set_param_hint` 方法会智能处理带/不带前缀的输入，自动剥离前缀后存储。

#### Q2：为什么 CompositeModel 的 param_hints 键名带前缀？

**A**：CompositeModel 可能包含多个子模型，这些子模型可能有同名参数（如多个 GaussianModel 都有 `sigma`）。必须通过前缀来区分来自不同子模型的参数提示。

#### Q3：为什么表达式中的参数名必须带前缀？

**A**：表达式最终在 `Parameters` 对象的上下文中执行。`Parameters` 对象的键始终是带前缀的完整参数名（如 `'g1_sigma'`）。如果表达式使用不带前缀的参数名（如 `'sigma'`），在符号表中找不到，会导致 `NameError`。

#### Q4：如何知道应该使用带前缀还是不带前缀的键名？

**A**：
- **访问 `param_hints`**：检查模型类型
  - 单模型（GaussianModel 等）：使用不带前缀的键名
  - CompositeModel：使用带前缀的键名
- **编写 `expr` 表达式**：**始终使用带前缀的完整参数名**（这是统一规则）

#### Q5：嵌套复合模型的行为是否一致？

**A**：是的，完全一致。无论嵌套多少层：
- `param_hints` 的键名始终带前缀（从第一层 CompositeModel 开始）
- 表达式始终保持不变，始终使用带前缀的参数名
- 嵌套层数不影响最终结果

### 决策流程图

```
用户操作
    │
    ├──► 设置 param_hint
    │         │
    │         ├──► 单模型内部
    │         │         │
    │         │         ├──► 键名：不带前缀（'sigma'）
    │         │         └──► 表达式：带前缀（'g1_sigma'）
    │         │
    │         └──► CompositeModel
    │                   │
    │                   ├──► 键名：带前缀（'g1_sigma'）
    │                   └──► 表达式：带前缀（'g1_sigma'）
    │
    └──► 访问 param_hints
              │
              ├──► 单模型内部
              │         └──► 使用不带前缀的键名：g1.param_hints['sigma']
              │
              └──► CompositeModel
                        └──► 使用带前缀的键名：model.param_hints['g1_sigma']
```

---

## 参考资料

### 源代码位置

| 功能 | 文件位置 | 关键代码 |
|------|---------|---------|
| `set_param_hint` | `model.py:620-676` | 前缀剥离逻辑 |
| `param_hints` 传播 | `model.py:1273-1276` | CompositeModel 初始化 |
| `fwhm_expr` | `models.py:31-34` | 表达式构建（含前缀） |
| `height_expr` | `models.py:37-40` | 表达式构建（含前缀） |
| `make_params` | `model.py:699-816` | 参数创建 |

### 测试文件

- `param_hints_verification.py`：本文档使用的对账测试脚本
- `tests/test_model.py`：官方测试文件

### 相关文档

- `model_prefix_analysis.md`：基础前缀机制分析
- `nested_composite_model_analysis.md`：嵌套复合模型分析
