# lmfit 方法名路由匹配规则精确勘误

> 本文档精确澄清 `minimize()` 方法的算法名匹配逻辑，修正之前报告中的错误示例。

---

## 核心匹配逻辑

### 源码定义

**`SCALAR_METHODS` 字典**（`minimizer.py:84-100`）：

```python
SCALAR_METHODS = {
    'nelder': 'Nelder-Mead',           # 位置 0
    'powell': 'Powell',                 # 位置 1
    'cg': 'CG',                         # 位置 2
    'bfgs': 'BFGS',                     # 位置 3
    'newton': 'Newton-CG',              # 位置 4
    'lbfgsb': 'L-BFGS-B',               # 位置 5
    'l-bfgsb': 'L-BFGS-B',              # 位置 6
    'tnc': 'TNC',                        # 位置 7
    'cobyla': 'COBYLA',                  # 位置 8
    'cobyqa': 'COBYQA',                  # 位置 9
    'slsqp': 'SLSQP',                    # 位置 10
    'dogleg': 'dogleg',                  # 位置 11
    'trust-ncg': 'trust-ncg',            # 位置 12
    'differential_evolution': 'differential_evolution',  # 位置 13
    'trust-constr': 'trust-constr',      # 位置 14
    'trust-exact': 'trust-exact',        # 位置 15
    'trust-krylov': 'trust-krylov'       # 位置 16 (最后)
}
```

**匹配逻辑**（`minimizer.py:2451-2454`）：

```python
user_method = method.lower()  # 用户输入转小写
# ...
for key, val in SCALAR_METHODS.items():
    if (key.lower().startswith(user_method) or
            val.lower().startswith(user_method)):
        kwargs['method'] = val
```

### 关键规则

| 规则 | 说明 |
|-----|------|
| **方向** | 字典的 `key` 或 `val` **以用户输入为前缀**（不是反过来） |
| **大小写** | 用户输入先转 `lower()`，然后比较也是用 `key.lower()` / `val.lower()` |
| **歧义处理** | 多个匹配时，按字典插入顺序，**最后一个匹配覆盖前面的** |
| **无匹配** | `kwargs['method']` 从未被设置，`scalar_minimize()` 使用默认值 `'Nelder-Mead'` |

---

## 精确匹配示例对照表

### 命中案例

| 用户输入 | 匹配过程 | 命中方法 | 说明 |
|---------|---------|---------|------|
| `'lbfgs'` | `'lbfgsb'.startswith('lbfgs')` → True | `'L-BFGS-B'` | `'lbfgsb'` 前5字符 = `'lbfgs'` |
| `'l'` | `'l-bfgsb'.startswith('l')` → True (最后匹配) | `'L-BFGS-B'` | 所有以 `'l'` 开头的键，最后一个是位置6 |
| `'trust'` | `'trust-krylov'.startswith('trust')` → True (最后匹配) | `'trust-krylov'` | 4个以 `'trust'` 开头的方法，最后一个 |
| `'bfg'` | `'bfgs'.startswith('bfg')` → True | `'BFGS'` | `'bfgs'` 前3字符 = `'bfg'` |
| `'c'` | `'cobyqa'.startswith('c')` → True (最后匹配) | `'COBYQA'` | 3个以 `'c'` 开头：`cg`, `cobyla`, `cobyqa` |
| `'t'` | `'trust-krylov'.startswith('t')` → True (最后匹配) | `'trust-krylov'` | 所有以 `'t'` 开头的方法 |
| `'diff'` | `'differential_evolution'.startswith('diff')` → True | `'differential_evolution'` | `'differential_evolution'` 前4字符 = `'diff'` |
| `'LBFGSB'` | `user_method='lbfgsb'`, `'lbfgsb'.startswith('lbfgsb')` → True | `'L-BFGS-B'` | 大小写不敏感 |
| `'Nelder-Mead'` | `'nelder'.startswith('nelder-mead')` 检查，实际匹配 `'Nelder-Mead'.lower().startswith('nelder-mead')` | `'Nelder-Mead'` | val 匹配 |

### 退化案例（退化为默认值 `'Nelder-Mead'`）

| 用户输入 | 为什么不匹配 | 结果 |
|---------|-------------|------|
| `'powells'` | `'powell'.startswith('powells')` → False<br>(`'powell'` 只有6个字符，无法以7字符的 `'powells'` 为前缀) | ❌ 退化为 `'Nelder-Mead'` |
| `'bfgss'` | `'bfgs'.startswith('bfgss')` → False<br>(`'bfgs'` 只有4个字符) | ❌ 退化为 `'Nelder-Mead'` |
| `'de'` | `'differential_evolution'.startswith('de')` → False<br>(`'differential'` 以 `'di'` 开头，不是 `'de'`) | ❌ 退化为 `'Nelder-Mead'` |
| `'lbfgs-'` | `'lbfgsb'.startswith('lbfgs-')` → False | ❌ 退化为 `'Nelder-Mead'` |
| `'unknown'` | 无任何 key/val 以 `'unknown'` 开头 | ❌ 退化为 `'Nelder-Mead'` |
| `'trust_ncg'` | 键是 `'trust-ncg'`（连字符），`'trust-ncg'.startswith('trust_ncg')` → False | ❌ 退化为 `'Nelder-Mead'` |

---

## 关键理解修正

### ❌ 之前报告中的错误示例

| 错误示例 | 错误原因 | 正确行为 |
|---------|---------|---------|
| `method='lbfgs'` → 退化为 `'Nelder-Mead'` | 错误理解 `startswith` 方向 | ✅ 命中 `'L-BFGS-B'` |
| `method='l'` → 退化为 `'Nelder-Mead'` | 错误理解匹配逻辑 | ✅ 命中 `'L-BFGS-B'` |

### ✅ 正确的退化场景

**真正会退化到 `'Nelder-Mead'` 的场景**：

1. **用户输入比字典键/值"更长"**：
   - `'powells'` (7字符) vs `'powell'` (6字符)
   - `startswith` 检查：短字符串 **不可能** 以长字符串为前缀

2. **用户输入的前缀不匹配任何键/值的开头**：
   - `'de'` 想匹配 `'differential_evolution'`
   - 但 `'differential_evolution'` 以 `'di'` 开头，不是 `'de'`

3. **拼写或分隔符不匹配**：
   - `'trust_ncg'` (下划线) vs `'trust-ncg'` (连字符)

---

## 完整路由流程图

```
用户输入 method='xxx'
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  第一级：专用方法检查（不会进入 SCALAR_METHODS 匹配）          │
├─────────────────────────────────────────────────────────────┤
│  'leasts'...     ──► leastsq()                               │
│  'least_s'...    ──► least_squares()                         │
│  == 'brute'      ──► brute()                                 │
│  == 'basinhopping' ──► basinhopping()                       │
│  == 'ampgo'      ──► ampgo()                                 │
│  == 'emcee'      ──► emcee()                                 │
│  == 'shgo'       ──► shgo()                                  │
│  == 'dual_annealing' ──► dual_annealing()                   │
│  == 'direct'     ──► direct()                                │
└─────────────────────────────────────────────────────────────┘
        │ 都不匹配
        ▼
┌─────────────────────────────────────────────────────────────┐
│  进入 scalar_minimize() 分支                                  │
│                                                               │
│  遍历 SCALAR_METHODS (按插入顺序):                            │
│    for key, val in SCALAR_METHODS.items():                   │
│        if (key.lower().startswith(user_method) or            │
│            val.lower().startswith(user_method)):              │
│            kwargs['method'] = val  ← 覆盖！                   │
│                                                               │
│  关键：检查 key/val 是否以 user_method 为前缀！               │
└─────────────────────────────────────────────────────────────┘
        │
        ├─────────────────────────────────┬─────────────────────┐
        │ 至少一个匹配                      │ 无任何匹配            │
        ▼                                 ▼                      │
┌───────────────────┐         ┌───────────────────────────────┐│
│ kwargs['method']  │         │ kwargs['method'] 未设置       ││
│ = 最后一个匹配的值 │         │ (保持默认 None)                ││
└───────────────────┘         └───────────────────────────────┘│
        │                                 │                       │
        ▼                                 ▼                       │
┌───────────────────────────────────────────────────────────────┐
│  scalar_minimize(method=kwargs.get('method', 'Nelder-Mead')) │
│                                                                 │
│  如果 method='differential_evolution':                         │
│      └──► 直接调用 scipy.optimize.differential_evolution      │
│  否则:                                                          │
│      └──► 调用 scipy.optimize.minimize(..., method=xxx)       │
└───────────────────────────────────────────────────────────────┘
```

---

## 速查表

### `SCALAR_METHODS` 插入顺序（Python 3.7+）

| 位置 | key | val | 以 `'t'` 开头 | 以 `'trust'` 开头 |
|-----|-----|-----|--------------|------------------|
| 0 | `'nelder'` | `'Nelder-Mead'` | ❌ | ❌ |
| 1 | `'powell'` | `'Powell'` | ❌ | ❌ |
| 2 | `'cg'` | `'CG'` | ❌ | ❌ |
| 3 | `'bfgs'` | `'BFGS'` | ❌ | ❌ |
| 4 | `'newton'` | `'Newton-CG'` | ❌ | ❌ |
| 5 | `'lbfgsb'` | `'L-BFGS-B'` | ❌ | ❌ |
| 6 | `'l-bfgsb'` | `'L-BFGS-B'` | ❌ | ❌ |
| 7 | `'tnc'` | `'TNC'` | ✅ | ❌ |
| 8 | `'cobyla'` | `'COBYLA'` | ❌ | ❌ |
| 9 | `'cobyqa'` | `'COBYQA'` | ❌ | ❌ |
| 10 | `'slsqp'` | `'SLSQP'` | ❌ | ❌ |
| 11 | `'dogleg'` | `'dogleg'` | ❌ | ❌ |
| 12 | `'trust-ncg'` | `'trust-ncg'` | ✅ | ✅ |
| 13 | `'differential_evolution'` | `'differential_evolution'` | ❌ | ❌ |
| 14 | `'trust-constr'` | `'trust-constr'` | ✅ | ✅ |
| 15 | `'trust-exact'` | `'trust-exact'` | ✅ | ✅ |
| 16 | `'trust-krylov'` | `'trust-krylov'` | ✅ | ✅ **(最后)** |

### `startswith` 方向速记

```
正确:  'lbfgsb'.startswith('lbfgs')   → True  (长字符串以短字符串为前缀)
错误:  'lbfgs'.startswith('lbfgsb')   → False (短字符串不可能以长字符串为前缀)

正确:  'trust-krylov'.startswith('trust')  → True
错误:  'trust'.startswith('trust-krylov')  → False
```

---

## 代码位置速查

| 内容 | 文件 | 行号 |
|-----|------|------|
| `SCALAR_METHODS` 字典定义 | `minimizer.py` | 84-100 |
| 分发路由逻辑 | `minimizer.py` | 2430-2455 |
| `scalar_minimize()` 默认参数 | `minimizer.py` | 820 |
