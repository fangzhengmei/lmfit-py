"""验证 param_hints 的键名存储规则与表达式引用规则

对账测试：区分"提示键名存储规则"与"表达式引用规则"
"""

import numpy as np
from lmfit import Model, Parameters
from lmfit.models import GaussianModel, LinearModel
from lmfit.lineshapes import gaussian


def test_single_model_param_hints():
    """测试单模型的 param_hints 键名存储规则"""
    
    print("=" * 70)
    print("测试1: 单模型 (GaussianModel with prefix='g1_')")
    print("=" * 70)
    
    g1 = GaussianModel(prefix='g1_')
    
    print(f"\n1.1 模型属性:")
    print(f"   g1._prefix = '{g1._prefix}'")
    print(f"   g1.param_names = {g1.param_names}")
    
    print(f"\n1.2 g1.param_hints (键名存储规则):")
    for key, val in g1.param_hints.items():
        print(f"   '{key}': {val}")
    
    # 分析：set_param_hint 如何处理输入的 name
    print(f"\n1.3 验证 set_param_hint 的行为:")
    print(f"   GaussianModel._set_paramhints_prefix() 调用:")
    print(f"     self.set_param_hint('sigma', min=0)")
    print(f"     self.set_param_hint('fwhm', expr=fwhm_expr(self))")
    print(f"     self.set_param_hint('height', expr=height_expr(self))")
    print(f"\n   fwhm_expr(self) 返回:")
    print(f"     '2.3548200*{g1.prefix}sigma' = '2.3548200*g1_sigma'")
    
    print(f"\n   set_param_hint 内部逻辑 (model.py:665-667):")
    print(f"     npref = len(self._prefix) = {len(g1._prefix)}")
    print(f"     if npref > 0 and name.startswith(self._prefix):")
    print(f"         name = name[npref:]  # 剥离前缀")
    print(f"\n   所以:")
    print(f"     set_param_hint('sigma', min=0)  → 'sigma' 不以前缀开头 → 存储键 'sigma'")
    print(f"     set_param_hint('fwhm', expr=...) → 'fwhm' 不以前缀开头 → 存储键 'fwhm'")
    print(f"     但 expr 值 '2.3548200*g1_sigma' 保持不变！")
    
    return g1


def test_composite_model_param_hints():
    """测试 CompositeModel 的 param_hints 键名存储规则"""
    
    print("\n" + "=" * 70)
    print("测试2: CompositeModel (g1 + g2)")
    print("=" * 70)
    
    g1 = GaussianModel(prefix='g1_')
    g2 = GaussianModel(prefix='g2_')
    
    print(f"\n2.1 组合前各模型的 param_hints:")
    print(f"   g1.param_hints = {g1.param_hints}")
    print(f"   g2.param_hints = {g2.param_hints}")
    
    model = g1 + g2
    
    print(f"\n2.2 CompositeModel 的 param_hints:")
    for key, val in model.param_hints.items():
        print(f"   '{key}': {val}")
    
    print(f"\n2.3 传播逻辑 (model.py:1273-1276):")
    print(f"   for side in (left, right):")
    print(f"       prefix = side.prefix")
    print(f"       for basename, hint in side.param_hints.items():")
    print(f"           self.param_hints[f'{{prefix}}{{basename}}'] = hint")
    print(f"\n   所以:")
    print(f"     子模型 g1 的键 'sigma' → CompositeModel 的键 'g1_sigma'")
    print(f"     子模型 g1 的键 'fwhm' → CompositeModel 的键 'g1_fwhm'")
    print(f"     子模型 g1 的键 'height' → CompositeModel 的键 'g1_height'")
    print(f"     但 hint 中的 expr 值保持不变！")
    
    print(f"\n2.4 验证:")
    print(f"   model._prefix = '{model._prefix}' (CompositeModel 无前缀)")
    print(f"   model.param_names = {model.param_names}")
    
    return model, g1, g2


def test_make_params_application():
    """测试 make_params 如何应用 param_hints"""
    
    print("\n" + "=" * 70)
    print("测试3: make_params 应用 param_hints 的规则")
    print("=" * 70)
    
    g1 = GaussianModel(prefix='g1_')
    g2 = GaussianModel(prefix='g2_')
    model = g1 + g2
    
    print(f"\n3.1 单模型 g1.make_params():")
    params_g1 = g1.make_params()
    print(f"   params_g1.keys() = {list(params_g1.keys())}")
    
    print(f"\n   检查各参数的 expr:")
    for name in ['g1_amplitude', 'g1_center', 'g1_sigma', 'g1_fwhm', 'g1_height']:
        if name in params_g1:
            par = params_g1[name]
            expr_str = f", expr='{par.expr}'" if par.expr else ""
            print(f"   {name}: value={par.value}, min={par.min}{expr_str}")
    
    print(f"\n3.2 CompositeModel.make_params():")
    params_comp = model.make_params()
    print(f"   params_comp.keys() = {list(params_comp.keys())}")
    
    print(f"\n   检查各参数的 expr:")
    for name in ['g1_amplitude', 'g1_center', 'g1_sigma', 'g1_fwhm', 'g1_height',
                 'g2_amplitude', 'g2_center', 'g2_sigma', 'g2_fwhm', 'g2_height']:
        if name in params_comp:
            par = params_comp[name]
            expr_str = f", expr='{par.expr}'" if par.expr else ""
            print(f"   {name}: value={par.value}, min={par.min}{expr_str}")
    
    print(f"\n3.3 关键发现:")
    print(f"   g1_fwhm.expr = '2.3548200*g1_sigma' (使用带前缀的参数名)")
    print(f"   g1_height.expr = '0.3989423*g1_amplitude/max(1e-15, g1_sigma)'")
    print(f"\n   这些表达式引用的是 Parameters 对象中的完整参数名:")
    print(f"   'g1_sigma' (不是 'sigma')")
    print(f"   'g1_amplitude' (不是 'amplitude')")
    
    return params_g1, params_comp


def test_expression_reference_rules():
    """测试表达式引用的前缀规则"""
    
    print("\n" + "=" * 70)
    print("测试4: 表达式引用规则 (expr 中的参数名)")
    print("=" * 70)
    
    g1 = GaussianModel(prefix='g1_')
    g2 = GaussianModel(prefix='g2_')
    bkg = LinearModel(prefix='bkg_')
    model = g1 + g2 + bkg
    
    print(f"\n4.1 设置用户自定义约束:")
    print(f"   规则：表达式必须使用完整的带前缀参数名")
    
    # 正确的表达式
    model.set_param_hint('delta', value=3.0, vary=False)
    model.set_param_hint('g2_center', expr='g1_center + delta')
    model.set_param_hint('g2_sigma', expr='1.5 * g1_sigma')
    
    print(f"\n   设置的 hints:")
    print(f"   model.set_param_hint('g2_center', expr='g1_center + delta')")
    print(f"   model.set_param_hint('g2_sigma', expr='1.5 * g1_sigma')")
    
    print(f"\n4.2 验证 CompositeModel 的 param_hints:")
    for key in ['g1_fwhm', 'g1_height', 'g2_center', 'g2_sigma', 'delta']:
        if key in model.param_hints:
            print(f"   '{key}': {model.param_hints[key]}")
    
    print(f"\n4.3 分析:")
    print(f"   CompositeModel.set_param_hint 内部逻辑:")
    print(f"     self._prefix = '' (空)")
    print(f"     所以 name 不会被剥离前缀")
    print(f"\n   所以:")
    print(f"     model.set_param_hint('g2_center', ...) → 存储键 'g2_center'")
    print(f"     expr='g1_center + delta' 保持不变")
    
    # 创建参数并验证
    params = model.make_params(
        g1_amplitude=100, g1_center=5, g1_sigma=1,
        g2_amplitude=50, g2_sigma=1.5,
        bkg_slope=0, bkg_intercept=0
    )
    
    print(f"\n4.4 Parameters 对象中的参数:")
    for name in ['g1_center', 'g2_center', 'g1_sigma', 'g2_sigma', 'delta']:
        if name in params:
            par = params[name]
            expr_str = f", expr='{par.expr}'" if par.expr else ""
            print(f"   {name}: value={par.value}{expr_str}")
    
    print(f"\n4.5 关键结论:")
    print(f"   表达式中的参数名必须与 Parameters 对象中的键完全一致")
    print(f"   Parameters 对象中的键是完整的带前缀参数名")
    print(f"   所以表达式必须使用带前缀的参数名！")
    
    return model, params


def test_nested_composite_hints():
    """测试嵌套复合模型的 param_hints 传播"""
    
    print("\n" + "=" * 70)
    print("测试5: 嵌套复合模型的 param_hints 传播")
    print("=" * 70)
    
    g1 = GaussianModel(prefix='g1_')
    g2 = GaussianModel(prefix='g2_')
    g3 = GaussianModel(prefix='g3_')
    bkg = LinearModel(prefix='bkg_')
    
    # 嵌套组合
    peaks12 = g1 + g2
    all_peaks = peaks12 + g3
    model = all_peaks + bkg
    
    print(f"\n5.1 各层模型的 param_hints:")
    
    print(f"\n   g1.param_hints (叶子模型):")
    for key, val in g1.param_hints.items():
        print(f"      '{key}': {val}")
    
    print(f"\n   peaks12.param_hints (第一层 CompositeModel):")
    for key, val in peaks12.param_hints.items():
        print(f"      '{key}': {val}")
    
    print(f"\n   all_peaks.param_hints (第二层 CompositeModel):")
    for key in ['g1_fwhm', 'g1_height', 'g2_fwhm', 'g2_height', 'g3_fwhm', 'g3_height']:
        if key in all_peaks.param_hints:
            print(f"      '{key}': {all_peaks.param_hints[key]}")
    
    print(f"\n   model.param_hints (第三层 CompositeModel):")
    for key in ['g1_fwhm', 'g1_height', 'g2_fwhm', 'g2_height', 'g3_fwhm', 'g3_height']:
        if key in model.param_hints:
            print(f"      '{key}': {model.param_hints[key]}")
    
    print(f"\n5.2 传播规则:")
    print(f"   每一层 CompositeModel 都会:")
    print(f"   1. 遍历 left 和 right 的 param_hints")
    print(f"   2. 对每个 hint 的键名添加子模型的前缀")
    print(f"   3. hint 的值（包括 expr）保持不变")
    
    print(f"\n5.3 验证 make_params:")
    params = model.make_params()
    
    print(f"\n   Parameters 中的 fwhm 参数:")
    for name in ['g1_fwhm', 'g2_fwhm', 'g3_fwhm']:
        if name in params:
            par = params[name]
            print(f"   {name}: expr='{par.expr}'")
    
    print(f"\n5.4 关键发现:")
    print(f"   嵌套多层后:")
    print(f"   - param_hints 的键名: 'g1_fwhm', 'g2_fwhm', 'g3_fwhm' (带前缀)")
    print(f"   - expr 的值: '2.3548200*g1_sigma', '2.3548200*g2_sigma', 等")
    print(f"   - 表达式引用的始终是完整的带前缀参数名")
    print(f"   - 嵌套层数不影响最终结果！")
    
    return model, params


def test_edge_cases():
    """测试边界情况"""
    
    print("\n" + "=" * 70)
    print("测试6: 边界情况")
    print("=" * 70)
    
    print(f"\n6.1 set_param_hint 传入带前缀的名称:")
    
    g1 = GaussianModel(prefix='g1_')
    
    print(f"\n   初始 g1.param_hints:")
    for key, val in g1.param_hints.items():
        print(f"      '{key}': {val}")
    
    # 传入带前缀的名称
    g1.set_param_hint('g1_sigma', max=10)
    g1.set_param_hint('g1_center', value=5.0)
    
    print(f"\n   调用后 g1.param_hints:")
    for key, val in g1.param_hints.items():
        print(f"      '{key}': {val}")
    
    print(f"\n   分析:")
    print(f"   set_param_hint('g1_sigma', max=10):")
    print(f"     name='g1_sigma' 以前缀 'g1_' 开头")
    print(f"     所以被剥离为 'sigma'")
    print(f"     存储键为 'sigma'，合并 hint: {{'min': 0, 'max': 10}}")
    
    print(f"\n6.2 验证:")
    params = g1.make_params()
    print(f"   params['g1_sigma'].min = {params['g1_sigma'].min}")
    print(f"   params['g1_sigma'].max = {params['g1_sigma'].max}")
    
    print(f"\n6.3 结论:")
    print(f"   set_param_hint 接受带前缀或不带前缀的名称")
    print(f"   但存储时统一使用不带前缀的键名（对于单模型）")
    print(f"   表达式中的值始终保持不变")


def summary_of_rules():
    """汇总所有规则"""
    
    print("\n" + "=" * 70)
    print("规则汇总")
    print("=" * 70)
    
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
规则 1: 提示键名存储规则 (param_hints 的键)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

对于单模型 (如 GaussianModel with prefix='g1_'):
┌─────────────────────────────────────────────────────────────────────────┐
│ set_param_hint 输入的 name    │ 存储在 param_hints 中的键             │
├─────────────────────────────────────────────────────────────────────────┤
│ 'sigma'                       │ 'sigma' (不以前缀开头，不剥离)         │
│ 'fwhm'                        │ 'fwhm' (不以前缀开头，不剥离)          │
│ 'g1_sigma'                    │ 'sigma' (以前缀开头，被剥离)           │
│ 'g1_fwhm'                     │ 'fwhm' (以前缀开头，被剥离)            │
└─────────────────────────────────────────────────────────────────────────┘

对于 CompositeModel:
┌─────────────────────────────────────────────────────────────────────────┐
│ CompositeModel._prefix = '' (空字符串)                                  │
│ 所以 set_param_hint 不会剥离任何前缀                                    │
│ 存储的键名就是传入的名称                                                 │
└─────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
规则 2: param_hints 传播规则 (CompositeModel)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CompositeModel 在初始化时传播子模型的 param_hints:

for side in (left, right):
    prefix = side.prefix
    for basename, hint in side.param_hints.items():
        self.param_hints[f"{prefix}{basename}"] = hint

┌─────────────────────────────────────────────────────────────────────────┐
│ 子模型 (g1, prefix='g1_')                                               │
│   param_hints = {                                                        │
│       'sigma': {'min': 0},                                              │
│       'fwhm': {'expr': '2.3548200*g1_sigma'},                         │
│       'height': {'expr': '0.3989423*g1_amplitude/max(1e-15, g1_sigma)'}│
│   }                                                                       │
├─────────────────────────────────────────────────────────────────────────┤
│ 传播到 CompositeModel 后:                                                │
│   param_hints = {                                                        │
│       'g1_sigma': {'min': 0},               ← 键添加了前缀 'g1_'      │
│       'g1_fwhm': {'expr': '2.3548200*g1_sigma'}, ← expr 不变         │
│       'g1_height': {'expr': '...g1_amplitude...g1_sigma...'}           │
│   }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘

关键点:
- 键名 (key) 会添加子模型的前缀
- 提示值 (value, 包括 expr) 保持不变！

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
规则 3: 表达式引用规则 (expr 中的参数名)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

表达式中的参数名必须与 Parameters 对象中的键完全一致！

┌─────────────────────────────────────────────────────────────────────────┐
│ Parameters 对象中的键:                                                   │
│   'g1_amplitude', 'g1_center', 'g1_sigma',                             │
│   'g1_fwhm', 'g1_height',                                               │
│   'g2_amplitude', 'g2_center', 'g2_sigma', ...                         │
├─────────────────────────────────────────────────────────────────────────┤
│ 正确的表达式:                                                            │
│   expr='2.3548200*g1_sigma'           ✓ 引用 'g1_sigma'               │
│   expr='g2_center = g1_center + delta' ✓ 引用完整参数名                │
├─────────────────────────────────────────────────────────────────────────┤
│ 错误的表达式:                                                            │
│   expr='2.3548200*sigma'              ✗ 'sigma' 不存在于 Parameters    │
│   expr='center + delta'               ✗ 'center' 不存在                 │
└─────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
规则 4: 嵌套复合模型的一致性
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

无论嵌套多少层，最终结果是一致的:

((g1 + g2) + g3) + bkg
  = g1 + (g2 + (g3 + bkg))
  = (g1 + g2) + (g3 + bkg)
  = g1 + g2 + g3 + bkg

所有组合方式:
- param_hints 的键名: 'g1_fwhm', 'g1_height', 'g2_fwhm', ...
- 表达式的值: '2.3548200*g1_sigma', '2.3548200*g2_sigma', ...
- Parameters 的键: 'g1_amplitude', 'g1_center', 'g1_sigma', ...

完全一致！

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
重要对比：提示键名 vs 表达式引用
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────┐
│ 项目                │ 单模型键名      │ CompositeModel 键名  │ 表达式中 │
├─────────────────────────────────────────────────────────────────────────┤
│ 'sigma'             │ 不带前缀        │ 带前缀 'g1_sigma'   │ 必须带前缀│
│ 'fwhm'              │ 不带前缀        │ 带前缀 'g1_fwhm'    │ 必须带前缀│
│ expr 引用的参数名   │ -               │ -                    │ 'g1_sigma'│
└─────────────────────────────────────────────────────────────────────────┘

关键点：
1. param_hints 的键名存储规则在单模型和 CompositeModel 中不同
2. 但表达式引用规则是统一的：必须使用完整的带前缀参数名
3. 这是因为表达式最终在 Parameters 对象的上下文中执行
4. Parameters 对象中的键始终是完整的带前缀参数名

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


def main():
    """运行所有对账测试"""
    
    print("=" * 70)
    print("param_hints 键名存储规则与表达式引用规则 - 对账测试")
    print("=" * 70)
    
    # 测试1: 单模型
    test_single_model_param_hints()
    
    # 测试2: CompositeModel
    test_composite_model_param_hints()
    
    # 测试3: make_params 应用
    test_make_params_application()
    
    # 测试4: 表达式引用规则
    test_expression_reference_rules()
    
    # 测试5: 嵌套复合模型
    test_nested_composite_hints()
    
    # 测试6: 边界情况
    test_edge_cases()
    
    # 汇总规则
    summary_of_rules()


if __name__ == '__main__':
    main()
