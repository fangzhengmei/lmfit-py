"""测试嵌套复合模型的前缀传播机制"""

import numpy as np
from lmfit import Model, Parameters
from lmfit.models import GaussianModel, LinearModel, ConstantModel
from lmfit.lineshapes import gaussian


def create_nested_composite():
    """创建嵌套复合模型"""
    print("=" * 70)
    print("测试1: 创建嵌套复合模型")
    print("=" * 70)
    
    g1 = GaussianModel(prefix='g1_')
    g2 = GaussianModel(prefix='g2_')
    g3 = GaussianModel(prefix='g3_')
    bkg = LinearModel(prefix='bkg_')
    
    peaks12 = g1 + g2
    print(f"\n第一层组合 (g1 + g2):")
    print(f"  类型: {type(peaks12)}")
    print(f"  param_names: {peaks12.param_names}")
    print(f"  components: {[type(c).__name__ for c in peaks12.components]}")
    print(f"  component prefixes: {[c.prefix for c in peaks12.components]}")
    
    all_peaks = peaks12 + g3
    print(f"\n第二层组合 ((g1 + g2) + g3):")
    print(f"  类型: {type(all_peaks)}")
    print(f"  param_names: {all_peaks.param_names}")
    print(f"  components: {[type(c).__name__ for c in all_peaks.components]}")
    print(f"  component prefixes: {[c.prefix for c in all_peaks.components]}")
    
    model = all_peaks + bkg
    print(f"\n第三层组合 (((g1 + g2) + g3) + bkg):")
    print(f"  类型: {type(model)}")
    print(f"  param_names: {model.param_names}")
    print(f"  组件数量: {len(model.components)}")
    print(f"  component prefixes: {[c.prefix for c in model.components]}")
    
    return model, g1, g2, g3, bkg, peaks12, all_peaks


def test_param_names_consistency():
    """测试参数命名的一致性"""
    print("\n" + "=" * 70)
    print("测试2: 参数命名的一致性")
    print("=" * 70)
    
    g1 = GaussianModel(prefix='g1_')
    g2 = GaussianModel(prefix='g2_')
    g3 = GaussianModel(prefix='g3_')
    bkg = LinearModel(prefix='bkg_')
    
    model1 = (g1 + g2) + (g3 + bkg)
    model2 = g1 + (g2 + (g3 + bkg))
    model3 = g1 + g2 + g3 + bkg
    
    print(f"\n不同组合方式的 param_names:")
    print(f"  model1 ((g1+g2)+(g3+bkg)): {model1.param_names}")
    print(f"  model2 (g1+(g2+(g3+bkg))): {model2.param_names}")
    print(f"  model3 (g1+g2+g3+bkg):        {model3.param_names}")
    print(f"\n  param_names 是否相同: {model1.param_names == model2.param_names == model3.param_names}")
    
    print(f"\ncomponents 是否相同 (不考虑顺序):")
    comps1 = set(c.prefix for c in model1.components)
    comps2 = set(c.prefix for c in model2.components)
    comps3 = set(c.prefix for c in model3.components)
    print(f"  components prefixes 相同: {comps1 == comps2 == comps3}")
    
    return model1


def test_make_params():
    """测试 make_params 的前缀一致性"""
    print("\n" + "=" * 70)
    print("测试3: make_params 的前缀一致性")
    print("=" * 70)
    
    g1 = GaussianModel(prefix='g1_')
    g2 = GaussianModel(prefix='g2_')
    g3 = GaussianModel(prefix='g3_')
    bkg = LinearModel(prefix='bkg_')
    
    model = g1 + g2 + g3 + bkg
    params = model.make_params()
    
    print(f"\nmodel.param_names:")
    for name in model.param_names:
        print(f"  {name}")
    
    print(f"\nparams.keys() (Parameters 对象):")
    for key in params.keys():
        print(f"  {key}")
    
    print(f"\n检查参数名一致性:")
    model_params_set = set(model.param_names)
    params_keys_set = set(params.keys())
    print(f"  model.param_names ⊆ params.keys(): {model_params_set.issubset(params_keys_set)}")
    
    return params


def test_eval_components():
    """测试 eval_components 的键名一致性"""
    print("\n" + "=" * 70)
    print("测试4: eval_components 的键名一致性")
    print("=" * 70)
    
    np.random.seed(42)
    x = np.linspace(0, 20, 201)
    
    g1 = GaussianModel(prefix='g1_')
    g2 = GaussianModel(prefix='g2_')
    g3 = GaussianModel(prefix='g3_')
    bkg = LinearModel(prefix='bkg_')
    
    model = g1 + g2 + g3 + bkg
    
    pars = model.make_params(
        g1_amplitude=100, g1_center=5, g1_sigma=1,
        g2_amplitude=80, g2_center=10, g2_sigma=1.5,
        g3_amplitude=60, g3_center=15, g3_sigma=2,
        bkg_slope=0.5, bkg_intercept=10
    )
    
    y = (gaussian(x, 100, 5, 1) + 
         gaussian(x, 80, 10, 1.5) + 
         gaussian(x, 60, 15, 2) + 
         0.5 * x + 10 + 
         np.random.normal(0, 1, size=x.size))
    
    result = model.fit(y, pars, x=x)
    
    print(f"\nModel components prefixes:")
    for i, comp in enumerate(model.components):
        print(f"  [{i}] {type(comp).__name__}: prefix='{comp.prefix}'")
    
    comps = result.eval_components(x=x)
    print(f"\neval_components() 返回的键:")
    for key in comps.keys():
        print(f"  '{key}'")
    
    print(f"\n检查键名与组件前缀的对应关系:")
    component_prefixes = [c.prefix for c in model.components]
    comp_keys = list(comps.keys())
    print(f"  组件前缀: {component_prefixes}")
    print(f"  eval_components 键: {comp_keys}")
    
    print(f"\n验证各组件的计算值:")
    for prefix in component_prefixes:
        if prefix in comps:
            val = comps[prefix]
            print(f"  '{prefix}': shape={val.shape}, first 3 values={val[:3]}")
    
    return result, comps


def test_constraints_with_prefix():
    """测试约束表达式中的前缀引用"""
    print("\n" + "=" * 70)
    print("测试5: 约束表达式中的前缀引用")
    print("=" * 70)
    
    g1 = GaussianModel(prefix='g1_')
    g2 = GaussianModel(prefix='g2_')
    bkg = LinearModel(prefix='bkg_')
    
    model = g1 + g2 + bkg
    
    print(f"\n设置约束:")
    print(f"  1. g2_center = g1_center + 5 (第二个峰在第一个峰右侧5个单位)")
    print(f"  2. g2_sigma = 1.5 * g1_sigma (第二个峰更宽)")
    print(f"  3. g2_amplitude = 0.8 * g1_amplitude (第二个峰更矮)")
    
    model.set_param_hint('delta_center', value=5.0, vary=True, min=0)
    model.set_param_hint('sigma_ratio', value=1.5, vary=False)
    model.set_param_hint('amp_ratio', value=0.8, vary=False)
    
    model.set_param_hint('g2_center', expr='g1_center + delta_center')
    model.set_param_hint('g2_sigma', expr='sigma_ratio * g1_sigma')
    model.set_param_hint('g2_amplitude', expr='amp_ratio * g1_amplitude')
    
    pars = model.make_params(
        g1_amplitude=100, g1_center=5, g1_sigma=1,
        bkg_slope=0, bkg_intercept=0
    )
    
    print(f"\nParameters 对象中的参数:")
    for name, par in pars.items():
        expr_str = f", expr='{par.expr}'" if par.expr else ""
        print(f"  {name}: value={par.value}{expr_str}")
    
    np.random.seed(42)
    x = np.linspace(0, 20, 201)
    y = (gaussian(x, 100, 5, 1) + 
         gaussian(x, 80, 10, 1.5) + 
         np.random.normal(0, 1, size=x.size))
    
    result = model.fit(y, pars, x=x)
    
    print(f"\n拟合后的参数 (带约束):")
    for name in ['g1_amplitude', 'g2_amplitude', 'g1_center', 'g2_center', 
                 'g1_sigma', 'g2_sigma', 'delta_center']:
        if name in result.params:
            par = result.params[name]
            expr_str = f", expr='{par.expr}'" if par.expr else ""
            print(f"  {name}: {par.value:.4f}{expr_str}")
    
    print(f"\n验证约束关系:")
    g1_amp = result.params['g1_amplitude'].value
    g2_amp = result.params['g2_amplitude'].value
    g1_cen = result.params['g1_center'].value
    g2_cen = result.params['g2_center'].value
    g1_sig = result.params['g1_sigma'].value
    g2_sig = result.params['g2_sigma'].value
    delta = result.params['delta_center'].value
    
    print(f"  g2_amplitude = {g2_amp:.4f}, expected = 0.8 * g1_amplitude = {0.8 * g1_amp:.4f}")
    print(f"  g2_center = {g2_cen:.4f}, expected = g1_center + delta_center = {g1_cen + delta:.4f}")
    print(f"  g2_sigma = {g2_sig:.4f}, expected = 1.5 * g1_sigma = {1.5 * g1_sig:.4f}")
    
    return result


def test_best_values_and_init_values():
    """测试 best_values 和 init_values 的前缀一致性"""
    print("\n" + "=" * 70)
    print("测试6: best_values 和 init_values 的前缀一致性")
    print("=" * 70)
    
    g1 = GaussianModel(prefix='g1_')
    g2 = GaussianModel(prefix='g2_')
    bkg = LinearModel(prefix='bkg_')
    
    model = g1 + g2 + bkg
    
    np.random.seed(42)
    x = np.linspace(0, 20, 201)
    y = (gaussian(x, 100, 5, 1) + 
         gaussian(x, 80, 10, 1.5) + 
         0.5 * x + 10 + 
         np.random.normal(0, 1, size=x.size))
    
    pars = model.make_params(
        g1_amplitude=50, g1_center=4, g1_sigma=0.5,
        g2_amplitude=40, g2_center=9, g2_sigma=1,
        bkg_slope=0, bkg_intercept=5
    )
    
    result = model.fit(y, pars, x=x)
    
    print(f"\nmodel.param_names:")
    for name in model.param_names:
        print(f"  {name}")
    
    print(f"\nresult.init_values (初始值):")
    for key, val in result.init_values.items():
        print(f"  {key}: {val}")
    
    print(f"\nresult.best_values (最佳拟合值):")
    for key, val in result.best_values.items():
        print(f"  {key}: {val:.4f}")
    
    print(f"\n检查一致性:")
    param_names_set = set(model.param_names)
    init_keys_set = set(result.init_values.keys())
    best_keys_set = set(result.best_values.keys())
    
    print(f"  model.param_names == init_values.keys(): {param_names_set == init_keys_set}")
    print(f"  model.param_names == best_values.keys(): {param_names_set == best_keys_set}")
    
    return result


def test_fit_report():
    """测试拟合报告中的参数名"""
    print("\n" + "=" * 70)
    print("测试7: 拟合报告中的参数名")
    print("=" * 70)
    
    g1 = GaussianModel(prefix='g1_')
    g2 = GaussianModel(prefix='g2_')
    bkg = LinearModel(prefix='bkg_')
    
    model = g1 + g2 + bkg
    
    np.random.seed(42)
    x = np.linspace(0, 20, 201)
    y = (gaussian(x, 100, 5, 1) + 
         gaussian(x, 80, 10, 1.5) + 
         0.5 * x + 10 + 
         np.random.normal(0, 1, size=x.size))
    
    pars = model.make_params(
        g1_amplitude=50, g1_center=5, g1_sigma=1,
        g2_amplitude=40, g2_center=10, g2_sigma=1.5,
        bkg_slope=0.5, bkg_intercept=10
    )
    
    result = model.fit(y, pars, x=x)
    
    report = result.fit_report()
    print("\n拟合报告 (部分):")
    print("-" * 70)
    lines = report.split('\n')
    for line in lines[:40]:
        print(line)
    print("-" * 70)
    if len(lines) > 40:
        print(f"... 省略 {len(lines) - 40} 行")
    
    print("\n检查报告中的参数名前缀:")
    expected_prefixes = ['g1_', 'g2_', 'bkg_']
    for prefix in expected_prefixes:
        count = report.count(prefix)
        print(f"  '{prefix}' 出现次数: {count}")
    
    return result


def test_nested_expressions():
    """测试嵌套约束表达式"""
    print("\n" + "=" * 70)
    print("测试8: 嵌套约束表达式 (多层引用)")
    print("=" * 70)
    
    g1 = GaussianModel(prefix='g1_')
    g2 = GaussianModel(prefix='g2_')
    g3 = GaussianModel(prefix='g3_')
    
    model = g1 + g2 + g3
    
    print(f"\n设置嵌套约束:")
    print(f"  g2_center = g1_center + 3")
    print(f"  g3_center = g2_center + 3  (间接引用 g1_center)")
    print(f"  g3_amplitude = g2_amplitude = 0.5 * g1_amplitude")
    
    model.set_param_hint('spacing', value=3.0, vary=False)
    model.set_param_hint('g2_center', expr='g1_center + spacing')
    model.set_param_hint('g3_center', expr='g2_center + spacing')
    model.set_param_hint('g2_amplitude', expr='0.5 * g1_amplitude')
    model.set_param_hint('g3_amplitude', expr='0.5 * g1_amplitude')
    
    pars = model.make_params(
        g1_amplitude=100, g1_center=5, g1_sigma=1,
        g2_sigma=1, g3_sigma=1
    )
    
    print(f"\n参数初始值:")
    for name in ['g1_amplitude', 'g2_amplitude', 'g3_amplitude',
                 'g1_center', 'g2_center', 'g3_center']:
        par = pars[name]
        expr_str = f", expr='{par.expr}'" if par.expr else ""
        print(f"  {name}: {par.value}{expr_str}")
    
    np.random.seed(42)
    x = np.linspace(0, 20, 201)
    y = (gaussian(x, 100, 5, 1) + 
         gaussian(x, 50, 8, 1) + 
         gaussian(x, 50, 11, 1) + 
         np.random.normal(0, 0.5, size=x.size))
    
    result = model.fit(y, pars, x=x)
    
    print(f"\n拟合结果:")
    for name in ['g1_amplitude', 'g2_amplitude', 'g3_amplitude',
                 'g1_center', 'g2_center', 'g3_center']:
        par = result.params[name]
        expr_str = f", expr='{par.expr}'" if par.expr else ""
        print(f"  {name}: {par.value:.4f}{expr_str}")
    
    print(f"\n验证嵌套约束:")
    g1_cen = result.params['g1_center'].value
    g2_cen = result.params['g2_center'].value
    g3_cen = result.params['g3_center'].value
    spacing = result.params['spacing'].value
    
    print(f"  g2_center = {g2_cen:.4f}, expected = g1_center + 3 = {g1_cen + spacing:.4f}")
    print(f"  g3_center = {g3_cen:.4f}, expected = g2_center + 3 = {g2_cen + spacing:.4f}")
    print(f"  g3_center 也等于 g1_center + 6 = {g1_cen + 2*spacing:.4f}")
    
    return result


def main():
    """运行所有测试"""
    print("=" * 70)
    print("嵌套复合模型的前缀传播机制测试")
    print("=" * 70)
    
    create_nested_composite()
    test_param_names_consistency()
    test_make_params()
    test_eval_components()
    test_constraints_with_prefix()
    test_best_values_and_init_values()
    test_fit_report()
    test_nested_expressions()
    
    print("\n" + "=" * 70)
    print("所有测试完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
