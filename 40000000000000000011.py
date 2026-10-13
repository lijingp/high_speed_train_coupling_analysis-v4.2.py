"""
================================================================================
高速列车耦合网络分析与解耦决策系统
理论-案例-代码 完全统一版本 v4.2 (含附录数据导出)
功能：CSP筛选 → 性能计算 → Pareto优化 → L0-L5耦合网络分析 → 附录数据导出
================================================================================
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product, combinations
from matplotlib.patches import Patch
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 全局输出目录
# =============================================================================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
APPENDIX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "appendix_data")
os.makedirs(APPENDIX_DIR, exist_ok=True)


def out_path(name):
    return os.path.join(OUTPUT_DIR, name)


def app_path(name):
    return os.path.join(APPENDIX_DIR, name)


def safe_save(fig, name):
    path = out_path(name)
    try:
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  [OK] {path}")
    except OSError as e:
        print(f"  [WARN] 保存 {path} 失败: {e}")
        try:
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  [OK] {path} (150 dpi)")
        except Exception as e2:
            print(f"  [FAIL] 仍然失败: {e2}")
    plt.close(fig)


print("=" * 80)
print("高速列车耦合网络分析与解耦决策系统")
print("完整流程：CSP → Pareto → L0-L5耦合网络分析 → 附录导出")
print(f"图像输出目录: {OUTPUT_DIR}")
print(f"附录数据输出目录: {APPENDIX_DIR}")
print("=" * 80)


# =============================================================================
# 第一部分：CSP筛选
# =============================================================================
class CSPFilter:
    def __init__(self):
        self.subsystems = {
            'CarBody': {
                'options': ['B0-1', 'B0-2', 'B0-3'],
                'params': {
                    'B0-1': {'cost': 55, 'noise': 66},
                    'B0-2': {'cost': 95, 'noise': 58},
                    'B0-3': {'cost': 75, 'noise': 60}
                }
            },
            'Bogie': {
                'options': ['B1-1', 'B1-2', 'B1-3'],
                'params': {
                    'B1-1': {'cost': 28, 'critical_speed': 350},
                    'B1-2': {'cost': 48, 'critical_speed': 420},
                    'B1-3': {'cost': 78, 'critical_speed': 500}
                }
            },
            'Traction': {
                'options': ['B2-1', 'B2-2', 'B2-3'],
                'params': {
                    'B2-1': {'cost': 62, 'efficiency': 0.95},
                    'B2-2': {'cost': 88, 'efficiency': 0.97},
                    'B2-3': {'cost': 125, 'efficiency': 0.99}
                }
            },
            'Braking': {
                'options': ['B3-1', 'B3-2', 'B3-3'],
                'params': {
                    'B3-1': {'cost': 42, 'residual_shock': 0.12, 'delay': 0.3},
                    'B3-2': {'cost': 62, 'residual_shock': 0.06, 'delay': 0.1},
                    'B3-3': {'cost': 92, 'residual_shock': 0.02, 'delay': 0.02}
                }
            },
            'Network': {
                'options': ['B4-1', 'B4-2', 'B4-3'],
                'params': {
                    'B4-1': {'cost': 18, 'residual_coupling': 0.08, 'delay_ms': 30},
                    'B4-2': {'cost': 43, 'residual_coupling': 0.03, 'delay_ms': 10},
                    'B4-3': {'cost': 68, 'residual_coupling': 0.01, 'delay_ms': 5}
                }
            },
            'Service': {
                'options': ['B5-1', 'B5-2', 'B5-3'],
                'params': {
                    'B5-1': {'cost': 24, 'comfort': 0.7},
                    'B5-2': {'cost': 38, 'comfort': 0.9},
                    'B5-3': {'cost': 52, 'comfort': 1.0}
                }
            },
            'Auxiliary': {
                'options': ['B6-1', 'B6-2', 'B6-3'],
                'params': {
                    'B6-1': {'cost': 10, 'lifespan': 4},
                    'B6-2': {'cost': 24, 'lifespan': 8},
                    'B6-3': {'cost': 38, 'lifespan': 15}
                }
            }
        }
        self.sub_names = ['CarBody', 'Bogie', 'Traction', 'Braking', 'Network', 'Service', 'Auxiliary']

    def is_feasible(self, combo):
        if self.subsystems['Bogie']['params'][combo['Bogie']]['critical_speed'] < 420:
            return False
        if self.subsystems['CarBody']['params'][combo['CarBody']]['noise'] > 62:
            return False
        if self.subsystems['Traction']['params'][combo['Traction']]['efficiency'] < 0.97:
            return False
        if combo['Traction'] == 'B2-3' and combo['Auxiliary'] == 'B6-1':
            return False
        if combo['Bogie'] == 'B1-3' and combo['Auxiliary'] == 'B6-1':
            return False
        if combo['Braking'] == 'B3-3' and combo['Auxiliary'] == 'B6-1':
            return False
        return True

    def run(self):
        print("\n" + "=" * 60)
        print("Step 1: CSP Screening")
        print("=" * 60)
        all_options = [self.subsystems[name]['options'] for name in self.sub_names]
        feasible_combos = []
        total_count = 0
        for combo_tuple in product(*all_options):
            total_count += 1
            combo = dict(zip(self.sub_names, combo_tuple))
            if self.is_feasible(combo):
                feasible_combos.append(combo)
        print(f"  Total combinations: {total_count}")
        print(f"  Feasible combinations: {len(feasible_combos)}")
        print(f"  Excluded combinations: {total_count - len(feasible_combos)}")
        self.feasible_combos = feasible_combos
        return feasible_combos


# =============================================================================
# 第二部分：性能计算
# =============================================================================
class PerformanceCalculator:
    def __init__(self, csp_filter):
        self.csp = csp_filter

    def calculate_performance(self, combo):
        cost = sum(self.csp.subsystems[sub]['params'][opt]['cost'] for sub, opt in combo.items())
        noise = self.csp.subsystems['CarBody']['params'][combo['CarBody']]['noise']
        brake_params = self.csp.subsystems['Braking']['params'][combo['Braking']]
        braking_distance = 5000 + 500 * brake_params['residual_shock'] + 200 * brake_params['delay']
        traction_eff = self.csp.subsystems['Traction']['params'][combo['Traction']]['efficiency']
        energy = 5.0 - 0.5 * (traction_eff - 0.95) * 10 + 0.01 * (noise - 58)
        energy = max(3.5, min(6.0, energy))
        bogie_params = self.csp.subsystems['Bogie']['params'][combo['Bogie']]
        if bogie_params['critical_speed'] >= 500:
            derailment = 0.55
        elif bogie_params['critical_speed'] >= 420:
            derailment = 0.65
        else:
            derailment = 0.75
        comfort = self.csp.subsystems['Service']['params'][combo['Service']]['comfort']
        lifespan = self.csp.subsystems['Auxiliary']['params'][combo['Auxiliary']]['lifespan']
        return {
            'cost': cost, 'noise': noise, 'braking_distance': braking_distance,
            'energy': energy, 'derailment_coefficient': derailment,
            'comfort': comfort, 'lifespan': lifespan
        }

    def calculate_all(self, combos):
        print("\n" + "=" * 60)
        print("Step 2: Performance Indicator Calculation")
        print("=" * 60)
        results = []
        for i, combo in enumerate(combos):
            perf = self.calculate_performance(combo)
            perf['combo'] = combo
            perf['id'] = i
            results.append(perf)
        print(f"  Calculated {len(results)} architectures")
        return results


# =============================================================================
# 第三部分：Pareto多目标优化
# =============================================================================
class ParetoOptimizer:
    def __init__(self, performance_results):
        self.results = performance_results

    def dominates(self, a, b):
        a_vals = [a['cost'], a['noise'], a['braking_distance'], a['energy'],
                  a['derailment_coefficient'], -a['comfort'], -a['lifespan']]
        b_vals = [b['cost'], b['noise'], b['braking_distance'], b['energy'],
                  b['derailment_coefficient'], -b['comfort'], -b['lifespan']]
        all_less_equal = all(a_vals[i] <= b_vals[i] + 1e-10 for i in range(7))
        any_strict = any(a_vals[i] < b_vals[i] - 1e-10 for i in range(7))
        return all_less_equal and any_strict

    def non_dominated_sort(self):
        print("\n" + "=" * 60)
        print("Step 3: Pareto Multi-Objective Optimization")
        print("=" * 60)
        for i, p in enumerate(self.results):
            p['dominated_by'] = 0
            p['dominates'] = []
            for j, q in enumerate(self.results):
                if i == j:
                    continue
                if self.dominates(p, q):
                    p['dominates'].append(j)
                elif self.dominates(q, p):
                    p['dominated_by'] += 1
        fronts = []
        current_front = [i for i, p in enumerate(self.results) if p['dominated_by'] == 0]
        fronts.append(current_front)
        while current_front:
            next_front = []
            for i in current_front:
                for j in self.results[i]['dominates']:
                    self.results[j]['dominated_by'] -= 1
                    if self.results[j]['dominated_by'] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front = next_front
        print(f"  Pareto front size: {len(fronts[0])}")
        print(f"  Total fronts: {len(fronts)}")
        self.fronts = fronts
        return fronts


# =============================================================================
# 第四部分：L0-L5耦合网络分析
# =============================================================================
class NodeData:
    def __init__(self):
        self.internal_nodes = {
            'C01': '车体', 'C02': '设备舱', 'C03': '司机室', 'C04': '客室地板',
            'C05': '车顶', 'C06': '车钩区', 'C07': '车窗密封', 'C08': '蒙皮气密结构',
            'B01': '构架', 'B02': '轮对', 'B03': '一系悬挂', 'B04': '二系悬挂',
            'B05': '牵引电机(机械)', 'B06': '齿轮箱联轴节', 'B07': '制动盘/卡钳',
            'B08': '减振器', 'B09': '抗侧滚扭杆', 'B10': '轴箱导框',
            'B11': '速度/温度传感器', 'B12': '接地装置',
            'P01': '牵引变压器', 'P02': '牵引变流器', 'P03': '牵引电机(电气)',
            'P04': 'TCU', 'P05': '受电弓', 'P06': '主断路器',
            'P07': '高压电缆连接器', 'P08': '制动电阻', 'P09': '冷却系统', 'P10': '接地开关',
            'R01': 'BCU', 'R02': 'EP阀', 'R03': '制动缸', 'R04': '防滑阀',
            'R05': '供风单元', 'R06': '主风缸', 'R07': '停放制动', 'R08': '间隙调整器',
            'N01': 'CCU', 'N02': '网关', 'N03': 'RIOM', 'N04': '列车总线',
            'N05': 'MMI', 'N06': '数据记录器',
            'A01': '辅助变流器', 'A02': '蓄电池', 'A03': '充电机', 'A04': '配电柜',
            'A05': '逆变器', 'A06': '照明控制',
            'S01': 'HVAC', 'S02': '座椅', 'S03': 'PIS', 'S04': '卫生单元',
            'S05': '供水系统', 'S06': '内装地板/墙板', 'S07': '厨房', 'S08': '紧急逃生装置',
            'I01': '轮轨力传感器', 'I02': '车体应变传感器', 'I03': '受电弓视觉',
            'I04': '主动悬架执行器', 'I05': '主动受电弓执行器', 'I06': '车-车通信模块',
        }
        self.boundary_nodes = {'RAIL': '轨道边界', 'AIR': '空气边界', 'CAT': '接触网边界'}
        self.all_nodes = list(self.internal_nodes.keys()) + list(self.boundary_nodes.keys())
        self.n = len(self.all_nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.all_nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(self.all_nodes)}
        self.subsystem = {}
        for node in self.internal_nodes:
            self.subsystem[node] = node[0] if node[0] in 'CBPRNAIS' else 'X'
        for node in self.boundary_nodes:
            self.subsystem[node] = 'BOUNDARY'
        self.field_tags = {
            'P02': ['热场', '电磁场'], 'P03': ['热场', '电磁场'], 'P05': ['流场', '电磁场'],
            'S01': ['热场', '振动场', '声场'], 'B02': ['振动场', '机械场', '声场'],
            'B07': ['热场', '机械场'], 'R01': ['电磁场', '时序场'], 'N04': ['电磁场', '时序场'],
            'C01': ['振动场', '声场', '机械场'], 'RAIL': ['机械场', '振动场', '声场'],
            'AIR': ['流场', '声场'], 'CAT': ['电磁场', '机械场'],
            'I01': ['机械场'], 'I05': ['电磁场', '流场'], 'B08': ['振动场'],
            'B04': ['振动场', '机械场'], 'B03': ['振动场', '机械场'],
            'N01': ['电磁场', '时序场'], 'N02': ['电磁场', '时序场'],
            'N03': ['电磁场', '时序场'], 'N06': ['电磁场'],
        }
        print(f"[L0] 节点: 内部{len(self.internal_nodes)}个 + 边界{len(self.boundary_nodes)}个 = {self.n}个")


class CouplingPathData:
    def __init__(self):
        self._define_raw_paths()
        self._aggregate_by_component_pair()
        self._print_summary()

    def _define_raw_paths(self):
        self.raw_explicit = [
            ('B01', 'C01', 0.8), ('B04', 'C01', 0.7), ('B03', 'C01', 0.7), ('B05', 'C01', 0.8),
            ('B02', 'B01', 0.8), ('B02', 'B03', 0.7), ('B03', 'B01', 0.8), ('B04', 'B01', 0.7),
            ('B05', 'B02', 0.7), ('B05', 'B06', 0.8), ('B07', 'B02', 0.8), ('B08', 'B01', 0.7),
            ('B09', 'B01', 0.7), ('B10', 'B01', 0.7), ('B11', 'B02', 0.5), ('B12', 'B01', 0.4),
            ('P01', 'P02', 0.7), ('P02', 'P03', 0.7), ('P02', 'P04', 0.5), ('P05', 'P01', 0.7),
            ('P05', 'P06', 0.7), ('P06', 'P01', 0.7), ('P07', 'P05', 0.7), ('P08', 'P02', 0.7),
            ('P09', 'P02', 0.4), ('P10', 'P01', 0.7), ('R01', 'R02', 0.7), ('R01', 'R03', 0.5),
            ('R01', 'R04', 0.5), ('R02', 'R03', 0.4), ('R05', 'R06', 0.4), ('R01', 'R07', 0.5),
            ('R07', 'R03', 0.7), ('N01', 'N04', 0.5), ('N02', 'N04', 0.5), ('N03', 'N04', 0.5),
            ('N01', 'N02', 0.5), ('N04', 'N05', 0.5), ('N04', 'N06', 0.5), ('P02', 'R01', 0.5),
            ('P04', 'N04', 0.5), ('N01', 'R01', 0.5), ('N02', 'R01', 0.5), ('P02', 'N04', 0.5),
            ('P03', 'N04', 0.5), ('R01', 'N04', 0.5), ('A01', 'P02', 0.7), ('B05', 'P03', 0.8),
            ('R02', 'B07', 0.4), ('R03', 'B07', 0.7), ('R04', 'B02', 0.5), ('I01', 'N04', 0.5),
            ('I02', 'N04', 0.5), ('I03', 'N04', 0.5), ('I04', 'N04', 0.5), ('I05', 'N04', 0.5),
        ]
        self.raw_implicit = [
            ('P02', 'S01', 0.9, 1, 0.99, 'Independent Bus', ['热场', '电磁场']),
            ('P03', 'S01', 0.7, 1, 0.99, 'Independent Bus', ['热场', '电磁场']),
            ('P02', 'N04', 0.8, 1, 1.00, 'Fiber Optic', ['电磁场']),
            ('P03', 'N04', 0.7, 1, 1.00, 'Fiber Optic', ['电磁场']),
            ('P02', 'N01', 0.6, 1, 1.00, 'Fiber Optic', ['电磁场']),
            ('P05', 'C01', 0.7, 1, 0.98, 'Aero Deflector', ['流场']),
            ('B05', 'C01', 0.8, 1, 0.95, 'Vibration Isolator', ['振动场']),
            ('N06', 'N05', 0.8, 1, 1.00, 'Separate Network', ['电磁场']),
            ('N06', 'N04', 0.7, 1, 1.00, 'Separate Network', ['电磁场']),
            ('P02', 'R01', 0.9, 1, 0.99, 'Unified Brake Module', ['电磁场', '时序场']),
            ('P03', 'R02', 0.8, 1, 0.99, 'Unified Brake Module', ['电磁场', '时序场']),
            ('B04', 'C01', 0.8, 1, 0.95, 'Damping Tuning', ['振动场', '机械场']),
            ('B03', 'C01', 0.7, 1, 0.95, 'Damping Tuning', ['振动场', '机械场']),
            ('R01', 'N03', 1.0, 2, 0.80, 'Timing Contract', ['时序场']),
            ('P04', 'N03', 1.0, 2, 0.80, 'Timing Contract', ['时序场']),
            ('R01', 'P04', 0.8, 2, 0.80, 'Timing Contract', ['时序场']),
            ('B07', 'B02', 0.9, 3, 0.90, 'Thermal Coordinator', ['热场', '机械场']),
            ('B02', 'I01', 0.8, 3, 0.90, 'Thermal Coordinator', ['热场', '机械场']),
            ('B08', 'I04', 0.9, 3, 0.90, 'Hunting Control', ['振动场']),
            ('B08', 'B02', 0.7, 3, 0.90, 'Hunting Control', ['振动场']),
            ('P05', 'I05', 0.9, 3, 0.90, 'Arc Suppression', ['电磁场', '流场']),
            ('P05', 'C05', 0.6, 3, 0.90, 'Arc Suppression', ['电磁场', '流场']),
            ('R01', 'C06', 0.8, 3, 0.85, 'Longitudinal Control', ['机械场', '时序场']),
            ('P04', 'C06', 0.7, 3, 0.85, 'Longitudinal Control', ['机械场', '时序场']),
            ('N02', 'C06', 0.6, 3, 0.85, 'Longitudinal Control', ['机械场', '时序场']),
            ('B02', 'S01', 0.7, 4, 0.95, 'Adaptive Acoustic', ['振动场', '声场']),
            ('B02', 'S06', 0.6, 4, 0.95, 'Adaptive Acoustic', ['振动场', '声场']),
            ('P05', 'C05', 0.7, 4, 0.95, 'Aero Fairing', ['声场', '流场']),
            ('B01', 'C01', 0.5, 4, 0.95, 'Active Isolator', ['振动场', '声场']),
            ('N01', 'N02', 0.9, 4, 0.98, 'Standard Power Unit', ['电磁场', '时序场']),
            ('N01', 'N03', 0.9, 4, 0.98, 'Standard Power Unit', ['电磁场', '时序场']),
            ('N02', 'N03', 0.8, 4, 0.98, 'Standard Power Unit', ['电磁场', '时序场']),
        ]
        self.raw_boundary = [
            ('RAIL', 'B02', 0.95, 0.70, ['机械场', '振动场']),
            ('RAIL', 'B02', 0.85, 0.70, ['机械场', '振动场']),
            ('RAIL', 'B10', 0.70, 0.70, ['机械场']),
            ('RAIL', 'I01', 0.80, 0.70, ['机械场']),
            ('RAIL', 'C01', 0.60, 0.70, ['振动场', '声场']),
            ('RAIL', 'S01', 0.40, 0.70, ['声场']),
            ('RAIL', 'B01', 0.55, 0.70, ['振动场']),
            ('RAIL', 'B04', 0.50, 0.70, ['振动场']),
            ('AIR', 'C01', 0.85, 0.75, ['流场']),
            ('AIR', 'C05', 0.80, 0.75, ['流场', '声场']),
            ('AIR', 'P05', 0.90, 0.75, ['流场']),
            ('AIR', 'I03', 0.60, 0.75, ['流场']),
            ('AIR', 'I05', 0.70, 0.75, ['流场']),
            ('AIR', 'C06', 0.40, 0.75, ['流场']),
            ('AIR', 'S02', 0.30, 0.75, ['声场']),
            ('AIR', 'B02', 0.35, 0.75, ['声场']),
            ('CAT', 'P05', 0.95, 0.80, ['电磁场', '机械场']),
            ('CAT', 'P05', 0.90, 0.80, ['电磁场']),
            ('CAT', 'P05', 0.85, 0.80, ['电磁场', '机械场']),
            ('CAT', 'I05', 0.80, 0.80, ['电磁场']),
            ('CAT', 'P02', 0.70, 0.80, ['电磁场']),
        ]

    def _aggregate_by_component_pair(self):
        agg = {}
        for u, v, w in self.raw_explicit:
            key = (u, v) if u < v else (v, u)
            if key not in agg:
                agg[key] = self._new_entry(key)
            agg[key]['explicit_max'] = max(agg[key]['explicit_max'], w)
            agg[key]['has_explicit'] = True

        for p in self.raw_implicit:
            u, v, w, typ, eta, dec, fields = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            key = (u, v) if u < v else (v, u)
            if key not in agg:
                agg[key] = self._new_entry(key)
            if w > agg[key]['implicit_max']:
                agg[key]['implicit_max'] = w
                agg[key]['eta_implicit'] = eta
                agg[key]['decoupling_implicit'] = dec
                agg[key]['type_implicit'] = typ
                agg[key]['fields_implicit'] = fields
            agg[key]['has_implicit'] = True

        for p in self.raw_boundary:
            u, v, w, eta, fields = p[0], p[1], p[2], p[3], p[4]
            key = (u, v) if u < v else (v, u)
            if key not in agg:
                agg[key] = self._new_entry(key)
            if w > agg[key]['boundary_max']:
                agg[key]['boundary_max'] = w
                agg[key]['eta_boundary'] = eta
                agg[key]['fields_boundary'] = fields
                agg[key]['decoupling_boundary'] = 'BoundaryDecoupler'
            agg[key]['has_boundary'] = True

        self.all_paths = []
        for key, data in agg.items():
            if data['has_boundary']:
                category = 'boundary'
                max_w = data['boundary_max']
                eta = data['eta_boundary']
                fields = data['fields_boundary']
                decoupling = data['decoupling_boundary']
                residual = max_w * (1 - eta)
            elif data['has_implicit']:
                category = 'implicit'
                max_w = data['implicit_max']
                eta = data['eta_implicit']
                fields = data['fields_implicit']
                decoupling = data['decoupling_implicit']
                residual = max_w * (1 - eta)
            else:
                category = 'explicit'
                max_w = data['explicit_max']
                eta = 1.0
                fields = []
                decoupling = '不可解耦'
                residual = max_w

            self.all_paths.append({
                'source': data['source'], 'target': data['target'],
                'weight': max_w, 'residual': residual,
                'category': category, 'eta': eta,
                'fields': fields, 'decoupling': decoupling,
                'type': data['type_implicit'],
                'has_explicit': data['has_explicit'],
                'has_implicit': data['has_implicit'],
                'has_boundary': data['has_boundary'],
            })
        self.total = len(self.all_paths)

    def _new_entry(self, key):
        return {
            'source': key[0], 'target': key[1],
            'explicit_max': 0.0, 'implicit_max': 0.0, 'boundary_max': 0.0,
            'eta_explicit': 1.0, 'eta_implicit': 1.0, 'eta_boundary': 1.0,
            'fields_explicit': [], 'fields_implicit': [], 'fields_boundary': [],
            'decoupling_explicit': '不可解耦', 'decoupling_implicit': '', 'decoupling_boundary': '',
            'type_implicit': None,
            'has_explicit': False, 'has_implicit': False, 'has_boundary': False,
        }

    def _print_summary(self):
        counts = {'explicit': 0, 'implicit': 0, 'boundary': 0}
        for p in self.all_paths:
            counts[p['category']] += 1
        print(f"[L1d] 聚合后: {self.total}条 (显式{counts['explicit']}, 隐式{counts['implicit']}, 边界{counts['boundary']})")


class LatentCouplingInference:
    def __init__(self, G, node_fields, node_subsystem, all_nodes, known_edges):
        self.G = G
        self.node_fields = node_fields
        self.node_subsystem = node_subsystem
        self.all_nodes = all_nodes
        self.known_edges = set(known_edges)
        self.alpha = 0.6
        self.beta = 0.3
        self.gamma = 0.1

    def predict(self, top_k=30, threshold=0.15):
        predictions = []
        for u, v in combinations(self.all_nodes, 2):
            if (u, v) in self.known_edges or (v, u) in self.known_edges:
                continue
            neigh_u = set(self.G.neighbors(u))
            neigh_v = set(self.G.neighbors(v))
            common = neigh_u & neigh_v
            topo_score = 0.0
            if common:
                topo_score = sum(1.0 / self.G.degree(w) for w in common)
            topo_score = min(1.0, topo_score / 1.5)

            fields_u = set(self.node_fields.get(u, []))
            fields_v = set(self.node_fields.get(v, []))
            if fields_u or fields_v:
                overlap = len(fields_u & fields_v)
                union = len(fields_u | fields_v)
                field_score = overlap / union if union > 0 else 0.0
            else:
                field_score = 0.0

            attr_score = 1.0 if self.node_subsystem.get(u) == self.node_subsystem.get(v) else 0.0
            raw_score = (self.alpha * topo_score + self.beta * field_score + self.gamma * attr_score)
            pred_strength = 1 / (1 + np.exp(-10 * (raw_score - 0.2)))
            if pred_strength > threshold:
                predictions.append({
                    'source': u, 'target': v,
                    'pred_strength': round(pred_strength, 4),
                    'topo_score': round(topo_score, 3),
                    'field_score': round(field_score, 3),
                    'common_neighbors': list(common),
                    'reasoning': f"共因邻居:{list(common) if common else '无'}, 共享场:{list(fields_u & fields_v)}"
                })
        predictions.sort(key=lambda x: x['pred_strength'], reverse=True)
        return predictions[:top_k]


class FourthOrderTensor:
    def __init__(self, A, B, D, node_to_idx, idx_to_node, path_info):
        self.A, self.B, self.D = A, B, D
        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node
        self.path_info = path_info
        self.path_keys = list(path_info.keys())
        self.n = len(path_info)
        self.delta = 0.01
        self.compute_all()

    def _compute_subspace(self, source_matrix, target_matrix):
        m = len(self.path_keys)
        X = np.zeros((m, m))
        for i, p in enumerate(self.path_keys):
            info_p = self.path_info[p]
            i_p, j_p = self.node_to_idx[info_p['source']], self.node_to_idx[info_p['target']]
            C_p = source_matrix[i_p, j_p]
            for j, q in enumerate(self.path_keys):
                if i == j:
                    X[i, j] = 1.0
                    continue
                info_q = self.path_info[q]
                i_q, j_q = self.node_to_idx[info_q['source']], self.node_to_idx[info_q['target']]
                C_q = target_matrix[i_q, j_q]
                if C_q < 1e-6 or C_p < 1e-6:
                    X[i, j] = 0
                    continue
                perturb = C_q * self.delta
                C_pert = source_matrix.copy()
                C_pert[i_q, j_q] += perturb
                C_pert[j_q, i_q] += perturb
                C_p_pert = C_pert[i_p, j_p]
                derivative = (C_p_pert - C_p) / perturb
                X[i, j] = derivative * (C_q / C_p)
        return X

    def compute_all(self):
        self.X_EE = self._compute_subspace(self.A, self.A)
        self.X_EI = self._compute_subspace(self.A, self.B)
        self.X_IE = self._compute_subspace(self.B, self.A)
        self.X_II = self._compute_subspace(self.B, self.B)
        self.X_DE = self._compute_subspace(self.D, self.A)
        self.X_ED = self._compute_subspace(self.A, self.D)
        self.X_total = self.X_EE + self.X_EI + self.X_IE + self.X_II + self.X_DE + self.X_ED
        return self.X_total

    def get_subspace_norms(self):
        return {
            'EE': np.linalg.norm(self.X_EE), 'EI': np.linalg.norm(self.X_EI),
            'IE': np.linalg.norm(self.X_IE), 'II': np.linalg.norm(self.X_II),
            'DE': np.linalg.norm(self.X_DE), 'ED': np.linalg.norm(self.X_ED),
        }


class CouplingNetworkAnalyzer:
    def __init__(self):
        print("\n" + "=" * 80)
        print("开始构建耦合网络...")
        print("=" * 80)
        self.nodes = NodeData()
        self.paths = CouplingPathData()
        self.n = self.nodes.n
        self.node_list = self.nodes.all_nodes
        self.node_to_idx = self.nodes.node_to_idx
        self.idx_to_node = self.nodes.idx_to_node
        self.m = len(self.paths.all_paths)
        self._compute_residual_matrix()
        self._build_network()
        self._latent_inference()
        self._compute_tensor_decomposition()
        self._detect_feedback_loops()
        self._compute_stability_index()
        self._compute_path_degree()
        self._generate_decoupling_suggestions()
        self._verify_results()

    def _compute_residual_matrix(self):
        print("\n[L1] 残差耦合矩阵 R")
        print("     R_ij = max(A_ij, B_ij·(1-η), D_ij·(1-η_boundary))")
        self.A = np.zeros((self.n, self.n))
        self.B = np.zeros((self.n, self.n))
        self.D = np.zeros((self.n, self.n))
        self.R = np.zeros((self.n, self.n))
        self.path_info = {}
        for idx, path in enumerate(self.paths.all_paths):
            u, v = path['source'], path['target']
            i, j = self.node_to_idx[u], self.node_to_idx[v]
            w = path['weight']
            eta = path['eta']
            cat = path['category']
            if cat == 'explicit':
                self.A[i, j] = self.A[j, i] = w
                residual = w
            elif cat == 'implicit':
                self.B[i, j] = self.B[j, i] = w
                residual = w * (1 - eta)
            else:
                self.D[i, j] = self.D[j, i] = w
                residual = w * (1 - eta)
            self.R[i, j] = max(self.R[i, j], residual)
            self.R[j, i] = max(self.R[j, i], residual)
            path_id = f"P{idx + 1:03d}"
            self.path_info[path_id] = {
                'source': u, 'target': v,
                'weight': w, 'residual': residual,
                'eta': eta, 'category': cat,
                'fields': path['fields'],
                'decoupling': path['decoupling'],
                'type': path['type'],
                'has_explicit': path['has_explicit'],
                'has_implicit': path['has_implicit'],
                'has_boundary': path['has_boundary'],
            }
        nz = int(np.count_nonzero(self.R))
        print(f"      R矩阵: {self.R.shape}, 非零元素: {nz}")
        self.R_nonzero = nz

    def _build_network(self):
        print("\n[L3] 耦合网络 G = (V, E, R, H, T)")
        self.G = nx.Graph()
        for node in self.node_list:
            self.G.add_node(node, subsystem=self.nodes.subsystem[node])
        edge_count = 0
        self.known_edges = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.R[i, j] > 0:
                    u, v = self.idx_to_node[i], self.idx_to_node[j]
                    self.G.add_edge(u, v, weight=self.R[i, j])
                    self.known_edges.append((u, v))
                    edge_count += 1
        degrees = dict(self.G.degree())
        hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"      节点数: {self.G.number_of_nodes()}, 边数: {edge_count}")
        print(f"      网络密度: {nx.density(self.G):.4f}")
        print("      枢纽节点 (Top 5):")
        for node, deg in hubs:
            print(f"        {node}: {deg}")
        self.edge_count = edge_count

    def _latent_inference(self):
        print("\n[L1.5] 潜在耦合推理 (基于链路预测)")
        infer = LatentCouplingInference(
            G=self.G, node_fields=self.nodes.field_tags,
            node_subsystem=self.nodes.subsystem,
            all_nodes=self.node_list, known_edges=self.known_edges
        )
        self.latent_predictions = infer.predict(top_k=20, threshold=0.15)
        print(f"      推理出 {len(self.latent_predictions)} 条潜在耦合路径 (得分>0.15)")
        for item in self.latent_predictions[:10]:
            u, v = item['source'], item['target']
            i, j = self.node_to_idx[u], self.node_to_idx[v]
            pred_strength = item['pred_strength']
            if self.B[i, j] < pred_strength:
                self.B[i, j] = self.B[j, i] = pred_strength
                if pred_strength > self.R[i, j]:
                    self.R[i, j] = self.R[j, i] = pred_strength
        print(f"      增强后B矩阵非零元素: {np.count_nonzero(self.B)}")
        for i, item in enumerate(self.latent_predictions[:5], 1):
            print(f"        {i}. {item['source']} ↔ {item['target']} 强度={item['pred_strength']:.3f}")

    def _compute_tensor_decomposition(self):
        print("\n[L2] 4阶张量 X 及其6子空间分解")
        self.path_keys = list(self.path_info.keys())
        tensor = FourthOrderTensor(self.A, self.B, self.D,
                                   self.node_to_idx, self.idx_to_node, self.path_info)
        self.X = tensor.compute_all()
        self.X_EE, self.X_EI, self.X_IE = tensor.X_EE, tensor.X_EI, tensor.X_IE
        self.X_II, self.X_DE, self.X_ED = tensor.X_II, tensor.X_DE, tensor.X_ED
        norms = tensor.get_subspace_norms()
        print(f"      X张量: {self.X.shape}, 非零元素: {np.count_nonzero(self.X)}")
        print("\n      各子空间贡献 (Frobenius范数):")
        for name, norm in norms.items():
            print(f"        X^{name}: {norm:.4f}")

    def _detect_feedback_loops(self):
        print("\n[L4] 正反馈环检测: X_pq · X_qp > 0 (排除同一组件对)")
        self.feedback_loops = []
        m = len(self.path_keys)
        for i in range(m):
            for j in range(i + 1, m):
                p, q = self.path_keys[i], self.path_keys[j]
                info_p, info_q = self.path_info[p], self.path_info[q]
                p_src, p_tgt = info_p['source'], info_p['target']
                q_src, q_tgt = info_q['source'], info_q['target']
                if (p_src == q_src and p_tgt == q_tgt) or (p_src == q_tgt and p_tgt == q_src):
                    continue
                product = self.X[i, j] * self.X[j, i]
                if product > 0.001:
                    self.feedback_loops.append({
                        'path1': p, 'path2': q,
                        'x_ij': self.X[i, j], 'x_ji': self.X[j, i],
                        'product': product,
                        'info1': info_p, 'info2': info_q,
                    })
        self.feedback_loops.sort(key=lambda x: x['product'], reverse=True)
        print(f"      发现 {len(self.feedback_loops)} 个有效正反馈环 (跨路径)")

    def _compute_stability_index(self):
        print("\n[L3] 网络稳定性指标 Ω")
        n_pos = len(self.feedback_loops) if hasattr(self, 'feedback_loops') else 0
        n_neg = max(0, 5 - n_pos)
        self.stability_index = 1 - n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 1.0
        print(f"      Ω = 1 - {n_pos}/({n_pos}+{n_neg}) = {self.stability_index:.3f}")

    def _compute_path_degree(self):
        print("\n[L4] 路径度 d_p = Σ|X_pq|")
        self.path_degree = {}
        m = len(self.path_keys)
        for i, p in enumerate(self.path_keys):
            self.path_degree[p] = np.sum(np.abs(self.X[i, :])) - 1.0
        sorted_paths = sorted(self.path_degree.items(), key=lambda x: x[1], reverse=True)
        print("      路径度 Top 5:")
        for i, (p, d) in enumerate(sorted_paths[:5], 1):
            info = self.path_info[p]
            print(f"        {i}. {info['source']}→{info['target']}: d_p={d:.4f}")

    def _generate_decoupling_suggestions(self):
        print("\n[L5] 解耦决策建议 (供设计师决策)")
        candidates = []
        for pid, info in self.path_info.items():
            if info['category'] == 'explicit':
                continue
            score = 0
            reason = []
            if info['category'] == 'boundary':
                score += 1000
                reason.append('边界耦合')
            in_loop = False
            for loop in self.feedback_loops:
                if loop['path1'] == pid or loop['path2'] == pid:
                    in_loop = True
                    score += 500 * loop['product']
                    reason.append(f'正反馈环(乘积={loop["product"]:.3f})')
                    break
            path_deg = self.path_degree.get(pid, 0)
            if path_deg > 1.0:
                score += 100 * path_deg
                reason.append(f'高路径度(d={path_deg:.2f})')
            if info['residual'] > 0.05:
                score += 50 * info['residual']
                reason.append(f'高残差(r={info["residual"]:.3f})')
            candidates.append({
                'path': pid, 'source': info['source'], 'target': info['target'],
                'residual': info['residual'], 'eta': info['eta'],
                'category': info['category'], 'fields': info['fields'],
                'decoupling': info['decoupling'], 'score': score,
                'reason': '+'.join(reason) if reason else '常规'
            })
        candidates.sort(key=lambda x: x['score'], reverse=True)
        self.decisions = candidates

    def _verify_results(self):
        print("\n" + "=" * 80)
        print("验证结果")
        print("=" * 80)
        total_w = total_r = implicit_w = implicit_r = boundary_w = boundary_r = explicit_w = explicit_r = 0
        for info in self.path_info.values():
            w, r = info['weight'], info['residual']
            total_w += w; total_r += r
            if info['category'] == 'explicit':
                explicit_w += w; explicit_r += r
            elif info['category'] == 'implicit':
                implicit_w += w; implicit_r += r
            elif info['category'] == 'boundary':
                boundary_w += w; boundary_r += r
        total_s = (1 - total_r / total_w) * 100 if total_w > 0 else 0
        implicit_s = (1 - implicit_r / implicit_w) * 100 if implicit_w > 0 else 0
        boundary_s = (1 - boundary_r / boundary_w) * 100 if boundary_w > 0 else 0
        print(f"\n  [验证1] 总耦合抑制率: {total_s:.1f}% ({total_w:.2f} → {total_r:.2f})")
        print(f"  [验证2] 隐式耦合抑制率: {implicit_s:.1f}% ({implicit_w:.2f} → {implicit_r:.2f})")
        print(f"  [验证3] 边界耦合抑制率: {boundary_s:.1f}% ({boundary_w:.2f} → {boundary_r:.2f})")
        print(f"  [验证4] 有效正反馈环: {len(self.feedback_loops)} 个")
        print(f"  [验证5] 潜在耦合预测: {len(self.latent_predictions)} 条")
        print(f"  [验证6] 网络稳定性: Ω = {self.stability_index:.3f}")
        print(f"  [验证7] R矩阵非零元素: {self.R_nonzero}")
        print(f"  [验证8] 耦合网络边数: {self.edge_count}")
        print("=" * 80)

        self.summary_stats = {
            'total_w': total_w, 'total_r': total_r, 'total_s': total_s,
            'implicit_w': implicit_w, 'implicit_r': implicit_r, 'implicit_s': implicit_s,
            'boundary_w': boundary_w, 'boundary_r': boundary_r, 'boundary_s': boundary_s,
            'explicit_w': explicit_w, 'explicit_r': explicit_r,
        }


# =============================================================================
# 第五部分：可视化
# =============================================================================
ENGLISH_ABBR = {
    'C01': 'Car Body', 'C02': 'Equip.Cabin', 'C03': 'Driver Cab', 'C04': 'Pass.Floor',
    'C05': 'Roof', 'C06': 'Coupler', 'C07': 'Window Seal', 'C08': 'Airtight',
    'B01': 'Bogie Frame', 'B02': 'Wheelset', 'B03': 'Primary Susp.', 'B04': 'Secondary Susp.',
    'B05': 'Trac.Motor(M)', 'B06': 'Gearbox', 'B07': 'Brake Disc', 'B08': 'Yaw Damper',
    'B09': 'Anti-Roll', 'B10': 'Axle Box', 'B11': 'Sensor', 'B12': 'Grounding',
    'P01': 'Trac.Trans.', 'P02': 'Trac.Converter', 'P03': 'Trac.Motor(E)', 'P04': 'TCU',
    'P05': 'Pantograph', 'P06': 'Circuit Brk.', 'P07': 'HV Cable', 'P08': 'Brake Resistor',
    'P09': 'Cooling', 'P10': 'Grounding Sw.',
    'R01': 'BCU', 'R02': 'EP Valve', 'R03': 'Brake Cyl.', 'R04': 'Anti-Skid',
    'R05': 'Air Supply', 'R06': 'Air Reservoir', 'R07': 'Parking Brake', 'R08': 'Slack Adj.',
    'N01': 'CCU', 'N02': 'Gateway', 'N03': 'RIOM', 'N04': 'Train Bus',
    'N05': 'MMI', 'N06': 'Data Rec.',
    'A01': 'Aux.Conv.', 'A02': 'Battery', 'A03': 'Charger', 'A04': 'Dist.Cabinet',
    'A05': 'Inverter', 'A06': 'Lighting Ctrl.',
    'S01': 'HVAC', 'S02': 'Seats', 'S03': 'PIS', 'S04': 'Sanitary',
    'S05': 'Water Supply', 'S06': 'Interior Panels', 'S07': 'Kitchen', 'S08': 'Escape',
    'I01': 'Wheel-Rail Force', 'I02': 'Strain Sensor', 'I03': 'Pantograph Vis.',
    'I04': 'Active Susp.Actuator', 'I05': 'Active Pantograph', 'I06': 'V2V Comm.',
    'RAIL': 'Track', 'AIR': 'Aerodynamic', 'CAT': 'Catenary',
}


def plot_residual_heatmap(analyzer, save_name='residual_heatmap.png'):
    print("  Generating residual_heatmap.png (L1 snapshot, recomputed)...")
    n = analyzer.n
    R_L1 = np.zeros((n, n))
    node_to_idx = analyzer.node_to_idx
    for path in analyzer.paths.all_paths:
        u, v = path['source'], path['target']
        i, j = node_to_idx[u], node_to_idx[v]
        w = path['weight']
        eta = path['eta']
        cat = path['category']
        if cat == 'explicit':
            residual = w
        elif cat == 'implicit':
            residual = w * (1.0 - eta)
        else:
            residual = w * (1.0 - eta)
        R_L1[i, j] = max(R_L1[i, j], residual)
        R_L1[j, i] = max(R_L1[j, i], residual)

    nonzero = int(np.count_nonzero(R_L1))
    edges = nonzero // 2
    density = 2 * edges / (n * (n - 1)) if n > 1 else 0.0
    print(f"      L1 R 非零元素: {nonzero}, 边数: {edges}, 密度: {density:.4f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(R_L1, cmap='Reds', aspect='auto', vmin=0, vmax=1)
    ax.set_title('Residual Coupling Matrix R (L1)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Component Index', fontsize=12)
    ax.set_ylabel('Component Index', fontsize=12)
    cbar = plt.colorbar(im, ax=ax, label='Coupling Strength', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    ax.text(0.02, 0.98, f'Non-zero: {nonzero}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    safe_save(fig, save_name)


def plot_coupling_network(analyzer, save_name='coupling_network.png', weight_type='residual'):
    G = nx.Graph()
    for node in analyzer.node_list:
        G.add_node(node, subsystem=analyzer.nodes.subsystem[node])
    for path in analyzer.paths.all_paths:
        u, v = path['source'], path['target']
        w = path['weight'] if weight_type == 'raw' else path['residual']
        if w > 0:
            G.add_edge(u, v, weight=w, category=path['category'])

    boundary = ['RAIL', 'AIR', 'CAT']
    sub_colors = {
        'C': '#4e79a7', 'B': '#f28e2b', 'P': '#e15759', 'R': '#59a14f',
        'N': '#edc948', 'A': '#b07aa1', 'S': '#ff9da7', 'I': '#9c755f',
        'BOUNDARY': '#e31a1c', 'default': '#cccccc'
    }

    fig, ax = plt.subplots(figsize=(22, 16))
    pos = nx.spring_layout(G, seed=42, k=3.0, iterations=500)

    edges_explicit = [(u, v) for u, v, d in G.edges(data=True) if d.get('category') == 'explicit']
    edges_implicit = [(u, v) for u, v, d in G.edges(data=True) if d.get('category') == 'implicit']
    edges_boundary = [(u, v) for u, v, d in G.edges(data=True) if d.get('category') == 'boundary']

    if edges_explicit:
        nx.draw_networkx_edges(G, pos, edgelist=edges_explicit, width=1.2, alpha=0.5,
                               edge_color='#4e79a7', ax=ax)
    if edges_implicit:
        nx.draw_networkx_edges(G, pos, edgelist=edges_implicit, width=2.0, alpha=0.7,
                               edge_color='#f28e2b', ax=ax)
    if edges_boundary:
        nx.draw_networkx_edges(G, pos, edgelist=edges_boundary, width=3.0, alpha=0.9,
                               edge_color='#e31a1c', ax=ax)

    for node in G.nodes():
        if node in boundary:
            size, ec, lw = 900, '#8b0000', 3.0
        else:
            size, ec, lw = 450, 'black', 1.2
        color = sub_colors.get(analyzer.nodes.subsystem.get(node, 'default'), sub_colors['default'])
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=size,
                               node_color=[color], edgecolors=ec, linewidths=lw,
                               alpha=0.85, ax=ax)

    labels = {n: ENGLISH_ABBR.get(n, n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10,
                            font_weight='bold', ax=ax)

    legend_elements = [
        Patch(facecolor='#e31a1c', edgecolor='#e31a1c', label='Boundary Coupling'),
        Patch(facecolor='#f28e2b', edgecolor='#f28e2b', label='Implicit Coupling'),
        Patch(facecolor='#4e79a7', edgecolor='#4e79a7', label='Explicit Coupling'),
        Patch(facecolor='#e31a1c', edgecolor='black', label='Boundary Node'),
        Patch(facecolor='#4e79a7', edgecolor='black', label='Car Body (C)'),
        Patch(facecolor='#f28e2b', edgecolor='black', label='Bogie (B)'),
        Patch(facecolor='#e15759', edgecolor='black', label='Traction (P)'),
        Patch(facecolor='#59a14f', edgecolor='black', label='Braking (R)'),
        Patch(facecolor='#edc948', edgecolor='black', label='Network (N)'),
        Patch(facecolor='#b07aa1', edgecolor='black', label='Auxiliary (A)'),
        Patch(facecolor='#ff9da7', edgecolor='black', label='Service (S)'),
        Patch(facecolor='#9c755f', edgecolor='black', label='Sensing (I)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, ncol=2,
              title='LEGEND', title_fontsize=12, frameon=True, fancybox=True, shadow=True)

    title_str = 'ORIGINAL COUPLING NETWORK' if weight_type == 'raw' else 'RESIDUAL COUPLING NETWORK'
    ax.set_title(f'{title_str}\n(67 Nodes: 64 Internal + 3 Boundary)',
                 fontsize=18, fontweight='bold', pad=20)

    total_edges = G.number_of_edges()
    cnt_e = sum(1 for _, _, d in G.edges(data=True) if d.get('category') == 'explicit')
    cnt_i = sum(1 for _, _, d in G.edges(data=True) if d.get('category') == 'implicit')
    cnt_b = sum(1 for _, _, d in G.edges(data=True) if d.get('category') == 'boundary')
    ax.text(0.5, -0.03,
            f'Edges: {total_edges} (Explicit: {cnt_e}, Implicit: {cnt_i}, Boundary: {cnt_b})  |  Density: {nx.density(G):.4f}',
            ha='center', transform=ax.transAxes, fontsize=11, style='italic', color='#444444')
    ax.axis('off')
    plt.tight_layout()
    safe_save(fig, save_name)


def plot_subspace_norms(analyzer, save_name='subspace_norms.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    norms = {
        'EE': np.linalg.norm(analyzer.X_EE), 'EI': np.linalg.norm(analyzer.X_EI),
        'IE': np.linalg.norm(analyzer.X_IE), 'II': np.linalg.norm(analyzer.X_II),
        'DE': np.linalg.norm(analyzer.X_DE), 'ED': np.linalg.norm(analyzer.X_ED),
    }
    labels_full = {'EE': 'Explicit→Explicit', 'EI': 'Explicit→Implicit', 'IE': 'Implicit→Explicit',
                   'II': 'Implicit→Implicit', 'DE': 'Boundary→Explicit', 'ED': 'Explicit→Boundary'}
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = ax.bar(norms.keys(), norms.values(), color=colors, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Frobenius Norm', fontsize=12, fontweight='bold')
    ax.set_xlabel('Interaction Subspace', fontsize=12, fontweight='bold')
    ax.set_title('Six-Component Decomposition of the 4th-Order Tensor', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, norms.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticklabels([labels_full[k] for k in norms.keys()], rotation=15, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(norms.values()) * 1.2)
    plt.tight_layout()
    safe_save(fig, save_name)


def plot_hub_importance(analyzer, save_name='hub_importance.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    degrees = dict(analyzer.G.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:12]
    nodes = [n for n, _ in sorted_nodes]
    degs = [d for _, d in sorted_nodes]
    labels = [ENGLISH_ABBR.get(n, n) for n in nodes]
    colors = ['#e41a1c' if n in ['RAIL', 'AIR', 'CAT'] else '#1f77b4' for n in nodes]
    y_pos = np.arange(len(nodes))
    bars = ax.barh(y_pos, degs, color=colors, edgecolor='black', linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Degree', fontsize=12, fontweight='bold')
    ax.set_ylabel('Component', fontsize=12, fontweight='bold')
    ax.set_title('Hub Nodes in the Coupling Network (L3)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, degs):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f'{val}', va='center', fontsize=10, fontweight='bold')
    legend_elements = [Patch(facecolor='#e41a1c', label='Boundary Node'),
                       Patch(facecolor='#1f77b4', label='Internal Component')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    plt.tight_layout()
    safe_save(fig, save_name)


def plot_latent_predictions(analyzer, save_name='latent_predictions.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    predictions = analyzer.latent_predictions[:10]
    if not predictions:
        plt.close(fig)
        return
    paths = [f"{ENGLISH_ABBR.get(i['source'], i['source'])} ↔ {ENGLISH_ABBR.get(i['target'], i['target'])}" for i in predictions]
    strengths = [i['pred_strength'] for i in predictions]
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(paths)))[::-1]
    y_pos = np.arange(len(paths))
    bars = ax.barh(y_pos, strengths, color=colors, edgecolor='black', linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(paths, fontsize=9)
    ax.set_xlabel('Predicted Coupling Strength', fontsize=12, fontweight='bold')
    ax.set_ylabel('Component Pair', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Latent Coupling Predictions (L1.5)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 1.0)
    for bar, val in zip(bars, strengths):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', fontsize=9)
    plt.tight_layout()
    safe_save(fig, save_name)


def plot_decoupling_priority(analyzer, save_name='decoupling_priority.png'):
    fig, ax = plt.subplots(figsize=(12, 7))
    candidates = []
    for pid, info in analyzer.path_info.items():
        if info['category'] == 'explicit' or info['residual'] < 0.001:
            continue
        score = 0
        if info['category'] == 'boundary':
            score += 1000
        in_loop = False
        for loop in analyzer.feedback_loops:
            if loop['path1'] == pid or loop['path2'] == pid:
                in_loop = True
                score += 500 * loop['product']
                break
        path_deg = analyzer.path_degree.get(pid, 0)
        if path_deg > 1.0:
            score += 100 * path_deg
        if info['residual'] > 0.05:
            score += 50 * info['residual']
        candidates.append({'pid': pid, 'source': info['source'], 'target': info['target'],
                           'score': score, 'residual': info['residual'],
                           'category': info['category'], 'in_loop': in_loop})
    candidates.sort(key=lambda x: x['score'], reverse=True)
    top = candidates[:10]
    if not top:
        plt.close(fig)
        return
    labels = [f"{ENGLISH_ABBR.get(c['source'], c['source'])} → {ENGLISH_ABBR.get(c['target'], c['target'])}" for c in top]
    scores = [c['score'] for c in top]
    colors = ['#d62728' if c['category'] == 'boundary' else '#ff7f0e' if c['in_loop'] else '#1f77b4' for c in top]
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, scores, color=colors, edgecolor='black', linewidth=1.0, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Priority Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coupling Path', fontsize=12, fontweight='bold')
    ax.set_title('Decoupling Priority Ranking (L4)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar, c in zip(bars, top):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f'R={c["residual"]:.3f}', va='center', fontsize=9)
    legend_elements = [Patch(facecolor='#d62728', label='Boundary Coupling (Priority 0)'),
                       Patch(facecolor='#ff7f0e', label='Positive Feedback Loop (Priority 1)'),
                       Patch(facecolor='#1f77b4', label='Other (Priority 2-3)')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    plt.tight_layout()
    safe_save(fig, save_name)


def visualize(analyzer):
    print("\n[Visualization] Generating figures...")
    plot_coupling_network(analyzer, 'coupling_network_raw.png', weight_type='raw')
    plot_coupling_network(analyzer, 'coupling_network_residual.png', weight_type='residual')
    plot_residual_heatmap(analyzer, 'residual_heatmap.png')
    plot_subspace_norms(analyzer, 'subspace_norms.png')
    plot_hub_importance(analyzer, 'hub_importance.png')
    plot_latent_predictions(analyzer, 'latent_predictions.png')
    plot_decoupling_priority(analyzer, 'decoupling_priority.png')
    print("\n[Visualization] All figures saved.")


# =============================================================================
# 第六部分：附录数据导出
# =============================================================================
def export_appendix_data(csp, feasible_combos, performance_results, pareto, best, analyzer):
    """导出附录所需的所有数据表到 CSV 和 Markdown"""
    print("\n" + "=" * 80)
    print("导出附录数据...")
    print("=" * 80)

    # 1. CSP 保留选项
    retained = []
    for sub in csp.sub_names:
        opts = csp.subsystems[sub]['options']
        retained.append({
            'Subsystem': sub,
            'Options': ', '.join(opts),
            'Count': len(opts)
        })
    pd.DataFrame(retained).to_csv(app_path('appendix_csp_retained.csv'),
                                  index=False, encoding='utf-8-sig')
    print(f"  [OK] appendix_csp_retained.csv")

    # 2. 可行架构性能表
    perf_rows = []
    for p in performance_results:
        perf_rows.append({
            'ID': p['id'],
            'CarBody': p['combo']['CarBody'],
            'Bogie': p['combo']['Bogie'],
            'Traction': p['combo']['Traction'],
            'Braking': p['combo']['Braking'],
            'Network': p['combo']['Network'],
            'Service': p['combo']['Service'],
            'Auxiliary': p['combo']['Auxiliary'],
            'Cost_10kCNY': p['cost'],
            'Noise_dB': p['noise'],
            'BrakingDistance_m': p['braking_distance'],
            'Energy_kWh': round(p['energy'], 4),
            'Derailment': p['derailment_coefficient'],
            'Comfort': p['comfort'],
            'Lifespan_y': p['lifespan']
        })
    pd.DataFrame(perf_rows).to_csv(app_path('appendix_feasible_architectures.csv'),
                                   index=False, encoding='utf-8-sig')
    print(f"  [OK] appendix_feasible_architectures.csv")

    # 3. Pareto 前沿解
    pareto_rows = []
    for rank, idx in enumerate(pareto.fronts[0], 1):
        p = performance_results[idx]
        pareto_rows.append({
            'Rank': rank,
            'ID': p['id'],
            'CarBody': p['combo']['CarBody'],
            'Bogie': p['combo']['Bogie'],
            'Traction': p['combo']['Traction'],
            'Braking': p['combo']['Braking'],
            'Network': p['combo']['Network'],
            'Service': p['combo']['Service'],
            'Auxiliary': p['combo']['Auxiliary'],
            'Cost_10kCNY': p['cost'],
            'BrakingDistance_m': p['braking_distance'],
            'Noise_dB': p['noise'],
            'Energy_kWh': round(p['energy'], 4),
            'Derailment': p['derailment_coefficient'],
            'Comfort': p['comfort'],
            'Lifespan_y': p['lifespan']
        })
    pd.DataFrame(pareto_rows).to_csv(app_path('appendix_pareto_front.csv'),
                                     index=False, encoding='utf-8-sig')
    print(f"  [OK] appendix_pareto_front.csv")

    # 4. 推荐方案
    if best is not None:
        rec = {
            'Module': list(best['combo'].keys()),
            'Selection': list(best['combo'].values()),
            'Cost_10kCNY': [best['cost']] * len(best['combo']),
            'BrakingDistance_m': [best['braking_distance']] * len(best['combo']),
            'Noise_dB': [best['noise']] * len(best['combo']),
            'Energy_kWh': [round(best['energy'], 4)] * len(best['combo']),
            'Derailment': [best['derailment_coefficient']] * len(best['combo']),
            'Comfort': [best['comfort']] * len(best['combo']),
            'Lifespan_y': [best['lifespan']] * len(best['combo']),
            'Score': [best.get('score', 'N/A')] * len(best['combo']),
        }
        pd.DataFrame(rec).to_csv(app_path('appendix_recommended.csv'),
                                 index=False, encoding='utf-8-sig')
        with open(app_path('appendix_recommended_perf.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Recommended Solution ID: {best['id']}\n")
            f.write(f"Configuration: {best['combo']}\n")
            f.write(f"Cost: {best['cost']} (10k CNY)\n")
            f.write(f"Braking Distance: {best['braking_distance']:.1f} m\n")
            f.write(f"Noise: {best['noise']} dB\n")
            f.write(f"Energy: {best['energy']:.4f} kWh/人/100km\n")
            f.write(f"Derailment Coefficient: {best['derailment_coefficient']}\n")
            f.write(f"Comfort: {best['comfort']}\n")
            f.write(f"Lifespan: {best['lifespan']} years\n")
            f.write(f"Score: {best.get('score', 'N/A')}\n")
        print(f"  [OK] appendix_recommended.csv")
        print(f"  [OK] appendix_recommended_perf.txt")

    # 5. 耦合路径表
    path_rows = []
    for pid, info in analyzer.path_info.items():
        path_rows.append({
            'PathID': pid,
            'Source': info['source'],
            'Target': info['target'],
            'Category': info['category'],
            'Type': info['type'] if info['type'] is not None else '',
            'Weight': round(info['weight'], 4),
            'Eta': round(info['eta'], 4),
            'Residual': round(info['residual'], 4),
            'Fields': ', '.join(info['fields']) if info['fields'] else '',
            'Decoupling': info['decoupling']
        })
    pd.DataFrame(path_rows).to_csv(app_path('appendix_coupling_paths.csv'),
                                   index=False, encoding='utf-8-sig')
    print(f"  [OK] appendix_coupling_paths.csv ({len(path_rows)} 条路径)")

    # 6. 残差矩阵
    pd.DataFrame(analyzer.R).to_csv(app_path('appendix_residual_matrix.csv'),
                                    index=False, encoding='utf-8-sig')
    print(f"  [OK] appendix_residual_matrix.csv (67x67)")

    # 7. 潜在耦合
    latent_rows = []
    for item in analyzer.latent_predictions:
        latent_rows.append({
            'Source': item['source'],
            'Target': item['target'],
            'PredStrength': item['pred_strength'],
            'TopoScore': item['topo_score'],
            'FieldScore': item['field_score'],
            'CommonNeighbors': ', '.join(item['common_neighbors']),
            'Reasoning': item['reasoning']
        })
    pd.DataFrame(latent_rows).to_csv(app_path('appendix_latent_predictions.csv'),
                                     index=False, encoding='utf-8-sig')
    print(f"  [OK] appendix_latent_predictions.csv ({len(latent_rows)} 条)")

    # 8. 路径度
    degree_rows = []
    for pid, d in sorted(analyzer.path_degree.items(), key=lambda x: x[1], reverse=True):
        info = analyzer.path_info[pid]
        degree_rows.append({
            'PathID': pid,
            'Source': info['source'],
            'Target': info['target'],
            'PathDegree': round(d, 6),
            'Category': info['category'],
            'Residual': round(info['residual'], 4)
        })
    pd.DataFrame(degree_rows).to_csv(app_path('appendix_path_degree.csv'),
                                     index=False, encoding='utf-8-sig')
    print(f"  [OK] appendix_path_degree.csv ({len(degree_rows)} 条)")

    # 9. 正反馈环
    loop_rows = []
    for loop in analyzer.feedback_loops:
        loop_rows.append({
            'Path1': loop['path1'],
            'Path2': loop['path2'],
            'X_ij': loop['x_ij'],
            'X_ji': loop['x_ji'],
            'Product': loop['product']
        })
    if not loop_rows:
        loop_rows.append({'Path1': 'None', 'Path2': 'None',
                          'X_ij': 0, 'X_ji': 0, 'Product': 0})
    pd.DataFrame(loop_rows).to_csv(app_path('appendix_feedback_loops.csv'),
                                   index=False, encoding='utf-8-sig')
    print(f"  [OK] appendix_feedback_loops.csv ({len(analyzer.feedback_loops)} 个正反馈环)")

    # 10. 解耦建议
    sug_rows = []
    for c in analyzer.decisions[:50]:
        sug_rows.append({
            'PathID': c['path'],
            'Source': c['source'],
            'Target': c['target'],
            'Category': c['category'],
            'Residual': round(c['residual'], 4),
            'Eta': round(c['eta'], 4),
            'Decoupling': c['decoupling'],
            'Score': round(c['score'], 3),
            'Reason': c['reason']
        })
    pd.DataFrame(sug_rows).to_csv(app_path('appendix_decoupling_suggestions.csv'),
                                  index=False, encoding='utf-8-sig')
    print(f"  [OK] appendix_decoupling_suggestions.csv ({len(sug_rows)} 条)")

    # 11. Markdown 汇总
    with open(app_path('appendix_summary.md'), 'w', encoding='utf-8') as f:
        f.write("# 附录数据汇总 (代码 v4.2 输出)\n\n")
        f.write("## 1. CSP 筛选\n\n")
        f.write(f"- 总组合数: 3^7 = 2187\n")
        f.write(f"- 性能约束后: 648\n")
        f.write(f"- 物理兼容排除: 180\n")
        f.write(f"- 最终可行架构: {len(feasible_combos)}\n")
        f.write(f"- 总排除: {2187 - len(feasible_combos)}\n\n")

        f.write("## 2. Pareto 优化\n\n")
        f.write(f"- Pareto 前沿解数: {len(pareto.fronts[0])}\n")
        f.write(f"- Pareto 前沿总数: {len(pareto.fronts)}\n\n")

        if best is not None:
            f.write("## 3. 推荐方案\n\n")
            f.write(f"- 方案 ID: {best['id']}\n")
            f.write(f"- 配置: {best['combo']}\n")
            f.write(f"- 成本: {best['cost']} 万元\n")
            f.write(f"- 制动距离: {best['braking_distance']:.1f} m\n")
            f.write(f"- 噪声: {best['noise']} dB\n")
            f.write(f"- 能耗: {best['energy']:.4f} kWh/人/100km\n")
            f.write(f"- 脱轨系数: {best['derailment_coefficient']}\n")
            f.write(f"- 舒适度: {best['comfort']}\n")
            f.write(f"- 寿命: {best['lifespan']} 年\n")
            f.write(f"- 综合得分: {best.get('score', 'N/A')}\n\n")

        f.write("## 4. 耦合网络 (L0-L5)\n\n")
        f.write(f"- 节点数: 67 (内部 64 + 边界 3)\n")
        f.write(f"- 耦合路径总数: {len(analyzer.path_info)} (显式 46, 隐式 31, 边界 18)\n")
        f.write(f"- 残差矩阵非零元素: {analyzer.R_nonzero}\n")
        f.write(f"- 耦合网络边数: {analyzer.edge_count}\n")
        f.write(f"- 网络密度: {nx.density(analyzer.G):.4f}\n")
        f.write(f"- 潜在耦合数: {len(analyzer.latent_predictions)}\n")
        f.write(f"- 正反馈环数: {len(analyzer.feedback_loops)}\n")
        f.write(f"- 网络稳定性 Ω: {analyzer.stability_index:.3f}\n\n")

        f.write("## 5. 抑制率\n\n")
        s = analyzer.summary_stats
        f.write(f"- 总抑制率: {s['total_s']:.1f}% ({s['total_w']:.2f} → {s['total_r']:.2f})\n")
        f.write(f"- 隐式抑制率: {s['implicit_s']:.1f}% ({s['implicit_w']:.2f} → {s['implicit_r']:.2f})\n")
        f.write(f"- 边界抑制率: {s['boundary_s']:.1f}% ({s['boundary_w']:.2f} → {s['boundary_r']:.2f})\n")
        f.write(f"- 显式耦合: {s['explicit_w']:.2f} → {s['explicit_r']:.2f} (不可解耦)\n\n")

        f.write("## 6. 路径度 Top 10\n\n")
        sorted_deg = sorted(analyzer.path_degree.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (pid, d) in enumerate(sorted_deg, 1):
            info = analyzer.path_info[pid]
            f.write(f"{i}. {info['source']}→{info['target']}: d_p = {d:.4f}\n")

    print(f"  [OK] appendix_summary.md")
    print(f"\n所有附录数据已保存到: {APPENDIX_DIR}")


# =============================================================================
# 第七部分：主程序
# =============================================================================
def main():
    print("\n" + "=" * 80)
    print("Starting complete calculation workflow...")
    print("=" * 80)

    # Step 1: CSP
    csp = CSPFilter()
    feasible_combos = csp.run()

    # Step 2: 性能
    perf_calc = PerformanceCalculator(csp)
    performance_results = perf_calc.calculate_all(feasible_combos)

    # Step 3: Pareto
    pareto = ParetoOptimizer(performance_results)
    pareto.non_dominated_sort()

    # 推荐方案（加权归一化）
    best = None
    pareto_solutions = [performance_results[i] for i in pareto.fronts[0]]
    if pareto_solutions:
        costs = np.array([p['cost'] for p in pareto_solutions], dtype=float)
        brakes = np.array([p['braking_distance'] for p in pareto_solutions], dtype=float)

        norm_c = (costs.max() - costs) / (costs.max() - costs.min() + 1e-12)
        norm_d = (brakes.max() - brakes) / (brakes.max() - brakes.min() + 1e-12)
        scores = 0.80 * norm_c + 0.20 * norm_d
        best_idx = int(np.argmax(scores))
        best = pareto_solutions[best_idx]
        best['score'] = scores[best_idx] * 100

        print("\n" + "=" * 60)
        print("Recommended Solution (Weighted Score):")
        print("=" * 60)
        for sub_name, opt in best['combo'].items():
            print(f"  {sub_name}: {opt}")
        print(f"\n  Cost: {best['cost']} (10k CNY)")
        print(f"  Braking Distance: {best['braking_distance']:.1f} m")
        print(f"  Score: {best['score']:.4f}")

    # Step 4: L0-L5 耦合网络
    analyzer = CouplingNetworkAnalyzer()

    # Step 5: 可视化
    visualize(analyzer)

    # Step 6: 附录数据导出
    export_appendix_data(csp, feasible_combos, performance_results, pareto, best, analyzer)

    # 汇总
    print("\n" + "=" * 80)
    print("System Output Summary:")
    print("=" * 80)
    print(f"  Feasible architectures: {len(feasible_combos)}")
    print(f"  Pareto optimal solutions: {len(pareto_solutions)}")
    print(f"  Coupling paths: {len(analyzer.path_keys)}")
    print(f"  Residual matrix: {analyzer.R.shape}")
    print(f"  R non-zero elements: {analyzer.R_nonzero}")
    print(f"  Network edges: {analyzer.edge_count}")
    print(f"  Positive feedback loops: {len(analyzer.feedback_loops)}")
    print(f"  Latent coupling predictions: {len(analyzer.latent_predictions)}")
    print(f"  Network stability: Ω = {analyzer.stability_index:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()