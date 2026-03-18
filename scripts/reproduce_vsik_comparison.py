"""
Module: reproduce_vsik_comparison.py
Description: 批量对比实验 - VS-IK vs Jacobian-DLS
复现论文中的对比实验，验证 VS-IK 方法的计算效率与成功率
"""
import numpy as np
import json
import time
import yaml
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from csm import CSM, VSIKSolver, CSMParameters, compute_fk
from csm.utils import normalize_vector, axis_angle_from_vectors


@dataclass
class IKResult:
    """IK 求解结果"""
    success: bool
    method: str
    solve_time: float  # 秒
    iterations: int
    final_error_pos: float  # 位置误差 (m)
    final_error_ori: float  # 方向误差 (rad)
    config: Optional[Dict] = None
    error_msg: str = ""


def sample_random_pose(params: CSMParameters, mode: str = 'ci1') -> Tuple[np.ndarray, np.ndarray]:
    """
    随机采样一个可达位姿
    
    参数:
        params: 机器人参数
        mode: 'ci1' 或 'ci2'
        
    返回:
        (位置, 方向向量)
    """
    if mode == 'ci1':
        # CI-1: L_s = 0, 两段都可弯曲
        L1 = np.random.uniform(0, params.L_10 * 0.8)
        L2 = params.L_20
    else:
        # CI-2: L_s > 0, 第一段不可伸长
        L1 = params.L_10
        L2 = params.L_20
    
    theta_1 = np.random.uniform(0, params.theta1_max * 0.8)
    theta_2 = np.random.uniform(0, params.theta2_max * 0.8)
    delta_1 = np.random.uniform(0, 2 * np.pi)
    delta_2 = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    
    if mode == 'ci1':
        L_s = 0
    else:
        L_s = np.random.uniform(0, params.L_s0 * 0.8) if params.L_s0 > 0 else 0
    
    # 创建 CSM 实例并计算 FK
    csm = CSM(
        L_10=params.L_10,
        L_20=params.L_20,
        L_r0=params.L_r0,
        L_s0=params.L_s0,
        L_tool=params.L_tool,
        theta1_max=params.theta1_max,
        theta2_max=params.theta2_max
    )
    
    # 设置模式
    csm_mode = 3 if mode == 'ci1' else 4
    csm.set_state(
        mode=csm_mode,
        phi=phi,
        L1=L1,
        L2=L2,
        Lr=params.L_r0,
        Ls=L_s,
        theta_1=theta_1,
        theta_2=theta_2,
        delta_1=delta_1,
        delta_2=delta_2
    )
    
    return csm.pose[:3], csm.pose[3:]


def solve_vsik(params: CSMParameters, target_pose: Tuple[np.ndarray, np.ndarray], 
                mode: str = 'ci1') -> IKResult:
    """
    使用 VS-IK 方法求解
    """
    start_time = time.perf_counter()
    
    solver = VSIKSolver(params)
    
    try:
        result = solver.solve(target_pose, mode=mode)
        
        solve_time = time.perf_counter() - start_time
        
        if result is not None:
            # 验证 FK
            fk_pos, fk_dir = compute_fk(params, result)
            
            error_pos = np.linalg.norm(fk_pos - target_pose[0])
            error_ori = np.arccos(np.clip(np.dot(fk_dir, target_pose[1]), -1, 1))
            
            return IKResult(
                success=True,
                method='VS-IK',
                solve_time=solve_time,
                iterations=1,  # VS-IK 是解析求解，不是迭代
                final_error_pos=error_pos,
                final_error_ori=error_ori,
                config=result
            )
        else:
            return IKResult(
                success=False,
                method='VS-IK',
                solve_time=solve_time,
                iterations=0,
                final_error_pos=-1,
                final_error_ori=-1,
                error_msg="No solution found"
            )
            
    except Exception as e:
        return IKResult(
            success=False,
            method='VS-IK',
            solve_time=time.perf_counter() - start_time,
            iterations=0,
            final_error_pos=-1,
            final_error_ori=-1,
            error_msg=str(e)
        )


def solve_jacobian_dls(csm: CSM, target_pose: np.ndarray, 
                       max_steps: int = 500, 
                       v_lim: float = 0.04,
                       w_lim: float = 2.0) -> IKResult:
    """
    使用 Jacobian-DLS 方法求解 (论文中的对比基线)
    """
    start_time = time.perf_counter()
    
    csm.target_pose = target_pose.copy()
    step_count = 0
    
    try:
        for step in range(max_steps):
            csm.check_transition()
            csm.update()
            csm.update_jacobians()
            
            # 计算控制输入
            v = normalize_vector(csm.target_pose[:3] - csm.pose[:3]) * v_lim
            axis_hat, _ = axis_angle_from_vectors(csm.pose[3:], csm.target_pose[3:])
            w = w_lim * axis_hat
            
            csm.get_dot_PHI(v, w)
            csm.step()
            
            step_count += 1
            
            # 检查收敛
            error_pos = np.linalg.norm(csm.pose[:3] - csm.target_pose[:3])
            error_ori = np.arccos(np.clip(np.dot(csm.pose[3:], csm.target_pose[3:]), -1, 1))
            
            if error_pos < 1e-3 and error_ori < 1e-2:
                solve_time = time.perf_counter() - start_time
                return IKResult(
                    success=True,
                    method='Jacobian-DLS',
                    solve_time=solve_time,
                    iterations=step_count,
                    final_error_pos=error_pos,
                    final_error_ori=error_ori,
                    config={
                        'mode': csm.mode,
                        'phi': csm.phi,
                        'theta_1': csm.theta_1,
                        'theta_2': csm.theta_2,
                        'delta_1': csm.delta_1,
                        'delta_2': csm.delta_2,
                        'L1': csm.L1,
                        'L2': csm.L2,
                        'Lr': csm.Lr,
                        'Ls': csm.Ls
                    }
                )
        
        # 超时
        solve_time = time.perf_counter() - start_time
        error_pos = np.linalg.norm(csm.pose[:3] - csm.target_pose[:3])
        error_ori = np.arccos(np.clip(np.dot(csm.pose[3:], csm.target_pose[3:]), -1, 1))
        
        return IKResult(
            success=False,
            method='Jacobian-DLS',
            solve_time=solve_time,
            iterations=step_count,
            final_error_pos=error_pos,
            final_error_ori=error_ori,
            error_msg=f"Max iterations ({max_steps}) reached"
        )
        
    except Exception as e:
        return IKResult(
            success=False,
            method='Jacobian-DLS',
            solve_time=time.perf_counter() - start_time,
            iterations=step_count,
            final_error_pos=-1,
            final_error_ori=-1,
            error_msg=str(e)
        )


def run_comparison_experiment(config_path: str = "./config/csm_cfg_3.4mm.yaml",
                             num_cases: int = 100,
                             mode: str = 'ci1',
                             output_dir: str = "./data") -> Dict:
    """
    运行对比实验
    
    参数:
        config_path: 配置文件路径
        num_cases: 测试案例数量
        mode: 'ci1' 或 'ci2'
        output_dir: 输出目录
        
    返回:
        统计结果字典
    """
    print("=" * 60)
    print(f"运行对比实验: VS-IK vs Jacobian-DLS")
    print(f"模式: {mode}, 案例数: {num_cases}")
    print("=" * 60)
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    params = CSMParameters.from_config(config)
    
    # 创建 CSM 实例 (用于 Jacobian-DLS)
    csm = CSM(
        L_10=params.L_10,
        L_20=params.L_20,
        L_r0=params.L_r0,
        L_s0=params.L_s0,
        L_tool=params.L_tool,
        theta1_max=params.theta1_max,
        theta2_max=params.theta2_max
    )
    
    # 存储结果
    vsik_results = []
    jaco_results = []
    
    print(f"\n生成 {num_cases} 个测试案例...")
    
    # 生成测试案例
    test_cases = []
    for i in range(num_cases):
        target_pos, target_dir = sample_random_pose(params, mode)
        target_pose = np.concatenate([target_pos, target_dir])
        test_cases.append(target_pose)
    
    print(f"开始对比实验...\n")
    
    # 运行 VS-IK
    print("运行 VS-IK 求解...")
    for i, target_pose in enumerate(test_cases):
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{num_cases}")
        
        result = solve_vsik(params, (target_pose[:3], target_pose[3:]), mode=mode)
        vsik_results.append(asdict(result))
    
    # 运行 Jacobian-DLS
    print("\n运行 Jacobian-DLS 求解...")
    for i, target_pose in enumerate(test_cases):
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{num_cases}")
        
        csm.reset()
        result = solve_jacobian_dls(csm, target_pose)
        jaco_results.append(asdict(result))
    
    # 统计结果
    print("\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)
    
    # VS-IK 统计
    vsik_success = sum(1 for r in vsik_results if r['success'])
    vsik_times = [r['solve_time'] for r in vsik_results]
    vsik_avg_time = np.mean(vsik_times) if vsik_times else 0
    vsik_errors = [r['final_error_pos'] for r in vsik_results if r['success'] and r['final_error_pos'] > 0]
    vsik_avg_error = np.mean(vsik_errors) if vsik_errors else 0
    
    print(f"\nVS-IK ({mode}):")
    print(f"  成功率: {vsik_success}/{num_cases} ({vsik_success/num_cases*100:.1f}%)")
    print(f"  平均求解时间: {vsik_avg_time*1000:.2f} ms")
    print(f"  平均位置误差: {vsik_avg_error*1000:.4f} mm")
    
    # Jacobian-DLS 统计
    jaco_success = sum(1 for r in jaco_results if r['success'])
    jaco_times = [r['solve_time'] for r in jaco_results]
    jaco_avg_time = np.mean(jaco_times) if jaco_times else 0
    jaco_errors = [r['final_error_pos'] for r in jaco_results if r['success'] and r['final_error_pos'] > 0]
    jaco_avg_error = np.mean(jaco_errors) if jaco_errors else 0
    jaco_iterations = [r['iterations'] for r in jaco_results if r['success']]
    jaco_avg_iterations = np.mean(jaco_iterations) if jaco_iterations else 0
    
    print(f"\nJacobian-DLS ({mode}):")
    print(f"  成功率: {jaco_success}/{num_cases} ({jaco_success/num_cases*100:.1f}%)")
    print(f"  平均求解时间: {jaco_avg_time*1000:.2f} ms")
    print(f"  平均位置误差: {jaco_avg_error*1000:.4f} mm")
    print(f"  平均迭代次数: {jaco_avg_iterations:.1f}")
    
    # 对比
    print(f"\n对比总结:")
    print(f"  成功率提升: {(vsik_success - jaco_success)/max(jaco_success, 1)*100:+.1f}%")
    if jaco_avg_time > 0:
        print(f"  时间效率提升: {(1 - vsik_avg_time/jaco_avg_time)*100:+.1f}%")
    
    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    result_data = {
        'config': {
            'mode': mode,
            'num_cases': num_cases,
            'config_path': config_path
        },
        'vsik': {
            'success_rate': vsik_success / num_cases,
            'avg_time': vsik_avg_time,
            'avg_error': vsik_avg_error,
            'results': vsik_results
        },
        'jacobian_dls': {
            'success_rate': jaco_success / num_cases,
            'avg_time': jaco_avg_time,
            'avg_error': jaco_avg_error,
            'avg_iterations': jaco_avg_iterations,
            'results': jaco_results
        }
    }
    
    output_file = output_path / f"vsik_comparison_{mode}.json"
    with open(output_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    return result_data


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='VS-IK vs Jacobian-DLS 对比实验')
    parser.add_argument('--config', type=str, default='./config/csm_cfg_3.4mm.yaml',
                       help='配置文件路径')
    parser.add_argument('--num-cases', type=int, default=100,
                       help='测试案例数量')
    parser.add_argument('--mode', type=str, default='ci1', choices=['ci1', 'ci2'],
                       help='配置模式')
    parser.add_argument('--output', type=str, default='./data',
                       help='输出目录')
    
    args = parser.parse_args()
    
    run_comparison_experiment(
        config_path=args.config,
        num_cases=args.num_cases,
        mode=args.mode,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
