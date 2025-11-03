import json
import numpy as np
from tqdm import tqdm
from csm import CSM
from csm_display import axis_angle_from_vectors, calculate_angular_velocity, get_random_target, normalize_vector, load_workspace_data

success_data_path = "./data/successes_play_2.json"
failure_data_path = "./data/failures_play_2.json"

def log(cnt, succ):
    tqdm.write(f"Finished {cnt} targets, {succ} successes, {cnt - succ} failures")

if __name__ == "__main__":
    step_count = 0
    target_cnt = 1
    succ_cnt = 0
    max_steps = 8000
    total_targets = 2500  # 总目标数
    failures = []
    successes = []
    delta_t = 0.001

    csm = CSM(0.04, 0.06, 0.02, 0.15, np.pi/2, 2*np.pi/3, delta_t)
    v_lim = 0.2
    w_lim = 2
    workspace_data = load_workspace_data("./data/workspace_data.json")
    mode, pose, label_config = get_random_target(workspace_data)
    csm.target_pose = pose
    print("space:", mode, "target:", pose)
    raw_mode = 0

    try:
        with tqdm(total=total_targets, desc="Processing Targets", unit="target") as pbar:
            while target_cnt <= total_targets:
                csm.check_transition()
                csm.update()
                csm.update_jacobians()

                v = normalize_vector(csm.target_pose[:3] - csm.pose[:3]) * v_lim
                axis_hat, _ = axis_angle_from_vectors(csm.pose[3:], csm.target_pose[3:])
                w = w_lim * axis_hat
                csm.get_dot_PHI(v, w)
                csm.step()

                step_count += 1

                # 成功到达目标
                if np.linalg.norm(csm.pose - csm.target_pose) < 1e-3:
                    raw_mode, new_target_pose, label_config = get_random_target(workspace_data)
                    csm.target_pose = new_target_pose

                    success_data = {
                        "id": target_cnt,
                        "mode": csm.mode,
                        "steps_taken": step_count
                    }
                    successes.append(success_data)
                    step_count = 0
                    target_cnt += 1
                    succ_cnt += 1
                    pbar.update(1)  # 更新进度条

                    if target_cnt % 200 == 0:
                        log(target_cnt, succ_cnt)

                # 超过最大步数则记录失败
                elif step_count > max_steps:
                    failure_data = {
                        "id": target_cnt,
                        "target_pose": csm.target_pose.tolist(),
                        "mode": csm.mode,
                        "phi": csm.phi,
                        "theta_1": csm.theta_1,
                        "theta_2": csm.theta_2,
                        "delta_1": csm.delta_1,
                        "delta_2": csm.delta_2,
                        "L1": csm.L1,
                        "L2": csm.L2,
                        "Ls": csm.Ls,
                        "Lr": csm.Lr,
                        "steps_taken": step_count,
                        "true_mode": raw_mode,
                        "label_config": label_config  # 参考正确目标配置
                    }
                    failures.append(failure_data)
                    csm.reset()
                    raw_mode, new_target_pose, label_config = get_random_target(workspace_data)
                    csm.target_pose = new_target_pose
                    step_count = 0
                    target_cnt += 1
                    pbar.update(1)  # 更新进度条

                    if target_cnt % 200 == 0:
                        log(target_cnt, succ_cnt)

        # 保存结果
        if successes:
            with open(success_data_path, "w") as f:
                json.dump(successes, f, indent=4)
            print(f"Recorded {len(successes)} successes to {success_data_path}")

        if failures:
            with open(failure_data_path, "w") as f:
                json.dump(failures, f, indent=4)
            print(f"Recorded {len(failures)} failures to {failure_data_path}")

    except KeyboardInterrupt:
        print("Interrupted by user.")
        if successes:
            with open(success_data_path, "w") as f:
                json.dump(successes, f, indent=4)
            print(f"Recorded {len(successes)} successes to {success_data_path}")
        if failures:
            with open(failure_data_path, "w") as f:
                json.dump(failures, f, indent=4)
            print(f"Recorded {len(failures)} failures to {failure_data_path}")
