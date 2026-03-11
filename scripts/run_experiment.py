"""
Module: run_experiment.py
Description: Main script to run the experiment, generating random targets and tracking success/failure.
"""
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from csm.model import CSM
from csm.utils import axis_angle_from_vectors, normalize_vector, load_workspace_data, get_random_target

success_data_path = "./data/successes_play.json"
failure_data_path = "./data/failures_play.json"

def log(cnt, succ):
    tqdm.write(f"Finished {cnt} targets, {succ} successes, {cnt - succ} failures")

if __name__ == "__main__":
    step_count = 0
    target_cnt = 1
    succ_cnt = 0
    max_steps = 8000
    total_targets = 1000
    failures = []
    successes = []
    delta_t = 0.001

    config_path = Path("./config/csm_config1.yaml")
    csm = CSM.from_config(config_path)

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

                if np.linalg.norm(csm.pose - csm.target_pose) < 1e-3:
                    raw_mode, new_target_pose, label_config = get_random_target(workspace_data)
                    csm.target_pose = new_target_pose
                    successes.append({"id": target_cnt, "mode": csm.mode, "steps_taken": step_count})
                    step_count = 0
                    target_cnt += 1
                    succ_cnt += 1
                    pbar.update(1)
                    if target_cnt % 200 == 0:
                        log(target_cnt, succ_cnt)

                elif step_count > max_steps:
                    failures.append({
                        "id": target_cnt, "target_pose": csm.target_pose.tolist(),
                        "mode": csm.mode, "phi": csm.phi,
                        "theta_1": csm.theta_1, "theta_2": csm.theta_2,
                        "delta_1": csm.delta_1, "delta_2": csm.delta_2,
                        "L1": csm.L1, "L2": csm.L2, "Ls": csm.Ls, "Lr": csm.Lr,
                        "steps_taken": step_count, "true_mode": raw_mode, "label_config": label_config
                    })
                    csm.reset()
                    raw_mode, new_target_pose, label_config = get_random_target(workspace_data)
                    csm.target_pose = new_target_pose
                    step_count = 0
                    target_cnt += 1
                    pbar.update(1)
                    if target_cnt % 200 == 0:
                        log(target_cnt, succ_cnt)

        # 保存结果
        if successes:
            with open(success_data_path, "w") as f:
                json.dump(successes, f, indent=4)
        if failures:
            with open(failure_data_path, "w") as f:
                json.dump(failures, f, indent=4)

    except KeyboardInterrupt:
        print("Interrupted by user.")
        if successes:
            with open(success_data_path, "w") as f:
                json.dump(successes, f, indent=4)
        if failures:
            with open(failure_data_path, "w") as f:
                json.dump(failures, f, indent=4)
