import json
import numpy as np
from csm.model import CSM
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from csm.utils import axis_angle_from_vectors, normalize_vector, load_workspace_data, get_random_target

def log(cnt, succ):
    print(f"Finished {cnt} targets, {succ} successes, {cnt - succ} failures")


def load_log_state(csm, data):
    csm.mode = data["mode"]
    csm.phi = data["phi"]
    csm.theta_1 = data["theta_1"]
    csm.theta_2 = data["theta_2"]
    csm.delta_1 = data["delta_1"]
    csm.delta_2 = data["delta_2"]
    csm.L1 = data["L1"]
    csm.L2 = data["L2"]
    csm.Ls = data["Ls"]
    csm.Lr = data["Lr"]
    csm.target_pose = data["target_pose"]

if __name__ == "__main__":
    # 统计步数和失败目标的列表
    step_count = 0
    target_cnt = 1
    succ_cnt = 0
    max_steps = 800  # 设置达到目标的最大步数
    failures = []
    successes = []
    csm = CSM(0.5, 0.5, 0.15, 0.15, 0.01)
    filename = "failures_play.json"
    with open(filename, 'r') as f:
        data = json.load(f)
    print("failure_len:", len(data))
    failure = data[0]
    load_log_state(csm, failure)

    try:
        while True:
            if target_cnt >= 78:
                break
            csm.check_transition()
            csm.update()
            csm.update_jacobians()

            v = normalize_vector(csm.target_pose[:3] - csm.pose[:3]) * 4
            w = calculate_angular_velocity(csm.pose[3:], csm.target_pose[3:], 0.5)
            csm.get_dot_PHI(v, w)
            csm.step()

            step_count += 1
            if np.linalg.norm(csm.pose - csm.target_pose) < 0.05:
                print("Reached target")
                new_log = data[target_cnt]
                load_log_state(csm, new_log)
                
                # 记录成功目标和当前的各个参数
                success_data = {
                    "id": target_cnt,
                    "mode": csm.mode,
                    "steps_taken": step_count
                }
                successes.append(success_data)
                step_count = 0  # 重置步数统计
                target_cnt += 1
                succ_cnt += 1
                # print("space:", mode, "target:", new_target_pose)
                if target_cnt % 200 == 0:
                    log(target_cnt, succ_cnt)
                if target_cnt >= 10000:
                    break
            elif step_count > max_steps:
                print("Failed to reach target after", step_count, "steps")
                # 记录失败目标和当前的各个参数
                failure_data = {
                    "id": target_cnt,
                    "target_pose": csm.target_pose,
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
                    "steps_taken": step_count
                }
                failures.append(failure_data)
                csm.reset()
                new_log = data[target_cnt]
                load_log_state(csm, new_log)

                step_count = 0  # 重置步数统计
                target_cnt += 1
                # print("space:", mode, "target:", new_target_pose)
                if target_cnt % 200 == 0:
                    log(target_cnt, succ_cnt)

        if successes:
            with open("successes_replay.json", "w") as f:
                json.dump(successes, f, indent=4)
            print(f"Recorded {len(successes)} successes to successes.json")
        
        if failures:
            with open("failures_replay.json", "w") as f:
                json.dump(failures, f, indent=4)
            print(f"Recorded {len(failures)} failures to failures.json")


    except KeyboardInterrupt:
        print("Interrupted")
        # 保存成功和失败数据
        if successes:
            with open("successes_replay.json", "w") as f:
                json.dump(successes, f, indent=4)
            print(f"Recorded {len(successes)} successes to successes.json")
        
        if failures:
            with open("failures_replay.json", "w") as f:
                json.dump(failures, f, indent=4)
            print(f"Recorded {len(failures)} failures to failures.json")
