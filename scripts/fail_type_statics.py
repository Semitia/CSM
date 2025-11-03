import json
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

# ==== 参数 ====
ANGLE_THRESHOLD = 0.02  # rad
LENGTH_THRESHOLD = 0.0001  # 0.1 mm

# 假定结构参数上下限（论文 Fig. 4）
LIMITS = {
    "theta_1": (-np.pi, np.pi),
    "theta_2": (-np.pi, np.pi),
    "phi": (-np.pi, np.pi),
    "delta_1": (-np.pi, np.pi),
    "delta_2": (-np.pi, np.pi),
    "L1": (0.0, 0.04),
    "L2": (0.0, 0.06),
    "Ls": (0.0, 0.02),
    "Lr": (0.0, 0.02),
}

# ==== 加载数据 ====
with open("failures_play.json", "r") as f:
    failures = json.load(f)

results = []
limit_cases = []
disconnected_cases = []

# ==== 遍历失败样本 ====
for fcase in failures:
    near_limits = []
    for key, (low, high) in LIMITS.items():
        val = fcase.get(key, None)
        if val is None:
            continue
        # 判断接近下限或上限
        if abs(val - low) < LENGTH_THRESHOLD or abs(val - high) < LENGTH_THRESHOLD:
            near_limits.append(key)
        elif key.startswith("theta") or key.startswith("delta") or key == "phi":
            if abs(val - low) < ANGLE_THRESHOLD or abs(val - high) < ANGLE_THRESHOLD:
                near_limits.append(key)

    if near_limits:
        results.append({"id": fcase["id"], "type": "range_limit", "vars": near_limits})
        limit_cases.append(near_limits)
    else:
        results.append({"id": fcase["id"], "type": "disconnected_orientation"})
        disconnected_cases.append(fcase["id"])

# ==== 统计 Table II 风格 ====
flat = list(itertools.chain.from_iterable(limit_cases))
var_names = list(LIMITS.keys())

# 统计上下限命中次数
stats = {v: flat.count(v) for v in var_names}
df = pd.DataFrame.from_dict(stats, orient="index", columns=["# of cases"]).sort_values("# of cases", ascending=False)

# ==== 输出结果 ====
print("📊 Range-limit cases:", len(limit_cases))
print("🌀 Disconnected-orientation cases:", len(disconnected_cases))
print("\n📋 Table II-style summary:")
print(df)

# ==== 可视化 ====
plt.figure(figsize=(8,4))
plt.bar(df.index, df["# of cases"])
plt.xticks(rotation=45)
plt.ylabel("Number of Cases")
plt.title("Statistics of Encountered Variable Limits (Table II style)")
plt.tight_layout()
plt.show()
