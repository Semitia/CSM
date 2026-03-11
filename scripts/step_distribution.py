"""
Module: step_distribution.py
Description: Script to plot the distribution of steps taken for successful trials.
"""
import json
import threading
import numpy as np
from csm.model import CSM
import matplotlib.pyplot as plt

success_file = "./data/successes_play.json"
failure_file = "./data/failures_play.json"   # 假设失败数据在这个文件里
with open(success_file, 'r') as f:
    successes = json.load(f)
with open(failure_file, 'r') as f:
    failures = json.load(f)

# ====== 统计数据 ======
success_count = len(successes)
failure_count = len(failures)
total_count = success_count + failure_count
success_rate = success_count / total_count * 100 if total_count > 0 else 0.0
print(f"count of success: {success_count}")
print(f"count of failure: {failure_count}")
print(f"total count: {total_count}")
print(f"success rate: {success_rate:.2f}%")

# ====== 步数分布统计 ======
steps = [success["steps_taken"] for success in successes]
plt.figure(figsize=(10, 6))
plt.hist(steps, bins=50, color='blue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Steps Taken for Successes')
plt.xlabel('Number of Steps')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
