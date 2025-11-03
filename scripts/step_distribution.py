import json
import threading
import numpy as np
from csm import CSM
import matplotlib.pyplot as plt

success_file = "./data/successes_play.json"
failure_file = "./data/failures_play.json"   # 假设失败数据在这个文件里
with open(success_file, 'r') as f:
    data = json.load(f)
with open(failure_file, 'r') as f:
    failures = json.load(f)

steps = [success["steps_taken"] for success in data]
# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(steps, bins=50, color='blue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Steps Taken for Successes')
plt.xlabel('Number of Steps')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
