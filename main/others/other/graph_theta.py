import matplotlib.pyplot as plt
import csv
import numpy as np

# CSVファイルの読み込み
time_stamps = []
error_x_values = []
error_y_values = []
theta_values = []

with open("errorall2.csv", mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # ヘッダーをスキップ
    for row in reader:
        time_stamps.append(float(row[0]))
        error_x_values.append(float(row[1]))
        error_y_values.append(float(row[2]))
        theta_values.append(float(row[3]))

# error_xのグラフ
plt.figure(figsize=(10, 6))
plt.plot(time_stamps[19:], error_x_values[19:], label="Error X", color='b', linestyle='-', linewidth=2)

# 最大値と最小値を取得
max_error_x = max(error_x_values[19:])
min_error_x = min(error_x_values[19:])

# 最大値と最小値をグラフに表示
plt.axhline(y=max_error_x, color='b', linestyle='--', label=f"Max Error X: {max_error_x:.2f}")
plt.axhline(y=min_error_x, color='b', linestyle='--', label=f"Min Error X: {min_error_x:.2f}")

# グラフのタイトルとラベル
plt.title("Error X vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Error X (mm)")
plt.legend()
plt.grid(True)

# error_yのグラフ
plt.figure(figsize=(10, 6))
plt.plot(time_stamps[19:], error_y_values[19:], label="Error Y", color='r', linestyle='-', linewidth=2)

# 最大値と最小値を取得
max_error_y = max(error_y_values[19:])
min_error_y = min(error_y_values[19:])

# 最大値と最小値をグラフに表示
plt.axhline(y=max_error_y, color='r', linestyle='--', label=f"Max Error Y: {max_error_y:.2f}")
plt.axhline(y=min_error_y, color='r', linestyle='--', label=f"Min Error Y: {min_error_y:.2f}")

# グラフのタイトルとラベル
plt.title("Error Y vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Error Y (mm)")
plt.legend()
plt.grid(True)

# thetaのグラフ（ラジアンから度に変換してプロット）
theta_degrees = [theta * 180 / np.pi for theta in theta_values]

plt.figure(figsize=(10, 6))
plt.plot(time_stamps[19:], theta_degrees[19:], label="Theta (Angle)", color='g', linestyle='-', linewidth=2)

# 最大値と最小値を取得
max_theta = max(theta_degrees[19:])
min_theta = min(theta_degrees[19:])

# 最大値と最小値をグラフに表示
plt.axhline(y=max_theta, color='g', linestyle='--', label=f"Max Theta: {max_theta:.2f}°")
plt.axhline(y=min_theta, color='g', linestyle='--', label=f"Min Theta: {min_theta:.2f}°")

# グラフのタイトルとラベル
plt.title("Theta (Angle) vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Theta (°)")
plt.legend()
plt.grid(True)

# グラフを表示
plt.show()

