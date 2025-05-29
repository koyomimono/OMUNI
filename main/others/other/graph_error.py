import matplotlib.pyplot as plt
import csv

# CSVファイルの読み込み
time_stamps = []
error_x_values = []
error_y_values = []

with open("error.csv", mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # ヘッダーをスキップ
    for row in reader:
        time_stamps.append(float(row[0]))
        error_x_values.append(float(row[1]))
        error_y_values.append(float(row[2]))

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

# グラフを表示
plt.show()

