import numpy as np
import csv

# A行列とM行列のデータを読み込む関数
def read_csv_to_matrix(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append([float(value) for value in row])
    return np.array(data)

# Bt.csvからデータを読み込み
filename = "BT.csv"
data = read_csv_to_matrix(filename)
print(f"Bt.csv 全データの形状: {data.shape}")

# A行列の定義（1列目から5列目まで、最初の9行を除外）
A = data[9:, 0:5]  # 行は10行目以降、列は1～5列
print(f"Aの形状: {A.shape}")

# M行列の定義（6列目のみ、最初の9行を除外）
M = data[9:, 5:6]  # 行は10行目以降、列は6列目のみ
print(f"Mの形状 (変更前): {M.shape}")

# M列に1/(35*√2)を掛ける
scaling_factor = 1 / (35 * np.sqrt(2))
M = M * scaling_factor
print(f"Mの形状 (変更後): {M.shape}")

# パラメータaの推定
A_T = A.T
A_T_A_pseudo_inv = np.linalg.pinv(np.dot(A_T, A))  # 擬似逆行列を使用
A_T_M = np.dot(A_T, M)
a_hat = np.dot(A_T_A_pseudo_inv, A_T_M)

print("推定された未知パラメータ a:")
print(a_hat)
