import csv

# 定数Rを定義（適宜設定してください）
R = 1.0  # 必要に応じて適切な値に変更

# M1, M2, M3を計算する関数
def calculate_M(u1, u3, u2, u4):
    M1 = (R / (2 ** 0.5)) * (u3 - u1)
    M2 = (R / (2 ** 0.5)) * (u4 - u2)
    M3 = (R / (2 ** 0.5)) * (u1 + u2 + u3 + u4)
    return [M1, M2, M3]

# 入力ファイルと出力ファイルの準備
input_file = "random_data.csv"
output_file = "calculated_data_vertical.csv"

# データの読み込みと計算結果の書き込み
with open(input_file, "r") as csvfile_in, open(output_file, "w", newline="") as csvfile_out:
    reader = csv.reader(csvfile_in)
    writer = csv.writer(csvfile_out)
    
    # 入力データを1行ずつ読み込み、calculate_M関数で計算
    for row in reader:
        u1, u2, u3, u4 = map(float, row)  # 文字列から浮動小数点数に変換
        M_values = calculate_M(u1, u2, u3, u4)
        
        # 計算結果を1つずつ1列に書き込み
        for M in M_values:
            writer.writerow([M])

print(f"{output_file}ファイルに計算結果を書き込みました．")
