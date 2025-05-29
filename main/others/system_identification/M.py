import csv
import random
import time

# ランダム選択の候補となるデータ
data_options = [
    [(1, -1, 0, 0), (-1, 1, 0, 0)],  # nのデータ
    [(0, 0, 1, -1), (0, 0, -1, 1)],  # n+1のデータ
    #[(1, 1, 1, 1), (-1, -1, -1, -1)]  # n+2のデータ
]

# CSVファイルの準備
with open("random_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # writer.writerow(["X1", "X2", "Y1", "Y2"])  # ヘッダー行を追加する場合はこちらの行を有効に

    index = 0  # データ順序のインデックス
    row_count = 0  # 書き込んだ行数を管理

    # 600行書き込むまでデータをランダムで選び、30Hzで出力
    try:
        while row_count < 60:
            # 順番にaとbのうちどれかをランダムに選択
            data = random.choice(data_options[index % 2])

            # CSVに書き込み
            writer.writerow(data)
            csvfile.flush()  # 確実に書き込むためにフラッシュ

            # インデックスと行数を更新
            index += 1
            row_count += 1

            # 30Hz（約0.033秒）間隔での書き込み
            time.sleep(0.033)

        print("60行のデータを書き込んだためプログラムを終了します。")

    except KeyboardInterrupt:
        print("データの書き込みを終了します。")