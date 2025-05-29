# client.py
import socket

# サーバのIPアドレスとポート番号を指定
HOST = '192.168.50.9'  # Raspberry PiのIPアドレスに置き換える
PORT = 50000              # サーバで指定したポート番号

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))  # サーバに接続
    while True:
        message = input("送信データを入力してください: ")
        s.sendall(message.encode())  # データを送信
        if message.lower() == 'exit':  # 'exit'と入力で終了
            break
