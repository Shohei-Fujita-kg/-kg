import os
import torch.nn as nn
import torch.optim as optim

# abc順(項目別, 自動設定される項目は別にする)

# 一般

## 手動設定
dataset_path = "dataset" # データセットのあるディレクトリ名
f_path = ".." # ipynbファイルのある場所からcodeのあるディレクトリまでのパス

## 自動設定
osname = os.name # os名
sep = os.linesep # 区切り文字

# スペクトログラム生成パラメータ
freqdim = 129 # 周波数の次元数
hoplength = 224 # 窓の移動幅のデータ数
mono = True # モノラル
nfft = 256 # 窓幅のデータ数
replaceinf = -50 # -inf[dB]を一定値[dB]に変換
sampling_rate = 16000 # サンプリングレート
timedim = 715 # 時間の次元数
window = "hann" # 窓関数

# 学習時のパラメータ

## 手動設定
batchsize = 1 # 学習時のバッチサイズ
end_factor = 0.0001 # 学習時最終学習率
epoch = 20 # モデル学習時の学習回数
start_factor = 0.001 # 学習時の初期学習率

## 自動設定
criterion = nn.CrossEntropyLoss() # 損失関数


