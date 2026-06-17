import re
import os
import random
import numpy as np
import pandas as pd
import csv
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# Configクラスの定義
class Config:
    def __init__(self,
                 base_data_path="/~自分の環境に合わせて変更~/speechai/data",
                 datasets_folder="datasets",
                 csv_folder="csv",
                 models_folder="models",
                 vctk_dataset_folder="VCTK-Corpus-0.92",
                 wav_relative_path="wav48_silence_trimmed",
                 sample_rate=16000,
                 n_mels=64,
                 n_fft=1024,
                 hop_length=512,
                 hidden_size=128,
                 num_layers=2,
                 dropout=0.3,
                 num_epochs=20,
                 batch_size=16,
                 learning_rate=1e-3):
        # ディレクトリ・パスの定義
        self.base_data_path = base_data_path
        self.datasets_folder = datasets_folder
        self.csv_folder = csv_folder
        self.models_folder = models_folder
        self.vctk_dataset_folder = vctk_dataset_folder
        self.wav_relative_path = wav_relative_path

        # 依存するパスを更新
        self.update_paths()

        # 音声処理用の定数
        self.sample_rate = sample_rate # 音声を読み込む際のサンプリング周波数(Hz)
        self.n_mels = n_mels # メルスペクトログラム計算時のメルフィルタバンク数
        self.n_fft = n_fft # STFTの窓関数のサイズ
        self.hop_length = hop_length # ホップ長(オーバーラップ制御)

        # モデルのハイパーパラメータ
        self.hidden_size = hidden_size # 隠れ層の次元数
        self.num_layers = num_layers # LSTM層のスタック数
        self.dropout = dropout # ドロップアウト率

        # 学習パラメータ
        self.num_epochs = num_epochs # エポック数
        self.batch_size = batch_size # バッチサイズ
        self.learning_rate = learning_rate # 学習率

        # その他()
        self.label2index = "" # collate_fnで使用

    def update_paths(self):
        """
        基本パスに依存する変数を再計算する。
        例えば、datasets_folder のみ外部から変更した場合など、このメソッドを呼び出すことで全体のパスが更新される。
        """
        self.datasets_path = os.path.join(self.base_data_path, self.datasets_folder)
        self.csv_path = os.path.join(self.base_data_path, self.csv_folder)
        self.models_path = os.path.join(self.base_data_path, self.models_folder)
        self.vctk_dataset_path = os.path.join(self.datasets_path, self.vctk_dataset_folder)
        self.wav_path = os.path.join(self.vctk_dataset_path, self.wav_relative_path)

        # CSVやモデルファイルのパス
        self.vctkinfo_csv = os.path.join(self.csv_path, "vctkinfo.csv")
        self.vctk_data_csv = os.path.join(self.csv_path, "vctk_data.csv")
        self.model_file = os.path.join(self.models_path, "best_model.pth")

# グローバルインスタンス
config = Config()

# VCTK情報テキストをCSVに変換する関数
def convert_vctkinfo(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    header = re.split(r'\s{2,}', lines[0].strip()) # 先頭行をヘッダーとする（2個以上のスペースで区切る）
    rows = []
    for line in lines[1:]:
        columns = re.split(r'\s{2,}', line.strip()) # 2個以上の連続するスペースで分割する
        if len(columns) < len(header): # ヘッダーの列数に満たない場合は空文字列で補完する
            columns += [''] * (len(header) - len(columns))
        rows.append(columns)
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"変換を完了 : {output_file}")

# CSVからアクセント情報を取得する関数
def get_accents_from_csv(csv_file, target_id):
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['ID'] == target_id:
                return row['ACCENTS']
    return None

# VCTKデータから音声ファイル名とアクセント情報を抽出する関数
def get_audio_filename(data_item, speaker_csv, mic="mic1", ext="flac"):
    speaker_id = data_item[3]
    record_id = data_item[4]
    filename = os.path.join(config.datasets_folder, config.vctk_dataset_folder, config.wav_relative_path, speaker_id, f"{speaker_id}_{record_id}_{mic}.{ext}")
    accents = get_accents_from_csv(speaker_csv, speaker_id)
    return filename, accents

# 全項目の音声ファイル名とアクセント情報をcsvに書き出す関数
def process_all_data(data, speaker_csv, output_csv):
    output_rows = []
    for item in data:
        audio_filename, accent = get_audio_filename(item, speaker_csv)
        output_rows.append([audio_filename, accent])
    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "label"])  # ヘッダー
        writer.writerows(output_rows)

    print(f"{output_csv} に書き出し完了")

# 音声ファイルから特徴量を抽出する関数
def extract_features(file_path, sample_rate=config.sample_rate, n_mels=config.n_mels, n_fft=config.n_fft, hop_length=config.hop_length):
    """
    指定したファイルパスの音声を読み込み、メルスペクトログラムを計算し、対数変換と正規化を行う
    """
    full_path = os.path.join(config.base_data_path, file_path)
    y, sr = librosa.load(full_path, sr=sample_rate)
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max) # 対数変換
    norm_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-6) # 正規化
    return norm_spec

# データセットの定義
class AccentDataset(Dataset):
    def __init__(self, samples, sample_rate=config.sample_rate, n_mels=config.n_mels, n_fft=config.n_fft, hop_length=config.hop_length):
        self.samples = samples
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        spec = extract_features(file_path,
                                sample_rate=self.sample_rate,
                                n_mels=self.n_mels,
                                n_fft=self.n_fft,
                                hop_length=self.hop_length) # 特徴量抽出 [n_mels, time]
        spec = torch.tensor(spec, dtype=torch.float32).transpose(0, 1) # LSTM入力用に転置 [time, n_mels]
        return spec, label

def collate_fn(batch):
    """
    コラテーション関数(pad_sequenceを用いて時系列長を統一する)
    """
    specs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    lengths = torch.tensor([spec.shape[0] for spec in specs], dtype=torch.int64)  # 各サンプルの時刻数
    padded_specs = pad_sequence(specs, batch_first=True) # [batch, max_time, n_mels] となる
    label2index = config.label2index
    numeric_labels = []
    for label in labels:
        if label in label2index:
            numeric_labels.append(label2index[label])
    labels_tensor = torch.tensor(numeric_labels, dtype=torch.long)
    return padded_specs, lengths, labels_tensor

# LSTMモデルの定義
class AccentClassifierLSTM(nn.Module):
    def __init__(self, num_classes, sample_rate=config.sample_rate, n_mels=config.n_mels, n_fft=config.n_fft, hop_length=config.hop_length,
                 hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout):
        super(AccentClassifierLSTM, self).__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.lstm = nn.LSTM(input_size=n_mels,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def extract_features(self, file_path):
        # 単一サンプル用特徴量抽出
        spec = extract_features(file_path,
                                sample_rate=self.sample_rate,
                                n_mels=self.n_mels,
                                n_fft=self.n_fft,
                                hop_length=self.hop_length)
        return spec

    def forward(self, x, lengths=None):
        if isinstance(x, str): # 単一ファイルの場合：内部で特徴量抽出
            spec = self.extract_features(x)  # [n_mels, time]
            spec = torch.tensor(spec, dtype=torch.float32).transpose(0, 1).unsqueeze(0)  # [1, time, n_mels]
            lengths = torch.tensor([spec.size(1)], dtype=torch.int64)
            packed_x = pack_padded_sequence(spec, lengths, batch_first=True, enforce_sorted=False)
            output, (h_n, _) = self.lstm(packed_x)
            last_hidden = h_n[-1]  # [1, hidden_size]
            logits = self.fc(last_hidden)
            return logits
        else: # バッチ入力の場合
            if lengths is not None:
                packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
                output, (h_n, _) = self.lstm(packed_x)
            else:
                output, (h_n, _) = self.lstm(x)
            last_hidden = h_n[-1]  # 最終層の隠れ状態
            logits = self.fc(last_hidden) # クラス数に合わせる線形層
            return logits

# 学習・評価・テストの定義
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train() # 学習モードにする
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        specs, lengths, labels = batch
        specs = specs.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(specs, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            specs, lengths, labels = batch
            specs = specs.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            outputs = model(specs, lengths)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# データの読み込み・分割、学習・評価・テストを一括で実行する関数
def execute_pipeline(csvname):
    df = pd.read_csv(csvname)
    df = df[~df['file_path'].str.startswith('s5')] # アクセント情報がないためs5は使用しません
    samples = list(zip(df['file_path'].tolist(), df['label'].tolist()))
    random.shuffle(samples)

    n_total = len(samples)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    train_samples = samples[:n_train]
    val_samples   = samples[n_train:n_train+n_val]
    test_samples  = samples[n_train+n_val:]

    train_dataset = AccentDataset(train_samples, config.sample_rate, config.n_mels, config.n_fft, config.hop_length)
    val_dataset   = AccentDataset(val_samples, config.sample_rate, config.n_mels, config.n_fft, config.hop_length)
    test_dataset  = AccentDataset(test_samples, config.sample_rate, config.n_mels, config.n_fft, config.hop_length)

    unique_labels = df["label"].dropna().unique()
    config.label2index = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    # アクセントクラス数はcsv内のラベルの種類数から自動で計算
    num_classes = df['label'].nunique()

    # モデル・最適化手法・損失関数の設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AccentClassifierLSTM(num_classes=num_classes,
                                 sample_rate=config.sample_rate,
                                 n_mels=config.n_mels,
                                 n_fft=config.n_fft,
                                 hop_length=config.hop_length,
                                 hidden_size=config.hidden_size,
                                 num_layers=config.num_layers,
                                 dropout=config.dropout)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 学習・検証のループ
    best_val_acc = 0.0
    for epoch in range(config.num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{config.num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Validation精度が向上した場合にモデルを保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.model_file)

    # テストデータで評価
    model.load_state_dict(torch.load(config.model_file))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# csvからラベルマッピングを作成する関数
def load_label_mapping(csv_path):
    df = pd.read_csv(csv_path)
    df = df[~df['file_path'].str.startswith('s5')]
    unique_labels = df["label"].dropna().unique() # label列からユニークなラベルを取得（NaNは除外）
    label2index_local = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    index2label = {idx: label for label, idx in label2index_local.items()}
    return label2index_local, index2label

# 学習済みモデルを用いて音声ファイルのラベルを推論する
def infer_label(model, audio_file, device, index2label):
    model.eval() # 評価モード
    with torch.no_grad():
        logits = model(audio_file) # モデルのforward()は、入力が文字列の場合は内部で特徴量抽出を行う
        pred_idx = torch.argmax(logits, dim=1).item()
        predicted_label = index2label[pred_idx]
    return predicted_label

# 推論
def inference(test_audio_file, model_path, csv_path):
    _, index2label = load_label_mapping(csv_path)
    num_classes = len(index2label)

    # モデルのインスタンス作成（学習時と同じハイパーパラメータ）
    model = AccentClassifierLSTM(num_classes=num_classes,
                                 sample_rate=config.sample_rate,
                                 n_mels=config.n_mels,
                                 n_fft=config.n_fft,
                                 hop_length=config.hop_length,
                                 hidden_size=config.hidden_size,
                                 num_layers=config.num_layers,
                                 dropout=config.dropout)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    predicted = infer_label(model, test_audio_file, device, index2label)
    print(f"推論結果: {test_audio_file} のラベルは '{predicted}' です。")
    return predicted

