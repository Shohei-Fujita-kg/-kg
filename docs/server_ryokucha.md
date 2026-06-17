# 2337室の研究用マシン(mugicha/ryokucha)の設定

---

# サーバーマシンの環境構築メモ

ipアドレスが133.80.184.115のルータにつながっている場合に使用できる手順

**全体的な注意点**

- バージョンが異なると実行するコマンドが異なることもあるので、ネット上の記事を参考にするときはできるだけ新しいものを参考にする
- ファイルに書き込む時に使用するviやemacsは操作が独特であり、誤った記述をしてしまったり元の記述を誤って消してしまったりするので慎重に行う(originalを別ファイルとして残すなども有効)

## サーバマシンの概要

ryokucha (2024.03.30~)

- ubuntu : 22.04.4 LTS
- CPU : AMD Ryzen7 3700x 8-core processor x 16
- Memory : 64.0GiB
- GPU : Geforce RTX 4060 ti

## 目次

- A. プロキシの設定(2022.09~ 不要になった)
- B. gitのインストール
- C. pyenvのインストール
- D. NVIDIAドライバー/CUDAのインストール
- E. pipのインストール
- F. jupyter labのインストールと設定
- G. ポートフォワードの設定

---

## A. プロキシの設定

2022.09~ プロキシ設定は不要になった

vimとかで
~/.gitconfigに以下を追記 //大学のネットワーク設定変更によりプロキシ設定の追加は不要 2022/9/1 ~

> ```
> [http]
>         proxy = http://cache.ccs.kogakuin.ac.jp:8080
> [https]
>         proxy = http://cache.ccs.kogakuin.ac.jp:8080
> [url "https://"]
>         insteadOf = git://
> ```

環境変数の設定

.bash_profile に以下を追記

> ```
> PROXY="cache.ccs.kogakuin.ac.jp:8080"
>
> export http_proxy="http://$PROXY"
> export https_proxy="https://$PROXY"  //http://かも
> export ftp_proxy="ftp://$PROXY"
> export HTTP_PROXY="http://$PROXY"
> export HTTPS_PROXY="https://$PROXY"　　//http://かも
> export FTP_PROXY="ftp://$PROXY"
> ```

source ~/.bash_profile を実行するとproxyが通るようになる

---

## B. gitのインストール (gitコマンドが実行できない場合のみ)

rootユーザーのシェルを実行

> ```
> sudo -s
> ```

インストール可能なパッケージの一覧を更新し、インストール済みのパッケージ更新をおこない、新しいバージョンにアップグレードする

> ```
> sudo apt-get update && sudo apt-get upgrade
> ```

gitコマンドが実行できない場合はインストールする

> ```
> sudo apt-get install git-all
> ```

一旦`sudo -s`でrootに入り、gitのインストール

> ```
> apt-get install git
> ```

C/C++コンパイラ、Make等の標準開発ツール一式をパッケージでインストール

> ```
> sudo apt install build-essential
> ```

---

## C. pyenvのインストール

pyenvをインストールしたいアカウントで作業

再び個人アカウントに戻る

> ```
> su ryokucha
> ```

pyenvをローカル環境に複製する [(参考リンク)](https://github.com/pyenv/pyenv#basic-github-checkout)

> gitコマンドが使えない場合`A. gitのインストール`を参照

> ```
> git clone https://github.com/pyenv/pyenv.git ~/.pyenv
> ```

> ```
> cd ~/.pyenv && src/configure && make -C src
> ```

~/.bashrcにコマンドを追加する

> ```
> echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
> ```

> ```
> cat ./../.bashrc
> ```

> ```
> echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
> ```

> ```
> echo 'eval "$(pyenv init -)"' >> ~/.bashrc
> ```

~/.profileにコマンドを追加する

> ```
> echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
> ```

> ```
> echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
> ```

> ```
> echo 'eval "$(pyenv init -)"' >> ~/.profile
> ```

~/.bash_profileにコマンドを追加する

> ```
> echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
> ```

> ```
> echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
> ```

> ```
> echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
> ```

sourceコマンドで各設定ファイルを反映させる

> ```
> source ~/.bash_profile
> ```

> ```
> source ~/.bashrc
> ```

> ```
> source ~/.profile
> ```

インストール可能なバージョンの確認用コマンド

> ```
> pyenv install -l
> ```

依存するパッケージのインストール(** not availableのエラー発生を回避できた:[参考](https://zenn.dev/chocomochi/articles/53dd04c091f381))

> ```
> sudo apt install libssl-dev libffi-dev libncurses5-dev zlib1g zlib1g-dev libreadline-dev libbz2-dev libsqlite3-dev make gcc liblzma-dev
> ```

バージョンを指定してインストールする

> ```
> pyenv install 3.12.2
> ```

pyディレクトリでのpythonバージョンを固定

> ```
> pyenv local 3.12.2
> ```



---

## D. NVIDIAドライバー/CUDAのインストール

ここからは[この記事](https://qiita.com/syoamakase/items/8b9570d79effbb458b10)、[この記事](https://qiita.com/syoamakase/items/8b9570d79effbb458b10)を参考にするといい

[NVIDIAドライバダウンロード](https://www.nvidia.co.jp/Download/index.aspx?lang=jp)でGPUに適合したドライバー(runfile)をダウンロード

[CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)で適切な情報を入力しrunfileを得るためのコマンドを取得

得られたコマンドを実行(以下2行は例)

> ```
> wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
> ```

> ```
> sudo sh cuda_12.4.0_550.54.14_linux.run
> ```

`End User License Agreement`が表示されるので、`accept`と入力する

`CUDA Installer`で`CUDA Toolkit 12.4`のみを選択し、カーソルを`CUDA Toolkit 12.4`に合わせてAキー(Advanced options)を押す

`Change Toolkit Install Path`にカーソルを合わせ、Enterを押す(インストールパスを入力)

そのほかのチェックを全て外し、`Done`を押す

ここからは[この記事](https://qiita.com/porizou1/items/74d8264d6381ee2941bd)を参考にするといい

OSSとして開発されているNVIDIA GPU向けドライバーであるNouveauを無効化する

> ```
> sudo gedit /etc/modprobe.d/blacklist-nouveau.conf
> ```

nouveauの設定ファイルを新規作成する

> ```
> vi /etc/modprobe.d/blacklist-nouveau.conf
> ```

以下を記入して保存

> ```
> blacklist nouveau
> ```

> ```
> options nouveau modeset=0
> ```

保存したら以下を実行

> ```
> sudo update-initramfs -u
> ```

ダウンロードしたNVIDIAドライバー(のインストーラー)を作業しているディレクトリに移動する（以下は例）

> ```
> mv ../ダウンロード/NVIDIA-Linux-x86_64-550.67.run ./
> ```

ファイルのパーミッションを変更する(すべてのユーザーに実行権限を与える)

> ```
> sudo chmod +x ./NVIDIA-Linux-x86_64-550.67.run
> ```

runfileを実行する

> ```
> sudo NVIDIA-Linux-x86_64-550.67.run
> ```

または

> ```
> sudo bash NVIDIA-Linux-x86_64-550.67.run
> ```

インストールしたNVIDIAドライバーのバージョンを確認

> ```
> nvidia-smi
> ```

**もし、うまくインストールできなかった場合は以下の方法を試す**

ドライバーの検索(`recommended`が付いたドライバーを探す)

> ```
> ubuntu-drivers devices
> ```

PPA(Personal Package Archive)をaptでインストールやアップデートできるようにする

> ```
> sudo add-apt-repository ppa:graphics-drivers/ppa
> ```

> ```
> sudo apt update
> ```

(アップグレード可能なパッケージの更新確認)

> ```
> apt list --upgradable
> ```

`recommended`のドライバーをインストールする

> ```
> sudo apt install nvidia-driver-535
> ```

再起動する

> ```
> sudo reboot
> ```



---

## E. pipのインストール

pyディレクトリを作成

> ```
> mkdir ~/py
> ```

pyディレクトリに移動

> ```
> cd ~/py
> ```

python3系のpipをインストールする場合([参考](https://wellknowledge.org/linux-pip/))

> ```
> wget https://bootstrap.pypa.io/get-pip.py
> ```

> ```
> python3 get-pip.py
> ```

No module named 'distutils.util'でget-pip.pyで使えない場合に実行([参考](https://qiita.com/rh_/items/80d082bafcdae31bc95c))

> ```
> sudo apt-get install python3-distutils
> ```

pipがbinは以下にあるか確認

> ```
> ls ~/.local/bin
> ```

PATHを通す

> ```
> export PATH=$PATH:~/.local/bin
> ```

numpy scikit-learn pandas jupyterlabのインストール

> ```
> pip install numpy scikit-learn pandas jupyterlab
> ```



---

## F. jupyter labのインストールと設定

`C. pyenvのインストール`,`D. NVIDIAドライバー/CUDAのインストール`,`E. pipのインストール`を終えてから行う

インストールされているCUDAのバージョンを確認

> ```
> ls -a ~/../../usr/local/
> ```

`https://pytorch.org`でインストールするpytorch(Stable(2.2.2), Linux, Pip, CUDA 12.1)を選択し、生成されるコマンドを実行(以下は例)

> ```
> pip3 install torch torchvision torchaudio
> ```

jupyter labを起動

> ```
> jupyter lab
> ```

起動したjupyter labのノートブック上で

```
import torch; torch.cuda.is_available()
```

をセルに入力して， ctl+Enterで実行． Trueが出ればGPUが使用できる



---

## G. ポートフォワードの設定

SSH接続でリモート利用できるようにする

参考にしたサイト: [Qiita: jupyter labをリモートから使う](https://qiita.com/RayDoe/items/e1ec21c63a15adb1a061), [blog: UbuntuにSSH接続できるようにする](https://www.xenos.jp/~zen/blog2/index.php/2019/01/27/post-1657/), [Jupyter notebookのパスワード](https://qiita.com/SaitoTsutomu/items/aee41edf1a990cad5be6), [Ubuntu: OpenSSHサーバー](https://ubuntu.com/server/docs/service-openssh)

emacsの操作方法は[Emacsの主要操作(早見表)](http://www.rsch.tuis.ac.jp/~ohmi/literacy/emacs/quick.html#:~:text=Emacs%E3%81%AE%E8%B5%B7%E5%8B%95%E3%81%A8%E7%B5%82%E4%BA%86&text=Emacs%E3%82%92%E7%B5%82%E4%BA%86%E3%81%99%E3%82%8B%E3%81%AB,%E3%81%A7C%2Dx%20C%2Dc%E3%81%A8%E6%89%93%E3%81%A4%EF%BC%8E)を参考にするといい

emacsのインストール(configを書き換えるだけなので、emacsでなくてもいいかも)

> ```
> sudo add-apt-repository ppa:kelleyk/emacs
> ```

> ```
> sudo apt install emacs
> ```

OpenSSHサーバーをインストール

> ```
> sudo apt install openssh-server
> ```

`sshd_config`を書き換える前にコピーを残しておく(誤操作をすると戻すのが大変になるため)

> ```
> sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.original
> ```

originalとしてコピーしたものを書き込み保護しておく

> ```
> sudo chmod a-w /etc/ssh/sshd_config.original
> ```

sshd_config内の`#PermitRootLogin no`を`PermitRootLogin no`に書き換えて保存(コメントアウトを外す)

> ```
> sudo emacs -nw /etc/ssh/sshd_config
> ```

sshを起動する

> ```
> sudo systemctl enable ssh
> ```

> ```
> sudo systemctl start ssh
> ```

statusを確認(`Active: active (running)`となっていればOK)

> ```
> sudo systemctl status ssh
> ```

jupyter labの設定ファイルを作成する

> ```
> jupyter lab --generate-config
> ```

`jupyter_lab_config`を編集する

> ```
> vi ./.jupyter/jupyter_lab_config.py
> ```

下記6行を追記する(viで最終行に移動するにはコマンドモードで`Shift+G`)

> ```
> c = get_config()
> ```

> ```
> c.IPKernelApp.pylab = 'inline'
> ```

> ```
> c.NotebookApp.ip = '0.0.0.0'
> ```

> ```
> c.NotebookApp.open_browser = False
> ```

> ```
> c.NotebookApp.port = 9999
> ```

> ```
> c.NotebookApp.password = 'sha1:b5b09012066d:2a03a9c26579f9cd7f18a8c0f7a03407ca574547'
> ```

`c.NotebookApp.password = ...`では、2022年3月3日にリリースされたversion8.1.0で`passwd`関数が削除されたが廃止されたことに注意

`c.NotebookApp.password = ...`でうまく機能しない場合は`token`を代わりに使うこともできる

> ```
> c.NotebookApp.token = 'password'
> ```

jupyter labを起動してコンソールに`http://ryokucha-desktop:2426/lab?token=...`または`http://127.0.0.1:2426/lab?token=...`と表示されていればOK

> ```
> jupyter lab
> ```

もし`jupyter lab`コマンドが認識されていない場合、`PATH`の設定をスルーしている可能性があるので以下のコマンドを実行する

> ```
> sudo apt update && sudo apt upgrade
> ```

> ```
> export PATH=$PATH:~/.local/bin
> ```

> ```
> sudo systemctl restart ssh
> ```


```
グループ : Group02
Internet側IPアドレス : 手動設定(133.80.184.115)
プロトコル : TCP/UDP(任意のTCPポート:****)
LAN側IPアドレス : 192.168.42.4
LAN側ポート : TCP/UDPポート:****

```

```
ポート番号
田上 2426
齋藤 5555
細野 77777
```

---

### 大学外から繋ぐときは...

puttyのユーザ名は**他の人と被らないように**することに注意

karasuを経由する必要あり

karasuはユーザsanelabでadduserでアカウント作れる

mugicha は mugichaアカウントが一番権限を持ってる

---

### (2024.10追記) VSCodeからSSHリモート接続する場合

ssh接続用の`config`に以下を記入

```
Host 133.80.184.115
  HostName 133.80.184.115
  User tagami

Host karasu
  HostName 133.80.183.86
  User tagami
  Port 443

Host gakugairdp
  HostName 133.80.184.115
  User tagami
  ProxyJump karasu

# X11転送を要求するソフトのために必要な設定（-X、-Yオプションに相当）
  ForwardX11 yes
  ForwardX11Trusted yes
```
