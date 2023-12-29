# 選択式クイズのファインチューニング

## 準備
condaでpython3.10の環境をセットアップ

```bash
conda create -n quiz python=3.10  
conda activate quiz
```

以降、このconda環境でプログラムを実行する。

## ファインチューニングに必要なライブラリ(Axolotl)をインストール
```bash
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl

pip3 install packaging
pip3 install -e '.[flash-attn,deepspeed]'
```

## open-calmのモデルをチューニングする

### 詳細
"qlora-opencalm.yml"がファインチューニングの設定ファイル。これを必要に応じて変更する必要がある。

修正が必要な行:
```
base_model: cyberagent/open-calm-small  ## huggingfaceのモデル名
output_dir: ./open-small-qlora-out      ## チューニング結果を出力するディレクトリ
```

以下の部分の"path"に訓練用のクイズファイルを指定する。
```
datasets:
  - path: datasets/train-5-inst.json
```

クイズファイルのフォーマットは以下の形式を想定しているので、クイズ王のファイルから変換する。
```json
  {
    "instruction": "以下の問題について、0, 1, 2, 3, 4の選択肢から選んで答えてください。\n\n問題: 格闘家ボブ・サップの出身国はどこでしょう? \n選択肢\n\n0: オレゴン州\n1: イギリス\n2: アメリカ合衆国\n3: コロラド州\n4: ミシガン州\n\n\n### 回答:\n",
    "output": "2: アメリカ合衆国"
  }
```

"instrucion"の部分に問題と選択肢をまとめて入れる。最後のキーワード「### 回答:」は、opencalmモデルの学習には必ず必要。
"output"は問題に対する回答。

### チューニングしたあとの評価
サンプルで実行するためのスクリプト(run_quiz.py)を実行する。

```bash
python3 run_quiz.py
```

このプログラムは"quiz.json"を読み込んで、問題ごとに推論をして回答が正解かを計算する。

フォーマットは以下を想定している。
```json
  {
    "instruction": "以下の問題について、0, 1, 2, 3, 4から選んで答えてください。\n問題: 和名をハダカカメガイといい、実は巻き貝の一種とされている、その姿から「流氷の天使」と呼ばれる動物は何でしょう?\n",
    "input": "0: ミズダコ\n1: オオイカリナマコ\n2: クリオネ\n3: イソギンチャクモエビ\n4: アカシマシラヒゲエビ\n\n\n",
    "output": "2: クリオネ"
  },
```

訓練用のフォーマットとは異なるので、それに気をつけてクイズ王のファイルから変換する。



