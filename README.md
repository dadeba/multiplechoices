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

## open-calmのモデルをファインチューニング(学習)する
このサンプルコードは、日本語LLM(opencalmのsmall/parameters 160M)を、量子化とLoRA(Low-Rank Adaptation)の組み合わせでファインチューニングする。

このsmallモデルの場合、モデルサイズが小さいので量子化は必ずしも必要ではないが、
より大きいモデルでメモリ量を削減して計算を高速にするため、量子化を適用している。

### 詳細
"qlora-opencalm.yml"がファインチューニングの設定ファイル。これを必要に応じて変更する必要がある。

修正が必要な行:
```
base_model: cyberagent/open-calm-small  ## huggingfaceのモデル名
output_dir: ./open-small-qlora-out      ## 学習結果を出力するディレクトリ
```

以下の部分の"path"に学習用のクイズファイルを指定する。
```
datasets:
  - path: datasets/train-5-inst.json
```

クイズファイルのフォーマットは以下の形式を想定しているので、クイズ王のファイル(train_questions.json)から変換する。
```json
  {
    "instruction": "以下の問題について、0, 1, 2, 3, 4の選択肢から選んで答えてください。\n\n問題: 格闘家ボブ・サップの出身国はどこでしょう? \n選択肢\n\n0: オレゴン州\n1: イギリス\n2: アメリカ合衆国\n3: コロラド州\n4: ミシガン州\n\n\n### 回答:\n",
    "output": "2: アメリカ合衆国"
  }
```

"instrucion"の部分に「問題」と「選択肢」をまとめて入れる。
最後のキーワード「### 回答:」は、opencalmモデルの学習には必ず必要。
"output"は問題に対する回答。

### ファインチューニングしたあとの評価
サンプルで実行するためのスクリプト(run_quiz.py)を実行する。

```bash
python3 run_quiz.py
```

このプログラムは"quiz.json"を読み込んで、問題ごとに推論をして回答が正解かを計算する。
49行の"model_id"の行は、学習結果のディレクトリを指定している。適宜変更する。

フォーマットは以下を想定している。
```json
  {
    "instruction": "以下の問題について、0, 1, 2, 3, 4から選んで答えてください。\n問題: 和名をハダカカメガイといい、実は巻き貝の一種とされている、その姿から「流氷の天使」と呼ばれる動物は何でしょう?\n",
    "input": "0: ミズダコ\n1: オオイカリナマコ\n2: クリオネ\n3: イソギンチャクモエビ\n4: アカシマシラヒゲエビ\n\n\n",
    "output": "2: クリオネ"
  }
```

学習のjsonファイルとは異なるので、それに気をつけてクイズ王のファイル(dev1_questions.json/dev2_questions.json)から変換する。
サンプルとして"dev1_questions.json"の一部を変換したファイルを"quiz.json"として含めている。

このプログラムを実行すると、"result.log"と"result.json"というファイルが出力される。
前者は、推論した結果の確認用で、後者は推論結果をjsonファイルとして保存している。
jsonファイルを読み込んで、正解率を計算するサンプルコードとして"analysis.py"を含めている。

```bash
python analysiy.py

result.json                                                  : 0.35
```

この場合正解率は35パーセントということになる。
正解率の評価は、"dev1_questions.json"と"dev2_questions.json"の両方で評価すること。

# 評価
サンプルコードはopencalmの最も小さいモデル(https://huggingface.co/cyberagent/open-calm-small)をファインチューニングする。
これはパラメータ数が小さいので、他のopencalmモデル(medium,large,1b,3b,7b)についても評価してみること。
