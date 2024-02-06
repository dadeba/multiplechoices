# 選択式クイズのファインチューニング

## 準備
指定された計算機にログインしたあと、condaでpython3.10の環境をセットアップ

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
  - path: train-5-inst.json
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

ファインチューニングの実行
```bash
accelerate launch -m axolotl.cli.train qlora-opencalm.yml
```

"accelerate"コマンドにpathが通ってない場合には、以下を一度実行する。
```bash
export PATH=$PATH:/home/$USER/.local/bin
```

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
サンプルコードはopencalmの最も小さいモデル( https://huggingface.co/cyberagent/open-calm-small )をファインチューニングする。
これはパラメータ数が小さいので、他のopencalmモデル(medium,large,1b,3b,7b)についても評価してみること。

# QMA
### Table for key: dev5
| Model | Environment | Score | Latency |
| --- | --- | --- | --- |
| swallow-13b                    | dev5 | 0.562152133580705 | 584.5554900169373 |
| nekomata-7b                    | dev5 | 0.26159554730983303 | 347.97372341156006 |
| stablelm-gamma-7b              | dev5 | 0.4712430426716141 | 819.9777934551239 |
| karasu-7B                      | dev5 | 0.4174397031539889 | 561.0078446865082 |
| youri-7b                       | dev5 | 0.3525046382189239 | 556.2371628284454 |
| nekomata-7b-instruction        | dev5 | 0.38589981447124305 | 304.2966306209564 |
| Xwin-LM-7b                     | dev5 | 0.3562152133580705 | 552.5936434268951 |
| Mistral-7B                     | dev5 | 0.29313543599257885 | 813.7870280742645 |
| calm2-7b                       | dev5 | 0.5306122448979592 | 528.282838344574 |
| chatntq-7b                     | dev5 | 0.4582560296846011 | 542.4657855033875 |
| LLaMA-Pro-8B-Instruct          | dev5 | 0.33766233766233766 | 675.1720423698425 |
| swallow-7b                     | dev5 | 0.5584415584415584 | 621.7504227161407 |
| ELYZA-13b-fast-instruct        | dev5 | 0.4267161410018553 | 555.7631332874298 |
| ELYZA-7b-fast-instruct         | dev5 | 0.3209647495361781 | 641.4171378612518 |

### Table for key: dev6
| Model | Environment | Score | Latency |
| --- | --- | --- | --- |
| youri-7b                       | dev6 | 0.42836041358936483 | 604.1651704311371 |
| swallow-7b                     | dev6 | 0.6174298375184638 | 714.3206160068512 |
| stablelm-gamma-7b              | dev6 | 0.5819793205317577 | 909.2231831550598 |
| Xwin-LM-7b                     | dev6 | 0.4342688330871492 | 602.2411026954651 |
| nekomata-7b                    | dev6 | 0.24224519940915806 | 382.3318078517914 |
| Mistral-7B                     | dev6 | 0.3663220088626292 | 904.7412843704224 |
| ELYZA-7b-fast-instruct         | dev6 | 0.44756277695716395 | 729.2293200492859 |
| chatntq-7b                     | dev6 | 0.5878877400295421 | 588.6175103187561 |
| LLaMA-Pro-8B-Instruct          | dev6 | 0.4224519940915805 | 732.0392067432404 |
| calm2-7b                       | dev6 | 0.5583456425406204 | 600.8704307079315 |
| ELYZA-13b-fast-instruct        | dev6 | 0.4977843426883309 | 623.7655727863312 |
| nekomata-7b-instruction        | dev6 | 0.4357459379615953 | 345.6771879196167 |
| karasu-7B                      | dev6 | 0.6026587887740029 | 611.303624868393 |
| swallow-13b                    | dev6 | 0.6868537666174298 | 657.2044730186462 |

### Table for key: dev7
| Model | Environment | Score | Latency |
| --- | --- | --- | --- |
| chatntq-7b                     | dev7 | 0.6382978723404256 | 433.068562746048 |
| swallow-13b                    | dev7 | 0.6808510638297872 | 481.2613091468811 |
| nekomata-7b                    | dev7 | 0.2572533849129594 | 283.23853397369385 |
| Mistral-7B                     | dev7 | 0.3307543520309478 | 674.0275475978851 |
| swallow-7b                     | dev7 | 0.655705996131528 | 527.1075551509857 |
| ELYZA-13b-fast-instruct        | dev7 | 0.4990328820116054 | 474.12286829948425 |
| LLaMA-Pro-8B-Instruct          | dev7 | 0.40425531914893614 | 540.8537085056305 |
| youri-7b                       | dev7 | 0.41392649903288203 | 446.0511381626129 |
| calm2-7b                       | dev7 | 0.6170212765957447 | 445.73718190193176 |
| stablelm-gamma-7b              | dev7 | 0.6150870406189555 | 677.1285538673401 |
| ELYZA-7b-fast-instruct         | dev7 | 0.4835589941972921 | 535.7477090358734 |
| nekomata-7b-instruction        | dev7 | 0.402321083172147 | 259.0196442604065 |
| karasu-7B                      | dev7 | 0.5454545454545454 | 450.07862305641174 |
| Xwin-LM-7b                     | dev7 | 0.44874274661508706 | 443.0171539783478 |

### Table for key: dev8
| Model | Environment | Score | Latency |
| --- | --- | --- | --- |
| Mistral-7B                     | dev8 | 0.2728380024360536 | 1208.1579804420471 |
| youri-7b                       | dev8 | 0.292326431181486 | 297.8437886238098 |
| nekomata-7b-instruction        | dev8 | 0.31912302070645554 | 212.33313512802124 |
| chatntq-7b                     | dev8 | 0.392204628501827 | 305.8956661224365 |
| swallow-7b                     | dev8 | 0.48112058465286234 | 922.4220526218414 |
| LLaMA-Pro-8B-Instruct          | dev8 | 0.3373934226552984 | 374.45323634147644 |
| ELYZA-7b-fast-instruct         | dev8 | 0.31790499390986604 | 953.2142858505249 |
| Xwin-LM-7b                     | dev8 | 0.3654080389768575 | 296.3972837924957 |
| nekomata-7b                    | dev8 | 0.24725943970767356 | 233.88761019706726 |
| ELYZA-13b-fast-instruct        | dev8 | 0.37149817295980514 | 335.39585399627686 |
| stablelm-gamma-7b              | dev8 | 0.4007308160779537 | 1202.4080970287323 |
| calm2-7b                       | dev8 | 0.45676004872107184 | 781.0882666110992 |
| swallow-13b                    | dev8 | 0.5030450669914738 | 365.8383319377899 |
| karasu-7B                      | dev8 | 0.3556638246041413 | 314.779967546463 |

### Table for key: dev9
| Model | Environment | Score | Latency |
| --- | --- | --- | --- |
| youri-7b                       | dev9 | 0.3252361673414305 | 264.11663365364075 |
| nekomata-7b-instruction        | dev9 | 0.3481781376518219 | 185.23205661773682 |
| LLaMA-Pro-8B-Instruct          | dev9 | 0.31713900134952766 | 334.7826278209686 |
| Xwin-LM-7b                     | dev9 | 0.33738191632928477 | 265.5044949054718 |
| Mistral-7B                     | dev9 | 0.2739541160593792 | 1076.481040239334 |
| calm2-7b                       | dev9 | 0.42645074224021595 | 676.4696221351624 |
| ELYZA-13b-fast-instruct        | dev9 | 0.38191632928475033 | 291.4464793205261 |
| chatntq-7b                     | dev9 | 0.44804318488529016 | 275.1987826824188 |
| stablelm-gamma-7b              | dev9 | 0.44129554655870445 | 1091.468374967575 |
| swallow-13b                    | dev9 | 0.5114709851551957 | 306.42541575431824 |
| swallow-7b                     | dev9 | 0.48717948717948717 | 808.2661917209625 |
| ELYZA-7b-fast-instruct         | dev9 | 0.3603238866396761 | 834.1513533592224 |
| karasu-7B                      | dev9 | 0.407557354925776 | 283.627121925354 |
| nekomata-7b                    | dev9 | 0.2321187584345479 | 205.74456930160522 |
