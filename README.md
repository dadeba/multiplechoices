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

フォーマットはファインチューニング用の入力ファイルと同じものを想定している。
```json
  {
    "instruction": "以下の問題について、0, 1, 2, 3, 4の選択肢から選んで答えてください。\n\n問題: 江戸時代に長崎の出島に置かれたオランダ商館長のことを何といったでしょう? \n選択肢\n\n0: アントン・ポートマン\n1: 長崎目付\n2: 通詞\n3: カピタン\n4: リチャード・コックス\n\n\n### 回答:\n",
    "output": "3: カピタン"
  }
```

クイズ王のファイル(dev1_questions.json/dev2_questions.json)から変換する。

このプログラムを実行すると、"result.log"と"result.json"というファイルが出力される。
前者は、推論した結果の確認用で、後者は推論結果をjsonファイルとして保存している。
jsonファイルを読み込んで、正解率を計算するサンプルコードとして"analysis.py"を含めている。

```bash
python analysiy.py

result.json                                                  : 0.35
```

この場合正解率は35パーセントということになる。
正解率の評価は、"dev1_questions.json"と"dev2_questions.json"の両方で評価すること。

#### TODO
評価時にはpromptを正規化しないと正解判定を間違う場合がある。
問題文には最後に疑問符があるが、open-calmは"?"(ASCIIの半角相当の疑問符)と"？"(UTF-8の全角相当の疑問符)が同じtokenのため、
出力からprompt部分を取り除くことができずに正解判定が正しく動作しない。
open-calmのtokenizerには数字でも同様の問題がある。

# 評価
サンプルコードはopencalmの最も小さいモデル( https://huggingface.co/cyberagent/open-calm-small )をファインチューニングする。
これはパラメータ数が小さいので、他のopencalmモデル(medium,large,1b,3b,7b)についても評価してみること。

# QA
### Table for key: AI王 dev1  
| Model | Environment | Score | Latency |
| --- | --- | --- | --- |
| swallow-13b                    | dev1 | 0.962 | 404.3 |
| swallow-7b                     | dev1 | 0.949 | 1100.0 |
| calm2-7b                       | dev1 | 0.926 | 927.7 |
| chatntq-7b                     | dev1 | 0.922 | 376.9 |
| stablelm-gamma-7b              | dev1 | 0.907 | 393.0 |
| karasu-7B                      | dev1 | 0.889 | 1523.0 |
| ELYZA-13b-fast-instruct        | dev1 | 0.87 | 1529.0 |
| Qwen1.5-14B                    | dev1 | 0.863 | 985.1 |
| ELYZA-7b-fast-instruct         | dev1 | 0.861 | 307.5 |
| open-calm-7b                   | dev1 | 0.841 | 752.6 |
| open-calm-3b                   | dev1 | 0.825 | 562.6 |
| youri-7b                       | dev1 | 0.825 | 1447.0 |
| Xwin-LM-7b                     | dev1 | 0.811 | 1438.0 |
| open-calm-1b                   | dev1 | 0.797 | 388.8 |
| nekomata-7b-instruction        | dev1 | 0.789 | 251.4 |
| Qwen1.5-7B                     | dev1 | 0.779 | 761.7 |
| open-calm-large                | dev1 | 0.765 | 365.9 |
| Qwen1.5-4B                     | dev1 | 0.71 | 316.8 |
| Mistral-7B                     | dev1 | 0.693 | 1495.0 |
| Qwen1.5-1.8B                   | dev1 | 0.624 | 193.3 |
| open-calm-medium               | dev1 | 0.569 | 343.4 |
| Qwen1.5-0.5B                   | dev1 | 0.542 | 515.8 |
| open-calm-small                | dev1 | 0.348 | 164.4 |

### Table for key: AI王 dev2
| Model | Environment | Score | Latency |
| --- | --- | --- | --- |
| swallow-13b                    | dev2 | 0.971 | 406.2 |
| swallow-7b                     | dev2 | 0.95 | 1100.0 |
| calm2-7b                       | dev2 | 0.928 | 935.5 |
| chatntq-7b                     | dev2 | 0.924 | 378.3 |
| stablelm-gamma-7b              | dev2 | 0.911 | 394.5 |
| karasu-7B                      | dev2 | 0.882 | 1529.0 |
| Qwen1.5-14B                    | dev2 | 0.862 | 992.9 |
| ELYZA-13b-fast-instruct        | dev2 | 0.857 | 1532.0 |
| ELYZA-7b-fast-instruct         | dev2 | 0.849 | 310.0 |
| youri-7b                       | dev2 | 0.848 | 1457.0 |
| open-calm-7b                   | dev2 | 0.844 | 752.0 |
| open-calm-3b                   | dev2 | 0.824 | 562.6 |
| open-calm-1b                   | dev2 | 0.797 | 390.4 |
| Xwin-LM-7b                     | dev2 | 0.791 | 1451.0 |
| nekomata-7b-instruction        | dev2 | 0.779 | 252.3 |
| Qwen1.5-7B                     | dev2 | 0.778 | 770.5 |
| open-calm-large                | dev2 | 0.76 | 365.2 |
| Mistral-7B                     | dev2 | 0.701 | 1500.0 |
| Qwen1.5-4B                     | dev2 | 0.67 | 321.5 |
| Qwen1.5-1.8B                   | dev2 | 0.603 | 194.9 |
| open-calm-medium               | dev2 | 0.536 | 340.6 |
| Qwen1.5-0.5B                   | dev2 | 0.521 | 521.3 |
| open-calm-small                | dev2 | 0.347 | 163.9 |

### Table for key: JcommonsenseQA-v1.1 valid dev3
| Model | Environment | Score | Latency |
| --- | --- | --- | --- |
| swallow-13b                    | dev3 | 0.908 | 386.4 |
| stablelm-gamma-7b              | dev3 | 0.895 | 363.6 |
| chatntq-7b                     | dev3 | 0.893 | 351.5 |
| swallow-7b                     | dev3 | 0.892 | 1039.0 |
| karasu-7B                      | dev3 | 0.885 | 1403.0 |
| Qwen1.5-14B                    | dev3 | 0.868 | 917.1 |
| ELYZA-13b-fast-instruct        | dev3 | 0.863 | 1500.0 |
| calm2-7b                       | dev3 | 0.859 | 907.5 |
| youri-7b                       | dev3 | 0.832 | 1346.0 |
| nekomata-7b-instruction        | dev3 | 0.812 | 237.0 |
| ELYZA-7b-fast-instruct         | dev3 | 0.796 | 281.2 |
| open-calm-7b                   | dev3 | 0.794 | 749.2 |
| Xwin-LM-7b                     | dev3 | 0.762 | 1329.0 |
| Qwen1.5-7B                     | dev3 | 0.76 | 679.9 |
| open-calm-3b                   | dev3 | 0.718 | 533.6 |
| Mistral-7B                     | dev3 | 0.702 | 1366.0 |
| open-calm-1b                   | dev3 | 0.679 | 365.2 |
| open-calm-large                | dev3 | 0.657 | 339.8 |
| Qwen1.5-4B                     | dev3 | 0.646 | 290.1 |
| Qwen1.5-1.8B                   | dev3 | 0.525 | 176.8 |
| Qwen1.5-0.5B                   | dev3 | 0.399 | 468.4 |
| open-calm-medium               | dev3 | 0.366 | 306.5 |
| open-calm-small                | dev3 | 0.221 | 152.2 |

### Table for key: JcommonsenseQA-v1.1 train dev4
| Model | Environment | Score | Latency |
| --- | --- | --- | --- |
| swallow-13b                    | dev4 | 0.897 | 3097.0 |
| swallow-7b                     | dev4 | 0.883 | 8289.0 |
| chatntq-7b                     | dev4 | 0.881 | 2783.0 |
| stablelm-gamma-7b              | dev4 | 0.878 | 2901.0 |
| karasu-7B                      | dev4 | 0.869 | 11159.0 |
| Qwen1.5-14B                    | dev4 | 0.861 | 7242.0 |
| calm2-7b                       | dev4 | 0.852 | 7239.0 |
| ELYZA-13b-fast-instruct        | dev4 | 0.841 | 11969.0 |
| youri-7b                       | dev4 | 0.814 | 10694.0 |
| nekomata-7b-instruction        | dev4 | 0.813 | 1911.0 |
| ELYZA-7b-fast-instruct         | dev4 | 0.808 | 2254.0 |
| open-calm-7b                   | dev4 | 0.769 | 5959.0 |
| Qwen1.5-7B                     | dev4 | 0.764 | 5510.0 |
| Xwin-LM-7b                     | dev4 | 0.76 | 10569.0 |
| open-calm-3b                   | dev4 | 0.715 | 4289.0 |
| Mistral-7B                     | dev4 | 0.705 | 10866.0 |
| open-calm-1b                   | dev4 | 0.665 | 2943.0 |
| open-calm-large                | dev4 | 0.644 | 2729.0 |
| Qwen1.5-4B                     | dev4 | 0.631 | 2288.0 |
| Qwen1.5-1.8B                   | dev4 | 0.51 | 1389.0 |
| Qwen1.5-0.5B                   | dev4 | 0.385 | 3693.0 |
| open-calm-medium               | dev4 | 0.354 | 2461.0 |
| open-calm-small                | dev4 | 0.205 | 1205.0 |

# QMA
### Table for 芸能
| Model | Environment | Score | Latency |
| --- | --- | --- | --- |
| swallow-13b                    | dev5 | 0.562 | 584.6 |
| swallow-7b                     | dev5 | 0.558 | 621.8 |
| calm2-7b                       | dev5 | 0.531 | 528.3 |
| stablelm-gamma-7b              | dev5 | 0.471 | 820.0 |
| chatntq-7b                     | dev5 | 0.458 | 542.5 |
| ELYZA-13b-fast-instruct        | dev5 | 0.427 | 555.8 |
| Qwen1.5-14B                    | dev5 | 0.423 | 239.8 |
| karasu-7B                      | dev5 | 0.417 | 561.0 |
| nekomata-7b-instruction        | dev5 | 0.386 | 304.3 |
| Xwin-LM-7b                     | dev5 | 0.356 | 552.6 |
| youri-7b                       | dev5 | 0.353 | 556.2 |
| LLaMA-Pro-8B-Instruct          | dev5 | 0.338 | 675.2 |
| open-calm-7b                   | dev5 | 0.336 | 423.6 |
| Qwen1.5-7B                     | dev5 | 0.33 | 417.1 |
| ELYZA-7b-fast-instruct         | dev5 | 0.321 | 641.4 |
| open-calm-3b                   | dev5 | 0.295 | 324.7 |
| Mistral-7B                     | dev5 | 0.293 | 813.8 |
| Qwen1.5-4B                     | dev5 | 0.288 | 492.7 |
| open-calm-1b                   | dev5 | 0.278 | 224.8 |
| open-calm-large                | dev5 | 0.263 | 213.5 |
| nekomata-7b                    | dev5 | 0.262 | 348.0 |
| open-calm-small                | dev5 | 0.249 | 96.79 |
| Qwen1.5-1.8B                   | dev5 | 0.224 | 296.8 |
| open-calm-medium               | dev5 | 0.223 | 202.3 |
| Qwen1.5-0.5B                   | dev5 | 0.204 | 295.3 |

### Table for 学問
| Model | Environment | Score | Latency |
| --- | --- | --- | --- |
| swallow-13b                    | dev6 | 0.687 | 657.2 |
| swallow-7b                     | dev6 | 0.617 | 714.3 |
| karasu-7B                      | dev6 | 0.603 | 611.3 |
| chatntq-7b                     | dev6 | 0.588 | 588.6 |
| Qwen1.5-14B                    | dev6 | 0.583 | 269.2 |
| stablelm-gamma-7b              | dev6 | 0.582 | 909.2 |
| calm2-7b                       | dev6 | 0.558 | 600.9 |
| ELYZA-13b-fast-instruct        | dev6 | 0.498 | 623.8 |
| Qwen1.5-7B                     | dev6 | 0.456 | 456.1 |
| ELYZA-7b-fast-instruct         | dev6 | 0.448 | 729.2 |
| nekomata-7b-instruction        | dev6 | 0.436 | 345.7 |
| Xwin-LM-7b                     | dev6 | 0.434 | 602.2 |
| youri-7b                       | dev6 | 0.428 | 604.2 |
| LLaMA-Pro-8B-Instruct          | dev6 | 0.422 | 732.0 |
| open-calm-7b                   | dev6 | 0.405 | 479.5 |
| Mistral-7B                     | dev6 | 0.366 | 904.7 |
| Qwen1.5-4B                     | dev6 | 0.359 | 540.7 |
| open-calm-large                | dev6 | 0.344 | 231.9 |
| open-calm-3b                   | dev6 | 0.34 | 358.6 |
| open-calm-1b                   | dev6 | 0.326 | 248.9 |
| Qwen1.5-1.8B                   | dev6 | 0.294 | 321.6 |
| open-calm-medium               | dev6 | 0.261 | 219.6 |
| open-calm-small                | dev6 | 0.254 | 105.9 |
| Qwen1.5-0.5B                   | dev6 | 0.254 | 320.8 |
| nekomata-7b                    | dev6 | 0.242 | 382.3 |

### Table for 雑学
| Model | Environment | Score | Latency |
| --- | --- | --- | --- |
| swallow-13b                    | dev7 | 0.681 | 481.3 |
| swallow-7b                     | dev7 | 0.656 | 527.1 |
| chatntq-7b                     | dev7 | 0.638 | 433.1 |
| calm2-7b                       | dev7 | 0.617 | 445.7 |
| stablelm-gamma-7b              | dev7 | 0.615 | 677.1 |
| karasu-7B                      | dev7 | 0.545 | 450.1 |
| Qwen1.5-14B                    | dev7 | 0.53 | 196.4 |
| ELYZA-13b-fast-instruct        | dev7 | 0.499 | 474.1 |
| ELYZA-7b-fast-instruct         | dev7 | 0.484 | 535.7 |
| Xwin-LM-7b                     | dev7 | 0.449 | 443.0 |
| Qwen1.5-7B                     | dev7 | 0.429 | 329.6 |
| open-calm-7b                   | dev7 | 0.427 | 355.7 |
| youri-7b                       | dev7 | 0.414 | 446.1 |
| LLaMA-Pro-8B-Instruct          | dev7 | 0.404 | 540.9 |
| open-calm-3b                   | dev7 | 0.402 | 263.3 |
| nekomata-7b-instruction        | dev7 | 0.402 | 259.0 |
| open-calm-1b                   | dev7 | 0.389 | 181.8 |
| open-calm-large                | dev7 | 0.352 | 169.9 |
| Qwen1.5-4B                     | dev7 | 0.35 | 388.5 |
| Mistral-7B                     | dev7 | 0.331 | 674.0 |
| Qwen1.5-1.8B                   | dev7 | 0.317 | 232.6 |
| open-calm-small                | dev7 | 0.279 | 78.21 |
| Qwen1.5-0.5B                   | dev7 | 0.263 | 231.2 |
| nekomata-7b                    | dev7 | 0.257 | 283.2 |
| open-calm-medium               | dev7 | 0.248 | 159.9 |

### Table for アニメ
| Model | Environment | Score | Latency |
| --- | --- | --- | --- |
| swallow-13b                    | dev8 | 0.503 | 365.8 |
| swallow-7b                     | dev8 | 0.481 | 922.4 |
| calm2-7b                       | dev8 | 0.457 | 781.1 |
| stablelm-gamma-7b              | dev8 | 0.401 | 1202.0 |
| chatntq-7b                     | dev8 | 0.392 | 305.9 |
| ELYZA-13b-fast-instruct        | dev8 | 0.371 | 335.4 |
| Xwin-LM-7b                     | dev8 | 0.365 | 296.4 |
| Qwen1.5-14B                    | dev8 | 0.362 | 355.7 |
| karasu-7B                      | dev8 | 0.356 | 314.8 |
| LLaMA-Pro-8B-Instruct          | dev8 | 0.337 | 374.5 |
| nekomata-7b-instruction        | dev8 | 0.319 | 212.3 |
| ELYZA-7b-fast-instruct         | dev8 | 0.318 | 953.2 |
| Qwen1.5-7B                     | dev8 | 0.313 | 607.4 |
| open-calm-7b                   | dev8 | 0.308 | 643.7 |
| Qwen1.5-4B                     | dev8 | 0.308 | 719.2 |
| youri-7b                       | dev8 | 0.292 | 297.8 |
| open-calm-1b                   | dev8 | 0.286 | 333.0 |
| open-calm-3b                   | dev8 | 0.278 | 476.7 |
| open-calm-large                | dev8 | 0.276 | 313.6 |
| Qwen1.5-1.8B                   | dev8 | 0.274 | 429.3 |
| Mistral-7B                     | dev8 | 0.273 | 1208.0 |
| open-calm-medium               | dev8 | 0.256 | 297.6 |
| nekomata-7b                    | dev8 | 0.247 | 233.9 |
| open-calm-small                | dev8 | 0.234 | 142.6 |
| Qwen1.5-0.5B                   | dev8 | 0.228 | 432.0 |

### Table for スポーツ
| Model | Environment | Score | Latency |
| --- | --- | --- | --- |
| swallow-13b                    | dev9 | 0.511 | 306.4 |
| swallow-7b                     | dev9 | 0.487 | 808.3 |
| chatntq-7b                     | dev9 | 0.448 | 275.2 |
| stablelm-gamma-7b              | dev9 | 0.441 | 1091.0 |
| calm2-7b                       | dev9 | 0.426 | 676.5 |
| karasu-7B                      | dev9 | 0.408 | 283.6 |
| Qwen1.5-14B                    | dev9 | 0.389 | 311.3 |
| ELYZA-13b-fast-instruct        | dev9 | 0.382 | 291.4 |
| ELYZA-7b-fast-instruct         | dev9 | 0.36 | 834.2 |
| Qwen1.5-7B                     | dev9 | 0.352 | 534.4 |
| nekomata-7b-instruction        | dev9 | 0.348 | 185.2 |
| Xwin-LM-7b                     | dev9 | 0.337 | 265.5 |
| youri-7b                       | dev9 | 0.325 | 264.1 |
| LLaMA-Pro-8B-Instruct          | dev9 | 0.317 | 334.8 |
| open-calm-1b                   | dev9 | 0.309 | 279.4 |
| open-calm-7b                   | dev9 | 0.306 | 539.5 |
| Qwen1.5-4B                     | dev9 | 0.286 | 634.2 |
| open-calm-3b                   | dev9 | 0.281 | 405.8 |
| Mistral-7B                     | dev9 | 0.274 | 1076.0 |
| Qwen1.5-1.8B                   | dev9 | 0.265 | 376.2 |
| open-calm-large                | dev9 | 0.26 | 263.5 |
| open-calm-medium               | dev9 | 0.24 | 249.9 |
| open-calm-small                | dev9 | 0.236 | 119.8 |
| nekomata-7b                    | dev9 | 0.232 | 205.7 |
| Qwen1.5-0.5B                   | dev9 | 0.211 | 376.9 |
