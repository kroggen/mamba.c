# mamba.c

<p align="center">
  <img src="assets/mamba-c.png" width="300" height="300" alt="Mamba C">
</p>

<p align="right"><a href="https://github.com/kroggen/mamba.c/blob/learning/README.md">English</a> | <a href="https://github.com/kroggen/mamba.c/blob/learning/README-zh.md">中文</a> | <a href="https://github.com/kroggen/mamba.c/blob/learning/README-ru.md">Русский</a></p>

純粋なCでのマンバモデルの推論

[llama2.c](https://github.com/karpathy/llama2.c)からインスピレーションを受け、そのコードを使用しています

これはマンバSSMのリカレントモードのみを実装しています

[関連するpytorchの実装](https://github.com/kroggen/mamba-cpu/tree/recurrent-only)と比較することができます

バッチには対応していません。コードは学習目的で最小限にしています。

それでも、CPU上のpytorchよりも速いです！！！

## 早速始める

```
python3 tokenizer.py
python3 export.py state-spaces/mamba-130m model.bin
make cuda
./mamba-cuda model.bin -n 20 -i "Customer Support should" -t 0.0
```
エクスポート部分で別のモデルを選択することができます

## モデル

[HuggingFace](https://huggingface.co/state-spaces)に保存されているこれらのモデルを使用することができます：

* `state-spaces/mamba-130m`
* `state-spaces/mamba-370m`
* `state-spaces/mamba-790m`
* `state-spaces/mamba-1.4b`
* `state-spaces/mamba-2.8b`
* `state-spaces/mamba-2.8b-slimpj`

モデル名を`export.py`スクリプトの引数として指定することができます

エクスポートスクリプトは、モデルを（まだダウンロードされていない場合）hugingfaceのキャッシュディレクトリにダウンロードします。

オプションとして、手動でダウンロードした場合はモデルファイルへのパスも指定できます。例：

```
wget https://huggingface.co/state-spaces/mamba-130m/resolve/main/config.json?download=true -O config.json
wget https://huggingface.co/state-spaces/mamba-130m/resolve/main/pytorch_model.bin?download=true -O pytorch_model.bin
python3 export.py . model.bin
```

## 内部状態

これはリカレントモデルであるため、内部状態を保存し、後でその状態に戻ることが可能です

内部状態のコピーを取得するには：

```c
  int state_size;
  char* state = get_internal_state(mamba, &state_size);
```

内部状態を設定するには：

```c
  set_internal_state(mamba, state, state_size);
```


## ブランチ

主に2つのブランチがあります：

* `learning` - 非常に基本的な
* `fused` - 基本的な機能を大きなものに統合

それらを[比較](https://github.com/kroggen/mamba.c/compare/learning..fused)することができます


## ノート

特殊文字に対しては、トークナイザーがさらに作業を必要とするかもしれません

自由に貢献し、PRを送ってください



## ライセンス

MIT