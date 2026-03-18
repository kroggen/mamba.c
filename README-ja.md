# mamba.c

<p align="center">
  <img src="assets/mamba-c.png" width="300" height="300" alt="Mamba C">
</p>

<p align="right"><a href="https://github.com/kroggen/mamba.c/blob/learning/README.md">English</a> | <a href="https://github.com/kroggen/mamba.c/blob/learning/README-zh.md">中文</a> | <a href="https://github.com/kroggen/mamba.c/blob/learning/README-ru.md">Русский</a></p>

純粋なCでのMamba 1、2 & 3モデルの推論

[llama2.c](https://github.com/karpathy/llama2.c)からインスピレーションを受け、そのコードを使用しています

これはマンバSSMのリカレントモードのみを実装しています

[関連するpytorchの実装](https://github.com/kroggen/mamba-cpu/tree/recurrent-only)と比較することができます

バッチには対応していません。コードは学習目的で最小限にしています。

それでも、CPU上のpytorchよりも速いです！！！

## 早速始める

Mamba-3モデルの重みが公開されたら（[モデル](#モデル)を参照）：

```
python3 tokenizer.py
python3 export.py state-spaces/mamba3-130m model.bin
make fast
./mamba model.bin -n 20 -i "Customer Support should" -t 0.0
```
Pythonは、トークン化器とモデルをよりシンプルな形式にエクスポートするためにのみ使用されます（transformersとpytorchが必要です）

## モデル

> **注:** 2026年3月現在、Mamba-3モデルの重みはまだ公開されていません。
> 論文（[arXiv:2603.15569](https://arxiv.org/abs/2603.15569)）は2026年3月16日に投稿されました。
> [state-spaces](https://huggingface.co/state-spaces) HuggingFace組織は現在、Mamba-1とMamba-2のチェックポイントのみをホストしています。
> 将来のMamba-3リリースについてはそのページをご確認ください。

重みが利用可能になったら、エクスポートスクリプトは`mamba3.py`で使用される`backbone.layers.N.mixer.*` / `backbone.layers.N.mlp.*`レイアウトの標準的なHuggingFaceチェックポイントを期待します。
その後、以下を実行できます：

```
python3 export.py state-spaces/mamba3-130m model.bin
```

または手動で：

```
python3 export.py /path/to/local/mamba3-model model.bin
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

コードは3つのバージョンで利用可能で、それぞれ別のブランチにあります：

* `learning` - 基本的なもの
* `fused` - 基本的な機能をより大きなものに統合（[比較](https://github.com/kroggen/mamba.c/compare/learning..fused)することができます）
* `cuda` - シンプルなGPU実装、理解しやすい

Mamba 2のコードもあります：

* `mamba2-learning` - 非常に基本的（[mamba1と比較](https://github.com/kroggen/mamba.c/compare/learning..mamba2-learning)）
* `mamba2-fused` - 統合された関数（[learningと比較](https://github.com/kroggen/mamba.c/compare/mamba2-learning..mamba2-fused) | [mamba1と比較](https://github.com/kroggen/mamba.c/compare/fused..mamba2-fused)）

そしてMamba 3（ICLR 2026）の場合：

* `mamba3-learning` - 非常に基本的（[mamba2と比較](https://github.com/kroggen/mamba.c/compare/mamba2-learning..mamba3-learning)）
* `mamba3-fused` - 統合された関数（[learningと比較](https://github.com/kroggen/mamba.c/compare/mamba3-learning..mamba3-fused) | [mamba2と比較](https://github.com/kroggen/mamba.c/compare/mamba2-fused..mamba3-fused)）

Mamba-3の主な変更点（Mamba-2との比較）：
- **台形離散化**: `h_t = α*h_{t-1} + β*B̄_{t-1}x_{t-1} + γ*B̄_t*x_t`（`prev_Bx`の追跡が必要）
- **データ依存RoPE**: BとCは、入力θとステップサイズΔから導出された累積角度によって回転されます
- **QK正規化**: 投影後にBとCに適用されるRMSNorm（ゲート付きRMSNorm出力ノルムを置き換え）
- **学習可能なBCバイアス**: QK-norm後にBとCに追加されるヘッド固有のバイアス、1で初期化
- **短い畳み込みなし**: 台形則 + バイアスによりconv1dが不要
- **Llamaスタイルのアーキテクチャ**: 各レイヤーは`RMSNorm → SSM → 残差 → RMSNorm → SwiGLU MLP → 残差`


## ノート

特殊文字に対しては、トークナイザーがさらに作業を必要とするかもしれません

自由に貢献し、PRを送ってください



## ライセンス

MIT