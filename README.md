# BungakuLLM — 日本文学モデル

ゼロから学ぶTransformerベースの言語モデル。文字レベルのGPTを最小構成で実装し、テキスト生成を体験できるプロジェクトです。

---

## LLMの基礎の基礎

### Transformerアーキテクチャ全体像

大規模言語モデル (LLM) の中核は **Transformer** アーキテクチャです。Transformerは2017年の論文 "Attention Is All You Need" で提案され、RNNやLSTMに代わる系列モデルとして急速に普及しました。

本プロジェクトが実装しているのは **Decoder-only Transformer** (GPTスタイル) です。構造は以下のとおりです:

```
入力トークン列
    ↓
トークン埋め込み + 位置埋め込み
    ↓
┌─────────────────────────────┐
│  Transformer Block × N層    │
│  ┌───────────────────────┐  │
│  │ LayerNorm             │  │
│  │ Multi-Head Attention  │  │
│  │ + 残差接続            │  │
│  ├───────────────────────┤  │
│  │ LayerNorm             │  │
│  │ Feed-Forward Network  │  │
│  │ + 残差接続            │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
    ↓
最終LayerNorm
    ↓
線形層 → logits (語彙サイズ次元)
    ↓
softmax → 次トークンの確率分布
```

### 自己注意機構 (Self-Attention)

Self-Attentionは「入力列の各トークンが、他のすべてのトークンとどれだけ関連するか」を計算する仕組みです。

**Q / K / V (Query, Key, Value)**

各トークンの埋め込みベクトルから、3つの線形変換で Q, K, V を生成します:

- **Query (Q)**: 「このトークンは何を探しているか」
- **Key (K)**: 「このトークンはどんな情報を持っているか」
- **Value (V)**: 「このトークンが提供する実際の情報」

**スケーリングドット積注意**

注意スコアの計算式:

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

- `Q × K^T` で各トークン対の類似度を計算
- `√d_k` でスケーリング (次元が大きいとドット積の値が大きくなりすぎ、softmaxが飽和するのを防ぐ)
- softmaxで確率分布に変換
- Vとの積で、関連度に応じた重み付き平均を取得

本プロジェクトでは `model.py` の `Head.forward()` でこの計算を行っています:

```python
wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
```

**因果マスク (Causal Mask)**

GPTはテキスト生成モデルなので、未来のトークンを参照してはいけません。下三角行列のマスクを適用し、未来方向の注意スコアを `-inf` にして、softmax後にゼロにします:

```python
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
```

### マルチヘッド注意 (Multi-Head Attention)

1つのAttentionヘッドだけでは、1種類の「関連性パターン」しか捉えられません。マルチヘッド注意では複数のヘッドを並列に走らせ、それぞれ異なるパターン (構文的関係、意味的関係など) を学習させます。

各ヘッドの出力を結合 (concatenate) し、線形変換で元の次元に戻します:

```python
out = torch.cat([h(x) for h in self.heads], dim=-1)
return self.dropout(self.proj(out))
```

ヘッド数が `n_head=4`、埋め込み次元が `n_embd=128` の場合、各ヘッドのサイズは `128 / 4 = 32` 次元です。

### トークン化 (Tokenization)

テキストをモデルが扱える数値列に変換するプロセスです。

**文字レベルトークン化 (本プロジェクトの方式)**

- 各文字を1トークンとして扱う
- 実装がシンプルで語彙サイズが小さい (数十〜数百)
- 長い文脈を扱う場合にトークン列が長くなる欠点がある

```python
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}  # 文字→整数
itos = {i: ch for i, ch in enumerate(chars)}  # 整数→文字
```

**サブワードトークン化 (実用LLMの方式)**

- BPE (Byte Pair Encoding) やSentencePieceが主流
- 頻出パターンを1トークンにまとめる (例: "the" → 1トークン)
- 語彙サイズと系列長のバランスが良い (典型的に32k〜100k語彙)

### 埋め込み (Embedding)

**トークン埋め込み**

各トークン (整数ID) を固定長のベクトルに変換するルックアップテーブルです。学習を通じて、意味的に近いトークンが近いベクトルになるように調整されます。

```python
self.token_emb = nn.Embedding(vocab_size, cfg.n_embd)  # (語彙サイズ, 128)
```

**位置埋め込み**

Transformerは系列の順序情報を持たないため、位置情報を明示的に加える必要があります。本プロジェクトでは学習可能な位置埋め込みを使用:

```python
self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # (128, 128)
```

最終的な入力は、トークン埋め込みと位置埋め込みの要素ごとの和です:

```python
x = tok_emb + pos_emb
```

### 残差接続とLayerNorm (Pre-LN)

**残差接続 (Residual Connection)**

深いネットワークでは勾配消失が問題になります。残差接続は入力をそのまま出力に加算することで、勾配が層を飛び越えて流れる経路を作ります:

```python
x = x + self.attn(self.ln1(x))  # 入力xをそのまま加算
x = x + self.ffn(self.ln2(x))
```

**LayerNorm (Pre-LN方式)**

本プロジェクトではAttentionやFFNの**前に** LayerNormを適用する Pre-LN 方式を採用しています。Post-LN (後に適用) よりも学習が安定することが知られています。

LayerNormは各トークンのベクトルを平均0・分散1に正規化し、学習可能なスケール・シフトパラメータで調整します。

### Feed-Forward Network (FFN)

各トークンの表現に対して独立に適用される2層のニューラルネットワークです。Attentionが「トークン間の関係」を処理するのに対し、FFNは「各トークンの表現を非線形変換」する役割を担います。

```python
self.net = nn.Sequential(
    nn.Linear(cfg.n_embd, 4 * cfg.n_embd),  # 128 → 512 に拡大
    nn.ReLU(),                                # 非線形活性化
    nn.Linear(4 * cfg.n_embd, cfg.n_embd),   # 512 → 128 に縮小
    nn.Dropout(cfg.dropout),
)
```

中間層を4倍に拡大するのはTransformerの標準的な設計です。

### 自己回帰生成と温度パラメータ

**自己回帰 (Autoregressive) 生成**

GPTは1トークンずつ順に生成します:

1. 現在のトークン列をモデルに入力
2. 最後の位置のlogitsから次トークンの確率分布を計算
3. 確率分布からサンプリングして次トークンを選択
4. 生成したトークンを列に追加し、1に戻る

```python
logits, _ = self(idx_cond)
logits = logits[:, -1, :] / temperature
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

**温度 (Temperature)**

温度パラメータはlogitsをsoftmaxに通す前にスケーリングします:

- **temperature = 1.0**: 元の確率分布をそのまま使用
- **temperature < 1.0** (例: 0.5): 分布が鋭くなり、高確率のトークンがさらに選ばれやすくなる → より保守的・一貫性のある出力
- **temperature > 1.0** (例: 1.5): 分布が平坦になり、低確率のトークンも選ばれやすくなる → より多様・創造的な出力

### 学習の仕組み

**交差エントロピー損失 (Cross-Entropy Loss)**

「モデルの予測確率分布」と「正解の次トークン」との差を測る損失関数です。モデルが正解トークンに高い確率を割り当てるほど損失は小さくなります:

```python
loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
```

**AdamW オプティマイザ**

勾配降下法の改良版で、以下の特徴があります:

- **適応的学習率**: パラメータごとに学習率を自動調整 (勾配の1次・2次モーメントを利用)
- **重み減衰 (Weight Decay)**: L2正則化を正しく適用し、過学習を抑制

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
```

学習ループでは、バッチごとに以下を繰り返します:

1. ランダムなバッチを取得
2. 順伝播 (forward) で損失を計算
3. 逆伝播 (backward) で勾配を計算
4. オプティマイザで重みを更新

---

## ファイル構成

| ファイル | 役割 |
|---|---|
| `config.py` | ハイパーパラメータをdataclassで一元管理 |
| `model.py` | Transformerモデル本体 (Head, MultiHeadAttention, FeedForward, Block, GPT) |
| `train.py` | データ読込、文字レベルトークナイザ、学習ループ、チェックポイント保存 |
| `generate.py` | チェックポイントからモデルを復元し、テキスト生成を行うCLI |
| `requirements.txt` | 依存パッケージ (PyTorch, NumPy) |

---

## 使い方

### インストール

```bash
pip install -r requirements.txt
```

### 学習

```bash
python train.py
```

`data/input.txt` がなければ青空文庫から日本文学12作品（夏目漱石、太宰治、芥川龍之介、宮沢賢治、森鷗外、島崎藤村）が自動でダウンロードされます（約120万文字）。学習完了後、`checkpoint.pt` が保存されます。

### テキスト生成

```bash
# デフォルト生成
python generate.py

# プロンプト指定・温度調整
python generate.py --prompt "To be" --max-tokens 300 --temperature 0.8
```

---

## ハイパーパラメータ

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `n_embd` | 256 | 埋め込みベクトルの次元数。大きいほど表現力が増すがメモリ・計算コストも増加 |
| `n_head` | 8 | マルチヘッド注意のヘッド数。`n_embd` の約数である必要がある |
| `n_layer` | 6 | Transformerブロックの積み重ね数。深いほど複雑なパターンを学習可能 |
| `block_size` | 128 | コンテキスト長 (一度に参照できる最大トークン数) |
| `dropout` | 0.1 | ドロップアウト率。過学習を防ぐための正則化 |
| `batch_size` | 64 | ミニバッチサイズ。大きいほど学習が安定するがメモリを多く消費 |
| `learning_rate` | 3e-4 | AdamWの学習率。大きすぎると発散、小さすぎると収束が遅い |
| `max_iters` | 20000 | 学習ステップ数 |
| `eval_interval` | 1000 | 検証損失を計算する間隔 (ステップ数) |
| `eval_iters` | 200 | 検証損失の推定に使うバッチ数 |
