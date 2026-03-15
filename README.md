# myownllm

このリポジトリは、最小構成の言語モデルを通して、LLM の基本を学ぶための実装です。ここでは、言語モデル作成の流れを 1 〜 6 の段階に分けて説明します。

# 1. 学習データの用意

## 1-1. 何をする段階か

学習データの用意は、**モデルに読ませる文章を集める段階**です。
言語モデルは文章の並び方を学ぶので、まずは学習元になる文章が必要です。

## 1-2. 何を用意するか

用意するものは、**大量のテキスト**です。
たとえば小説、会話、ニュース、ソースコードなどです。
今回のような最小構成のモデルでは、1つまたは複数のテキストファイルをまとめたものを使います。

## 1-3. なぜ必要か

モデルは、何も文章を与えなければ何も学べません。
そのため、まずは **「どういう文章の続きを予測したいのか」** に対応するテキストを用意する必要があります。

## 1-4. この段階でのデータの状態

この段階では、データはまだ **ただの文章** です。
まだ token でも ID でも tensor でもありません。

例:

```text
吾輩は猫である。
名前はまだ無い。
```

## 1-5. この段階で決めること

### 1-5-1. どんな文章を使うか
何を学ばせたいかに応じて、使う文章の種類を決めます。

実際のコード（`train.py`）:

```python
AOZORA_WORKS = [
    ("https://www.aozora.gr.jp/cards/000148/files/752_ruby_2438.zip", "坊っちゃん"),
    ("https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip", "吾輩は猫である"),
    ("https://www.aozora.gr.jp/cards/000035/files/1567_ruby_4948.zip", "走れメロス"),
    ...
]
```

### 1-5-2. どれくらいの量を使うか
学習データが少なすぎると、モデルは十分なパターンを覚えにくくなります。

### 1-5-3. 前処理するか
不要な記号、注釈、ルビ、ヘッダ、フッタなどを除くかを決めます。
今回のような文学テキストでは、本文以外を削ることがあります。

実際のコード（`train.py`）:

```python
def clean_aozora(text: str) -> str:
    text = text.replace("｜", "")           # ルビ記号除去
    text = re.sub(r"《[^》]*》", "", text)   # ルビ除去
    text = re.sub(r"［＃[^］]*］", "", text) # 注記除去
```

## 1-6. この段階の完成状態

この段階が終わると、**学習に使うテキスト本文**が手元にあります。
ただしまだ、モデルが直接食べられる形にはなっていません。

実際のコード（`train.py`）:

```python
text = load_data()
```

# 2. データの token化・ID化・ベクトル化

## 2-1. 何をする段階か

この段階では、**文章をモデルが扱える数字の形に変える**準備をします。
文章のままでは計算できないので、順番に変換します。

流れはこうです。

**文章**
→ **token 化**
→ **ID 化**
→ **ベクトル化**

## 2-2. token 化

### 2-2-1. 用語: token
**意味:** 文章を分けた部品。
**なぜ必要か:** モデルは文章全体を一気に扱うのではなく、部品の並びとして扱うから。
**どう使うか:** 文章を token の列に分ける。

### 2-2-2. 今回の token 化
今回は **1文字 = 1 token** です。

例:

```text
吾 / 輩 / は / 猫 / で / あ / る
```

実際のコード（`train.py`）:

```python
chars = sorted(set(text))
```

### 2-2-3. なぜ token 化するか
文章を計算対象にするには、まず「どの単位で扱うか」を決める必要があるからです。
今回は最小構成なので、文字単位にします。

## 2-3. ID 化

### 2-3-1. 用語: token ID
**意味:** token に付けた番号。
**なぜ必要か:** token は文字列のままだと計算しにくいので、番号にするため。
**どう使うか:** token ごとに一意な番号を付け、token の列を番号列に変える。

### 2-3-2. どうやるか
まず、使われている token を重複なく集めます。
そのあと、それぞれに番号を振ります。

例:

```text
吾 -> 0
輩 -> 1
は -> 2
猫 -> 3
で -> 4
あ -> 5
る -> 6
```

すると文章は、

```text
[0, 1, 2, 3, 4, 5, 6]
```

のような **token ID 列** になります。

実際のコード（`train.py`）:

```python
stoi = {ch: i for i, ch in enumerate(chars)}  # string to integer
itos = {i: ch for i, ch in enumerate(chars)}  # integer to string
encode = lambda s: [stoi[c] for c in s]       # 文章 → token ID列
decode = lambda l: "".join([itos[i] for i in l])  # token ID列 → 文章
```

### 2-3-3. 用語: 語彙数
**意味:** token の種類数。
**なぜ必要か:** モデルが何種類の token を扱うかに関わるから。
**どう使うか:** embedding table の大きさや出力の候補数に使う。

実際のコード（`train.py`）:

```python
vocab_size = len(chars)
```

## 2-4. tensor 化

### 2-4-1. 用語: tensor
**意味:** 数字をまとめて入れる箱。
**なぜ必要か:** token ID の列を、まとめて計算できる形にするため。
**どう使うか:** token ID の列を tensor にする。

例:

```text
[0, 1, 2, 3, 4, 5, 6]
```

を、数値 tensor として持ちます。

実際のコード（`train.py`）:

```python
data = torch.tensor(encode(text), dtype=torch.long)
```

## 2-5. ベクトル化

### 2-5-1. 用語: ベクトル
**意味:** 数字を横に並べたもの。
**なぜ必要か:** token ID のままだと、token の特徴を表しにくいから。
**どう使うか:** token ID をベクトルへ変える。

### 2-5-2. 用語: embedding
**意味:** token ID をベクトルに変えること。
**なぜ必要か:** 番号だけではなく、計算に向いた内部表現が必要だから。
**どう使うか:** token ID を添字として embedding table の対応行を取り出す。

実際のコード（`model.py`）:

```python
self.token_emb = nn.Embedding(vocab_size, cfg.n_embd)
```

### 2-5-3. 用語: embedding table
**意味:** token ID ごとのベクトル一覧表。
**なぜ必要か:** 各 token に対応する内部表現を持つため。
**どう使うか:** たとえば ID が `7` なら、table の 7 行目を取る。

例:

```text
7 -> [0.2, -0.5, 1.1, ...]
```

実際のコード（`model.py`）:

```python
self.token_emb = nn.Embedding(vocab_size, cfg.n_embd)
```

## 2-6. この段階でのデータの形

この段階の終わりには、文章は最終的に

- token の列
- token ID の列
- token ID tensor
- embedding 後のベクトル列

として扱えるようになります。

## 2-7. この段階の完成状態

この段階が終わると、**文章がモデルに入力できる数字の形**になっています。
まだモデル本体は作っていませんが、**学習データをモデルに渡す準備**はできています。

# 3. モデルの用意

## 3-1. モデルの役割を決める

### 3-1-1. 用語: logits
**意味:** 各候補 token が「次に来そうか」を表す可能性の点数。確率そのものではない。
**なぜ必要か:** 言語モデルの仕事は、次 token 候補を比較できる形で出すことだから。
**どう使うか:** モデルは、各候補 token ごとの logits を返す。

このモデルの役割は、
**過去の token 列を見て、各位置で次 token 候補の logits を出すこと**
です。

## 3-2. 入力の仕様を決める

### 3-2-1. 用語: token ID
**意味:** token に付けた番号。
**なぜ必要か:** 文章そのままではなく、番号列としてモデルに入れるため。
**どう使うか:** 入力は token ID の列にする。

### 3-2-2. 用語: tensor
**意味:** 数字をまとめて入れる箱。
**なぜ必要か:** 複数の token ID 列をまとめて計算するため。
**どう使うか:** 入力は token ID tensor にする。

### 3-2-3. 用語: shape `(B, T)`
**意味:**
- `B` = まとめて入れる系列の本数
- `T` = 1本あたりの token 数
**なぜ必要か:** モデルがどの形の入力を受け取るかを決めるため。
**どう使うか:** 入力は通常 `(B, T)` の tensor にする。

例: `(64, 256)`
= **256 token の系列を 64 本まとめて入れる**

## 3-3. 出力の仕様を決める

### 3-3-1. 用語: shape `(B, T, V)`
**意味:**
- `B` = 系列の本数
- `T` = 各位置
- `V` = token の種類数
**なぜ必要か:** 各位置ごとに、候補 token 全員分の logits を返したいから。
**どう使うか:** 出力は通常 `(B, T, V)` の logits tensor にする。

つまり出力は、
**各系列の各位置について、次 token 候補 V 個ぶんの logits**
です。

## 3-4. 中の部品を決める

ここでは、**入力 token ID を logits に変えるまでの中身**を決めます。

### 3-4-1. Token Embedding

**用語: embedding**
**意味:** token ID をベクトルに変えること。
**なぜ必要か:** token ID はただの番号で、そのままだと token の特徴を表しにくいから。
**どう使うか:** token ID を、対応するベクトルへ置き換える。

**用語: embedding table**
**意味:** token ID ごとのベクトル一覧表。
**なぜ必要か:** 各 token の内部表現を持つため。
**どう使うか:** ID を添字にして、その行のベクトルを取り出す。

結果として、入力 `(B, T)` は
`(B, T, C)` のベクトル列になります。

実際のコード（`model.py`）:

```python
self.token_emb = nn.Embedding(vocab_size, cfg.n_embd)
```

### 3-4-2. Position Embedding

**用語: position embedding**
**意味:** 「何番目の token か」を表すベクトル。
**なぜ必要か:** 同じ token でも位置が違えば役割が違うから。
**どう使うか:** token embedding に足して、位置情報を持たせる。

実際のコード（`model.py`）:

```python
self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
# ...
x = tok_emb + pos_emb
```

### 3-4-3. Causal Mask

**用語: causal mask**
**意味:** 今の位置より未来を見ないようにする制約。
**なぜ必要か:** 言語モデルは「次」を予測するので、未来を見たらズルになるから。
**どう使うか:** 後ろの位置を参照できないようにする。

実際のコード（`model.py`）:

```python
self.register_buffer(
    "tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
)
# ...
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
```

### 3-4-4. Attention

**用語: attention**
**意味:** 今の位置が、過去のどの位置をどれだけ見るかを決める仕組み。
**なぜ必要か:** 次 token を決めるには、今の token だけでなく前の文脈が必要だから。
**どう使うか:** 過去の各位置への「見る強さ」を決めて、その情報を混ぜる。
具体的には、各位置のベクトルから query（何を知りたいか）、key（何を持っているか）、value（実際の情報）の3つを作る。
query と key の内積で「見る強さ」を計算し、スケーリングで値の大きさを調整したあと、softmax で合計1の比率に変える。
その比率で value を混ぜ合わせたものが、文脈込みのベクトルになる。

結果として、各位置のベクトルは
**文脈込みのベクトル** になります。

実際のコード（`model.py`）:

```python
k = self.key(x)    # (B, T, head_size)
q = self.query(x)  # (B, T, head_size)
wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
v = self.value(x)
out = wei @ v       # (B, T, head_size)
```

### 3-4-5. FeedForward

**用語: feedforward**
**意味:** attention の後に、各位置のベクトルをさらに加工する部品。
**なぜ必要か:** attention で集めた情報を、さらに予測向きの形に変えたいから。
**どう使うか:** 各位置のベクトルを、位置ごとに追加で変換する。
中では、まずベクトルを広げてから ReLU（活性化関数）を通し、元の幅に戻す。
ReLU は負の値を 0 にする関数で、線形変換だけでは表現できないパターンを扱えるようにする。
Dropout は学習時にランダムに一部の値を 0 にすることで、特定のパターンに頼りすぎるのを防ぐ。

実際のコード（`model.py`）:

```python
self.net = nn.Sequential(
    nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
    nn.ReLU(),
    nn.Linear(4 * cfg.n_embd, cfg.n_embd),
    nn.Dropout(cfg.dropout),
)
```

### 3-4-6. LayerNorm

**用語: layer norm**
**意味:** ベクトルの値の偏りを整える部品。
**なぜ必要か:** 値が偏りすぎると計算が不安定になりやすいから。
**どう使うか:** attention や feedforward の前後で使って、ベクトルを整える。

実際のコード（`model.py`）:

```python
self.ln1 = nn.LayerNorm(cfg.n_embd)
```

### 3-4-7. Residual Connection

**用語: residual connection**
**意味:** ある段の入力を、その段の出力に足し戻す仕組み。
**なぜ必要か:** 元の情報を失いにくくし、深いモデルでも扱いやすくするため。
**どう使うか:** 新しいベクトルに元のベクトルを足す。

実際のコード（`model.py`）:

```python
x = x + self.attn(self.ln1(x))
```

### 3-4-8. Block

**用語: block**
**意味:** attention・feedforward・layer norm・residual connection をまとめた1段ぶんの処理単位。
**なぜ必要か:** 同じ種類の処理を何段も積んで、各位置の表現を少しずつ良くするため。
**どう使うか:** block を1個通るごとに、各位置のベクトルがより文脈に合った表現になる。

実際のコード（`model.py`）:

```python
class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffn = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
```

### 3-4-9. 最終出力層

**用語: 最終出力層**
**意味:** 最後のベクトルを、候補 token 全員分の logits に変える部品。
**なぜ必要か:** 最後に欲しいのは「次 token 候補の可能性の点数」だから。
**どう使うか:** `(B, T, C)` のベクトル列を `(B, T, V)` の logits に変える。

実際のコード（`model.py`）:

```python
self.lm_head = nn.Linear(cfg.n_embd, vocab_size)
# ...
logits = self.lm_head(x)  # (B, T, vocab_size)
```

## 3-5. モデルの大きさを決める

### 3-5-1. `V`
**意味:** token の種類数。
**なぜ必要か:** embedding table の行数と、出力の候補数になるから。
**どう使うか:** 語彙数として使う。

実際のコード（`train.py`）:

```python
vocab_size = len(chars)
```

### 3-5-2. `T` の最大値
**意味:** 一度に見られる最大 token 数。
**なぜ必要か:** モデルが扱える長さの上限になるから。
**どう使うか:** 最大入力長として使う。

### 3-5-3. `C`
**意味:** 1 token を表すベクトルの長さ。
**なぜ必要か:** 内部表現の豊かさを決めるから。
**どう使うか:** embedding や中間表現の幅として使う。

### 3-5-4. attention の本数
**意味:** 文脈を見る視点の数。
**なぜ必要か:** 1種類だけでなく、複数の見方で文脈を見たいから。
**どう使うか:** 複数の attention を並列に使う。

### 3-5-5. block の段数
**意味:** block を何個積むか。
**なぜ必要か:** 表現を何段階で作り直すかを決めるから。
**どう使うか:** model の深さになる。

実際のコード（`config.py`）:

```python
n_embd: int = 256       # C — ベクトルの長さ
n_head: int = 8         # attentionの本数
n_layer: int = 6        # blockの段数
block_size: int = 128   # T の最大値
```

## 3-6. 重みを持たせる

**用語: weight（重み）**
**意味:** モデルの答え方を決める内部の数字。
**なぜ必要か:** 同じ入力でも、重みが違えば出てくる logits が変わるから。
**どう使うか:** model の中の各部品に持たせる。

重みがある場所は主に

- embedding table
- position embedding
- attention の中
- feedforward の中
- 最終出力層

です。

## 3-7. 重みに最初の値を入れる

**用語: 初期化**
**意味:** 重みに最初の値を入れること。
**なぜ必要か:** 重みの置き場だけでは model が動かないから。
**どう使うか:** 最初の値を入れて、未学習だが動く model にする。

この時点では、
**入力を入れれば logits は返るが、まだ当てにならない**。

## 3-8. 順方向と逆方向に計算できる形にする

**用語: forward**
**意味:** 入力から出力まで順方向に計算すること。
**なぜ必要か:** token ID から logits を出せないと model にならないから。
**どう使うか:**
`token ID → embedding → position embedding → block 群 → logits`
という流れを定義する。

**用語: backward**
**意味:** 結果から重みへ逆方向にたどれること。
**なぜ必要か:** あとで model を調整可能にするため。
**どう使うか:** backward できる形で model を組む。

実際のコード（`model.py`）:

```python
def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_emb = self.token_emb(idx)                          # Token Embedding
    pos_emb = self.pos_emb(torch.arange(T, device=idx.device))  # Position Embedding
    x = tok_emb + pos_emb
    x = self.blocks(x)        # block群を通す
    x = self.ln_f(x)          # 最終LayerNorm
    logits = self.lm_head(x)  # 最終出力層でlogitsに変換
```

## 3-9. モデルを1個作る

**用語: 実体化**
**意味:** 設計した model を、実際のオブジェクトとして作ること。
**なぜ必要か:** 設計図だけでは使えず、実物が必要だから。
**どう使うか:** 重みを持った未学習 model を 1 個生成する。

実際のコード（`train.py`）:

```python
model = Bungaku(cfg, vocab_size).to(cfg.device)
```

# 4. 学習

## 4-1. 何をする段階か

学習は、**3で用意した model の weight を、次 token を当てやすい方向へ変える段階**です。

3の時点では model は

- 入力を受け取れる
- logits を返せる
- backward できる

状態でした。
4では、その model に実際のデータを何度も通して、**weight を調整**します。

## 4-2. 学習に使う入力と正解を作る

### 4-2-1. 用語: input
**意味:** model に入れる token ID 列。
**なぜ必要か:** model が「ここまでの文章」を見るため。
**どう使うか:** たとえば `[A, B, C, D]` のような列を model に入れる。

### 4-2-2. 用語: target
**意味:** 正解の次 token。
**なぜ必要か:** model の出した logits が合っているか比べるため。
**どう使うか:** 入力を1つ右にずらした列を正解にする。

例:

入力
`[A, B, C, D]`

正解
`[B, C, D, E]`

意味は、

- `A` を見たら次は `B`
- `A, B` を見たら次は `C`
- `A, B, C` を見たら次は `D`
- `A, B, C, D` を見たら次は `E`

です。

### 4-2-3. 用語: batch
**意味:** まとめて処理する複数の入力列。
**なぜ必要か:** 1本ずつより、まとめて計算した方が効率がいいから。
**どう使うか:** shape `(B, T)` の input と target をまとめて model に入れる。

実際のコード（`train.py`）:

```python
ix = torch.randint(len(d) - cfg.block_size, (cfg.batch_size,))
x = torch.stack([d[i : i + cfg.block_size] for i in ix])      # input (B, T)
y = torch.stack([d[i + 1 : i + cfg.block_size + 1] for i in ix])  # target (B, T)
```

## 4-3. 順方向に計算する

### 4-3-1. 用語: forward
**意味:** 入力から logits まで順方向に計算すること。
**なぜ必要か:** まず model が何を予測したか出さないと、正解と比べられないから。
**どう使うか:** input を model に入れて logits を得る。

ここで model は、各位置について
**次 token 候補の logits** を返します。

実際のコード（`train.py`）:

```python
_, loss = model(xb, yb)
```

## 4-4. 外れ具合を1個の数字にする

### 4-4-1. 用語: loss
**意味:** model の予測がどれだけ外れているかを表す数字。
**なぜ必要か:** 外れ具合を1個の数字にしないと、weight をどちらに直すべきか決めにくいから。
**どう使うか:** logits と target を比べて計算する。

### 4-4-2. 用語: softmax
**意味:** logits を、合計1になる確率っぽい値に変える関数。
**なぜ必要か:** 候補ごとの強さを、比率として扱いやすくするため。
**どう使うか:** loss を計算するときに、内部で使われることが多い。

### 4-4-3. 用語: cross entropy
**意味:** 正解 token の確率が高いか低いかをもとに計算する loss。
**なぜ必要か:** 言語モデルの next-token prediction と相性がよいから。
**どう使うか:** logits と target を入れて、1個の loss を出す。

ここで大事なのは、
**model が直接出すのは logits** で、
**loss を作る段階で softmax 相当の処理が使われる**
ことです。

実際のコード（`model.py`）:

```python
B, T, C = logits.shape
loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
```

## 4-5. 逆方向に計算する

### 4-5-1. 用語: gradient（勾配）
**意味:** 各 weight を、どっち向きにどれくらい動かせば loss が下がるかを表す数字。
**なぜ必要か:** weight を適当に変えるのではなく、loss が下がる方向へ動かすため。
**どう使うか:** backward によって各 weight に対して計算される。

### 4-5-2. 用語: backward
**意味:** loss から逆向きにたどって、各 weight の gradient を計算すること。
**なぜ必要か:** どの weight をどう直すべきか知るため。
**どう使うか:** loss が出たあとに backward する。

ここで計算されるのは
**新しい weight** ではなく、
**weight の直し方** です。

実際のコード（`train.py`）:

```python
loss.backward()
```

## 4-6. weight を更新する

### 4-6-1. 用語: optimizer
**意味:** gradient を使って、実際に weight を更新する仕組み。
**なぜ必要か:** backward は直し方を出すだけで、実際の変更はしないから。
**どう使うか:** gradient を見て weight を少し動かす。

実際のコード（`train.py`）:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
```

### 4-6-2. 用語: learning rate
**意味:** weight を1回でどれくらい動かすかを決める大きさ。
**なぜ必要か:** 大きすぎると不安定になり、小さすぎると進みが遅いから。
**どう使うか:** optimizer が更新量を決めるときに使う。

### 4-6-3. 用語: step
**意味:** optimizer による1回の更新。
**なぜ必要か:** gradient を実際の weight 変更に変えるため。
**どう使うか:** backward のあとに step する。

実際のコード（`train.py`）:

```python
optimizer.step()
```

## 4-7. これを繰り返す

### 4-7-1. 用語: iteration / step
**意味:** 1回の
**input 作成 → forward → loss → backward → update**
の流れ。
**なぜ必要か:** 1回だけではほとんど学べないから。
**どう使うか:** 何千回、何万回と繰り返す。
各 iteration の先頭では、前回の gradient をゼロにリセットする。
リセットしないと前回の gradient が残ったまま加算されてしまうため。

学習とは結局、
**loss が下がる方向へ weight を何回も直し続けること**
です。

実際のコード（`train.py`）:

```python
for step in range(cfg.max_iters):
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)           # forward + loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()                    # backward
    optimizer.step()                   # weight更新
```

## 4-8. 成績を見る

### 4-8-1. 用語: train data
**意味:** weight を更新するのに使うデータ。
**なぜ必要か:** model を実際に学ばせるため。
**どう使うか:** forward / backward / update に使う。

### 4-8-2. 用語: validation data
**意味:** weight 更新には使わず、成績確認だけに使うデータ。
**なぜ必要か:** 学習用データだけで上手く見えても、本当に良くなっているとは限らないから。
**どう使うか:** loss を測るだけに使う。

実際のコード（`train.py`）:

```python
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

# 5. 保存

## 5-1. 何をする段階か

保存は、**学習が進んだ model をあとで再利用できるように残す段階**です。

学習には時間がかかるので、毎回最初からやり直さないようにします。

## 5-2. 何を保存するか

### 5-2-1. 用語: weight
**意味:** model の答え方を決める内部の数字。
**なぜ必要か:** これが学習の成果そのものだから。
**どう使うか:** 一番重要な保存対象になる。

### 5-2-2. 用語: checkpoint
**意味:** 学習途中または学習後の状態をまとめて保存したもの。
**なぜ必要か:** 後で同じ model を復元するため。
**どう使うか:** file として保存しておく。

実際のコード（`train.py`）:

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "vocab_size": vocab_size,
    "chars": chars,
    "config": cfg,
}
torch.save(checkpoint, cfg.checkpoint_path)
```

## 5-3. 一緒に保存する情報

### 5-3-1. 用語: 語彙情報
**意味:** token と token ID の対応。
**なぜ必要か:** 同じ weight でも、どの ID がどの token か分からないと使えないから。
**どう使うか:** `stoi` や `itos` のような対応表を保存する。

### 5-3-2. 用語: 設定値
**意味:** model の大きさや構造を決める値。
**なぜ必要か:** 復元時に同じ形の model を作る必要があるから。
**どう使うか:** `vocab_size`、`block_size`、`n_embd` などを保存する。

実際のコード（`train.py`）:

```python
checkpoint = {
    "model_state_dict": model.state_dict(),  # weight
    "vocab_size": vocab_size,                # 設定値
    "chars": chars,                          # 語彙情報
    "config": cfg,                           # 設定値
}
```

## 5-4. 保存の完成状態

この段階が終わると、

- 学習済み weight
- token と ID の対応
- model の設定

が file に残ります。
これで、あとから同じ model を読み込めます。

# 6. 生成

## 6-1. 何をする段階か

生成は、**保存した model を使って、新しい token を1個ずつ出していく段階**です。

## 6-2. 最初の入力を用意する

### 6-2-1. 用語: prompt
**意味:** 生成の出発点として与える最初の token 列。
**なぜ必要か:** 何も入力がないと、どんな続きを出せばいいか決めにくいから。
**どう使うか:** たとえば「吾輩は」のような最初の文字列を入れる。

### 6-2-2. どう使うか
prompt を token 化し、ID 化して、token ID tensor にして model に入れる。

実際のコード（`generate.py`）:

```python
if args.prompt:
    context = torch.tensor([encode(args.prompt)], dtype=torch.long, device=cfg.device)
else:
    context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
```

## 6-3. model に通す

### 6-3-1. 用語: forward
**意味:** 入力から logits を出す順方向の計算。
**なぜ必要か:** まず次 token 候補の logits を出さないと選べないから。
**どう使うか:** 現在の token 列を model に入れて logits を得る。

ここで model の出力は、定義どおり `(B, T, V)` の logits です。

実際のコード（`model.py`）:

```python
logits, _ = self(idx_cond)
```

## 6-4. 最後の位置だけ使う

### 6-4-1. なぜ最後だけ使うか
生成では、**今ほしいのは「次の1 token」** だけだからです。
なので、各位置の logits 全部は出ていても、使うのは **最後の位置の logits** です。

### 6-4-2. どう使うか
`(B, T, V)` の中から、最後の位置の `(B, V)` を取り出します。
これが、**今この瞬間の次 token 候補の logits** です。

取り出した logits を temperature で割る。
temperature が低いと差が大きくなり（確信度の高い token が選ばれやすい）、高いと差が小さくなり（いろんな token が選ばれやすい）。

実際のコード（`model.py`）:

```python
logits = logits[:, -1, :] / temperature
```

## 6-5. 次 token を選ぶ

### 6-5-1. 用語: softmax
**意味:** logits を、合計1になる確率っぽい値に変える関数。
**なぜ必要か:** 候補の強さを、選びやすい形にしたいから。
**どう使うか:** 最後の位置の logits に softmax をかける。

### 6-5-2. 用語: 確率分布
**意味:** 各候補 token の選ばれやすさを表す比率。合計は1。
**なぜ必要か:** どの token を出すか決めるため。
**どう使うか:** softmax の結果として得る。

### 6-5-3. 用語: sampling
**意味:** 確率分布にしたがって、次 token を1個選ぶこと。
**なぜ必要か:** 毎回同じ token だけでなく、確率に応じて選べるようにするため。
**どう使うか:** softmax 後の分布から1個選ぶ。

実際のコード（`model.py`）:

```python
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

## 6-6. 選んだ token を後ろに足す

### 6-6-1. 何をするか
選んだ次 token を、今の列の後ろに追加します。

例:

今の列
`[A, B, C]`

次 token に `D` を選んだら

`[A, B, C, D]`

になります。

実際のコード（`model.py`）:

```python
idx = torch.cat([idx, idx_next], dim=1)
```

## 6-7. これを繰り返す

### 6-7-1. 用語: autoregressive
**意味:** 自分で出した token を、次の入力にまた使うやり方。
**なぜ必要か:** 1 token ずつ続きを伸ばしていくため。
**どう使うか:**
`forward → 最後の位置の logits → softmax → 次 token を選ぶ → 後ろに足す`
を繰り返す。

これにより、1文字だけでなく何文字でも生成できます。

実際のコード（`model.py`）:

```python
for _ in range(max_new_tokens):
    idx_cond = idx[:, -self.cfg.block_size:]
    logits, _ = self(idx_cond)
    logits = logits[:, -1, :] / temperature
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
    idx = torch.cat([idx, idx_next], dim=1)
```

## 6-8. 生成の完成状態

この段階が終わると、model は prompt の続きとして token 列を伸ばし、最終的に文章を生成します。
最後は token ID を文字や token に戻して、人間が読める形にします。

実際のコード（`generate.py`）:

```python
print(decode(output[0].tolist()))
```
