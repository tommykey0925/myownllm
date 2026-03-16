"""Transformer (Bungaku) モデル本体 — 日本語文学向け言語モデル"""

# ============================================================
# 段階 3: モデルの用意（README セクション 3）
# ------------------------------------------------------------
# この段階では、入力 token ID を logits に変えるモデルを組み立てる。
# モデルの役割は「過去のtoken列を見て、各位置で次token候補のlogitsを出すこと」。
# （README 3-1）
#
# 入力: token ID tensor, shape (B, T)  — README 3-2
#   B = まとめて入れる系列の本数, T = 1本あたりのtoken数
# 出力: logits tensor, shape (B, T, V) — README 3-3
#   V = tokenの種類数。各位置ごとに候補token全員分のlogitsを返す。
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


# ============================================================
# Head — 単一の自己注意ヘッド（README 3-4-3, 3-4-4）
# ------------------------------------------------------------
# README 3-4-4 Attention:
#   今の位置が、過去のどの位置をどれだけ見るかを決める仕組み。
#   次tokenを決めるには、今のtokenだけでなく前の文脈が必要。
#   過去の各位置への「見る強さ」を決めて、その情報を混ぜる。
#
# README 3-4-3 Causal Mask:
#   今の位置より未来を見ないようにする制約。
#   言語モデルは「次」を予測するので、未来を見たらズルになる。
# ============================================================
class Head(nn.Module):
    """単一の自己注意ヘッド"""

    def __init__(self, cfg: Config, head_size: int):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        # causal mask — 未来の位置を見えなくする三角行列（README 3-4-3）
        self.register_buffer(
            "tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        # スケーリングドット積注意 — 「見る強さ」を計算（README 3-4-4）
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # 因果マスク
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)  # (B, T, head_size)
        return wei @ v      # (B, T, head_size) — 文脈込みのベクトル


# ============================================================
# MultiHeadAttention — 複数ヘッドを並列実行（README 3-5-4）
# ------------------------------------------------------------
# README 3-5-4: attentionの本数 — 文脈を見る視点の数。
# 1種類だけでなく、複数の見方で文脈を見たい。
# 複数のattentionを並列に使い、結果を結合する。
# ============================================================
class MultiHeadAttention(nn.Module):
    """複数ヘッドを並列実行し結合"""

    def __init__(self, cfg: Config):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


# ============================================================
# FeedForward — 位置ごとのFFN（README 3-4-5）
# ------------------------------------------------------------
# README 3-4-5: feedforward — attentionの後に、各位置のベクトルをさらに加工する部品。
# attentionで集めた情報を、さらに予測向きの形に変える。
# 各位置のベクトルを、位置ごとに追加で変換する。
# ============================================================
class FeedForward(nn.Module):
    """位置ごとのFFN"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Block — Transformerブロック（README 3-4-8, 3-4-6, 3-4-7）
# ------------------------------------------------------------
# README 3-4-8 Block:
#   attention・feedforward・layer norm・residual connectionをまとめた
#   1段ぶんの処理単位。blockを1個通るごとに、各位置のベクトルが
#   より文脈に合った表現になる。
#
# README 3-4-6 LayerNorm:
#   ベクトルの値の偏りを整える部品。値が偏りすぎると計算が不安定になりやすい。
#
# README 3-4-7 Residual Connection:
#   ある段の入力を、その段の出力に足し戻す仕組み。
#   元の情報を失いにくくし、深いモデルでも扱いやすくする。
# ============================================================
class Block(nn.Module):
    """Transformer ブロック: LayerNorm → Attention → 残差 → LayerNorm → FFN → 残差"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)       # LayerNorm（README 3-4-6）
        self.attn = MultiHeadAttention(cfg)         # Attention（README 3-4-4）
        self.ln2 = nn.LayerNorm(cfg.n_embd)        # LayerNorm（README 3-4-6）
        self.ffn = FeedForward(cfg)                 # FeedForward（README 3-4-5）

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # residual connection（README 3-4-7）
        x = x + self.ffn(self.ln2(x))   # residual connection（README 3-4-7）
        return x


# ============================================================
# Bungaku — 日本語文学言語モデル本体（README 3-4〜3-9）
# ============================================================
class Bungaku(nn.Module):
    """Bungaku — 日本語文学言語モデル"""

    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.cfg = cfg

        # ------------------------------------------------------------
        # README 3-6: 重みを持たせる / README 3-7: 重みに最初の値を入れる
        # weight（重み）= modelの答え方を決める内部の数字。
        # 初期化 = 重みに最初の値を入れること。未学習だが動くmodelにする。
        # ------------------------------------------------------------

        # Token Embedding（README 3-4-1）
        # token IDをベクトルに変える。IDを添字にしてembedding tableの対応行を取り出す。
        # 入力 (B, T) → (B, T, C)
        self.token_emb = nn.Embedding(vocab_size, cfg.n_embd)

        # Position Embedding（README 3-4-2）
        # 「何番目のtokenか」を表すベクトル。token embeddingに足して位置情報を持たせる。
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)

        # Block群（README 3-4-8, 3-5-5）
        # blockをn_layer個積む。各blockを通るごとに表現が良くなる。
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])

        # 最終LayerNorm
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # 最終出力層（README 3-4-10）
        # (B, T, C)のベクトル列を(B, T, V)のlogitsに変える。
        # 次token候補の可能性の点数を出す。
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)

    # ============================================================
    # 段階 3-8: 順方向計算 — forward（README 3-8）
    # ------------------------------------------------------------
    # token ID → embedding → position embedding → block群 → logits
    # という流れを定義する。
    #
    # README 3-1 logits:
    #   各候補tokenが「次に来そうか」を表す可能性の点数。
    #   モデルは各候補tokenごとのlogitsを返す。
    # ============================================================
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)                          # (B, T, n_embd) — Token Embedding
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))  # (T, n_embd) — Position Embedding
        x = tok_emb + pos_emb        # token embeddingにposition情報を足す
        x = self.blocks(x)           # block群を通す
        x = self.ln_f(x)             # 最終LayerNorm
        logits = self.lm_head(x)     # (B, T, vocab_size) — 最終出力層でlogitsに変換

        loss = None
        if targets is not None:
            # README 4-4: loss — modelの予測がどれだけ外れているかを表す数字
            # cross entropy: 正解tokenの確率が高いか低いかをもとに計算するloss
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        return logits, loss

    # ============================================================
    # 段階 6: 生成（README セクション 6-3〜6-7）
    # ------------------------------------------------------------
    # README 6-7-1 autoregressive:
    #   自分で出したtokenを、次の入力にまた使うやり方。
    #   forward → 最後の位置のlogits → softmax → 次tokenを選ぶ → 後ろに足す
    #   を繰り返す。
    # ============================================================
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """温度付き自己回帰生成"""
        for _ in range(max_new_tokens):
            # block_size以内にクロップ（README 3-5-2: Tの最大値）
            idx_cond = idx[:, -self.cfg.block_size:]

            # README 6-3: modelに通す — forwardでlogitsを得る
            logits, _ = self(idx_cond)

            # README 6-4: 最後の位置だけ使う
            # 生成では「次の1 token」だけほしいので、最後の位置の(B, V)を取り出す
            logits = logits[:, -1, :] / temperature

            # README 6-5: 次tokenを選ぶ
            # softmax: logitsを合計1になる確率っぽい値に変える
            probs = F.softmax(logits, dim=-1)
            # sampling: 確率分布にしたがって次tokenを1個選ぶ
            idx_next = torch.multinomial(probs, num_samples=1)

            # README 6-6: 選んだtokenを後ろに足す
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
