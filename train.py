"""データ読込・学習ループ・チェックポイント保存"""

import os
import re
import torch

from config import Config
from model import Bungaku

cfg = Config()


# ============================================================
# 段階 1: 学習データの用意（README セクション 1）
# ------------------------------------------------------------
# README 1-1: 学習データの用意は、モデルに読ませる文章を集める段階。
# 言語モデルは文章の並び方を学ぶので、まず学習元になる文章が必要。
# README 1-4: この段階では、データはまだ「ただの文章」。
#   まだtokenでもIDでもtensorでもない。
# ============================================================


# ------------------------------------------------------------
# README 1-5-3: 前処理するか
# 不要な記号、注釈、ルビ、ヘッダ、フッタなどを除くかを決める。
# 今回のような文学テキストでは、本文以外を削る。
# ------------------------------------------------------------
def clean_aozora(text: str) -> str:
    """青空文庫テキストからルビ・注記・ヘッダー/フッターを除去する"""
    # ヘッダー除去（「-------...」区切り以降が本文）
    parts = re.split(r"-{5,}", text)
    if len(parts) >= 3:
        # 最初のセクション（タイトル・凡例）を除去
        text = "".join(parts[2:])
    # フッター除去（「底本：」以降を削除）
    text = re.split(r"底本：", text)[0]

    # ルビ記号 ｜ を除去
    text = text.replace("｜", "")
    # ルビ《...》を除去
    text = re.sub(r"《[^》]*》", "", text)
    # 注記 ［＃...］ を除去
    text = re.sub(r"［＃[^］]*］", "", text)
    # 連続空行を1つに
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ------------------------------------------------------------
# README 1-5-1: どんな文章を使うか
# 何を学ばせたいかに応じて、使う文章の種類を決める。
# → 青空文庫の著作権切れ日本文学作品を使用。
#
# README 1-5-2: どれくらいの量を使うか
# 学習データが少なすぎると、モデルは十分なパターンを覚えにくくなる。
# → 12作品を使用。
# ------------------------------------------------------------
AOZORA_WORKS = [
    ("https://www.aozora.gr.jp/cards/000148/files/752_ruby_2438.zip", "坊っちゃん"),
    ("https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip", "吾輩は猫である"),
    ("https://www.aozora.gr.jp/cards/000035/files/1567_ruby_4948.zip", "走れメロス"),
    ("https://www.aozora.gr.jp/cards/000879/files/127_ruby_150.zip", "羅生門"),
    ("https://www.aozora.gr.jp/cards/000879/files/92_ruby_164.zip", "蜘蛛の糸"),
    ("https://www.aozora.gr.jp/cards/000081/files/456_ruby_145.zip", "銀河鉄道の夜"),
    ("https://www.aozora.gr.jp/cards/000129/files/2078_ruby_15898.zip", "舞姫"),
    ("https://www.aozora.gr.jp/cards/000148/files/773_ruby_5968.zip", "こころ"),
    ("https://www.aozora.gr.jp/cards/000148/files/794_ruby_4237.zip", "三四郎"),
    ("https://www.aozora.gr.jp/cards/000158/files/1502_ruby_24534.zip", "破戒"),
    ("https://www.aozora.gr.jp/cards/000035/files/301_ruby_5915.zip", "人間失格"),
    ("https://www.aozora.gr.jp/cards/000035/files/1565_ruby_8220.zip", "斜陽"),
]


# ------------------------------------------------------------
# README 1-1〜1-6: 学習データの用意
# テキスト本文を手元に用意する。まだモデルが直接食べられる形ではない。
# README 1-6: この段階の完成状態 — 学習に使うテキスト本文が手元にある。
# ------------------------------------------------------------
def load_data():
    if os.path.exists(cfg.data_path):
        with open(cfg.data_path, "r", encoding="utf-8") as f:
            return f.read()
    # 青空文庫から複数作品をダウンロード
    print(f"{cfg.data_path} が見つかりません。青空文庫からダウンロードします...")
    import urllib.request
    import zipfile
    import io

    os.makedirs(os.path.dirname(cfg.data_path) or ".", exist_ok=True)
    texts = []
    for url, title in AOZORA_WORKS:
        try:
            print(f"ダウンロード中: {title} ({url})")
            response = urllib.request.urlopen(url)
            zip_data = response.read()
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                txt_names = [n for n in zf.namelist() if n.endswith(".txt")]
                if not txt_names:
                    print(f"  スキップ: {title} — ZIPにテキストファイルなし")
                    continue
                raw = zf.read(txt_names[0]).decode("shift_jis")
            cleaned = clean_aozora(raw)
            texts.append(cleaned)
            print(f"  完了: {title} ({len(cleaned):,} 文字)")
        except Exception as e:
            print(f"  スキップ: {title} — {e}")
            continue

    if not texts:
        raise RuntimeError("1作品もダウンロードできませんでした")

    text = "\n\n".join(texts)
    with open(cfg.data_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"保存完了: {cfg.data_path} ({len(text):,} 文字, {len(texts)}作品)")
    return text


text = load_data()

# ============================================================
# 段階 2: データのtoken化・ID化・tensor化（README セクション 2）
# ------------------------------------------------------------
# README 2-1: 文章をモデルが扱える数字の形に変える。
#   文章 → token化 → ID化 → tensor化
# ============================================================

# --- README 2-2: token化 ---
# README 2-2-1: token = 文章を分けた部品。モデルは部品の並びとして扱う。
# README 2-2-2: 今回は1文字 = 1 token（文字レベルトークナイザ）。
chars = sorted(set(text))

# --- README 2-3-3: 語彙数 ---
# tokenの種類数。embedding tableの大きさや出力の候補数に使う。
vocab_size = len(chars)

# --- README 2-3: ID化 ---
# README 2-3-1: token ID = tokenに付けた番号。文字列のままだと計算しにくいので番号にする。
# README 2-3-2: 使われているtokenを重複なく集め、それぞれに番号を振る。
stoi = {ch: i for i, ch in enumerate(chars)}  # string to integer
itos = {i: ch for i, ch in enumerate(chars)}  # integer to string
encode = lambda s: [stoi[c] for c in s]       # 文章 → token ID列
decode = lambda l: "".join([itos[i] for i in l])  # token ID列 → 文章

print(f"語彙サイズ: {vocab_size}, テキスト長: {len(text):,} 文字")

# --- README 2-4: tensor化 ---
# README 2-4-1: tensor = 数字をまとめて入れる箱。
# token IDの列を、まとめて計算できる形にする。
data = torch.tensor(encode(text), dtype=torch.long)

# --- train/val分割 ---
# README 4-8-1: train data — weightを更新するのに使うデータ。
# README 4-8-2: validation data — weight更新には使わず、成績確認だけに使うデータ。
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# ============================================================
# 段階 4: 学習（README セクション 4）
# ------------------------------------------------------------
# README 4-1: 学習は、modelのweightを次tokenを当てやすい方向へ変える段階。
# ============================================================


# ------------------------------------------------------------
# README 4-2: 学習に使う入力と正解を作る
# README 4-2-1: input — modelに入れるtoken ID列。
# README 4-2-2: target — 正解の次token。入力を1つ右にずらした列。
# README 4-2-3: batch — まとめて処理する複数の入力列。
#   shape (B, T)のinputとtargetをまとめてmodelに入れる。
# ------------------------------------------------------------
def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([d[i : i + cfg.block_size] for i in ix])      # input (B, T)
    y = torch.stack([d[i + 1 : i + cfg.block_size + 1] for i in ix])  # target (B, T)
    return x.to(cfg.device), y.to(cfg.device)


# ------------------------------------------------------------
# README 4-8: 成績を見る
# train dataとvalidation dataそれぞれのlossを測って、学習の進み具合を確認する。
# ------------------------------------------------------------
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


# ============================================================
# 段階 3-9: モデルを1個作る（README 3-9）
# ------------------------------------------------------------
# README 3-9: 設計したmodelを実際のオブジェクトとして作る。
# 重みを持った未学習modelを1個生成する。
# ============================================================
model = Bungaku(cfg, vocab_size).to(cfg.device)
param_count = sum(p.numel() for p in model.parameters())
print(f"パラメータ数: {param_count:,}")
print(f"デバイス: {cfg.device}")

# --- README 4-6-1: optimizer ---
# gradient（勾配）を使って、実際にweightを更新する仕組み。
# backwardは直し方を出すだけで、実際の変更はしない。optimizerがgradientを見てweightを少し動かす。
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

# ============================================================
# 段階 4-7: これを繰り返す — 学習ループ（README 4-7）
# ------------------------------------------------------------
# README 4-7-1: iteration = 1回の
#   input作成 → forward → loss → backward → update の流れ。
# 学習とは結局「lossが下がる方向へweightを何回も直し続けること」。
# ============================================================
print(f"\n学習開始 ({cfg.max_iters} ステップ)...")
for step in range(cfg.max_iters):
    # --- README 4-8: 成績を見る ---
    if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
        losses = estimate_loss(model)
        print(
            f"ステップ {step:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    # --- README 4-3: forward — 入力からlogitsまで順方向に計算する ---
    # --- README 4-4: loss — modelの予測がどれだけ外れているかを1個の数字にする ---
    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)

    # --- README 4-5: backward — lossから逆向きにたどって各weightのgradientを計算する ---
    # README 4-5-1: gradient = 各weightをどっち向きにどれくらい動かせばlossが下がるかを表す数字。
    loss.backward()

    # --- README 4-6: weight更新 ---
    # README 4-6-3: step — optimizerによる1回の更新。gradientを実際のweight変更に変える。
    optimizer.step()

# ============================================================
# 段階 5: 保存（README セクション 5）
# ------------------------------------------------------------
# README 5-1: 学習が進んだmodelをあとで再利用できるように残す段階。
# 学習には時間がかかるので、毎回最初からやり直さないようにする。
#
# README 5-2-2: checkpoint — 学習途中または学習後の状態をまとめて保存したもの。
# README 5-3-1: 語彙情報 — tokenとtoken IDの対応（stoi, itos）。
# README 5-3-2: 設定値 — modelの大きさや構造を決める値。
# README 5-4: 保存の完成状態 — 学習済みweight, token対応, model設定がfileに残る。
# ============================================================
checkpoint = {
    "model_state_dict": model.state_dict(),  # README 5-2-1: weight — 学習の成果そのもの
    "vocab_size": vocab_size,                # README 5-3-2: 設定値
    "chars": chars,                          # README 5-3-1: 語彙情報
    "config": cfg,                           # README 5-3-2: 設定値
}
torch.save(checkpoint, cfg.checkpoint_path)
print(f"\nチェックポイント保存: {cfg.checkpoint_path}")

# --- サンプル生成 ---
print("\n--- サンプル生成 ---")
context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
generated = decode(model.generate(context, max_new_tokens=200)[0].tolist())
print(generated)
