"""データ読込・学習ループ・チェックポイント保存"""

import os
import re
import torch

from config import Config
from model import Bungaku

cfg = Config()


# --- テキストクリーニング（青空文庫用） ---
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


# --- 作品リスト（青空文庫・著作権切れ作品） ---
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


# --- データ読込 ---
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

# --- 文字レベルトークナイザ ---
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

print(f"語彙サイズ: {vocab_size}, テキスト長: {len(text):,} 文字")

# --- train/val分割 ---
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([d[i : i + cfg.block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + cfg.block_size + 1] for i in ix])
    return x.to(cfg.device), y.to(cfg.device)


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


# --- モデル作成 ---
model = Bungaku(cfg, vocab_size).to(cfg.device)
param_count = sum(p.numel() for p in model.parameters())
print(f"パラメータ数: {param_count:,}")
print(f"デバイス: {cfg.device}")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

# --- 学習ループ ---
print(f"\n学習開始 ({cfg.max_iters} ステップ)...")
for step in range(cfg.max_iters):
    if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
        losses = estimate_loss(model)
        print(
            f"ステップ {step:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --- チェックポイント保存 ---
checkpoint = {
    "model_state_dict": model.state_dict(),
    "vocab_size": vocab_size,
    "chars": chars,
    "config": cfg,
}
torch.save(checkpoint, cfg.checkpoint_path)
print(f"\nチェックポイント保存: {cfg.checkpoint_path}")

# --- サンプル生成 ---
print("\n--- サンプル生成 ---")
context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
generated = decode(model.generate(context, max_new_tokens=200)[0].tolist())
print(generated)
