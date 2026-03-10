"""データ読込・学習ループ・チェックポイント保存"""

import os
import torch

from config import Config
from model import GPT

cfg = Config()


# --- データ読込 ---
def load_data():
    if os.path.exists(cfg.data_path):
        with open(cfg.data_path, "r", encoding="utf-8") as f:
            return f.read()
    # フォールバック: Tiny Shakespeareをダウンロード
    print(f"{cfg.data_path} が見つかりません。Tiny Shakespeareをダウンロードします...")
    import urllib.request

    os.makedirs(os.path.dirname(cfg.data_path), exist_ok=True)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, cfg.data_path)
    with open(cfg.data_path, "r", encoding="utf-8") as f:
        return f.read()


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
model = GPT(cfg, vocab_size).to(cfg.device)
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
