"""テキスト生成CLI"""

import argparse
import torch

from config import Config
from model import Bungaku


def main():
    parser = argparse.ArgumentParser(description="Bungaku テキスト生成")
    parser.add_argument("--prompt", type=str, default="", help="生成の開始テキスト")
    parser.add_argument("--max-tokens", type=int, default=200, help="生成トークン数")
    parser.add_argument("--temperature", type=float, default=1.0, help="温度 (低い=保守的)")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt", help="チェックポイントパス")
    args = parser.parse_args()

    # チェックポイント読込
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    chars = checkpoint["chars"]
    vocab_size = checkpoint["vocab_size"]

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    # モデル再構築
    model = Bungaku(cfg, vocab_size).to(cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 入力エンコード
    if args.prompt:
        context = torch.tensor([encode(args.prompt)], dtype=torch.long, device=cfg.device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)

    # 生成
    output = model.generate(context, max_new_tokens=args.max_tokens, temperature=args.temperature)
    print(decode(output[0].tolist()))


if __name__ == "__main__":
    main()
