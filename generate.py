"""テキスト生成CLI"""

# ============================================================
# 段階 6: 生成（README セクション 6）
# ------------------------------------------------------------
# README 6-1: 生成は、保存したmodelを使って新しいtokenを1個ずつ出していく段階。
# ============================================================

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

    # ------------------------------------------------------------
    # README 5-2-2: checkpoint — 学習後の状態をまとめて保存したもの。
    # 後で同じmodelを復元するために読み込む。
    # ------------------------------------------------------------
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]       # README 5-3-2: 設定値 — modelの大きさや構造を決める値
    chars = checkpoint["chars"]      # README 5-3-1: 語彙情報 — tokenとtoken IDの対応
    vocab_size = checkpoint["vocab_size"]

    # ------------------------------------------------------------
    # README 5-3-1: 語彙情報の再構築
    # 同じweightでも、どのIDがどのtokenか分からないと使えない。
    # stoi, itosのような対応表を復元する。
    # ------------------------------------------------------------
    stoi = {ch: i for i, ch in enumerate(chars)}  # string to integer
    itos = {i: ch for i, ch in enumerate(chars)}  # integer to string
    encode = lambda s: [stoi[c] for c in s]       # 文章 → token ID列
    decode = lambda l: "".join([itos[i] for i in l])  # token ID列 → 文章

    # ------------------------------------------------------------
    # README 5-3-2: 設定値を使ってモデル再構築
    # 復元時に同じ形のmodelを作る必要がある。
    # ------------------------------------------------------------
    model = Bungaku(cfg, vocab_size).to(cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ------------------------------------------------------------
    # README 6-2: 最初の入力を用意する
    # README 6-2-1: prompt — 生成の出発点として与える最初のtoken列。
    # 何も入力がないと、どんな続きを出せばいいか決めにくい。
    # README 6-2-2: promptをtoken化し、ID化して、token ID tensorにしてmodelに入れる。
    # ------------------------------------------------------------
    if args.prompt:
        context = torch.tensor([encode(args.prompt)], dtype=torch.long, device=cfg.device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)

    # ------------------------------------------------------------
    # README 6-3〜6-7: 生成ループ
    # model.generate()の中で以下が繰り返される:
    #   6-3: modelに通す（forward）→ logitsを得る
    #   6-4: 最後の位置だけ使う → (B, V)を取り出す
    #   6-5: softmax → sampling → 次tokenを1個選ぶ
    #   6-6: 選んだtokenを後ろに足す
    #   6-7: これを繰り返す（autoregressive）
    # ------------------------------------------------------------
    output = model.generate(context, max_new_tokens=args.max_tokens, temperature=args.temperature)

    # ------------------------------------------------------------
    # README 6-8: 生成の完成状態
    # token IDを文字に戻して、人間が読める形にする。
    # ------------------------------------------------------------
    print(decode(output[0].tolist()))


if __name__ == "__main__":
    main()
