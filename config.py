"""ハイパーパラメータ設定"""

from dataclasses import dataclass
import torch


@dataclass
class Config:
    # ============================================================
    # model パラメータ
    # ============================================================

    # n_embd (number of embedding dimensions)
    # README 3-5-3: C — 1 tokenを表すベクトルの長さ。内部表現の豊かさを決める。
    # README 2-5-1: ベクトル — 数字を横に並べたもの。
    n_embd: int = 256

    # n_head (number of attention heads)
    # README 3-5-4: attentionの本数 — 文脈を見る視点の数。
    # 複数の見方で文脈を見るために、複数のattentionを並列に使う。
    n_head: int = 8

    # n_layer (number of layers/blocks)
    # README 3-5-5: blockの段数 — blockを何個積むか。modelの深さになる。
    # 表現を何段階で作り直すかを決める。
    n_layer: int = 6

    # block_size (maximum sequence length)
    # README 3-5-2: Tの最大値 — 一度に見られる最大token数。
    # モデルが扱える長さの上限。
    block_size: int = 128

    # dropout (dropout rate)
    # 学習時にランダムにニューロンを無効化する割合。過学習を防ぐ。
    dropout: float = 0.1

    # ============================================================
    # training パラメータ
    # ============================================================

    # batch_size
    # README 4-2-3: batch — まとめて処理する複数の入力列。
    # 1本ずつより、まとめて計算した方が効率がいい。
    batch_size: int = 64

    # learning_rate
    # README 4-6-2: learning rate — weightを1回でどれくらい動かすかを決める大きさ。
    # 大きすぎると不安定、小さすぎると進みが遅い。
    learning_rate: float = 3e-4

    # max_iters (maximum iterations)
    # README 4-7-1: iteration — forward → loss → backward → updateの繰り返し回数。
    max_iters: int = 20000

    # eval_interval (evaluation interval)
    # README 4-8: 何ステップごとに成績（train/val loss）を確認するか。
    eval_interval: int = 1000

    # eval_iters (evaluation iterations)
    # README 4-8: 成績確認時に何回分のlossを平均するか。
    eval_iters: int = 200

    # ============================================================
    # paths
    # ============================================================

    # data_path — 学習データの保存先（README 1-6: 学習に使うテキスト本文）
    data_path: str = "data/input.txt"

    # checkpoint_path — チェックポイントの保存先（README 5-2-2: checkpoint）
    checkpoint_path: str = "checkpoint.pt"

    # ============================================================
    # device
    # ============================================================

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
