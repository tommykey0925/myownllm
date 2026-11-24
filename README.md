```text
入力トークン
    ↓
Embedding（ベクトル化）
    ↓
第1層
 ├─ Attention（Wq, Wk, Wv, Wo）
 └─ FFN（W1, W2）
    ↓
第2層
 ├─ Attention（別のWq, Wk, Wv, Wo）
 └─ FFN（別のW1, W2）
    ↓
   ...
    ↓
第32層
 ├─ Attention
 └─ FFN
    ↓
出力変換（重み行列）
    ↓
Logits（語彙ごとのスコア）
    ↓
Softmax（確率分布に変換）
    ↓
Sampling（トークン選択）
    ↓
次のトークン確定
    ↓
入力トークン列に追加
    ↓
最初に戻る（生成終了まで繰り返す）
```
※正規化・残差接続・KVキャッシュなどは省略。
