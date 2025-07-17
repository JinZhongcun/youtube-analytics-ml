# 実験実行ガイド - どのコードをどの順番で実行したか

## 概要
このドキュメントは、YouTube動画再生数予測プロジェクトで実際に使用したPythonコードの実行順序と、各段階での結果をまとめたものです。他のチームメンバーがレポートを書く際の参考資料として作成しました。

## 実験の流れ

### Phase 1: 初期分析（R² = 0.21）

#### 1. 初期データ探索
```bash
python youtube_analysis.py
```
- **入力**: `youtube_top_jp.csv` (767件)
- **出力**: `youtube_eda_results.png`, `pca_results.json`
- **結果**: 基本的な統計量とPCA分析
- **発見**: データ量不足、画像特徴なし

#### 2. SVM実装（課題要件）
```bash
python svm_analysis.py
```
- **入力**: `youtube_top_jp.csv`
- **出力**: `svm_results.png`, `svm_results.json`
- **結果**: R² = 0.21（低性能）
- **理由**: 特徴量が不足

### Phase 2: データ拡張と画像分析（R² = 0.34）

#### 3. 画像特徴抽出
```bash
python simple_image_analysis.py
```
- **入力**: `thumbnails/` ディレクトリ（14,612枚）
- **出力**: `simple_image_analysis_results.json`
- **処理内容**:
  - 顔検出（Haar Cascade）
  - 色分析（brightness, colorfulness）
  - エッジ検出
  - テキスト領域検出

#### 4. データマージ試行（問題発生）
```bash
python merge_and_improve.py
```
- **入力**: 
  - `youtube_top_jp.csv` (767件、subscribersあり)
  - `youtube_top_new.csv` (6,078件、subscribersなし)
- **問題**: 90%のデータでsubscribers欠損
- **結果**: 607件のみ完全データ → R² = 0.44

#### 5. Subscribersなしモデル
```bash
python no_subscribers_model.py
```
- **入力**: `youtube_top_new.csv` (6,078件)
- **出力**: `no_subscribers_model_results.png`
- **結果**: R² = 0.34（subscribersなしでも実用的）

### Phase 3: 完全データでの最終分析（R² = 0.4528）

#### 6. 包括的モデル比較
```bash
python comprehensive_model_comparison.py
# または並列版（高速）
python comprehensive_model_comparison_parallel.py
```
- **比較モデル**: LightGBM, XGBoost, Random Forest, Linear Regression
- **結果**: LightGBMが最高性能

#### 7. データセット比較分析
```bash
python comprehensive_dataset_comparison.py
```
- **目的**: 3つのデータセット間での性能比較
- **出力**: `comprehensive_comparison_results.json`
- **重要な発見**:
  - youtube_top_jp.csv (767件): R² = 0.3239
  - youtube_top_new.csv (6,078件): R² = 0.2696
  - youtube_top_new_complete.csv (6,062件、subscribersなし): R² = 0.2575
  - youtube_top_new_complete.csv (6,062件、subscribersあり): R² = 0.4528

#### 8. 最終モデル
```bash
python final_correct_analysis.py
```
- **入力**: `drive-download-20250717T063336Z-1-001/youtube_top_new.csv`
- **最終結果**: CV R² = 0.4528
- **特徴量重要度**:
  1. subscribers (24.5%)
  2. video_duration (13.9%)
  3. colorfulness (13.5%)

## 実行環境

### Dockerを使用した実行
```bash
# Docker環境の構築
docker build -t youtube-6078 -f Dockerfile_6078 .

# 分析の実行
docker run --rm -v $(pwd):/app -w /app youtube-6078 python3 [スクリプト名]
```

### 一括実行スクリプト
```bash
bash run_analysis.sh
```

## データフロー図

```
youtube_top_jp.csv (767)
    ↓
[youtube_analysis.py / svm_analysis.py]
    ↓ R² = 0.21
    
＋ youtube_top_new.csv (6,078) + thumbnails/
    ↓
[simple_image_analysis.py]
    ↓
[merge_and_improve.py] → 607件のみ → R² = 0.44
[no_subscribers_model.py] → 6,078件 → R² = 0.34
    ↓
    
＋ youtube_top_new_complete.csv (6,062, subscribers復活)
    ↓
[comprehensive_dataset_comparison.py]
[final_correct_analysis.py]
    ↓
最終結果: R² = 0.4528
```

## 使用しなかったデータと理由

### 意図的に使用しなかった特徴量

#### 1. likes（いいね数）
- **理由**: 視聴後に発生するエンゲージメント指標
- **問題**: 動画アップロード時点では存在しない
- **影響**: 含めると非現実的に高い精度になる（データリーケージ）

#### 2. comment_count（コメント数）
- **理由**: likesと同様、視聴後の結果
- **問題**: 予測時点で利用不可能
- **補足**: Meng Siyuan氏も「人気の結果であり、原因ではない」と指摘

#### 3. subscriber_per_view（登録者数/再生数）
- **計算式**: `subscribers / views`
- **問題**: ターゲット変数（views）を含む
- **結果**: R² = 0.995という異常に高い精度（明らかなリーケージ）
- **教訓**: viewsを含む派生特徴量は全て除外すべき

#### 4. time_duration
- **内容**: データ収集時点からの経過日数
- **問題**: データ収集タイミングに依存
- **代替**: published_atから計算したdays_since_publishを使用

#### 5. title（動画タイトル）
- **理由**: テキスト分析（NLP）未実装
- **将来**: BERTなどで実装可能だが、今回は範囲外

### データセットの使用状況

#### 完全に使用したデータセット
1. **youtube_top_new_complete.csv** (6,062件)
   - 最終モデルで使用
   - 全特徴量が揃った完全版
   - R² = 0.4528達成

#### 部分的に使用したデータセット
1. **youtube_top_jp.csv** (767件)
   - 初期分析で使用
   - サンプル数が少なく性能限界
   - R² = 0.21-0.32

2. **youtube_top_new.csv** (6,078件)
   - subscribersなしモデルで使用
   - 90%でsubscribers欠損が問題
   - R² = 0.27-0.34

#### マージ時の問題
```
元データ: 6,078件
新データ: 6,062件
共通video_id: 4,482件のみ
失われたデータ: 1,596件
```
- **原因**: 異なる時点でのデータ収集
- **影響**: 98.4%でviewsが更新されていた

### 画像データの使用状況

#### 使用した画像特徴
- brightness（明度）
- colorfulness（色彩度）
- object_complexity（物体数）
- element_complexity（要素数）
- has_face（顔検出）
- edge_density（エッジ密度）

#### 使用しなかった高度な画像分析
- **CNN/Deep Learning**: 計算コストと解釈性の問題
- **CLIP埋め込み**: 実装の複雑さ
- **詳細な顔分析**: 顔ありが逆効果と判明したため

### データ品質の問題

#### tags_countの問題
- **58.7%がゼロ**: 多くの動画でタグ未設定
- **対処**: ゼロ値をそのまま使用

#### description_lengthの問題  
- **35.3%がゼロ**: 説明文なしの動画が多い
- **対処**: log変換時にlog(0+1)=0として処理

## 重要な注意点

### データリーケージの回避
- ❌ 使用してはいけない: `subscriber_per_view = subscribers / views`
- ✅ 使用可能: `subscribers` 単体
- ❌ 使用してはいけない: `likes`, `comment_count`（視聴後の結果）

### 特徴量エンジニアリング
```python
# 時間特徴量
df['hour_published'] = df['published_at'].dt.hour
df['weekday_published'] = df['published_at'].dt.weekday

# ログ変換
df['log_subscribers'] = np.log1p(df['subscribers'])
df['log_duration'] = np.log1p(df['video_duration'])

# 画像特徴（OpenCV）
brightness = np.mean(hsv[:,:,2])
colorfulness = calculate_colorfulness(img)
```

### モデルパラメータ（最終版）
```python
lgb_params = {
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 30,
    'lambda_l2': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'random_state': 42
}
```

## レポート作成のための重要数値

| 指標 | 値 |
|------|-----|
| 最終CV R² | 0.4528 ± 0.0158 |
| Test R² | 0.4550 |
| Train R² | 0.6385 |
| 過学習度 | 0.1835 |
| subscribersの効果 | +0.1953 (75.8%改善) |
| 中央相対誤差 | 68.8% |
| サンプル数 | 6,062 |
| 特徴量数 | 12 |

## 各スクリプトの役割

- `youtube_analysis.py`: 初期EDA、PCA分析
- `svm_analysis.py`: SVM実装（課題要件）
- `simple_image_analysis.py`: OpenCVでサムネイル画像特徴抽出
- `merge_and_improve.py`: データマージと初期最良モデル
- `no_subscribers_model.py`: subscribersなしでの実用モデル
- `comprehensive_model_comparison.py`: 複数MLモデルの比較
- `comprehensive_dataset_comparison.py`: データセット間の性能比較
- `final_correct_analysis.py`: 最終モデル実装
- `advanced_youtube_model.py`: 高度な特徴量エンジニアリング（参考）

## まとめ

1. **初期**: R² = 0.21（データ不足）
2. **画像追加**: R² = 0.34-0.44（改善）
3. **最終**: R² = 0.4528（subscribers復活で最高性能）

subscribersが最も重要な特徴量（24.5%）であり、それなしでは性能が大幅に低下することが判明。