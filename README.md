# YouTube Analytics ML Project

## 🚨 重要：データリーケージ検証済み (2025-01-23更新)

**最新の検証結果：**
- ✅ **メインモデルはリークフリー**: `final_correct_analysis.py`、`comprehensive_dataset_comparison.py`
- ⚠️ **一部にリーク発見**: `complete_data_results.json` でR² > 0.99（`subscriber_per_view`が原因）
- ✅ **正しい性能**: LightGBM R² = **0.4528** (6,062件、クリーンデータ)
- ❌ **古い図**: GitHub上の一部モデル比較図は767件の古いデータを使用

**使用すべき正しい結果:**
- **最終性能**: CV R² = 0.4528 (LightGBM)
- **データセット**: youtube_top_new_complete.csv (6,062件)
- **特徴量**: subscribers使用可（アップロード時点の情報）
- **除外済み**: likes, comment_count（視聴後のデータ）

---

> 📚 **論文・レポート執筆用の詳細版は [`README_detailed_analysis.md`](README_detailed_analysis.md) を参照してください**
> 
> 🔬 **実験の実行手順とコードの使い方は [`EXPERIMENT_EXECUTION_GUIDE.md`](EXPERIMENT_EXECUTION_GUIDE.md) を参照してください**
> 
> For academic/research purposes, please refer to:
> - Detailed analysis: [`README_detailed_analysis.md`](README_detailed_analysis.md)
> - Experiment execution guide: [`EXPERIMENT_EXECUTION_GUIDE.md`](EXPERIMENT_EXECUTION_GUIDE.md)

## 📊 Project Overview / プロジェクト概要

This project analyzes YouTube video performance using machine learning, focusing on predicting view counts through thumbnail image analysis and metadata features.

YouTubeの動画パフォーマンスを機械学習で分析し、サムネイル画像とメタデータから再生回数を予測するプロジェクトです。

### Key Achievements / 主な成果
- **Final Best Performance**: R² = 0.4528 with complete dataset
- **6,062 Videos Analyzed**: Full dataset with restored subscribers
- **75.8% Improvement**: Adding subscribers improved R² from 0.2575 to 0.4528
- **No Deep Learning Required**: Achieved strong results with classical ML and OpenCV

## 🎯 Results Summary / 結果まとめ

### Model Performance Comparison / モデル性能比較

| Model Type | Data Used | R² Score | Key Features |
|------------|-----------|----------|--------------|
| Initial Model | 767 videos | 0.21 | Basic metadata only |
| **Best Model** | 607 videos | **0.44** | All features + images |
| No-Subscribers Model | 6,078 videos | 0.34 | Images + metadata only |

### Top Predictive Features (Final Model) / 最終モデルの重要特徴量
1. **Subscribers** (1041, 24.5%) - チャンネル登録者数
2. **Video Duration** (590, 13.9%) - 動画の長さ
3. **Colorfulness** (576, 13.5%) - サムネイルの色彩
4. **Brightness** (522, 12.3%) - サムネイルの明度
5. **Description Length** (485, 11.4%) - 説明文の長さ
6. **Hour Published** (425, 10.0%) - 投稿時刻
7. **Tags Count** (313, 7.4%) - タグ数
8. **Object Complexity** (245, 5.8%) - オブジェクトの複雑さ

## 🔍 Key Findings / 重要な発見

### What Works / 効果的な要素
- **Short Videos Win**: 90 seconds average (7.7x more views)
- **Colorful Thumbnails**: Optimal brightness ~70
- **5M+ Subscribers**: Critical threshold
- **Strategic Tags**: 10-15 tags optimal

### What Doesn't Work / 逆効果な要素
- **Faces in Thumbnails**: Surprisingly decrease performance
- **Overly Bright Images**: Brightness > 80 performs worse
- **Long Videos**: 689+ seconds severely limit reach

## 📁 Project Journey / プロジェクトの経緯

### 🇯🇵 日本語版

#### 1. 初期状況
- **データ**: 767件のYouTube動画（日本）
- **目標**: 再生数予測モデルの構築
- **初期性能**: R² = 0.21（低い）
- **問題**: データ量不足、画像情報なし

#### 2. データ拡張
- **新規取得**: 6,078件の動画データ + サムネイル画像
- **問題発覚**: 新データにsubscribers/likes/commentsがない
- **マージ結果**: 607件のみ完全データ（10%）

#### 3. 画像分析実装
- **手法**: OpenCVで画像特徴抽出（CNNは使わず）
  - 顔検出（Haar Cascade）
  - 色分析（HSV色空間）
  - エッジ密度、テキスト領域
- **結果**: R² = 0.44（2倍改善！）

#### 4. 重要な発見
- **最重要特徴**:
  1. colorfulness（色の鮮やかさ）: 0.226
  2. video_duration（動画長）: 0.207
  3. subscribers（登録者数）: 0.199
- **意外な事実**: 顔ありサムネイルは逆効果

#### 5. データ問題への対処
- **subscribersなしモデル**: R² = 0.34
- **全6,078件使用可能**になった
- **画像特徴だけでも実用的**な精度

#### 6. 最終成果
- **最良モデル**: R² = 0.4528（6,062件、全特徴）
- **subscribersなし**: R² = 0.2575（6,062件、画像のみ）
- **subscribersの効果**: +0.1953（75%改善）

### 🇺🇸 English Version

#### 1. Initial Situation
- **Data**: 767 YouTube videos (Japan)
- **Goal**: Build view count prediction model
- **Initial performance**: R² = 0.21 (poor)
- **Issues**: Insufficient data, no image information

#### 2. Data Expansion
- **New acquisition**: 6,078 videos + thumbnail images
- **Problem found**: New data lacks subscribers/likes/comments
- **Merge result**: Only 607 complete records (10%)

#### 3. Image Analysis Implementation
- **Method**: OpenCV feature extraction (no CNN)
  - Face detection (Haar Cascade)
  - Color analysis (HSV space)
  - Edge density, text regions
- **Result**: R² = 0.44 (2x improvement!)

#### 4. Key Findings
- **Top features**:
  1. colorfulness: 0.226
  2. video_duration: 0.207
  3. subscribers: 0.199
- **Surprising fact**: Faces in thumbnails decrease views

#### 5. Handling Data Issues
- **No-subscribers model**: R² = 0.34
- **All 6,078 videos usable** now
- **Image features alone are practical**

#### 6. Final Achievements
- **Final best model**: R² = 0.4528 (6,062 videos, all features)
- **Without subscribers**: R² = 0.2575 (6,062 videos, images only)
- **Subscribers effect**: +0.1953 (75% improvement)

## 📊 Data Usage Documentation / データ使用詳細

### Features USED in Models / モデルで使用した特徴量
1. **video_duration** - 動画の長さ（秒）
2. **tags_count** - タグ数
3. **description_length** - 説明文の長さ
4. **subscribers** - チャンネル登録者数（利用可能な場合）
5. **object_complexity** - サムネイルの複雑さスコア
6. **element_complexity** - 視覚要素の複雑さ
7. **brightness** - サムネイルの明度
8. **colorfulness** - サムネイルの色の多様性
9. **category_id** - 動画カテゴリ（ワンホットエンコーディング）
10. **published_at** - days_since_publishに変換
11. **OpenCV抽出画像特徴**:
    - hue_mean, hue_std（色相）
    - saturation_mean, saturation_std（彩度）
    - edge_density（エッジ密度）
    - has_face（顔検出）
    - text_area_ratio（テキスト領域）
    - その他15+特徴量

### Features INTENTIONALLY EXCLUDED / 意図的に除外した特徴量
1. **likes** - いいね数（データリーケージ防止）
2. **comment_count** - コメント数（データリーケージ防止）
3. **title** - タイトル（テキスト分析未実装）
4. **thumbnail_link** - URL（予測に無関係）

### Why Excluded? / 除外理由
**likes**と**comment_count**は視聴後の結果であり：
- 予測時点（アップロード前）には存在しない
- 含めると非現実的な高精度になる
- Meng Siyuan氏の指摘通り「人気の結果であり、原因ではない」

## 🛠️ Technical Implementation / 技術実装

### Image Analysis (OpenCV) / 画像解析
```python
# Face Detection / 顔検出
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Color Analysis / 色分析
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
colorfulness = np.std(hsv)

# Edge Detection / エッジ検出
edges = cv2.Canny(gray, 50, 150)
edge_density = np.sum(edges > 0) / size
```

### Models Used / 使用モデル
- **LightGBM** - Best single model
- **XGBoost** - Strong alternative
- **Random Forest** - Baseline comparison
- **Ensemble** - Voting regressor

## 📈 Detailed Performance / 詳細性能

### Performance by Feature Type / 特徴量タイプ別性能
- **Metadata only**: R² = 0.21
- **Metadata + Subscribers**: R² = 0.35
- **Metadata + Images**: R² = 0.34
- **All features**: R² = 0.44

### Final Dataset Comparison / 最終データセット比較

| Dataset | Samples | Subscribers | CV R² |
|---------|---------|-------------|-------|
| youtube_top_jp.csv | 767 | Yes | 0.3239 |
| youtube_top_new.csv | 6,078 | No | 0.2696 |
| youtube_top_new_complete.csv | 6,062 | No | 0.2575 |
| **youtube_top_new_complete.csv** | **6,062** | **Yes** | **0.4528** |

### Key Findings / 重要な発見
- **Subscribers impact**: +0.1953 (75% improvement)
- **Data size effect**: Minimal (6,078 vs 767 similar performance)
- **Best configuration**: Full dataset with subscribers

## 🚀 Recommendations / 推奨事項

### For Content Creators / コンテンツ制作者向け
1. **Keep videos under 90 seconds** / 90秒以内に収める
2. **Use moderately colorful thumbnails** / 適度にカラフルなサムネイル
3. **Avoid faces in thumbnails** / サムネイルに顔は避ける
4. **Build to 5M+ subscribers** / 登録者500万人を目指す
5. **Use 10-15 relevant tags** / 関連タグを10-15個使用

## 📊 Detailed Analysis Documentation / 詳細分析ドキュメント

### For Academic/Research Purposes
Complete detailed analysis is available in `README_detailed_analysis.md` including:
- Comprehensive methodology
- Statistical analysis
- Feature engineering details
- Model evaluation metrics
- Discussion and implications

論文執筆用の詳細な分析は `README_detailed_analysis.md` を参照してください。

### Experimental Design Summary / 実験設計概要

#### Feature Engineering Pipeline
1. **Basic Metadata**: video_duration, tags_count, description_length
2. **Image Features** (OpenCV):
   - Color: brightness, colorfulness, HSV statistics
   - Structure: object_complexity, element_complexity, edge_density
   - Content: face detection, text area ratio
3. **Temporal Features**: hour_published, weekday_published, days_since_publish
4. **Channel Features**: subscribers, log_subscribers

#### Model Configuration
```python
# Optimized LightGBM parameters
lgb_params = {
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 30,
    'lambda_l2': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'learning_rate': 0.05,
    'n_estimators': 200
}
```

#### Evaluation Methodology
- **Cross-validation**: 5-fold CV (KFold with random_state=42)
- **Train/Test Split**: 80/20
- **Metric**: R² (coefficient of determination)
- **Data Leakage Prevention**: Excluded views-derived features

## 📁 Repository Structure / リポジトリ構造

```
├── youtube_analysis.py          # Initial EDA / 初期分析
├── svm_analysis.py             # SVM implementation / SVM実装
├── simple_image_analysis.py     # Image features / 画像特徴抽出
├── merge_and_improve.py        # Best model / 最良モデル (R² = 0.44)
├── final_correct_analysis.py   # Final analysis / 最終分析 (R² = 0.4528)
├── comprehensive_dataset_comparison.py  # Dataset comparison / データセット比較
├── youtube_top_jp.csv          # Original data / 元データ (767)
├── youtube_top_new.csv         # Extended data / 拡張データ (6,078)
├── youtube_top_new_complete.csv # Complete data / 完全データ (6,062)
├── README_detailed_analysis.md # Detailed documentation / 詳細ドキュメント
└── thumbnails/                 # 14,612 images / サムネイル画像
```

## 🔧 Requirements / 必要環境

```bash
pip install pandas numpy scikit-learn lightgbm xgboost opencv-python matplotlib seaborn
```

## 📝 Assignment Context / 課題背景

University Assignment 2 requiring:
- Principal Component Analysis (PCA)
- Support Vector Machine (SVM)
- Actionable insights for content creators

大学の課題2の要件：
- 主成分分析（PCA）
- サポートベクターマシン（SVM）
- コンテンツ制作者への実用的な洞察

## 🤝 Collaboration / 協力

Completed with team member Meng Siyuan who provided the extended dataset.

チームメンバーのMeng Siyuanが拡張データセットを提供。

---

*Academic project demonstrating practical ML for social media analytics*
*ソーシャルメディア分析のための実用的MLを示す学術プロジェクト*