# YouTube動画分析プロジェクト - 進捗報告

## 日本語版

Meng Siyuanさん

お疲れ様です。中村です。

YouTube動画分析プロジェクトの現状を報告します。

### 実施内容
1. **データセット**: 767件のYouTube動画データ（2025年6-7月）を使用
2. **モデル比較**: LightGBM、XGBoost、Random Forest、SVM等7種類のモデルを実装
3. **特徴量**: サムネイル分析（明度、色彩度、複雑度）を含む9つの特徴量を使用

### 現在の結果
- **最良モデル**: LightGBM
- **精度**: R² = 0.21（正直、低いです）
- **主要な発見**:
  - 90秒以内の短い動画が7.7倍多く視聴される
  - チャンネル登録者数が最重要（500万人以上推奨）
  - サムネイルは明度70前後が最適

### 問題点と提案
現在の精度（R²=0.21）では実用的とは言えません。精度向上のため、以下を提案します：

**重要なお願い**：
- **サムネイル画像そのもの**が必要です
- 現在は数値化された特徴量（brightness等）のみで、実際の画像データがありません
- 画像があれば、CNNやVision Transformerで直接画像から特徴を抽出できます

### 今後の方針
1. サムネイル画像データの収集
2. より多くのデータ（最低5000件以上）の収集
3. 時系列データ（日次再生回数の推移）の活用

GitHubリポジトリ: https://github.com/JinZhongcun/youtube-analytics-ml

ご確認お願いします。

中村

---

## English Version

Hi Meng Siyuan,

Here's the progress report on our YouTube video analysis project.

### What We've Done
1. **Dataset**: 767 YouTube videos (June-July 2025)
2. **Models Tested**: 7 models including LightGBM, XGBoost, Random Forest, SVM
3. **Features**: 9 features including thumbnail analysis (brightness, colorfulness, complexity)

### Current Results
- **Best Model**: LightGBM
- **Accuracy**: R² = 0.21 (honestly, quite low)
- **Key Findings**:
  - Videos under 90 seconds get 7.7x more views
  - Subscriber count is the most important factor (5M+ recommended)
  - Optimal thumbnail brightness is around 70

### Issues and Proposals
The current accuracy (R²=0.21) is not practical. To improve accuracy, I propose:

**Important Request**:
- We need **actual thumbnail images**
- Currently we only have numerical features (brightness, etc.), not the actual image data
- With images, we can use CNN or Vision Transformers to extract features directly

### Next Steps
1. Collect thumbnail image data
2. Gather more data (at least 5000+ videos)
3. Use time-series data (daily view count progression)

GitHub Repository: https://github.com/JinZhongcun/youtube-analytics-ml

Please let me know your thoughts.

Best regards,
Nakamura