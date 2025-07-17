# YouTube動画分析プロジェクト - 最終成果報告

## 日本語版

@MENG SIYUAN @here

YouTube動画分析プロジェクトの最終成果を報告します。

### 🎯 主要成果

**R²スコアが0.21→0.44に大幅改善（2倍以上！）**

### 📊 実施内容

1. **データ統合**
   - 旧データ（767件、subscribers付き）
   - 新データ（6,078件、画像付き）
   - 統合成功：607件

2. **画像分析の実装**
   - 顔検出（OpenCV）
   - エッジ密度（複雑さ指標）
   - 色の多様性
   - HSV色空間分析

3. **モデル性能**

| モデル | R² | 改善率 |
|:---|:---:|:---:|
| Random Forest + 画像 | **0.4416** | +110% |
| Ensemble | 0.4334 | +106% |
| XGBoost + 画像 | 0.4103 | +95% |
| 初期モデル | 0.2102 | - |

### 🔑 重要な発見

1. **色彩度（colorfulness）が最重要特徴量に**
2. **顔なしサムネイルの方が高パフォーマンス**（意外！）
3. **subscribersデータの重要性を再確認**

### 📈 実用的な示唆

コンテンツ制作者への提言：
- 90秒以内の短い動画
- 色彩豊かなサムネイル（ただし派手すぎない）
- 顔より内容を表現するビジュアル
- チャンネル登録者500万人が分岐点

### 🔗 成果物

GitHub: https://github.com/JinZhongcun/youtube-analytics-ml

すべてのコード、データ、分析結果を公開しています。

ご協力ありがとうございました！

---

## English Version

@MENG SIYUAN @here

Final results of our YouTube video analysis project.

### 🎯 Key Achievement

**R² score improved from 0.21 to 0.44 (more than 2x!)**

### 📊 What We Did

1. **Data Integration**
   - Old data (767 videos with subscribers)
   - New data (6,078 videos with images)
   - Successfully merged: 607 videos

2. **Image Analysis Implementation**
   - Face detection (OpenCV)
   - Edge density (complexity metric)
   - Color diversity
   - HSV color space analysis

3. **Model Performance**

| Model | R² | Improvement |
|:---|:---:|:---:|
| Random Forest + Images | **0.4416** | +110% |
| Ensemble | 0.4334 | +106% |
| XGBoost + Images | 0.4103 | +95% |
| Initial Model | 0.2102 | - |

### 🔑 Key Findings

1. **Colorfulness became the top feature**
2. **Thumbnails without faces perform better** (surprising!)
3. **Confirmed importance of subscriber data**

### 📈 Practical Insights

Recommendations for content creators:
- Keep videos under 90 seconds
- Use colorful thumbnails (but not too flashy)
- Focus on content visualization over faces
- 5M subscribers is the key threshold

### 🔗 Deliverables

GitHub: https://github.com/JinZhongcun/youtube-analytics-ml

All code, data, and analysis results are available.

Thank you for the great collaboration!

Best,
Nakamura