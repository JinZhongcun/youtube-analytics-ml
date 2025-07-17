# YouTube動画予測プロジェクト - 最終報告書（修正版）

## プロジェクト概要
- 大阪大学 Group 4 - 国際融合科学論/先端融合科学論 課題2
- 目的：YouTube動画の再生数（views）を予測
- 手法：機械学習（LightGBM）+ サムネイル画像分析

## 重要な修正事項

### 誤解していた点
- ❌ **誤**: subscribersは時系列リーケージのため使用不可
- ✅ **正**: subscribersは使用可能。ただし`subscriber_per_view = subscribers/views`のようなviewsを含む計算は不可

### データリーケージの正しい理解
- **問題となる特徴量**: viewsを直接含む計算結果
  - `subscriber_per_view = subscribers / views`
  - `views_per_day = views / days`
- **問題ない特徴量**: viewsを含まない
  - `subscribers`（単体）
  - `log_subscribers`
  - その他全てのメタデータ

## 最終結果

### モデル性能
- **CV R² = 0.4470** ± 0.0132
- **Test R² = 0.4550**
- 過学習度: 0.1835（許容範囲）

### 特徴量重要度TOP5
1. **subscribers**: 1041（最重要）
2. **video_duration**: 590
3. **colorfulness**: 576
4. **brightness**: 522
5. **description_length**: 485

## 使用データ
- **youtube_top_new_complete.csv**: 6,062件
- Meng Siyuanさんが修復したsubscribers完備版
- 全ての動画にsubscribers情報あり

## 実験経緯

### Phase 1: 初期分析
- データ: youtube_top_jp.csv（767件）
- R² = 0.21（低性能）

### Phase 2: 画像分析導入
- データ: youtube_top_new.csv（6,078件）
- 90%でsubscribers欠損
- R² = 0.34（画像特徴のみ）

### Phase 3: subscribers復活
- データ: youtube_top_new_complete.csv（6,062件）
- 初期結果: R² = 0.995（subscriber_per_view使用）
- 修正後: R² = 0.447（正しい使い方）

## 結論

1. **subscribersは重要な特徴量**
   - チャンネルの人気度を表す正当な指標
   - 最も重要度の高い特徴量

2. **サムネイル画像も有効**
   - colorfulness、brightnessが上位
   - OpenCVによる特徴抽出が効果的

3. **R² ≈ 0.45は妥当な性能**
   - YouTubeアルゴリズムやトレンドは予測不可能
   - 利用可能な情報での限界に近い

## 教訓
- 高すぎる精度（R² > 0.9）は疑うべき
- データリーケージは「ターゲット変数を直接含む」場合に発生
- 批判的思考の重要性

---
*2025年7月17日*
*大阪大学 Group 4 課題2 完了*