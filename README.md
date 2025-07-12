# YouTube Analytics ML

YouTube動画の再生回数予測のための機械学習分析プロジェクト

## 概要

本プロジェクトでは、YouTube動画の各種特徴量から再生回数を予測するモデルを構築します。主成分分析（PCA）による次元削減と、複数の機械学習アルゴリズムの比較を行いました。

## データセット

- **データ数**: 767件のYouTube動画
- **期間**: 2025年6月〜7月
- **カテゴリ**: 22（エンターテインメント）、24（エンターテインメント）、10（音楽）など

## 使用した特徴量

### 含まれる特徴量
1. `video_duration` - 動画の長さ（秒）
2. `tags_count` - タグ数
3. `description_length` - 概要欄の文字数
4. `subscribers` - チャンネル登録者数
5. `object_complexity` - サムネイルのオブジェクト複雑度
6. `element_complexity` - サムネイルの要素複雑度
7. `brightness` - サムネイルの明るさ
8. `colorfulness` - サムネイルの色彩度
9. `days_since_publish` - 投稿からの経過日数

### 除外した特徴量（データリーク防止）
- `likes` - 高評価数
- `comment_count` - コメント数

## 分析手法

### 1. 次元削減
- **PCA（主成分分析）**: 5次元に削減（累積寄与率73.47%）
- **t-SNE**: 可視化用に2次元に削減

### 2. 予測モデル
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Support Vector Machine (RBF)
- LightGBM
- XGBoost

## 主要な結果

### モデル性能（テストデータR²スコア）
1. **LightGBM**: R² = 0.2102
2. **Random Forest**: R² = 0.1933
3. **XGBoost**: R² = 0.1892

### 重要な特徴量（LightGBM基準）
1. チャンネル登録者数（subscribers）
2. 動画の長さ（video_duration）
3. サムネイルの明るさ（brightness）

### 高再生回数動画の特徴
- **短い動画**: 平均90秒（低再生: 689秒）
- **多くのチャンネル登録者**: 平均583万人（低再生: 114万人）
- **適度な色彩度**: 平均38.2（低再生: 47.7）

## 環境構築

### 必要なパッケージ
```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost
```

### Docker使用時
```bash
docker build -t youtube-analytics .
docker run --rm -v $(pwd):/work youtube-analytics
```

## ファイル構成

```
.
├── README.md                              # このファイル
├── Dockerfile                             # Docker環境定義
├── youtube_top_jp.csv                     # 元データ
├── youtube_analysis.py                    # EDA・PCA分析
├── svm_analysis.py                        # SVM分析
├── comprehensive_model_comparison.py      # 包括的モデル比較
├── comprehensive_model_comparison_parallel.py  # 並列処理版
└── 出力ファイル/
    ├── correlation_matrix.png             # 相関行列
    ├── pca_variance.png                   # PCA寄与率
    ├── comprehensive_model_comparison.png # モデル比較結果
    └── comprehensive_results.json         # 詳細な数値結果
```

## 実行方法

### 1. 基本的な分析（EDA + PCA）
```bash
python youtube_analysis.py
```

### 2. SVMモデルの構築
```bash
python svm_analysis.py
```

### 3. 包括的なモデル比較
```bash
python comprehensive_model_comparison_parallel.py
```

## 今後の改善点

1. **特徴量エンジニアリング**
   - カテゴリ別の特徴量作成
   - 時系列特徴の追加
   - 交互作用項の検討

2. **モデルの改善**
   - ハイパーパラメータの最適化
   - アンサンブル学習の適用
   - ディープラーニングモデルの検討

3. **評価の改善**
   - より多くのデータ収集
   - 時系列分割での検証
   - カテゴリ別の予測精度評価

## ライセンス

本プロジェクトはMITライセンスの下で公開されています。