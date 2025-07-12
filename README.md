# YouTube Analytics ML

YouTube動画の再生回数予測 - 767動画を分析

## 結果

### モデル性能（テストR²）
| 順位 | モデル | R² | MSE | MAE |
|:---:|:---|:---:|:---:|:---:|
| 1 | LightGBM | **0.2102** | 0.2721 | 0.4229 |
| 2 | Random Forest | 0.1933 | 0.2779 | 0.4302 |
| 3 | XGBoost | 0.1892 | 0.2793 | 0.4300 |
| 4 | PCA+Random Forest | 0.1843 | 0.2810 | 0.4352 |
| 5 | SVM (RBF) | 0.1687 | 0.2865 | 0.4375 |

### 高再生回数の特徴
| 特徴 | 高再生（上位25%） | 低再生（下位50%） | 差 |
|:---|---:|---:|---:|
| 動画の長さ | 90秒 | 689秒 | **-87%** |
| チャンネル登録者 | 583万人 | 114万人 | **+411%** |
| サムネイル明るさ | 71 | 80 | -11% |
| サムネイル色彩度 | 38 | 48 | -21% |
| タグ数 | 6個 | 7個 | -14% |

### 重要度ランキング
1. **brightness** (0.226) - サムネイルの明るさ
2. **video_duration** (0.207) - 動画の長さ
3. **subscribers** (0.199) - チャンネル登録者数
4. **tags_count** (0.170) - タグ数
5. **object_complexity** (0.168) - サムネイルの複雑さ

## 結論
- **90秒以内の短い動画**が7.7倍多く視聴される
- **チャンネル登録者数**が最重要（500万人以上推奨）
- **サムネイルは明るすぎず**（明度70前後）**シンプルに**

## データセット

### 収集方法
- **YouTube Data API v3**を使用して収集
- **期間**: 2025年6月〜7月の日本のトレンド動画
- **データ数**: 767件

### データ内容（youtube_top_jp.csv）
- 動画メタデータ: video_id, title, category_id, video_duration, tags_count
- 統計データ: views, likes, comment_count, subscribers
- サムネイル分析: brightness, colorfulness, object_complexity, element_complexity
- その他: description_length, published_at, keyword

### カテゴリ分布
- カテゴリ22（エンターテインメント）: 317件
- カテゴリ24（エンターテインメント）: 147件
- カテゴリ10（音楽）: 33件
- その他: 270件

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