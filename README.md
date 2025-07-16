# YouTube Analytics ML

YouTube動画の再生回数予測 - 6,078動画 + サムネイル画像分析

**講義課題**: YouTubeの再生回数に影響を与える要因を特定し、コンテンツ制作者が視聴者数を増やすための戦略を立案（PCAとSVMを使用）

## 最新アップデート（2025/7/16）
- 6,078件の動画データと全サムネイル画像を追加分析
- 画像解析（顔検出、エッジ検出、色解析）を実装
- **R²スコアが0.21→0.44に大幅改善（2倍以上！）**

## 結果

### モデル性能（テストR²）

#### 最新結果（サムネイル画像統合後）
| 順位 | モデル | R² | 改善率 | データ |
|:---:|:---|:---:|:---:|:---|
| 1 | **Random Forest + 画像** | **0.4416** | +110% | 統合データ |
| 2 | Ensemble (3モデル) | 0.4334 | +106% | 統合データ |
| 3 | XGBoost + 画像 | 0.4103 | +95% | 統合データ |
| 4 | LightGBM + 画像 | 0.3652 | +74% | 統合データ |

#### 初期結果（比較用）
| モデル | R² | データ |
|:---|:---:|:---|
| LightGBM（初期） | 0.2102 | 767件 |
| 画像のみ | 0.1526 | 6,078件（subscribersなし） |

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
- **初期データ**: 767件（subscribers含む）
- **追加データ**: 6,078件（サムネイル画像含む）
- **統合データ**: 607件（両方の情報を持つ）

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
├── youtube_top_jp.csv                     # 初期データ（767件）
├── youtube_top_new.csv                    # 追加データ（6,078件）
├── thumbnails/                            # サムネイル画像フォルダ
│   └── *.jpg                             # 6,078枚の画像
├── youtube_analysis.py                    # EDA・PCA分析
├── svm_analysis.py                        # SVM分析
├── simple_image_analysis.py               # 画像特徴抽出分析
├── merge_and_improve.py                   # データ統合・改良モデル
├── comprehensive_model_comparison.py      # 包括的モデル比較
├── comprehensive_model_comparison_parallel.py  # 並列処理版
└── 出力ファイル/
    ├── correlation_matrix.png             # 相関行列
    ├── simple_image_analysis.png          # 画像分析結果
    ├── merged_model_analysis.png          # 統合モデル結果
    └── *.json                            # 各種分析結果
```

## 実行方法

### 1. 基本的な分析（EDA + PCA）
```bash
python youtube_analysis.py
```

### 2. 画像特徴抽出分析
```bash
python simple_image_analysis.py
```

### 3. データ統合と改良モデル（推奨）
```bash
python merge_and_improve.py
```

### 4. 包括的なモデル比較
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