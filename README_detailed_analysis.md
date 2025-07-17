# YouTube動画再生数予測プロジェクト - 詳細分析報告書

## 1. 研究背景と目的

### 1.1 背景
YouTube動画の再生数は、コンテンツクリエイターにとって最も重要な指標の一つである。しかし、どのような要因が再生数に影響するかは必ずしも明確ではない。本研究では、機械学習を用いて動画のメタデータとサムネイル画像から再生数を予測するモデルを構築し、重要な要因を特定することを目的とする。

### 1.2 研究目的
1. YouTube動画の再生数を予測する機械学習モデルの構築
2. 再生数に影響する重要な特徴量の特定
3. サムネイル画像の視覚的特徴が再生数に与える影響の分析

### 1.3 使用技術
- 機械学習: LightGBM, XGBoost, Random Forest
- 画像処理: OpenCV (顔検出、色分析、エッジ検出)
- データ処理: pandas, numpy, scikit-learn

## 2. データセット

### 2.1 データ収集と変遷

#### Phase 1: 初期データセット
- **ファイル**: youtube_top_jp.csv
- **サンプル数**: 767件
- **特徴**: 
  - 基本メタデータ（video_duration, tags_count, description_length）
  - チャンネル情報（subscribers）
  - エンゲージメント指標（views, likes, comment_count）
  - サムネイル画像特徴量

#### Phase 2: 拡張データセット
- **ファイル**: youtube_top_new.csv
- **サンプル数**: 6,078件
- **問題点**: subscribers列が90%欠損
- **解決策**: サムネイル画像特徴量で補完

#### Phase 3: 完全データセット
- **ファイル**: youtube_top_new_complete.csv
- **サンプル数**: 6,062件
- **改善点**: Meng Siyuan氏によりsubscribers情報が復元
- **特徴**: 全ての動画で完全な特徴量セット

### 2.2 データ品質の検証

```
データセット間の共通video_id: 4,482件
views更新率: 98.4%（異なる時点でのデータ収集）
平均views変化率: +15.2%
```

## 3. 特徴量エンジニアリング

### 3.1 基本メタデータ特徴量
1. **video_duration**: 動画の長さ（秒）
   - 平均: 423.2秒
   - 分布: 右に歪んだ分布（log変換適用）
   
2. **tags_count**: タグ数
   - 平均: 6.5個
   - ゼロ率: 58.7%（多くの動画がタグなし）
   
3. **description_length**: 説明文の長さ
   - 平均: 479文字
   - ゼロ率: 35.3%

### 3.2 サムネイル画像特徴量（OpenCV抽出）

#### 色特徴量
```python
# HSV色空間での分析
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue_mean = np.mean(hsv[:,:,0])
saturation_mean = np.mean(hsv[:,:,1])
brightness = np.mean(hsv[:,:,2])

# 色の多様性（Colorfulness）
rg = img[:,:,2] - img[:,:,1]
yb = 0.5 * (img[:,:,2] + img[:,:,1]) - img[:,:,0]
colorfulness = np.sqrt(np.mean(rg**2) + np.mean(yb**2))
```

#### 構造特徴量
- **object_complexity**: YOLOv3で検出されたオブジェクト数
- **element_complexity**: 輪郭検出によるビジュアル要素数
- **edge_density**: Cannyエッジ検出によるエッジ密度
- **text_area_ratio**: OCRで検出されたテキスト領域の割合

#### 顔検出
```python
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
has_face = 1 if len(faces) > 0 else 0
face_area = sum(w*h for x,y,w,h in faces) / (img.shape[0] * img.shape[1])
```

### 3.3 時間特徴量
- **days_since_publish**: 公開からの経過日数
- **hour_published**: 公開時刻（0-23）
- **weekday_published**: 公開曜日（0-6）

### 3.4 チャンネル特徴量
- **subscribers**: チャンネル登録者数
- **log_subscribers**: log(subscribers + 1)

### 3.5 使用しなかった特徴量とその理由

#### データリーケージを防ぐため除外
1. **likes**: 視聴後のエンゲージメント（予測時点で存在しない）
2. **comment_count**: 同上
3. **subscriber_per_view**: views/subscribersはターゲット変数を含む
4. **dislike_count**: APIで取得不可（YouTube側で非公開化）

#### 実装上の理由で除外
1. **title**: テキスト分析（NLP）未実装
2. **description**: 同上（length のみ使用）
3. **tags**: 個別タグの分析未実装（count のみ使用）
4. **video_id, thumbnail_link**: 予測に無関係なID/URL

#### データ品質の問題
1. **time_duration**: データ収集時点依存のため除外
2. **keyword**: category_idと重複

## 4. 実験設定

### 4.1 モデル設定

#### LightGBMパラメータ（最適化済み）
```python
lgb_params = {
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 30,
    'lambda_l2': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'random_state': 42
}
```

### 4.2 評価方法
- **交差検証**: 5-fold Cross Validation
- **評価指標**: R²（決定係数）
- **Train/Test分割**: 80/20

### 4.3 データリーケージの防止
- **subscriber_per_view = subscribers / views**: 使用禁止（viewsを含むため）
- **likes, comment_count**: 使用禁止（視聴後の結果）
- **subscribers単体**: 使用可能（viewsを含まない）

## 5. 実験結果

### 5.1 データセット比較実験

| データセット | サンプル数 | subscribers | CV R² | 標準偏差 |
|------------|-----------|------------|-------|---------|
| youtube_top_jp.csv | 767 | あり | 0.3239 | 0.0916 |
| youtube_top_new.csv | 6,078 | なし | 0.2696 | 0.0168 |
| youtube_top_new_complete.csv | 6,062 | なし | 0.2575 | 0.0259 |
| **youtube_top_new_complete.csv** | **6,062** | **あり** | **0.4528** | **0.0158** |

### 5.2 特徴量重要度分析

#### 最終モデル（CV R² = 0.4528）での重要度
1. **subscribers**: 1041 (24.5%)
2. **video_duration**: 590 (13.9%)
3. **colorfulness**: 576 (13.5%)
4. **brightness**: 522 (12.3%)
5. **description_length**: 485 (11.4%)
6. **hour_published**: 425 (10.0%)
7. **tags_count**: 313 (7.4%)
8. **object_complexity**: 245 (5.8%)
9. **log_subscribers**: 209 (4.9%)
10. **element_complexity**: 180 (4.2%)

### 5.3 subscribersの効果分析

```
subscribersなし: CV R² = 0.2575
subscribersあり: CV R² = 0.4528
改善幅: +0.1953 (75.8%改善)
```

### 5.4 過学習の評価

| モデル | Train R² | Test R² | 差分 |
|--------|----------|---------|------|
| 最終モデル | 0.6385 | 0.4550 | 0.1835 |

過学習度0.18は許容範囲内（一般的に0.2以下が望ましい）

### 5.5 予測精度の実用性評価

```
中央相対誤差: 68.8%
90%以内の予測: 67.8%の動画
極端な予測ミス（500%以上）: 11.1%
```

## 6. 考察

### 6.1 重要な発見

#### 1. チャンネル登録者数の決定的重要性
- 最も重要な特徴量（重要度24.5%）
- 単独で約75%の性能改善
- チャンネルのブランド力を表す指標

#### 2. サムネイル画像特徴の有効性
- colorfulness（13.5%）とbrightness（12.3%）が上位
- 視覚的に魅力的なサムネイルが重要
- ただし、顔検出は予想に反して負の影響

#### 3. 動画の長さの影響
- video_duration（13.9%）が2番目に重要
- 短い動画（90秒以下）が高パフォーマンス
- 視聴者の注意力の限界を反映

#### 4. 投稿タイミングの重要性
- hour_published（10.0%）が意外に高い重要度
- 最適な投稿時間帯の存在を示唆

### 6.2 データ量の影響

```
767件 → 6,078件: R²の改善なし（0.32 → 0.27）
サンプル数増加だけでは性能向上しない
特徴量の質（特にsubscribers）が決定的
```

### 6.3 モデルの限界

1. **予測精度の限界**: R² = 0.45が上限
   - YouTubeアルゴリズムの影響
   - トレンドやバイラル性の予測不可能性
   - 外部要因（SNS拡散等）の考慮不足

2. **時系列要素の欠如**
   - 動画公開後の成長曲線を考慮せず
   - 季節性やトレンドの変化に対応できない

3. **コンテンツ内容の未考慮**
   - タイトルのテキスト分析なし
   - 動画内容の質的評価なし

## 7. 実用的示唆

### 7.1 コンテンツクリエイター向け推奨事項

1. **チャンネル成長の重要性**
   - 500万人以上の登録者が閾値
   - 個別動画より継続的なチャンネル育成

2. **サムネイルデザイン**
   - 適度な色彩（colorfulness）
   - 明るさは70程度が最適
   - 顔の使用は慎重に

3. **動画の長さ**
   - 90秒以下を推奨
   - 長時間動画（689秒以上）は避ける

4. **投稿タイミング**
   - 時間帯の最適化が重要
   - ターゲット視聴者のアクティブ時間を考慮

## 8. 結論

本研究では、YouTube動画の再生数予測モデルを構築し、CV R² = 0.4528の性能を達成した。最も重要な発見は、チャンネル登録者数が圧倒的に重要な予測因子であることである。サムネイル画像の視覚的特徴も有意な影響を持つが、その効果は限定的である。

モデルの予測精度には限界があるが、これはYouTubeプラットフォームの複雑性と、我々が観測できない要因（アルゴリズム、トレンド等）の存在を反映している。しかし、本研究で特定された要因は、コンテンツクリエイターが戦略的に動画を最適化する上で有用な指針となる。

## 9. 今後の課題

1. **時系列モデルの導入**: 動画公開後の成長パターンの予測
2. **テキスト分析の追加**: タイトルと説明文のNLP分析
3. **外部データの統合**: SNSでの言及、Google Trendsデータ等
4. **因果推論の適用**: 相関から因果への移行

## 付録: 技術的詳細

### A. データ前処理
```python
# 対数変換（歪んだ分布の正規化）
df['log_views'] = np.log1p(df['views'])
df['log_subscribers'] = np.log1p(df['subscribers'])
df['log_duration'] = np.log1p(df['video_duration'])

# カテゴリカル変数のエンコーディング
df_encoded = pd.get_dummies(df, columns=['category_id'], prefix='cat')
```

### B. 画像処理パイプライン
```python
def extract_image_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    features = {
        'brightness': np.mean(hsv[:,:,2]),
        'colorfulness': calculate_colorfulness(img),
        'edge_density': calculate_edge_density(gray),
        'has_face': detect_faces(gray),
        'text_ratio': detect_text_area(img)
    }
    return features
```

### C. モデル学習コード
```python
from sklearn.model_selection import cross_val_score, KFold
import lightgbm as lgb

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
model = lgb.LGBMRegressor(**lgb_params)
cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
print(f"CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```