#!/usr/bin/env python3
"""
YouTube動画分析 - サムネイル画像を含む統合モデル
CNN/Vision Transformerによる画像特徴抽出
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB0, VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("YouTube動画分析 - サムネイル画像統合モデル")
print("="*60)

# データ読み込み
df = pd.read_csv('youtube_top_new.csv')
print(f"データ数: {len(df)}件")

# サムネイル画像の存在確認
thumbnails_dir = 'thumbnails'
df['thumbnail_exists'] = df['video_id'].apply(lambda x: os.path.exists(os.path.join(thumbnails_dir, f'{x}.jpg')))
df_with_thumbnails = df[df['thumbnail_exists']].copy()
print(f"サムネイル画像が存在するデータ: {len(df_with_thumbnails)}件")

# 時間関連の特徴量
df_with_thumbnails['published_at'] = pd.to_datetime(df_with_thumbnails['published_at']).dt.tz_localize(None)
df_with_thumbnails['days_since_publish'] = (pd.Timestamp.now() - df_with_thumbnails['published_at']).dt.days

# 数値特徴量
numerical_features = [
    'video_duration', 'tags_count', 'description_length',
    'object_complexity', 'element_complexity', 'brightness', 
    'colorfulness', 'days_since_publish'
]

# ターゲット変数（対数変換）
y = np.log10(df_with_thumbnails['views'] + 1)

# データ分割（画像処理の前に分割）
indices = np.arange(len(df_with_thumbnails))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

print(f"\n訓練: {len(train_idx)}件, 検証: {len(val_idx)}件, テスト: {len(test_idx)}件")

# 画像読み込み関数
def load_and_preprocess_image(video_id, size=(224, 224)):
    """サムネイル画像を読み込み前処理"""
    img_path = os.path.join(thumbnails_dir, f'{video_id}.jpg')
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(size)
        img_array = img_to_array(img) / 255.0
        return img_array
    except:
        return np.zeros((*size, 3))

# 効率的なバッチ処理のため、画像を事前に読み込み
print("\n画像データを読み込み中...")
image_arrays = []
for i, video_id in enumerate(df_with_thumbnails['video_id'].values):
    if i % 1000 == 0:
        print(f"  {i}/{len(df_with_thumbnails)}件処理済み")
    image_arrays.append(load_and_preprocess_image(video_id))

image_arrays = np.array(image_arrays)
print(f"画像データ形状: {image_arrays.shape}")

# 数値特徴量の準備
X_numerical = df_with_thumbnails[numerical_features].fillna(0).values
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)

# データセットの分割
X_img_train = image_arrays[train_idx]
X_img_val = image_arrays[val_idx]
X_img_test = image_arrays[test_idx]

X_num_train = X_numerical_scaled[train_idx]
X_num_val = X_numerical_scaled[val_idx]
X_num_test = X_numerical_scaled[test_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]
y_test = y.iloc[test_idx]

# 1. CNN特徴抽出器の構築
print("\n=== CNN特徴抽出器の構築 ===")

# 事前学習済みモデル（ImageNetで訓練済み）
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # 転移学習のため固定

# 画像特徴抽出モデル
img_input = Input(shape=(224, 224, 3), name='image_input')
x = base_model(img_input, training=False)
x = GlobalAveragePooling2D()(x)
img_features = Dense(128, activation='relu')(x)
img_features = Dropout(0.3)(img_features)

# 数値特徴入力
num_input = Input(shape=(len(numerical_features),), name='numerical_input')
num_features = Dense(64, activation='relu')(num_input)
num_features = Dropout(0.3)(num_features)

# 特徴量の結合
combined = concatenate([img_features, num_features])
x = Dense(128, activation='relu')(combined)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(1)(x)

# 統合モデル
model = Model(inputs=[img_input, num_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

print(model.summary())

# モデルの訓練
print("\n=== モデルの訓練 ===")
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

history = model.fit(
    [X_img_train, X_num_train], y_train,
    validation_data=([X_img_val, X_num_val], y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 2. CNN特徴量の抽出（他のMLモデル用）
print("\n=== CNN特徴量の抽出 ===")

# 中間層の出力を取得するモデル
feature_extractor = Model(inputs=model.inputs, 
                         outputs=model.get_layer('concatenate').output)

# 特徴量を抽出
train_features = feature_extractor.predict([X_img_train, X_num_train])
val_features = feature_extractor.predict([X_img_val, X_num_val])
test_features = feature_extractor.predict([X_img_test, X_num_test])

print(f"抽出された特徴量の次元: {train_features.shape[1]}")

# 3. 抽出した特徴量でLightGBMとXGBoostを訓練
print("\n=== 抽出特徴量でのモデル比較 ===")

models = {
    'CNN (End-to-End)': None,  # 既に訓練済み
    'LightGBM on CNN features': lgb.LGBMRegressor(n_estimators=300, random_state=42, verbosity=-1),
    'XGBoost on CNN features': xgb.XGBRegressor(n_estimators=300, random_state=42, verbosity=0),
    'Random Forest on CNN features': RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
}

results = {}

# CNN (End-to-End)の評価
cnn_pred = model.predict([X_img_test, X_num_test]).flatten()
results['CNN (End-to-End)'] = {
    'r2': r2_score(y_test, cnn_pred),
    'mse': mean_squared_error(y_test, cnn_pred),
    'mae': mean_absolute_error(y_test, cnn_pred)
}

# 他のモデルの訓練と評価
for name, ml_model in models.items():
    if ml_model is not None:
        print(f"\n{name}を訓練中...")
        ml_model.fit(train_features, y_train)
        pred = ml_model.predict(test_features)
        
        results[name] = {
            'r2': r2_score(y_test, pred),
            'mse': mean_squared_error(y_test, pred),
            'mae': mean_absolute_error(y_test, pred)
        }

# 結果の表示
print("\n=== 最終結果（テストセット） ===")
print(f"{'Model':<30} {'R²':>8} {'MSE':>8} {'MAE':>8}")
print("-" * 60)
for name, metrics in sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True):
    print(f"{name:<30} {metrics['r2']:>8.4f} {metrics['mse']:>8.4f} {metrics['mae']:>8.4f}")

# 4. 重要な洞察の可視化
plt.figure(figsize=(20, 12))

# 学習曲線
plt.subplot(2, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CNN Training History')
plt.legend()

# 予測vs実測（最良モデル）
best_model_name = max(results.items(), key=lambda x: x[1]['r2'])[0]
if best_model_name == 'CNN (End-to-End)':
    best_pred = cnn_pred
else:
    best_model = models[best_model_name]
    best_pred = best_model.predict(test_features)

plt.subplot(2, 3, 2)
plt.scatter(y_test, best_pred, alpha=0.5, s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Log10(Views + 1)')
plt.ylabel('Predicted Log10(Views + 1)')
plt.title(f'{best_model_name} (R² = {results[best_model_name]["r2"]:.4f})')
plt.grid(True, alpha=0.3)

# カテゴリ別の性能
plt.subplot(2, 3, 3)
test_df = df_with_thumbnails.iloc[test_idx].copy()
test_df['predicted'] = best_pred
test_df['actual'] = y_test
category_performance = test_df.groupby('category_id').apply(
    lambda x: r2_score(x['actual'], x['predicted']) if len(x) > 10 else np.nan
).dropna().sort_values(ascending=False)[:10]

category_performance.plot(kind='barh')
plt.xlabel('R² Score')
plt.title('Performance by Category (Top 10)')

# サンプルサムネイル表示（高再生回数）
plt.subplot(2, 3, 4)
high_views_idx = test_df.nlargest(9, 'views').index
sample_images = []
for idx in high_views_idx[:9]:
    video_id = df_with_thumbnails.loc[idx, 'video_id']
    img = load_and_preprocess_image(video_id, size=(100, 100))
    sample_images.append(img)

grid = np.array(sample_images).reshape(3, 3, 100, 100, 3)
grid = grid.transpose(0, 2, 1, 3, 4).reshape(300, 300, 3)
plt.imshow(grid)
plt.axis('off')
plt.title('High View Count Thumbnails')

# サンプルサムネイル表示（低再生回数）
plt.subplot(2, 3, 5)
low_views_idx = test_df.nsmallest(9, 'views').index
sample_images = []
for idx in low_views_idx[:9]:
    video_id = df_with_thumbnails.loc[idx, 'video_id']
    img = load_and_preprocess_image(video_id, size=(100, 100))
    sample_images.append(img)

grid = np.array(sample_images).reshape(3, 3, 100, 100, 3)
grid = grid.transpose(0, 2, 1, 3, 4).reshape(300, 300, 3)
plt.imshow(grid)
plt.axis('off')
plt.title('Low View Count Thumbnails')

# 特徴量の重要度（LightGBMの場合）
if 'LightGBM on CNN features' in models and models['LightGBM on CNN features'] is not None:
    plt.subplot(2, 3, 6)
    lgb_model = models['LightGBM on CNN features']
    feature_importance = lgb_model.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-20:]
    plt.barh(range(20), feature_importance[top_features_idx])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 CNN Features (LightGBM)')

plt.tight_layout()
plt.savefig('thumbnail_cnn_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 結果の保存
import json
final_results = {
    'data_info': {
        'total_samples': len(df),
        'samples_with_thumbnails': len(df_with_thumbnails),
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx)
    },
    'model_results': results,
    'best_model': best_model_name,
    'cnn_architecture': 'EfficientNetB0 + Custom Dense Layers',
    'numerical_features': numerical_features
}

with open('thumbnail_cnn_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print("\n✅ 分析完了！")
print(f"最良モデル: {best_model_name}")
print(f"最高R²スコア: {results[best_model_name]['r2']:.4f}")
print("結果は thumbnail_cnn_results.json に保存されました。")