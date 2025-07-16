#!/usr/bin/env python3
"""
subscribersを一切使わないモデル
画像特徴とメタデータのみで予測
"""

import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Subscribersを使わないモデル（全6,078件使用可能）")
print("="*60)

# データ読み込み
df = pd.read_csv('youtube_top_new.csv')
print(f"全データ数: {len(df)}件")

# 画像特徴抽出関数
def extract_image_features(video_id):
    """画像から詳細な特徴を抽出"""
    img_path = f'thumbnails/{video_id}.jpg'
    
    if not os.path.exists(img_path):
        return None
    
    try:
        # OpenCVで読み込み
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 基本統計量
        height, width, _ = img.shape
        aspect_ratio = width / height
        
        # 色空間の統計
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # エッジ検出
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # 顔検出
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        has_face = len(faces) > 0
        num_faces = len(faces)
        
        # テキスト領域の推定
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        text_area_ratio = np.sum(binary > 0) / (height * width)
        
        # 色の多様性
        unique_colors = len(np.unique(img.reshape(-1, 3), axis=0))
        color_diversity = unique_colors / (height * width)
        
        # 中心部の明るさ
        center_y, center_x = height // 2, width // 2
        center_region = v[center_y-50:center_y+50, center_x-50:center_x+50]
        center_brightness = np.mean(center_region) if center_region.size > 0 else np.mean(v)
        
        # LAB色空間での色の鮮やかさ
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        color_vibrancy = np.std(a) + np.std(b)
        
        # コントラスト
        contrast = np.std(gray)
        
        # 4分割領域の特徴
        h_mid, w_mid = height // 2, width // 2
        quadrants = [
            gray[:h_mid, :w_mid],
            gray[:h_mid, w_mid:],
            gray[h_mid:, :w_mid],
            gray[h_mid:, w_mid:]
        ]
        quadrant_brightness_std = np.std([np.mean(q) for q in quadrants])
        
        return {
            'img_aspect_ratio': aspect_ratio,
            'img_hue_mean': np.mean(h),
            'img_hue_std': np.std(h),
            'img_saturation_mean': np.mean(s),
            'img_saturation_std': np.std(s),
            'img_brightness_mean': np.mean(v),
            'img_brightness_std': np.std(v),
            'img_edge_density': edge_density,
            'img_has_face': int(has_face),
            'img_num_faces': num_faces,
            'img_text_area_ratio': text_area_ratio,
            'img_color_diversity': color_diversity,
            'img_center_brightness': center_brightness,
            'img_color_vibrancy': color_vibrancy,
            'img_contrast': contrast,
            'img_quadrant_brightness_std': quadrant_brightness_std
        }
    except:
        return None

# サンプリングして画像特徴を抽出（高速化のため）
print("\n画像特徴を抽出中...")
sample_size = min(2000, len(df))  # 最大2000件
sample_indices = np.random.choice(df.index, sample_size, replace=False)

image_features_list = []
for i, idx in enumerate(sample_indices):
    if i % 200 == 0:
        print(f"  {i}/{sample_size}件処理済み")
    video_id = df.loc[idx, 'video_id']
    features = extract_image_features(video_id)
    if features:
        features['index'] = idx
        image_features_list.append(features)

# 画像特徴をDataFrameに変換
image_features_df = pd.DataFrame(image_features_list).set_index('index')
df_sample = df.loc[sample_indices].join(image_features_df)
df_sample = df_sample.dropna()

print(f"\n画像特徴抽出完了: {len(df_sample)}件")

# 時間特徴量
df_sample['published_at'] = pd.to_datetime(df_sample['published_at']).dt.tz_localize(None)
df_sample['days_since_publish'] = (pd.Timestamp.now() - df_sample['published_at']).dt.days
df_sample['hour_published'] = pd.to_datetime(df_sample['published_at']).dt.hour
df_sample['weekday_published'] = pd.to_datetime(df_sample['published_at']).dt.dayofweek

# カテゴリのワンホットエンコーディング
category_dummies = pd.get_dummies(df_sample['category_id'], prefix='category')
df_sample = pd.concat([df_sample, category_dummies], axis=1)

# 特徴量エンジニアリング
df_sample['log_duration'] = np.log10(df_sample['video_duration'] + 1)
df_sample['log_tags'] = np.log10(df_sample['tags_count'] + 1)
df_sample['log_desc_length'] = np.log10(df_sample['description_length'] + 1)
df_sample['tags_per_second'] = df_sample['tags_count'] / (df_sample['video_duration'] + 1)
df_sample['desc_per_second'] = df_sample['description_length'] / (df_sample['video_duration'] + 1)

# 特徴量の選択（subscribersを除外）
feature_cols = [
    # 基本特徴
    'video_duration', 'tags_count', 'description_length',
    'object_complexity', 'element_complexity', 'brightness', 'colorfulness',
    'log_duration', 'log_tags', 'log_desc_length',
    'tags_per_second', 'desc_per_second',
    'days_since_publish', 'hour_published', 'weekday_published',
    # 画像特徴
    'img_aspect_ratio', 'img_hue_mean', 'img_hue_std',
    'img_saturation_mean', 'img_saturation_std',
    'img_brightness_mean', 'img_brightness_std',
    'img_edge_density', 'img_has_face', 'img_num_faces',
    'img_text_area_ratio', 'img_color_diversity',
    'img_center_brightness', 'img_color_vibrancy',
    'img_contrast', 'img_quadrant_brightness_std'
] + [col for col in df_sample.columns if col.startswith('category_')]

# データ準備
X = df_sample[feature_cols]
y = np.log10(df_sample['views'] + 1)

# データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n訓練データ: {len(X_train)}件")
print(f"テストデータ: {len(X_test)}件")

# スケーリング
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# モデル1: LightGBM
print("\n=== LightGBM ===")
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
r2_lgb = r2_score(y_test, y_pred_lgb)
print(f"R²: {r2_lgb:.4f}")

# モデル2: XGBoost
print("\n=== XGBoost ===")
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"R²: {r2_xgb:.4f}")

# モデル3: Random Forest
print("\n=== Random Forest ===")
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"R²: {r2_rf:.4f}")

# アンサンブル
print("\n=== アンサンブル ===")
ensemble = VotingRegressor([
    ('lgb', lgb_model),
    ('xgb', xgb_model),
    ('rf', rf_model)
])

ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_test)
r2_ensemble = r2_score(y_test, y_pred_ensemble)
print(f"R²: {r2_ensemble:.4f}")

# 結果まとめ
print("\n" + "="*60)
print("【結果まとめ】Subscribersなしモデル")
print("="*60)
print(f"LightGBM: {r2_lgb:.4f}")
print(f"XGBoost: {r2_xgb:.4f}")
print(f"Random Forest: {r2_rf:.4f}")
print(f"アンサンブル: {r2_ensemble:.4f}")
print(f"\n最良スコア: {max(r2_lgb, r2_xgb, r2_rf, r2_ensemble):.4f}")

# 重要な特徴量TOP10
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

print("\n【重要な特徴量TOP10】")
for idx, row in feature_importance.iterrows():
    print(f"{row['feature']}: {row['importance']:.1f}")