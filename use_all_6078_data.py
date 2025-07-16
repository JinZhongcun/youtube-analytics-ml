#!/usr/bin/env python3
"""
6,078件全てのサムネイルデータを活用した改良モデル
subscribersがないデータも含めて学習
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import cv2
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("6,078件全データを使用した改良モデル")
print("="*60)

# データ読み込み
df_old = pd.read_csv('youtube_top_jp.csv', skiprows=1)
df_new = pd.read_csv('youtube_top_new.csv')

print(f"旧データ（subscribers有）: {len(df_old)}件")
print(f"新データ（全サムネイル）: {len(df_new)}件")

# 画像特徴抽出（高速化のため並列処理）
def extract_image_features(video_id):
    """画像から特徴を抽出"""
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
            'img_brightness_contrast': np.std(v) / (np.mean(v) + 1e-6)
        }
    except:
        return None

# 全データに画像特徴を追加
print("\n画像特徴を抽出中...")
image_features_list = []
for i, video_id in enumerate(df_new['video_id']):
    if i % 500 == 0:
        print(f"  {i}/{len(df_new)}件処理済み")
    features = extract_image_features(video_id)
    image_features_list.append(features)

# 画像特徴をDataFrameに変換
image_features_df = pd.DataFrame(image_features_list)
df_new = pd.concat([df_new, image_features_df], axis=1)

# subscribersデータをマージ（ある場合のみ）
df_merged = pd.merge(df_new, df_old[['video_id', 'subscribers']], 
                     on='video_id', how='left')

print(f"\nsubscribersデータがある動画: {df_merged['subscribers'].notna().sum()}件")
print(f"subscribersデータがない動画: {df_merged['subscribers'].isna().sum()}件")

# 特徴量エンジニアリング
df_merged['log_views'] = np.log10(df_merged['views'] + 1)
df_merged['log_duration'] = np.log10(df_merged['video_duration'] + 1)
df_merged['log_tags_count'] = np.log10(df_merged['tags_count'] + 1)
df_merged['log_desc_length'] = np.log10(df_merged['description_length'] + 1)

# カテゴリのワンホットエンコーディング
category_dummies = pd.get_dummies(df_merged['category_id'], prefix='category')
df_merged = pd.concat([df_merged, category_dummies], axis=1)

# 特徴量の選択
feature_cols = [
    # 基本特徴
    'video_duration', 'tags_count', 'description_length',
    'object_complexity', 'element_complexity', 'brightness', 'colorfulness',
    'log_duration', 'log_tags_count', 'log_desc_length',
    # 画像特徴
    'img_aspect_ratio', 'img_hue_mean', 'img_hue_std',
    'img_saturation_mean', 'img_saturation_std',
    'img_brightness_mean', 'img_brightness_std',
    'img_edge_density', 'img_has_face', 'img_num_faces',
    'img_text_area_ratio', 'img_color_diversity',
    'img_center_brightness', 'img_color_vibrancy',
    'img_brightness_contrast'
] + [col for col in df_merged.columns if col.startswith('category_')]

# subscribersは特徴量として使う（ある場合のみ）
if 'subscribers' in df_merged.columns:
    df_merged['log_subscribers'] = np.log10(df_merged['subscribers'].fillna(1) + 1)
    df_merged['has_subscribers'] = (~df_merged['subscribers'].isna()).astype(int)
    feature_cols.extend(['log_subscribers', 'has_subscribers'])

# データの準備
X = df_merged[feature_cols].fillna(0)
y = df_merged['log_views']

# データ分割（stratifyのためにviewsの分位数を使用）
y_bins = pd.qcut(y, q=5, labels=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_bins
)

print(f"\n訓練データ: {len(X_train)}件")
print(f"テストデータ: {len(X_test)}件")

# スケーリング
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# モデル1: LightGBM（全データ対応）
print("\n=== LightGBM（全データ版） ===")
lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

y_pred_lgb = lgb_model.predict(X_test)
r2_lgb = r2_score(y_test, y_pred_lgb)
print(f"LightGBM R²: {r2_lgb:.4f}")

# モデル2: XGBoost（全データ対応）
print("\n=== XGBoost（全データ版） ===")
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              early_stopping_rounds=50,
              verbose=False)

y_pred_xgb = xgb_model.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"XGBoost R²: {r2_xgb:.4f}")

# モデル3: Random Forest（全データ対応）
print("\n=== Random Forest（全データ版） ===")
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest R²: {r2_rf:.4f}")

# アンサンブルモデル
print("\n=== アンサンブルモデル（全データ版） ===")
ensemble = VotingRegressor([
    ('lgb', lgb_model),
    ('xgb', xgb_model),
    ('rf', rf_model)
])

ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_test)
r2_ensemble = r2_score(y_test, y_pred_ensemble)
print(f"アンサンブル R²: {r2_ensemble:.4f}")

# 結果のまとめ
print("\n" + "="*60)
print("【結果まとめ】6,078件全データ使用")
print("="*60)
print(f"LightGBM R²: {r2_lgb:.4f}")
print(f"XGBoost R²: {r2_xgb:.4f}")
print(f"Random Forest R²: {r2_rf:.4f}")
print(f"アンサンブル R²: {r2_ensemble:.4f}")
print(f"\n最良モデル: {max(r2_lgb, r2_xgb, r2_rf, r2_ensemble):.4f}")
print(f"改善率: {(max(r2_lgb, r2_xgb, r2_rf, r2_ensemble) / 0.44 - 1) * 100:.1f}%")

# 特徴量の重要度（LightGBM）
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

print("\n【重要な特徴量TOP15】")
for idx, row in feature_importance.iterrows():
    print(f"{row['feature']}: {row['importance']:.3f}")

# 結果を保存
results = {
    'data_count': len(df_merged),
    'train_count': len(X_train),
    'test_count': len(X_test),
    'models': {
        'lightgbm': r2_lgb,
        'xgboost': r2_xgb,
        'random_forest': r2_rf,
        'ensemble': r2_ensemble
    },
    'best_r2': max(r2_lgb, r2_xgb, r2_rf, r2_ensemble),
    'improvement_from_607': (max(r2_lgb, r2_xgb, r2_rf, r2_ensemble) / 0.44 - 1) * 100
}

import json
with open('all_6078_data_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n結果をall_6078_data_results.jsonに保存しました。")