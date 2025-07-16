#!/usr/bin/env python3
"""
サムネイル画像の簡易分析
CNNを使わず、画像の基本的な特徴を抽出
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import cv2
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("サムネイル画像の簡易分析")
print("="*60)

# データ読み込み
df_old = pd.read_csv('youtube_top_jp.csv', skiprows=1)
df_new = pd.read_csv('youtube_top_new.csv')

print(f"旧データ: {len(df_old)}件")
print(f"新データ: {len(df_new)}件")

# 新旧データの比較
print("\n=== データセットの比較 ===")
print("旧データのカラム:", df_old.columns.tolist())
print("\n新データのカラム:", df_new.columns.tolist())

# 新データにない重要なカラム
missing_columns = set(df_old.columns) - set(df_new.columns)
print(f"\n新データにないカラム: {missing_columns}")

# サムネイル画像の存在確認
thumbnails_dir = 'thumbnails'
df_new['thumbnail_exists'] = df_new['video_id'].apply(
    lambda x: os.path.exists(os.path.join(thumbnails_dir, f'{x}.jpg'))
)
print(f"\nサムネイル画像が存在: {df_new['thumbnail_exists'].sum()}/{len(df_new)}件")

# 画像が存在するデータのみ使用
df = df_new[df_new['thumbnail_exists']].copy()
print(f"分析対象データ: {len(df)}件")

# 追加の画像特徴を抽出する関数
def extract_advanced_image_features(video_id):
    """画像から詳細な特徴を抽出"""
    img_path = os.path.join(thumbnails_dir, f'{video_id}.jpg')
    
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
        
        # エッジ検出（複雑さの指標）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # 顔検出（簡易版）
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        has_face = len(faces) > 0
        num_faces = len(faces)
        
        # テキスト領域の推定（高コントラスト領域）
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        text_area_ratio = np.sum(binary > 0) / (height * width)
        
        # 色の多様性
        unique_colors = len(np.unique(img.reshape(-1, 3), axis=0))
        color_diversity = unique_colors / (height * width)
        
        # 中心部の明るさ（注目度）
        center_y, center_x = height // 2, width // 2
        center_region = v[center_y-50:center_y+50, center_x-50:center_x+50]
        center_brightness = np.mean(center_region) if center_region.size > 0 else np.mean(v)
        
        return {
            'aspect_ratio': aspect_ratio,
            'hue_mean': np.mean(h),
            'hue_std': np.std(h),
            'saturation_mean': np.mean(s),
            'saturation_std': np.std(s),
            'value_mean': np.mean(v),
            'value_std': np.std(v),
            'edge_density': edge_density,
            'has_face': int(has_face),
            'num_faces': num_faces,
            'text_area_ratio': text_area_ratio,
            'color_diversity': color_diversity,
            'center_brightness': center_brightness
        }
    except Exception as e:
        print(f"Error processing {video_id}: {e}")
        return {
            'aspect_ratio': 1.77,
            'hue_mean': 0, 'hue_std': 0,
            'saturation_mean': 0, 'saturation_std': 0,
            'value_mean': 0, 'value_std': 0,
            'edge_density': 0, 'has_face': 0, 'num_faces': 0,
            'text_area_ratio': 0, 'color_diversity': 0,
            'center_brightness': 0
        }

# サンプル画像で特徴抽出をテスト（全データは時間がかかる）
print("\n=== 画像特徴の抽出（サンプル） ===")
sample_size = min(1000, len(df))
sample_df = df.sample(n=sample_size, random_state=42).copy()

# 画像特徴を抽出
image_features_list = []
for i, video_id in enumerate(sample_df['video_id']):
    if i % 100 == 0:
        print(f"  {i}/{sample_size}件処理済み")
    features = extract_advanced_image_features(video_id)
    image_features_list.append(features)

# DataFrameに変換
image_features_df = pd.DataFrame(image_features_list)
sample_df = pd.concat([sample_df.reset_index(drop=True), image_features_df], axis=1)

# 時間関連の特徴量
sample_df['published_at'] = pd.to_datetime(sample_df['published_at']).dt.tz_localize(None)
sample_df['days_since_publish'] = (pd.Timestamp.now() - sample_df['published_at']).dt.days

# 特徴量の準備
feature_columns = [
    # 既存の特徴量
    'video_duration', 'tags_count', 'description_length',
    'object_complexity', 'element_complexity', 'brightness', 'colorfulness',
    'days_since_publish',
    # 新しい画像特徴量
    'aspect_ratio', 'hue_mean', 'hue_std', 'saturation_mean', 'saturation_std',
    'value_mean', 'value_std', 'edge_density', 'has_face', 'num_faces',
    'text_area_ratio', 'color_diversity', 'center_brightness'
]

X = sample_df[feature_columns].fillna(0)
y = np.log10(sample_df['views'] + 1)

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n訓練: {len(X_train)}件, テスト: {len(X_test)}件")

# モデルの訓練と評価
print("\n=== モデルの訓練と評価 ===")
models = {
    'LightGBM': lgb.LGBMRegressor(n_estimators=200, random_state=42, verbosity=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
}

results = {}
for name, model in models.items():
    print(f"\n{name}を訓練中...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    results[name] = {'r2': r2, 'mse': mse}
    print(f"  R²: {r2:.4f}, MSE: {mse:.4f}")

# 特徴量の重要度（LightGBM）
lgb_model = models['LightGBM']
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== 特徴量の重要度（Top 10） ===")
for _, row in feature_importance.head(10).iterrows():
    print(f"{row['feature']:<25} {row['importance']:>8.0f}")

# 可視化
plt.figure(figsize=(20, 15))

# 1. モデル性能比較
plt.subplot(3, 3, 1)
model_names = list(results.keys())
r2_scores = [results[name]['r2'] for name in model_names]
plt.bar(model_names, r2_scores)
plt.ylabel('R² Score')
plt.title('Model Performance Comparison')
plt.ylim(0, max(r2_scores) * 1.2)

# 2. 特徴量重要度
plt.subplot(3, 3, 2)
top_features = feature_importance.head(15)
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance (LightGBM)')

# 3. 顔検出と再生回数
plt.subplot(3, 3, 3)
face_stats = sample_df.groupby('has_face')['views'].agg(['mean', 'median', 'count'])
face_stats.index = ['No Face', 'Has Face']
face_stats['mean'].plot(kind='bar')
plt.ylabel('Average Views')
plt.title('Views by Face Detection')
plt.xticks(rotation=0)

# 4. 色彩度の分布
plt.subplot(3, 3, 4)
plt.hist(sample_df['colorfulness'], bins=30, alpha=0.7, label='Original')
plt.hist(sample_df['saturation_mean'], bins=30, alpha=0.7, label='HSV Saturation')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Color Saturation Distribution')
plt.legend()

# 5. エッジ密度と再生回数
plt.subplot(3, 3, 5)
plt.scatter(sample_df['edge_density'], sample_df['views'], alpha=0.5, s=20)
plt.xlabel('Edge Density')
plt.ylabel('Views')
plt.yscale('log')
plt.title('Views vs Edge Density (Complexity)')

# 6. カテゴリ別の平均特徴量
plt.subplot(3, 3, 6)
category_features = sample_df.groupby('category_id')[['has_face', 'edge_density', 'text_area_ratio']].mean()
category_features.plot(kind='bar')
plt.xlabel('Category ID')
plt.ylabel('Average Value')
plt.title('Image Features by Category')
plt.legend()

# 7. 予測vs実測（最良モデル）
plt.subplot(3, 3, 7)
best_model_name = max(results.items(), key=lambda x: x[1]['r2'])[0]
best_model = models[best_model_name]
y_pred = best_model.predict(X_test_scaled)
plt.scatter(y_test, y_pred, alpha=0.5, s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Log10(Views + 1)')
plt.ylabel('Predicted Log10(Views + 1)')
plt.title(f'{best_model_name} (R² = {results[best_model_name]["r2"]:.4f})')

# 8. 中心部の明るさと再生回数
plt.subplot(3, 3, 8)
plt.scatter(sample_df['center_brightness'], sample_df['views'], alpha=0.5, s=20)
plt.xlabel('Center Brightness')
plt.ylabel('Views')
plt.yscale('log')
plt.title('Views vs Center Brightness')

# 9. 高再生回数動画の特徴
plt.subplot(3, 3, 9)
high_views = sample_df[sample_df['views'] > sample_df['views'].quantile(0.9)]
low_views = sample_df[sample_df['views'] < sample_df['views'].quantile(0.1)]

features_to_compare = ['has_face', 'edge_density', 'text_area_ratio', 'color_diversity']
high_avg = high_views[features_to_compare].mean()
low_avg = low_views[features_to_compare].mean()

x = np.arange(len(features_to_compare))
width = 0.35

plt.bar(x - width/2, high_avg, width, label='High Views (Top 10%)')
plt.bar(x + width/2, low_avg, width, label='Low Views (Bottom 10%)')
plt.xlabel('Features')
plt.ylabel('Average Value')
plt.title('Feature Comparison: High vs Low Views')
plt.xticks(x, features_to_compare, rotation=45)
plt.legend()

plt.tight_layout()
plt.savefig('simple_image_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 最終レポート
print("\n" + "="*60)
print("分析結果のまとめ")
print("="*60)
print(f"最良モデル: {best_model_name} (R² = {results[best_model_name]['r2']:.4f})")
print(f"\n主要な発見:")
print(f"- 顔検出あり: 平均{face_stats.loc['Has Face', 'mean']/face_stats.loc['No Face', 'mean']:.2f}倍の再生回数")
print(f"- 最重要特徴: {feature_importance.iloc[0]['feature']}")
print(f"- データ数増加の効果: 前回R²=0.21 → 今回R²={max(r2_scores):.4f}")

# 結果の保存
import json
final_results = {
    'data_info': {
        'total_new_data': len(df_new),
        'analyzed_samples': len(sample_df),
        'feature_count': len(feature_columns)
    },
    'model_results': results,
    'top_features': feature_importance.head(10).to_dict('records'),
    'face_detection_impact': {
        'with_face_avg_views': float(face_stats.loc['Has Face', 'mean']),
        'without_face_avg_views': float(face_stats.loc['No Face', 'mean']),
        'ratio': float(face_stats.loc['Has Face', 'mean'] / face_stats.loc['No Face', 'mean'])
    }
}

with open('simple_image_analysis_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print("\n結果を simple_image_analysis_results.json に保存しました。")