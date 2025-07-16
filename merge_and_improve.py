#!/usr/bin/env python3
"""
新旧データの統合と改良モデル
画像特徴量 + 既存特徴量 + subscribersを組み合わせ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("新旧データ統合による改良モデル")
print("="*60)

# データ読み込み
df_old = pd.read_csv('youtube_top_jp.csv', skiprows=1)
df_new = pd.read_csv('youtube_top_new.csv')

print(f"旧データ: {len(df_old)}件（subscribersあり）")
print(f"新データ: {len(df_new)}件（画像あり）")

# video_idで結合を試みる
df_merged = pd.merge(df_new, df_old[['video_id', 'subscribers', 'likes', 'comment_count']], 
                     on='video_id', how='left', suffixes=('', '_old'))

# 結合成功率の確認
merged_with_subs = df_merged['subscribers'].notna().sum()
print(f"\nsubscribersデータと結合成功: {merged_with_subs}/{len(df_merged)}件")

# 画像特徴抽出（高速版）
def extract_image_features_fast(video_id):
    """高速な画像特徴抽出"""
    img_path = os.path.join('thumbnails', f'{video_id}.jpg')
    
    try:
        # 画像読み込み（小さいサイズで）
        img = cv2.imread(img_path)
        img = cv2.resize(img, (150, 100))  # 処理高速化
        
        # HSV変換
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 基本統計
        features = {
            'hue_mean': np.mean(h),
            'saturation_mean': np.mean(s),
            'value_mean': np.mean(v),
            'hue_std': np.std(h),
            'saturation_std': np.std(s),
            'value_std': np.std(v)
        }
        
        # エッジ検出（簡易版）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        features['edge_ratio'] = np.sum(edges > 0) / edges.size
        
        # 顔検出（簡易版）
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        features['has_face'] = int(len(faces) > 0)
        
        return features
    except:
        return {
            'hue_mean': 0, 'saturation_mean': 0, 'value_mean': 0,
            'hue_std': 0, 'saturation_std': 0, 'value_std': 0,
            'edge_ratio': 0, 'has_face': 0
        }

# 戦略1: subscribersがあるデータのみで高精度モデル
print("\n=== 戦略1: Subscribers付きデータで高精度モデル ===")
df_with_subs = df_merged[df_merged['subscribers'].notna()].copy()
print(f"使用データ数: {len(df_with_subs)}件")

if len(df_with_subs) > 100:
    # 画像特徴を抽出（サンプル）
    print("画像特徴を抽出中...")
    sample_size = min(500, len(df_with_subs))
    sample_indices = np.random.choice(df_with_subs.index, sample_size, replace=False)
    
    image_features_list = []
    for i, idx in enumerate(sample_indices):
        if i % 100 == 0:
            print(f"  {i}/{sample_size}件処理済み")
        video_id = df_with_subs.loc[idx, 'video_id']
        features = extract_image_features_fast(video_id)
        image_features_list.append(features)
    
    # DataFrameに変換
    image_features_df = pd.DataFrame(image_features_list, index=sample_indices)
    df_with_subs = df_with_subs.join(image_features_df)
    
    # 時間特徴量
    df_with_subs['published_at'] = pd.to_datetime(df_with_subs['published_at']).dt.tz_localize(None)
    df_with_subs['days_since_publish'] = (pd.Timestamp.now() - df_with_subs['published_at']).dt.days
    
    # 特徴量選択
    feature_columns = [
        # 基本特徴量
        'video_duration', 'tags_count', 'description_length', 'subscribers',
        'object_complexity', 'element_complexity', 'brightness', 'colorfulness',
        'days_since_publish',
        # 画像特徴量
        'hue_mean', 'saturation_mean', 'value_mean', 'edge_ratio', 'has_face'
    ]
    
    # 欠損値処理
    df_with_subs_clean = df_with_subs[feature_columns + ['views']].dropna()
    X = df_with_subs_clean[feature_columns]
    y = np.log10(df_with_subs_clean['views'] + 1)
    
    print(f"\n最終データ数: {len(X)}件")
    
    # モデル訓練
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 複数モデルの訓練
    models = {
        'LightGBM': lgb.LGBMRegressor(n_estimators=200, random_state=42, verbosity=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0),
        'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    }
    
    results_with_subs = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        results_with_subs[name] = r2
        print(f"{name}: R² = {r2:.4f}")
    
    # アンサンブルモデル
    ensemble = VotingRegressor([
        ('lgb', models['LightGBM']),
        ('xgb', models['XGBoost']),
        ('rf', models['Random Forest'])
    ])
    ensemble.fit(X_train_scaled, y_train)
    y_pred_ensemble = ensemble.predict(X_test_scaled)
    r2_ensemble = r2_score(y_test, y_pred_ensemble)
    results_with_subs['Ensemble'] = r2_ensemble
    print(f"Ensemble: R² = {r2_ensemble:.4f}")

# 戦略2: 全データを使った転移学習アプローチ
print("\n=== 戦略2: 転移学習アプローチ ===")
print("1. Subscribersありデータで基本モデルを訓練")
print("2. 画像特徴量の重要度を学習")
print("3. Subscribersなしデータに適用")

# 可視化
if len(df_with_subs) > 100:
    plt.figure(figsize=(15, 10))
    
    # 1. モデル性能比較
    plt.subplot(2, 3, 1)
    model_names = list(results_with_subs.keys())
    r2_scores = list(results_with_subs.values())
    bars = plt.bar(model_names, r2_scores)
    plt.ylabel('R² Score')
    plt.title('Model Performance with Subscribers Data')
    plt.ylim(0, max(r2_scores) * 1.2)
    
    # 色分け
    for bar, score in zip(bars, r2_scores):
        if score > 0.3:
            bar.set_color('green')
        elif score > 0.2:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # 2. 特徴量重要度
    plt.subplot(2, 3, 2)
    lgb_model = models['LightGBM']
    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.barh(importance['feature'][:10], importance['importance'][:10])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance')
    
    # 3. Subscribers vs Views
    plt.subplot(2, 3, 3)
    plt.scatter(df_with_subs_clean['subscribers'], df_with_subs_clean['views'], 
                alpha=0.5, s=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Subscribers')
    plt.ylabel('Views')
    plt.title('Views vs Subscribers (Log Scale)')
    
    # 4. 顔検出の効果
    plt.subplot(2, 3, 4)
    if 'has_face' in df_with_subs_clean.columns:
        face_stats = df_with_subs_clean.groupby('has_face')['views'].agg(['mean', 'count'])
        face_stats.index = ['No Face', 'Has Face']
        face_stats['mean'].plot(kind='bar')
        plt.ylabel('Average Views')
        plt.title('Views by Face Detection')
        plt.xticks(rotation=0)
    
    # 5. 予測vs実測
    plt.subplot(2, 3, 5)
    plt.scatter(y_test, y_pred_ensemble, alpha=0.5, s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Log10(Views + 1)')
    plt.ylabel('Predicted Log10(Views + 1)')
    plt.title(f'Ensemble Model (R² = {r2_ensemble:.4f})')
    
    # 6. データ統合の効果
    plt.subplot(2, 3, 6)
    comparison_data = {
        'Old Model\n(R²=0.21)': 0.21,
        'Image Only\n(R²=0.15)': 0.15,
        'Combined\n(R²={:.3f})'.format(r2_ensemble): r2_ensemble
    }
    bars = plt.bar(comparison_data.keys(), comparison_data.values())
    plt.ylabel('R² Score')
    plt.title('Model Evolution')
    
    # 色分け
    colors = ['blue', 'orange', 'green' if r2_ensemble > 0.21 else 'red']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    plt.savefig('merged_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# 最終レポート
print("\n" + "="*60)
print("分析結果まとめ")
print("="*60)

if len(df_with_subs) > 100:
    print(f"統合データでの最良R²: {max(results_with_subs.values()):.4f}")
    print(f"最良モデル: {max(results_with_subs.items(), key=lambda x: x[1])[0]}")
    print(f"\n重要な発見:")
    print(f"- Subscribersありデータ: {len(df_with_subs)}件")
    print(f"- 画像特徴量の追加効果: {'向上' if r2_ensemble > 0.21 else '限定的'}")
    print(f"- 最重要特徴量: {importance.iloc[0]['feature']}")
else:
    print("統合可能なデータが少なすぎます。")
    print("新データにsubscribersの追加が必要です。")

# 結果の保存
import json
final_results = {
    'data_summary': {
        'old_data_count': len(df_old),
        'new_data_count': len(df_new),
        'merged_count': merged_with_subs,
        'analyzed_count': len(df_with_subs) if 'df_with_subs' in locals() else 0
    },
    'model_results': results_with_subs if 'results_with_subs' in locals() else {},
    'conclusion': 'Subscribers data is critical for accurate prediction'
}

with open('merged_model_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print("\n結果を merged_model_results.json に保存しました。")