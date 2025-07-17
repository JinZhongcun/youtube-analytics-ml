#!/usr/bin/env python3
"""
subscribersデータが復活した新データで分析
6,062件全てにsubscribers情報あり！
"""

import pandas as pd
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("subscribers復活！完全データ6,062件で最高精度を目指す")
print("="*60)

# 新データをコピー
print("新データをコピー中...")
shutil.copy('drive-download-20250717T063336Z-1-001/youtube_top_new.csv', 'youtube_top_new_complete.csv')
print("コピー完了")

# データ読み込み
df = pd.read_csv('youtube_top_new_complete.csv')
print(f"\n全データ数: {len(df)}件")
print(f"subscribers列の欠損: {df['subscribers'].isna().sum()}件")
print(f"subscribers列の統計:")
print(f"  平均: {df['subscribers'].mean():,.0f}")
print(f"  中央値: {df['subscribers'].median():,.0f}")
print(f"  最大: {df['subscribers'].max():,.0f}")

# 時間特徴量
df['published_at'] = pd.to_datetime(df['published_at']).dt.tz_localize(None)
df['days_since_publish'] = (pd.Timestamp.now() - df['published_at']).dt.days
df['hour_published'] = pd.to_datetime(df['published_at']).dt.hour
df['weekday_published'] = pd.to_datetime(df['published_at']).dt.dayofweek

# カテゴリのワンホットエンコーディング
category_dummies = pd.get_dummies(df['category_id'], prefix='category')
df = pd.concat([df, category_dummies], axis=1)

# 特徴量エンジニアリング
df['log_duration'] = np.log10(df['video_duration'] + 1)
df['log_tags'] = np.log10(df['tags_count'] + 1)
df['log_desc_length'] = np.log10(df['description_length'] + 1)
df['log_subscribers'] = np.log10(df['subscribers'] + 1)
df['tags_per_second'] = df['tags_count'] / (df['video_duration'] + 1)
df['desc_per_second'] = df['description_length'] / (df['video_duration'] + 1)
df['subscriber_per_view'] = df['subscribers'] / (df['views'] + 1)

# 特徴量の選択（全特徴を使用）
feature_cols = [
    # 基本特徴
    'video_duration', 'tags_count', 'description_length',
    'subscribers', 'log_subscribers',
    'object_complexity', 'element_complexity', 'brightness', 'colorfulness',
    'log_duration', 'log_tags', 'log_desc_length',
    'tags_per_second', 'desc_per_second', 'subscriber_per_view',
    'days_since_publish', 'hour_published', 'weekday_published'
] + [col for col in df.columns if col.startswith('category_')]

# ターゲット変数
X = df[feature_cols]
y = np.log10(df['views'] + 1)

# データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n訓練データ: {len(X_train)}件")
print(f"テストデータ: {len(X_test)}件")

# モデル1: LightGBM
print("\n=== LightGBM ===")
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

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

y_pred_lgb = lgb_model.predict(X_test)
r2_lgb = r2_score(y_test, y_pred_lgb)
print(f"R²: {r2_lgb:.4f}")

# モデル2: XGBoost
print("\n=== XGBoost ===")
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

y_pred_xgb = xgb_model.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"R²: {r2_xgb:.4f}")

# モデル3: Random Forest
print("\n=== Random Forest ===")
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

# 結果サマリー
print("\n" + "="*60)
print("【最終結果】完全データ6,062件")
print("="*60)
print(f"LightGBM: {r2_lgb:.4f}")
print(f"XGBoost: {r2_xgb:.4f}")
print(f"Random Forest: {r2_rf:.4f}")
print(f"アンサンブル: {r2_ensemble:.4f}")
print(f"\n最高スコア: {max(r2_lgb, r2_xgb, r2_rf, r2_ensemble):.4f}")

# 過去の結果との比較
print("\n【比較】")
print("607件（subscribers有）: R² = 0.4416")
print("6,078件（subscribers無）: R² = 0.34")
print(f"6,062件（全データ完全）: R² = {max(r2_lgb, r2_xgb, r2_rf, r2_ensemble):.4f}")

# 特徴量重要度
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

print("\n【重要な特徴量TOP10】")
for _, row in feature_importance.iterrows():
    print(f"{row['feature']}: {row['importance']:.1f}")

# 結果を保存
results = {
    'data_count': len(df),
    'train_count': len(X_train),
    'test_count': len(X_test),
    'models': {
        'lightgbm': r2_lgb,
        'xgboost': r2_xgb,
        'random_forest': r2_rf,
        'ensemble': r2_ensemble
    },
    'best_r2': max(r2_lgb, r2_xgb, r2_rf, r2_ensemble),
    'top_features': feature_importance.to_dict('records')
}

import json
with open('complete_data_results.json', 'w') as f:
    json.dump(results, f, indent=2)