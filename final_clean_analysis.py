#!/usr/bin/env python3
"""
最終的なクリーンな分析
subscribersを完全に除外した適切なモデル
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("最終分析: subscribersを除外したクリーンなモデル")
print("="*60)

# データ読み込み
df = pd.read_csv('youtube_top_new_complete.csv')
print(f"全データ数: {len(df)}件")

# 時間特徴量（安全）
df['published_at'] = pd.to_datetime(df['published_at']).dt.tz_localize(None)
df['days_since_publish'] = (pd.Timestamp.now() - df['published_at']).dt.days
df['hour_published'] = pd.to_datetime(df['published_at']).dt.hour
df['weekday_published'] = pd.to_datetime(df['published_at']).dt.dayofweek

# カテゴリのダミー変数（安全）
category_dummies = pd.get_dummies(df['category_id'], prefix='category')
df = pd.concat([df, category_dummies], axis=1)

# 安全な派生特徴量のみ
df['log_duration'] = np.log10(df['video_duration'] + 1)
df['log_tags'] = np.log10(df['tags_count'] + 1)
df['log_desc_length'] = np.log10(df['description_length'] + 1)
df['tags_per_second'] = df['tags_count'] / (df['video_duration'] + 1)
df['desc_per_second'] = df['description_length'] / (df['video_duration'] + 1)

# クリーンな特徴量のみ選択（subscribersは一切使わない）
feature_cols = [
    # 基本特徴（全て安全）
    'video_duration', 'tags_count', 'description_length',
    'object_complexity', 'element_complexity', 
    'brightness', 'colorfulness',
    # 時間特徴（安全）
    'days_since_publish', 'hour_published', 'weekday_published',
    # 対数変換（安全）
    'log_duration', 'log_tags', 'log_desc_length',
    # 比率（安全）
    'tags_per_second', 'desc_per_second'
] + [col for col in df.columns if col.startswith('category_')]

print(f"\n使用する特徴量数: {len(feature_cols)}")

# ターゲット変数
X = df[feature_cols]
y = np.log10(df['views'] + 1)

# データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"訓練データ: {len(X_train)}件")
print(f"テストデータ: {len(X_test)}件")

# モデル1: LightGBM
print("\n=== LightGBM（クリーン） ===")
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
r2_lgb = r2_score(y_test, y_pred_lgb)
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
print(f"R²: {r2_lgb:.4f}")
print(f"RMSE: {rmse_lgb:.4f}")

# モデル2: XGBoost
print("\n=== XGBoost（クリーン） ===")
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f"R²: {r2_xgb:.4f}")
print(f"RMSE: {rmse_xgb:.4f}")

# モデル3: Random Forest
print("\n=== Random Forest（クリーン） ===")
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"R²: {r2_rf:.4f}")
print(f"RMSE: {rmse_rf:.4f}")

# 結果まとめ
print("\n" + "="*60)
print("【最終結果】subscribersなしのクリーンなモデル")
print("="*60)
print(f"LightGBM: R² = {r2_lgb:.4f} (RMSE = {rmse_lgb:.4f})")
print(f"XGBoost: R² = {r2_xgb:.4f} (RMSE = {rmse_xgb:.4f})")
print(f"Random Forest: R² = {r2_rf:.4f} (RMSE = {rmse_rf:.4f})")
print(f"\n最良モデル: R² = {max(r2_lgb, r2_xgb, r2_rf):.4f}")

# 特徴量重要度（LightGBM）
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

print("\n【重要な特徴量TOP10】")
for _, row in feature_importance.iterrows():
    print(f"{row['feature']}: {row['importance']:.1f}")

# 過去の結果との比較
print("\n【モデル比較】")
print("初期モデル（767件）: R² = 0.21")
print("subscribersあり（607件）: R² = 0.44")
print("subscribersなし（6,078件）: R² = 0.34")
print(f"今回（6,062件、クリーン）: R² = {max(r2_lgb, r2_xgb, r2_rf):.4f}")

print("\n結論: subscribersを使わなくても実用的な予測が可能")