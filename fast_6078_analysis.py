#!/usr/bin/env python3
"""
6,078件全データを高速処理する簡易版
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("6,078件全データ 高速版")
print("="*60)

# データ読み込み
print("データ読み込み中...")
df_old = pd.read_csv('youtube_top_jp.csv', skiprows=1)
df_new = pd.read_csv('youtube_top_new.csv')

# subscribersデータをマージ
df_merged = pd.merge(df_new, df_old[['video_id', 'subscribers']], 
                     on='video_id', how='left')

print(f"全データ: {len(df_merged)}件")
print(f"subscribers有: {df_merged['subscribers'].notna().sum()}件")
print(f"subscribers無: {df_merged['subscribers'].isna().sum()}件")

# ターゲット変数
y = np.log10(df_merged['views'] + 1)

# 特徴量（既存の数値特徴とサムネイル特徴のみ使用）
feature_cols = [
    'video_duration', 'tags_count', 'description_length',
    'object_complexity', 'element_complexity', 'brightness', 'colorfulness'
]

# subscribersがある場合は追加
df_merged['has_subscribers'] = (~df_merged['subscribers'].isna()).astype(int)
df_merged['log_subscribers'] = np.log10(df_merged['subscribers'].fillna(1) + 1)
feature_cols.extend(['has_subscribers', 'log_subscribers'])

X = df_merged[feature_cols].fillna(0)

# データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n訓練データ: {len(X_train)}件")
print(f"テストデータ: {len(X_test)}件")

# LightGBMモデル（シンプル版）
print("\n=== LightGBM モデル訓練中 ===")
model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=8,
    num_leaves=31,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

model.fit(X_train, y_train)

# 予測と評価
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*60)
print("結果")
print("="*60)
print(f"R² スコア: {r2:.4f}")
print(f"前回の607件モデル: R² = 0.4416")
print(f"改善率: {(r2 / 0.4416 - 1) * 100:+.1f}%")

# 特徴量の重要度
print("\n重要な特徴量:")
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

# subscribersの有無での性能差
print("\n【subscribersデータの影響】")
test_has_subs = X_test['has_subscribers'] == 1
if test_has_subs.sum() > 0:
    r2_with_subs = r2_score(y_test[test_has_subs], y_pred[test_has_subs])
    print(f"subscribers有のデータ: R² = {r2_with_subs:.4f} (n={test_has_subs.sum()})")
    
test_no_subs = X_test['has_subscribers'] == 0
if test_no_subs.sum() > 0:
    r2_without_subs = r2_score(y_test[test_no_subs], y_pred[test_no_subs])
    print(f"subscribers無のデータ: R² = {r2_without_subs:.4f} (n={test_no_subs.sum()})")