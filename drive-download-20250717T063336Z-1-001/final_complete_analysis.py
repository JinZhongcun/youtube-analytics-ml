#!/usr/bin/env python3
"""
最終完全分析：正しい理解に基づく分析
- subscribersは使用可能（viewsを含む計算はダメ）
- 全データセットで統一的に評価
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score
import lightgbm as lgb

print("="*80)
print("最終完全分析：正しい理解に基づく評価")
print("="*80)

# 統一されたモデルパラメータ
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
    'random_state': 42,
    'verbosity': -1
}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# 完全データ（subscribers込み）を使用
print("\n【メインデータセット分析】")
df = pd.read_csv('drive-download-20250717T063336Z-1-001/youtube_top_new.csv')
print(f"データ数: {len(df)}件")
print(f"subscribers欠損: {df['subscribers'].isna().sum()}件")

# 特徴量（subscribersを含む）
feature_cols = [
    'video_duration', 'tags_count', 'description_length',
    'brightness', 'colorfulness', 'object_complexity', 'element_complexity',
    'subscribers'
]

# 時間特徴量
df['published_at'] = pd.to_datetime(df['published_at'])
df['hour_published'] = df['published_at'].dt.hour
df['weekday_published'] = df['published_at'].dt.weekday
df['days_since_publish'] = (pd.Timestamp.now(tz='UTC') - df['published_at']).dt.days
feature_cols.extend(['hour_published', 'weekday_published', 'days_since_publish'])

# ログ変換
df['log_subscribers'] = np.log1p(df['subscribers'])
df['log_duration'] = np.log1p(df['video_duration'])
df['log_desc_length'] = np.log1p(df['description_length'])
feature_cols.extend(['log_subscribers', 'log_duration', 'log_desc_length'])

print(f"\n使用特徴量数: {len(feature_cols)}個")

# 1. Subscribersありの分析
print("\n【分析1: Subscribersあり（正しい使い方）】")
X_with_sub = df[feature_cols]
y = np.log1p(df['views'])

model_with_sub = lgb.LGBMRegressor(**lgb_params)
cv_scores_with = cross_val_score(model_with_sub, X_with_sub, y, cv=kfold, scoring='r2')

print(f"CV R²: {cv_scores_with.mean():.4f} ± {cv_scores_with.std():.4f}")
print(f"各fold: {[f'{s:.4f}' for s in cv_scores_with]}")

# Train/Test評価
X_train, X_test, y_train, y_test = train_test_split(
    X_with_sub, y, test_size=0.2, random_state=42
)
model_with_sub.fit(X_train, y_train)
test_r2_with = r2_score(y_test, model_with_sub.predict(X_test))
print(f"Test R²: {test_r2_with:.4f}")

# 2. Subscribersなしの分析（比較のため）
print("\n【分析2: Subscribersなし（ベースライン）】")
feature_cols_no_sub = [col for col in feature_cols if 'subscriber' not in col.lower()]
X_no_sub = df[feature_cols_no_sub]

model_no_sub = lgb.LGBMRegressor(**lgb_params)
cv_scores_no = cross_val_score(model_no_sub, X_no_sub, y, cv=kfold, scoring='r2')

print(f"CV R²: {cv_scores_no.mean():.4f} ± {cv_scores_no.std():.4f}")

# 3. 誤った使い方の例（警告として）
print("\n【警告: 誤った使い方の例】")
print("× subscriber_per_view = subscribers / views")
print("これはviewsを含むためデータリーケージ！")
print("○ subscribers単体は問題なし")

# 特徴量重要度
print("\n【特徴量重要度TOP10（正しいモデル）】")
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_with_sub.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in importance_df.head(10).iterrows():
    print(f"{i+1:2d}. {row['feature']:25s}: {row['importance']:.0f}")

# まとめ
print("\n" + "="*80)
print("【最終結論】")
print("="*80)
print(f"✓ Subscribersあり: R² = {cv_scores_with.mean():.4f}")
print(f"✓ Subscribersなし: R² = {cv_scores_no.mean():.4f}")
print(f"✓ 改善幅: +{cv_scores_with.mean() - cv_scores_no.mean():.4f} ({(cv_scores_with.mean() - cv_scores_no.mean())/cv_scores_no.mean()*100:.1f}%)")
print("\n重要：")
print("- subscribersは適切に使用可能")
print("- subscriber_per_viewなどviewsを含む計算は禁止")
print("- R² ≈ 0.45が現実的な性能")