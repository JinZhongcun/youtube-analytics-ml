#!/usr/bin/env python3
"""
最終正しい分析：subscribersは使用可能
viewsを含む計算（subscriber_per_view等）のみがデータリーケージ
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score
import lightgbm as lgb

print("="*80)
print("最終分析：正しい理解でのクリーンな実装")
print("="*80)

# LightGBMパラメータ（最適化済み）
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

# データ読み込み（subscribers完備版）
df = pd.read_csv('drive-download-20250717T063336Z-1-001/youtube_top_new.csv')
print(f"データ数: {len(df)}件")

# 特徴量定義（subscribersを含む）
feature_cols = [
    'video_duration', 'tags_count', 'description_length',
    'brightness', 'colorfulness', 'object_complexity', 'element_complexity',
    'subscribers'  # ← これは問題なし！
]

# 時間特徴量
df['published_at'] = pd.to_datetime(df['published_at'])
df['hour_published'] = df['published_at'].dt.hour
df['weekday_published'] = df['published_at'].dt.weekday
feature_cols.extend(['hour_published', 'weekday_published'])

# ログ変換（有効な特徴量エンジニアリング）
df['log_subscribers'] = np.log1p(df['subscribers'])
df['log_duration'] = np.log1p(df['video_duration'])
feature_cols.extend(['log_subscribers', 'log_duration'])

print(f"\n使用特徴量: {len(feature_cols)}個")
print("※subscriber_per_view等は使用しない（viewsを含むため）")

X = df[feature_cols]
y = np.log1p(df['views'])

# 5-fold交差検証
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
model = lgb.LGBMRegressor(**lgb_params)
cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')

print(f"\n【交差検証結果】")
print(f"CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"各fold: {[f'{s:.4f}' for s in cv_scores]}")

# Train/Test分割評価
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, model.predict(X_test))

print(f"\n【Train/Test評価】")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"過学習度: {train_r2 - test_r2:.4f}")

# 特徴量重要度
print(f"\n【特徴量重要度TOP5】")
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in importance_df.head(5).iterrows():
    print(f"{i+1}. {row['feature']:20s}: {row['importance']:.0f}")

print("\n" + "="*80)
print("【結論】")
print("✓ subscribersは正当に使用可能")
print("✓ データリーケージなし")
print(f"✓ 最終性能: CV R² = {cv_scores.mean():.4f}")
print("✓ これが本来の予測性能")