#!/usr/bin/env python3
"""
データリーケージを防いだ分析
subscriber_per_viewなどの怪しい特徴量を除外
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
print("データリーケージを防いだ適切な分析")
print("="*60)

# データ読み込み
df = pd.read_csv('youtube_top_new_complete.csv')
print(f"データ数: {len(df)}件")

# 時間特徴量
df['published_at'] = pd.to_datetime(df['published_at']).dt.tz_localize(None)
df['days_since_publish'] = (pd.Timestamp.now() - df['published_at']).dt.days

# ログ変換（外れ値の影響を減らす）
df['log_subscribers'] = np.log10(df['subscribers'] + 1)
df['log_duration'] = np.log10(df['video_duration'] + 1)
df['log_tags'] = np.log10(df['tags_count'] + 1)

# カテゴリのダミー変数
category_dummies = pd.get_dummies(df['category_id'], prefix='category')
df = pd.concat([df, category_dummies], axis=1)

# 適切な特徴量のみ選択（subscriber_per_viewは使わない！）
feature_cols = [
    # 基本特徴
    'video_duration', 'tags_count', 'description_length',
    'log_subscribers', 'log_duration', 'log_tags',
    'object_complexity', 'element_complexity', 
    'brightness', 'colorfulness',
    'days_since_publish'
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

# シンプルなLightGBM（過学習を防ぐパラメータ）
print("\n=== LightGBM（適切なパラメータ）===")
lgb_model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,  # 浅くする
    num_leaves=31,
    min_child_samples=50,  # 増やす
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.1,  # L1正則化
    reg_lambda=0.1,  # L2正則化
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R²: {r2:.4f}")

# 特徴量重要度
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

print("\n【重要な特徴量TOP10】")
for _, row in feature_importance.iterrows():
    print(f"{row['feature']}: {row['importance']:.1f}")

# 予測値と実際値の差を確認
residuals = y_test - y_pred
print(f"\n【予測誤差の統計】")
print(f"平均誤差: {np.mean(residuals):.4f}")
print(f"誤差の標準偏差: {np.std(residuals):.4f}")
print(f"最大誤差: {np.max(np.abs(residuals)):.4f}")

# データの確認
print("\n【使用した特徴量の相関】")
important_features = feature_importance.head(5)['feature'].tolist()
if 'views' in df.columns:
    for feat in important_features[:3]:
        if feat in df.columns:
            corr = df[feat].corr(df['views'])
            print(f"{feat} と views の相関: {corr:.4f}")