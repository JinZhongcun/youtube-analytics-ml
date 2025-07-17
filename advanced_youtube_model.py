#!/usr/bin/env python3
"""
YouTube動画分析 - 改良版モデル
特徴量エンジニアリング、ディープラーニング、アンサンブル学習を含む
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

print("="*60)
print("YouTube動画分析 - 高精度モデルの構築")
print("="*60)

# データの読み込み
df = pd.read_csv('youtube_top_jp.csv', skiprows=1)
print(f"データ数: {len(df)}件")

# 時間関連の特徴量を作成
df['published_at'] = pd.to_datetime(df['published_at']).dt.tz_localize(None)
df['hour'] = df['published_at'].dt.hour
df['day_of_week'] = df['published_at'].dt.dayofweek
df['month'] = df['published_at'].dt.month
df['days_since_publish'] = (pd.Timestamp.now() - df['published_at']).dt.days

# 新しい特徴量エンジニアリング
print("\n=== 特徴量エンジニアリング ===")

# 1. 比率・対数変換特徴量
df['views_per_day'] = df['views'] / (df['days_since_publish'] + 1)
df['likes_per_view'] = df['likes'] / (df['views'] + 1)
df['comments_per_view'] = df['comment_count'] / (df['views'] + 1)
df['engagement_rate'] = (df['likes'] + df['comment_count']) / (df['views'] + 1)
df['tags_per_minute'] = df['tags_count'] / (df['video_duration'] / 60 + 1)
df['desc_per_minute'] = df['description_length'] / (df['video_duration'] / 60 + 1)
df['log_subscribers'] = np.log10(df['subscribers'] + 1)
df['log_duration'] = np.log10(df['video_duration'] + 1)

# 2. サムネイル関連の交互作用特徴量
df['visual_complexity'] = df['object_complexity'] * df['element_complexity']
df['color_brightness_interaction'] = df['colorfulness'] * df['brightness']
df['thumbnail_score'] = (df['brightness'] / 100) * (1 - df['colorfulness'] / 100) * (1 / (df['object_complexity'] + 1))

# 3. カテゴリベースの統計量
category_stats = df.groupby('category_id').agg({
    'views': ['mean', 'median', 'std'],
    'likes': ['mean', 'median'],
    'subscribers': ['mean', 'median']
}).reset_index()
category_stats.columns = ['category_id'] + [f'cat_{col[0]}_{col[1]}' for col in category_stats.columns[1:]]
df = df.merge(category_stats, on='category_id', how='left')

# 4. 相対的な特徴量（カテゴリ内での相対位置）
df['views_vs_category'] = df['views'] / (df['cat_views_mean'] + 1)
df['subs_vs_category'] = df['subscribers'] / (df['cat_subscribers_mean'] + 1)

# 5. 時間帯別の特徴量
time_features = pd.get_dummies(pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=['night', 'morning', 'afternoon', 'evening']), prefix='time')
df = pd.concat([df, time_features], axis=1)

# 6. 動画長カテゴリ
duration_categories = pd.get_dummies(pd.cut(df['video_duration'], bins=[0, 60, 180, 600, 10000], labels=['short', 'medium', 'long', 'very_long']), prefix='duration')
df = pd.concat([df, duration_categories], axis=1)

# 7. キーワード特徴量
keyword_dummies = pd.get_dummies(df['keyword'], prefix='keyword')
df = pd.concat([df, keyword_dummies], axis=1)

# 特徴量の選択
feature_columns = [
    # 基本特徴量
    'video_duration', 'tags_count', 'description_length', 'subscribers',
    'object_complexity', 'element_complexity', 'brightness', 'colorfulness',
    'days_since_publish', 'hour', 'day_of_week', 'month',
    
    # エンジニアリング特徴量
    'views_per_day', 'engagement_rate', 'tags_per_minute', 'desc_per_minute',
    'log_subscribers', 'log_duration', 'visual_complexity', 
    'color_brightness_interaction', 'thumbnail_score',
    'views_vs_category', 'subs_vs_category',
    
    # カテゴリ統計量
    'cat_views_mean', 'cat_views_median', 'cat_likes_mean', 
    'cat_subscribers_mean', 'cat_subscribers_median'
] + [col for col in df.columns if col.startswith(('time_', 'duration_', 'keyword_'))]

# NaNを除外
feature_columns = [col for col in feature_columns if col in df.columns]
X = df[feature_columns].fillna(0)
y = np.log10(df['views'] + 1)

print(f"特徴量数: {X.shape[1]}")

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# スケーリング
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"\n訓練: {X_train.shape[0]}件, 検証: {X_val.shape[0]}件, テスト: {X_test.shape[0]}件")

# 1. LightGBMの最適化
print("\n=== LightGBMの最適化 ===")
lgb_params = {
    'n_estimators': [300, 500, 700],
    'max_depth': [5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 100],
    'min_child_samples': [5, 10, 20],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

lgb_model = lgb.LGBMRegressor(random_state=42, verbosity=-1, n_jobs=-1)
lgb_grid = GridSearchCV(lgb_model, lgb_params, cv=5, scoring='r2', n_jobs=-1, verbose=1)
lgb_grid.fit(X_train_scaled, y_train)

print(f"LightGBM最適パラメータ: {lgb_grid.best_params_}")
print(f"LightGBM CV R²: {lgb_grid.best_score_:.4f}")

# 2. XGBoostの最適化
print("\n=== XGBoostの最適化 ===")
xgb_params = {
    'n_estimators': [300, 500, 700],
    'max_depth': [5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb_model = xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)
xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='r2', n_jobs=-1, verbose=1)
xgb_grid.fit(X_train_scaled, y_train)

print(f"XGBoost最適パラメータ: {xgb_grid.best_params_}")
print(f"XGBoost CV R²: {xgb_grid.best_score_:.4f}")

# 3. ディープニューラルネットワーク
print("\n=== ディープニューラルネットワーク ===")

def create_nn_model(input_dim):
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

nn_model = create_nn_model(X_train_scaled.shape[1])

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)

history = nn_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=0
)

nn_train_pred = nn_model.predict(X_train_scaled).flatten()
nn_val_pred = nn_model.predict(X_val_scaled).flatten()
nn_test_pred = nn_model.predict(X_test_scaled).flatten()

print(f"NN Train R²: {r2_score(y_train, nn_train_pred):.4f}")
print(f"NN Val R²: {r2_score(y_val, nn_val_pred):.4f}")

# 4. アンサンブルモデル
print("\n=== アンサンブルモデル ===")

# 基本モデル
base_models = [
    ('lgb', lgb_grid.best_estimator_),
    ('xgb', xgb_grid.best_estimator_),
    ('rf', RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=42))
]

# スタッキングアンサンブル
meta_model = Ridge(alpha=1.0)
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1
)

stacking_model.fit(X_train_scaled, y_train)

# 5. 最終的な加重平均アンサンブル
print("\n=== 最終アンサンブル（加重平均） ===")

# 各モデルの予測
lgb_pred = lgb_grid.best_estimator_.predict(X_test_scaled)
xgb_pred = xgb_grid.best_estimator_.predict(X_test_scaled)
nn_pred = nn_model.predict(X_test_scaled).flatten()
stack_pred = stacking_model.predict(X_test_scaled)

# 検証セットで最適な重みを探索
val_lgb = lgb_grid.best_estimator_.predict(X_val_scaled)
val_xgb = xgb_grid.best_estimator_.predict(X_val_scaled)
val_nn = nn_model.predict(X_val_scaled).flatten()
val_stack = stacking_model.predict(X_val_scaled)

best_r2 = 0
best_weights = None

for w1 in np.arange(0, 1.1, 0.1):
    for w2 in np.arange(0, 1.1 - w1, 0.1):
        for w3 in np.arange(0, 1.1 - w1 - w2, 0.1):
            w4 = 1 - w1 - w2 - w3
            if w4 >= 0:
                val_ensemble = w1 * val_lgb + w2 * val_xgb + w3 * val_nn + w4 * val_stack
                r2 = r2_score(y_val, val_ensemble)
                if r2 > best_r2:
                    best_r2 = r2
                    best_weights = [w1, w2, w3, w4]

print(f"最適な重み: LGB={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, NN={best_weights[2]:.2f}, Stack={best_weights[3]:.2f}")

# 最終予測
final_pred = (best_weights[0] * lgb_pred + 
              best_weights[1] * xgb_pred + 
              best_weights[2] * nn_pred + 
              best_weights[3] * stack_pred)

# 結果の評価
results = {
    'LightGBM': r2_score(y_test, lgb_pred),
    'XGBoost': r2_score(y_test, xgb_pred),
    'Neural Network': r2_score(y_test, nn_pred),
    'Stacking': r2_score(y_test, stack_pred),
    'Final Ensemble': r2_score(y_test, final_pred)
}

print("\n=== 最終結果（テストR²） ===")
for model, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model}: {score:.4f}")

# 特徴量重要度
feature_importance = lgb_grid.best_estimator_.feature_importances_
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n=== Top 10 重要特徴量 ===")
for idx, row in importance_df.head(10).iterrows():
    print(f"{row['feature']}: {row['importance']:.0f}")

# 結果の保存
final_results = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'feature_count': X.shape[1],
    'train_size': len(X_train),
    'test_size': len(X_test),
    'model_scores': results,
    'best_model': max(results.items(), key=lambda x: x[1])[0],
    'best_score': max(results.values()),
    'ensemble_weights': {
        'lightgbm': best_weights[0],
        'xgboost': best_weights[1],
        'neural_network': best_weights[2],
        'stacking': best_weights[3]
    },
    'top_features': importance_df.head(10).to_dict('records')
}

with open('advanced_model_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

# 予測vs実測値のプロット
plt.figure(figsize=(15, 10))

models_to_plot = {
    'LightGBM': lgb_pred,
    'XGBoost': xgb_pred,
    'Neural Network': nn_pred,
    'Final Ensemble': final_pred
}

for i, (name, pred) in enumerate(models_to_plot.items(), 1):
    plt.subplot(2, 2, i)
    plt.scatter(y_test, pred, alpha=0.5, s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Log10(Views + 1)')
    plt.ylabel('Predicted Log10(Views + 1)')
    plt.title(f'{name} (R² = {r2_score(y_test, pred):.4f})')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('advanced_model_predictions.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ 分析完了！")
print(f"最終R²スコア: {results['Final Ensemble']:.4f}")
print("結果は advanced_model_results.json に保存されました。")