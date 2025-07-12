#!/usr/bin/env python3
"""
YouTube動画データのSVM予測モデル
PCAで次元削減した特徴量を使用
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import warnings
warnings.filterwarnings('ignore')

# データの読み込み
print("データを読み込んでいます...")
df = pd.read_csv('youtube_top_jp.csv', skiprows=1)

# published_atの処理
df['published_at'] = pd.to_datetime(df['published_at']).dt.tz_localize(None)
df['days_since_publish'] = (pd.Timestamp.now() - df['published_at']).dt.days

# 特徴量の選択（likes, comment_countは除外）
numerical_features = [
    'video_duration', 'tags_count', 'description_length', 
    'subscribers', 'object_complexity', 'element_complexity', 
    'brightness', 'colorfulness', 'days_since_publish'
]

# 特徴量とターゲットの準備
X = df[numerical_features].copy()
y = np.log10(df['views'] + 1)  # 対数変換したviewsを予測

# 欠損値の処理
X = X.fillna(X.mean())

# データの分割（訓練用70%、テスト用30%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"訓練データ: {X_train.shape[0]}件")
print(f"テストデータ: {X_test.shape[0]}件")

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCAによる次元削減
pca = PCA(n_components=5)  # 累積寄与率80%を達成する主成分数
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\nPCA後の特徴量数: {X_train_pca.shape[1]}")
print(f"累積寄与率: {pca.explained_variance_ratio_.sum():.2%}")

# SVMモデルの構築
print("\n=== SVMモデルの構築 ===")

# グリッドサーチでハイパーパラメータを最適化
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01],
    'epsilon': [0.01, 0.1, 0.2]
}

svm = SVR(kernel='rbf')
grid_search = GridSearchCV(
    svm, param_grid, cv=5, 
    scoring='neg_mean_squared_error',
    n_jobs=-1, verbose=1
)

print("グリッドサーチを実行中...")
grid_search.fit(X_train_pca, y_train)

# 最良のモデル
best_svm = grid_search.best_estimator_
print(f"\n最適なパラメータ: {grid_search.best_params_}")

# 予測
y_train_pred = best_svm.predict(X_train_pca)
y_test_pred = best_svm.predict(X_test_pca)

# モデルの評価
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\n=== モデルの評価 ===")
print(f"訓練データ - MSE: {train_mse:.4f}, R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
print(f"テストデータ - MSE: {test_mse:.4f}, R²: {test_r2:.4f}, MAE: {test_mae:.4f}")

# 予測vs実測値のプロット
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.5, s=20)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Log10(Views + 1)')
plt.ylabel('Predicted Log10(Views + 1)')
plt.title(f'Training Data (R² = {train_r2:.3f})')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.5, s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Log10(Views + 1)')
plt.ylabel('Predicted Log10(Views + 1)')
plt.title(f'Test Data (R² = {test_r2:.3f})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('svm_predictions.png', dpi=300, bbox_inches='tight')
plt.close()

print("予測結果プロットを svm_predictions.png に保存しました。")

# 残差分析
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train_pred, residuals_train, alpha=0.5, s=20)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Log10(Views + 1)')
plt.ylabel('Residuals')
plt.title('Residual Plot - Training Data')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(residuals_test, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution - Test Data')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('svm_residuals.png', dpi=300, bbox_inches='tight')
plt.close()

print("残差分析プロットを svm_residuals.png に保存しました。")

# 特徴量の重要度（PCA寄与度を考慮）
feature_importance = np.abs(pca.components_).T @ pca.explained_variance_ratio_[:5]
feature_importance_df = pd.DataFrame({
    'feature': numerical_features,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
plt.xlabel('Importance (PCA-weighted)')
plt.title('Feature Importance in SVM Model')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance_svm.png', dpi=300, bbox_inches='tight')
plt.close()

print("特徴量重要度プロットを feature_importance_svm.png に保存しました。")

# カテゴリ別の予測精度
df_test = df.iloc[X_test.index].copy()
df_test['predicted_log_views'] = y_test_pred
df_test['actual_log_views'] = y_test
df_test['residual'] = residuals_test

category_performance = df_test.groupby('category_id').agg({
    'residual': ['mean', 'std', 'count']
}).round(3)

print("\n=== カテゴリ別の予測精度 ===")
print(category_performance.head(10))

# 結果の保存
results = {
    'model_params': grid_search.best_params_,
    'performance': {
        'train': {'mse': float(train_mse), 'r2': float(train_r2), 'mae': float(train_mae)},
        'test': {'mse': float(test_mse), 'r2': float(test_r2), 'mae': float(test_mae)}
    },
    'feature_importance': feature_importance_df.to_dict('records'),
    'pca_variance_ratio': pca.explained_variance_ratio_.tolist()
}

with open('svm_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nSVM分析結果を svm_results.json に保存しました。")

# 実用的な洞察
print("\n=== 実用的な洞察 ===")
print("\n再生回数を増やすための重要な要因（重要度順）:")
for i, row in feature_importance_df.head(5).iterrows():
    print(f"{i+1}. {row['feature']}: {row['importance']:.3f}")

# 高再生回数と低再生回数の特徴比較
high_views_mask = y > y.median()
low_views_mask = ~high_views_mask

comparison_df = pd.DataFrame({
    'High Views (Above Median)': X[high_views_mask].mean(),
    'Low Views (Below Median)': X[low_views_mask].mean()
})

print("\n高再生回数動画 vs 低再生回数動画の特徴比較:")
print(comparison_df.round(2))

# 最適な動画特性の提案
print("\n=== 推奨される動画特性 ===")
optimal_features = X[y > np.percentile(y, 75)].mean()
print("上位25%の高再生回数動画の平均的特徴:")
for feature in numerical_features[:5]:  # 上位5つの特徴を表示
    print(f"- {feature}: {optimal_features[feature]:.1f}")