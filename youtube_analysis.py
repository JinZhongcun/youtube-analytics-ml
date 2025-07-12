#!/usr/bin/env python3
"""
YouTube動画データのPCA分析とSVM予測モデル
課題: Assignment 2 - YouTube再生回数予測
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 日本語フォントの設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# データの読み込み
print("データを読み込んでいます...")
df = pd.read_csv('youtube_top_jp.csv', skiprows=1)

# データの基本情報を表示
print(f"\nデータセットの形状: {df.shape}")
print(f"\nカラム一覧:\n{df.columns.tolist()}")
print(f"\n基本統計量:")
print(df.describe())

# 欠損値の確認
print(f"\n欠損値の確認:")
print(df.isnull().sum())

# カテゴリ別の動画数を確認
print(f"\nカテゴリ別動画数:")
category_counts = df['category_id'].value_counts()
print(category_counts)

# 1. 探索的データ分析（EDA）
print("\n=== 探索的データ分析（EDA） ===")

# 数値特徴量の選択（likes, comment_countは除外）
numerical_features = [
    'video_duration', 'tags_count', 'description_length', 
    'subscribers', 'object_complexity', 'element_complexity', 
    'brightness', 'colorfulness'
]

# time_durationを日数に変換
df['published_at'] = pd.to_datetime(df['published_at'])
# タイムゾーンを削除して比較可能にする
df['published_at'] = df['published_at'].dt.tz_localize(None)
df['days_since_publish'] = (pd.Timestamp.now() - df['published_at']).dt.days

# 相関行列の計算と可視化
plt.figure(figsize=(12, 10))
correlation_matrix = df[numerical_features + ['views']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, fmt='.2f')
plt.title('Feature Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("相関行列を correlation_matrix.png に保存しました。")

# 再生回数の分布
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['views'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Views')
plt.ylabel('Frequency')
plt.title('Distribution of Views')
plt.ticklabel_format(style='plain', axis='x')

plt.subplot(1, 2, 2)
plt.hist(np.log10(df['views'] + 1), bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Log10(Views + 1)')
plt.ylabel('Frequency')
plt.title('Distribution of Log-transformed Views')

plt.tight_layout()
plt.savefig('views_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("再生回数の分布を views_distribution.png に保存しました。")

# カテゴリ別の平均再生回数
category_views = df.groupby('category_id')['views'].agg(['mean', 'median', 'count'])
category_views = category_views.sort_values('mean', ascending=False)

plt.figure(figsize=(12, 6))
x = range(len(category_views))
plt.bar(x, category_views['mean'], alpha=0.7, label='Mean')
plt.bar(x, category_views['median'], alpha=0.7, label='Median')
plt.xticks(x, category_views.index, rotation=45)
plt.xlabel('Category ID')
plt.ylabel('Views')
plt.title('Average Views by Category')
plt.legend()
plt.tight_layout()
plt.savefig('category_views.png', dpi=300, bbox_inches='tight')
plt.close()

print("カテゴリ別平均再生回数を category_views.png に保存しました。")

# 2. PCA用のデータ前処理
print("\n=== PCA用のデータ前処理 ===")

# PCA用の特徴量データを準備
X = df[numerical_features].copy()

# 欠損値の処理（平均値で補完）
X = X.fillna(X.mean())

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"標準化後のデータ形状: {X_scaled.shape}")

# 3. 主成分分析（PCA）の実施
print("\n=== 主成分分析（PCA） ===")

# PCAの実行
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 寄与率の計算
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# 寄与率の可視化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Component')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
plt.axhline(y=0.8, color='r', linestyle='--', label='80% variance')
plt.axhline(y=0.9, color='g', linestyle='--', label='90% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_variance.png', dpi=300, bbox_inches='tight')
plt.close()

print("PCA寄与率を pca_variance.png に保存しました。")

# 第1・第2主成分の因子負荷量
loadings = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])
loadings_df = pd.DataFrame(
    loadings,
    columns=['PC1', 'PC2'],
    index=numerical_features
)

plt.figure(figsize=(10, 8))
for i, feature in enumerate(numerical_features):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
              head_width=0.05, head_length=0.05, fc='blue', ec='blue')
    plt.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, feature, 
             ha='center', va='center', fontsize=10)

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel(f'PC1 ({explained_variance_ratio[0]:.2%})')
plt.ylabel(f'PC2 ({explained_variance_ratio[1]:.2%})')
plt.title('PCA Loadings Plot')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.savefig('pca_loadings.png', dpi=300, bbox_inches='tight')
plt.close()

print("PCA因子負荷量プロットを pca_loadings.png に保存しました。")

# 主成分得点の散布図（再生回数で色分け）
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=np.log10(df['views'] + 1), 
                     cmap='viridis', alpha=0.6, s=50)
plt.colorbar(scatter, label='Log10(Views + 1)')
plt.xlabel(f'PC1 ({explained_variance_ratio[0]:.2%})')
plt.ylabel(f'PC2 ({explained_variance_ratio[1]:.2%})')
plt.title('PCA Score Plot (colored by views)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_scores.png', dpi=300, bbox_inches='tight')
plt.close()

print("PCA得点プロットを pca_scores.png に保存しました。")

# 重要な主成分の特定
n_components_80 = np.argmax(cumulative_variance_ratio >= 0.8) + 1
n_components_90 = np.argmax(cumulative_variance_ratio >= 0.9) + 1

print(f"\n累積寄与率80%に必要な主成分数: {n_components_80}")
print(f"累積寄与率90%に必要な主成分数: {n_components_90}")

# 各主成分の解釈
print("\n各主成分の解釈（上位3つの寄与特徴量）:")
for i in range(min(3, len(pca.components_))):
    print(f"\n第{i+1}主成分 (寄与率: {explained_variance_ratio[i]:.2%}):")
    component_loadings = pd.Series(pca.components_[i], index=numerical_features)
    top_features = component_loadings.abs().nlargest(3)
    for feature, loading in component_loadings[top_features.index].items():
        print(f"  {feature}: {loading:.3f}")

# 分析結果の保存
results = {
    'explained_variance_ratio': explained_variance_ratio.tolist(),
    'cumulative_variance_ratio': cumulative_variance_ratio.tolist(),
    'n_components_80': int(n_components_80),
    'n_components_90': int(n_components_90),
    'loadings': loadings_df.to_dict(),
    'feature_importance': {}
}

# 特徴量の重要度（第1主成分の絶対値負荷量）
feature_importance = pd.Series(
    np.abs(pca.components_[0]), 
    index=numerical_features
).sort_values(ascending=False)

results['feature_importance'] = feature_importance.to_dict()

print("\n=== PCA分析完了 ===")
print(f"\n最も重要な特徴量（第1主成分の負荷量）:")
for feature, importance in feature_importance.head().items():
    print(f"  {feature}: {importance:.3f}")

# 結果をJSONファイルに保存
import json
with open('pca_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nPCA分析結果を pca_results.json に保存しました。")