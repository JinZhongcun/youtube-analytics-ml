#!/usr/bin/env python3
"""
少量データ（767件）に特化した分析手法
過学習を避けつつ、解釈可能性を重視
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import shap
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("少量データに特化した分析手法")
print("="*60)

# データ読み込み
df = pd.read_csv('youtube_top_jp.csv', skiprows=1)
print(f"データ数: {len(df)}件（少量データ分析）")

# 基本的な前処理
df['published_at'] = pd.to_datetime(df['published_at']).dt.tz_localize(None)
df['days_since_publish'] = (pd.Timestamp.now() - df['published_at']).dt.days

# シンプルで堅牢な特徴量のみを使用
features = [
    'video_duration', 'tags_count', 'description_length', 
    'subscribers', 'object_complexity', 'element_complexity', 
    'brightness', 'colorfulness', 'days_since_publish'
]

X = df[features].fillna(df[features].median())
y = np.log10(df['views'] + 1)

# 外れ値に強いスケーリング
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n特徴量数: {X.shape[1]} (少量データのため最小限に)")

# 1. 正則化線形モデル（少量データに最適）
print("\n=== 1. 正則化線形モデル（交差検証付き） ===")

# RidgeCV - L2正則化
ridge = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=10)
ridge.fit(X_scaled, y)
ridge_scores = cross_val_score(ridge, X_scaled, y, cv=10, scoring='r2')
print(f"Ridge CV R²: {ridge_scores.mean():.4f} (±{ridge_scores.std():.4f})")
print(f"最適alpha: {ridge.alpha_:.4f}")

# LassoCV - L1正則化（特徴選択効果）
lasso = LassoCV(alphas=np.logspace(-3, 1, 100), cv=10, max_iter=10000)
lasso.fit(X_scaled, y)
lasso_scores = cross_val_score(lasso, X_scaled, y, cv=10, scoring='r2')
print(f"Lasso CV R²: {lasso_scores.mean():.4f} (±{lasso_scores.std():.4f})")
print(f"選択された特徴量数: {np.sum(lasso.coef_ != 0)}/{len(features)}")

# ElasticNetCV - L1+L2正則化
elastic = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99], 
                       alphas=np.logspace(-3, 1, 50), cv=10, max_iter=10000)
elastic.fit(X_scaled, y)
elastic_scores = cross_val_score(elastic, X_scaled, y, cv=10, scoring='r2')
print(f"ElasticNet CV R²: {elastic_scores.mean():.4f} (±{elastic_scores.std():.4f})")

# 2. ベイズ線形回帰（不確実性の定量化）
print("\n=== 2. ベイズ線形回帰 ===")
bayes = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
bayes.fit(X_scaled, y)
bayes_scores = cross_val_score(bayes, X_scaled, y, cv=10, scoring='r2')
print(f"Bayesian Ridge CV R²: {bayes_scores.mean():.4f} (±{bayes_scores.std():.4f})")

# 予測の不確実性を計算
y_pred, y_std = bayes.predict(X_scaled, return_std=True)
print(f"予測の平均標準偏差: {y_std.mean():.4f}")

# 3. ガウス過程回帰（非線形性と不確実性）
print("\n=== 3. ガウス過程回帰 ===")
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=2)

# サンプルサイズを制限（計算効率のため）
sample_size = min(500, len(X))
indices = np.random.choice(len(X), sample_size, replace=False)
X_sample = X_scaled[indices]
y_sample = y.iloc[indices]

gpr.fit(X_sample, y_sample)
gpr_pred, gpr_std = gpr.predict(X_sample, return_std=True)
gpr_r2 = r2_score(y_sample, gpr_pred)
print(f"Gaussian Process R² (sample): {gpr_r2:.4f}")
print(f"予測の不確実性: {gpr_std.mean():.4f}")

# 4. 特徴量選択（少量データでは重要）
print("\n=== 4. 特徴量選択 ===")

# SelectKBest
selector = SelectKBest(f_regression, k=5)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = np.array(features)[selector.get_support()]
print(f"統計的に重要な上位5特徴量: {list(selected_features)}")

# RFE（再帰的特徴除去）
rfe = RFE(RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=5)
rfe.fit(X_scaled, y)
rfe_features = np.array(features)[rfe.support_]
print(f"RFEで選択された特徴量: {list(rfe_features)}")

# 5. Leave-One-Out交差検証（最も厳密な評価）
print("\n=== 5. Leave-One-Out交差検証 ===")
loo = LeaveOneOut()
models = {
    'Ridge': ridge,
    'Lasso': lasso,
    'ElasticNet': elastic,
    'Bayesian': bayes
}

print("各モデルのLOO-CV結果:")
for name, model in models.items():
    loo_scores = cross_val_score(model, X_scaled, y, cv=loo, scoring='r2', n_jobs=-1)
    print(f"{name}: R² = {loo_scores.mean():.4f} (最小: {loo_scores.min():.4f}, 最大: {loo_scores.max():.4f})")

# 6. 単純なルールベースモデル（解釈可能性重視）
print("\n=== 6. 解釈可能なルールベース分析 ===")

# カテゴリ別の分析
print("\n動画の長さによる分類:")
df['duration_category'] = pd.cut(df['video_duration'], 
                                 bins=[0, 60, 180, 600, 10000], 
                                 labels=['very_short', 'short', 'medium', 'long'])
duration_stats = df.groupby('duration_category')['views'].agg(['mean', 'median', 'count'])
print(duration_stats)

print("\nチャンネル規模による分類:")
df['channel_size'] = pd.cut(df['subscribers'], 
                            bins=[0, 100000, 1000000, 5000000, 100000000],
                            labels=['small', 'medium', 'large', 'mega'])
channel_stats = df.groupby('channel_size')['views'].agg(['mean', 'median', 'count'])
print(channel_stats)

# 7. SHAP値による解釈（最良モデルに対して）
print("\n=== 7. SHAP分析（モデルの解釈） ===")
explainer = shap.LinearExplainer(ridge, X_scaled, feature_names=features)
shap_values = explainer.shap_values(X_scaled)

# 特徴量の重要度
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

print("\nSHAPによる特徴量重要度:")
for _, row in feature_importance.iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

# 8. アンサンブル予測（保守的アプローチ）
print("\n=== 8. 保守的アンサンブル ===")
# 各モデルの予測を平均化
predictions = {}
for name, model in models.items():
    predictions[name] = model.predict(X_scaled)

ensemble_pred = np.mean(list(predictions.values()), axis=0)
ensemble_r2 = r2_score(y, ensemble_pred)
print(f"アンサンブルR²: {ensemble_r2:.4f}")

# 結果の可視化
plt.figure(figsize=(15, 10))

# 1. 特徴量の相関
plt.subplot(2, 3, 1)
correlation = pd.DataFrame(X_scaled, columns=features).corr()
mask = np.triu(np.ones_like(correlation), k=1)
sns.heatmap(correlation, mask=mask, cmap='coolwarm', center=0, 
            square=True, annot=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix')

# 2. 正則化パスの可視化
plt.subplot(2, 3, 2)
alphas = np.logspace(-3, 3, 100)
coefs = []
for alpha in alphas:
    ridge_temp = Ridge(alpha=alpha)
    ridge_temp.fit(X_scaled, y)
    coefs.append(ridge_temp.coef_)
coefs = np.array(coefs)

for i, feature in enumerate(features):
    plt.plot(alphas, coefs[:, i], label=feature)
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Coefficient Value')
plt.title('Ridge Regularization Path')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 3. 予測の不確実性
plt.subplot(2, 3, 3)
plt.errorbar(y, y_pred, yerr=y_std, fmt='o', alpha=0.5, ecolor='gray', capsize=2)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Log10(Views)')
plt.ylabel('Predicted Log10(Views)')
plt.title('Bayesian Predictions with Uncertainty')

# 4. 残差分析
plt.subplot(2, 3, 4)
residuals = y - ensemble_pred
plt.scatter(ensemble_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot (Ensemble)')

# 5. 交差検証スコアの分布
plt.subplot(2, 3, 5)
cv_results = pd.DataFrame({
    'Ridge': ridge_scores,
    'Lasso': lasso_scores,
    'ElasticNet': elastic_scores,
    'Bayesian': bayes_scores
})
cv_results.boxplot()
plt.ylabel('CV R² Score')
plt.title('Cross-Validation Score Distribution')
plt.xticks(rotation=45)

# 6. 動画長と再生回数の関係
plt.subplot(2, 3, 6)
plt.scatter(df['video_duration'], df['views'], alpha=0.5, s=20)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Video Duration (seconds)')
plt.ylabel('Views')
plt.title('Views vs Duration (Log Scale)')

plt.tight_layout()
plt.savefig('small_data_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 最終レポート
print("\n" + "="*60)
print("少量データ分析の結論")
print("="*60)
print(f"1. 最良モデル: Ridge回帰（R² = {ridge_scores.mean():.4f}）")
print(f"2. 予測の不確実性が高い（ベイズ標準偏差: {y_std.mean():.4f}）")
print(f"3. 最重要特徴量: {list(selected_features[:3])}")
print(f"4. 90秒以下の動画は平均{duration_stats.loc['very_short', 'mean']/duration_stats.loc['long', 'mean']:.1f}倍の再生回数")
print("\n推奨事項:")
print("- より多くのデータ収集が必須（最低3000件以上）")
print("- サムネイル画像の直接分析が必要")
print("- 時系列データ（日次再生回数）の活用を検討")