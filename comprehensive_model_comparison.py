#!/usr/bin/env python3
"""
YouTube動画データの包括的なモデル比較
複数の次元削減手法と予測モデルを比較
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
import json
import warnings
warnings.filterwarnings('ignore')

# 結果を保存する辞書
results_summary = {}

# データの読み込み
print("="*60)
print("データを読み込んでいます...")
print("="*60)
df = pd.read_csv('youtube_top_jp.csv', skiprows=1)

print(f"\n総データ数: {len(df)}件")
print(f"カラム数: {len(df.columns)}列")

# 使用するカラムの明示的な確認
print("\n使用可能なカラム:")
for i, col in enumerate(df.columns):
    print(f"{i+1:2d}. {col}")

# published_atの処理
df['published_at'] = pd.to_datetime(df['published_at']).dt.tz_localize(None)
df['days_since_publish'] = (pd.Timestamp.now() - df['published_at']).dt.days

# 使用する特徴量（likes, comment_countは意図的に除外）
print("\n【重要】使用する特徴量（likes, comment_countは除外）:")
numerical_features = [
    'video_duration',      # 動画の長さ
    'tags_count',         # タグ数
    'description_length', # 概要欄の長さ
    'subscribers',        # チャンネル登録者数
    'object_complexity',  # オブジェクトの複雑さ
    'element_complexity', # 要素の複雑さ
    'brightness',         # 明るさ
    'colorfulness',       # 色彩度
    'days_since_publish'  # 投稿からの日数
]

for i, feat in enumerate(numerical_features):
    print(f"  {i+1}. {feat}")

print("\n【重要】除外したカラム（データリークの可能性）:")
excluded_columns = ['likes', 'comment_count', 'video_id', 'title', 'thumbnail_link', 
                   'keyword', 'thumbnail_path', 'time_duration', 'published_at']
for col in excluded_columns:
    if col in df.columns:
        print(f"  - {col}")

# 特徴量とターゲットの準備
X = df[numerical_features].copy()
y = np.log10(df['views'] + 1)  # 対数変換

# 欠損値の処理
X = X.fillna(X.mean())

# データの分割（70:30）
print(f"\nデータを訓練用70%、テスト用30%に分割...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)

print(f"訓練データ: {X_train.shape[0]}件")
print(f"テストデータ: {X_test.shape[0]}件")

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# クロスバリデーション設定
cv = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "="*60)
print("次元削減手法の比較")
print("="*60)

# 1. PCA
print("\n1. PCA (Principal Component Analysis)")
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"   - 累積寄与率: {pca.explained_variance_ratio_.sum():.2%}")
print(f"   - 各主成分の寄与率: {pca.explained_variance_ratio_}")

# 2. t-SNE
print("\n2. t-SNE (t-distributed Stochastic Neighbor Embedding)")
print("   - 計算時間の関係で2次元に削減")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_train_tsne = tsne.fit_transform(X_train_scaled[:500])  # サンプルサイズを制限
print(f"   - t-SNE完了（訓練データの一部を使用）")

print("\n" + "="*60)
print("予測モデルの比較")
print("="*60)

# モデルのリスト
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVM (RBF)': SVR(kernel='rbf', C=1, gamma='scale'),
    'LightGBM': lgb.LGBMRegressor(random_state=42, verbosity=-1),
    'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0)
}

# 各モデルの評価
model_results = {}

for name, model in models.items():
    print(f"\n{name}を評価中...")
    
    # クロスバリデーション（元の特徴量）
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, 
                               scoring='r2', n_jobs=-1)
    
    # モデルの訓練
    model.fit(X_train_scaled, y_train)
    
    # 予測
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # 評価指標
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    model_results[name] = {
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae
    }
    
    print(f"  - CV R² (平均±標準偏差): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  - 訓練 R²: {train_r2:.4f}, テスト R²: {test_r2:.4f}")
    print(f"  - 訓練 MSE: {train_mse:.4f}, テスト MSE: {test_mse:.4f}")

# PCA + 各モデルの評価
print("\n" + "="*60)
print("PCA + 予測モデルの組み合わせ")
print("="*60)

pca_model_results = {}

for name, model in models.items():
    print(f"\nPCA + {name}を評価中...")
    
    # クロスバリデーション
    cv_scores = cross_val_score(model, X_train_pca, y_train, cv=cv, 
                               scoring='r2', n_jobs=-1)
    
    # モデルの訓練
    model.fit(X_train_pca, y_train)
    
    # 予測
    y_train_pred = model.predict(X_train_pca)
    y_test_pred = model.predict(X_test_pca)
    
    # 評価指標
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    pca_model_results[f"PCA + {name}"] = {
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'test_r2': test_r2,
        'test_mse': test_mse
    }
    
    print(f"  - CV R² (平均±標準偏差): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  - テスト R²: {test_r2:.4f}")

# 結果のランキング
print("\n" + "="*60)
print("モデル性能ランキング（テストR²スコア）")
print("="*60)

all_results = {**model_results, **pca_model_results}
sorted_models = sorted(all_results.items(), 
                      key=lambda x: x[1]['test_r2'], 
                      reverse=True)

print("\n順位 | モデル名 | CV R² (平均±SD) | テスト R²")
print("-" * 60)
for i, (name, metrics) in enumerate(sorted_models):
    cv_score = f"{metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}"
    print(f"{i+1:2d}   | {name:25s} | {cv_score:15s} | {metrics['test_r2']:.4f}")

# 最良モデルの詳細分析
best_model_name = sorted_models[0][0]
print(f"\n最良モデル: {best_model_name}")
print(f"テストR²スコア: {sorted_models[0][1]['test_r2']:.4f}")

# 結果の可視化
plt.figure(figsize=(15, 10))

# モデル比較のバープロット
plt.subplot(2, 2, 1)
model_names = [name for name, _ in sorted_models[:10]]
test_r2_scores = [metrics['test_r2'] for _, metrics in sorted_models[:10]]
colors = ['green' if score > 0.15 else 'orange' if score > 0.1 else 'red' 
          for score in test_r2_scores]

bars = plt.barh(model_names, test_r2_scores, color=colors)
plt.xlabel('Test R² Score')
plt.title('Model Performance Comparison (Top 10)')
plt.xlim(0, max(test_r2_scores) * 1.1)

# 値をバーの上に表示
for i, (bar, score) in enumerate(zip(bars, test_r2_scores)):
    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{score:.4f}', va='center')

# CVスコアの箱ひげ図
plt.subplot(2, 2, 2)
cv_data = []
cv_labels = []
for name in ['Random Forest', 'LightGBM', 'XGBoost', 'SVM (RBF)', 'Linear Regression']:
    if name in models:
        model = models[name]
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='r2')
        cv_data.append(scores)
        cv_labels.append(name)

plt.boxplot(cv_data, labels=cv_labels)
plt.ylabel('R² Score')
plt.title('Cross-Validation Score Distribution')
plt.xticks(rotation=45)

# 特徴量の重要度（LightGBM）
plt.subplot(2, 2, 3)
lgb_model = lgb.LGBMRegressor(random_state=42, verbosity=-1)
lgb_model.fit(X_train_scaled, y_train)
feature_importance = lgb_model.feature_importances_
indices = np.argsort(feature_importance)[::-1]

plt.barh(np.array(numerical_features)[indices], feature_importance[indices])
plt.xlabel('Feature Importance')
plt.title('Feature Importance (LightGBM)')

# 学習曲線
plt.subplot(2, 2, 4)
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes_abs = (train_sizes * len(X_train)).astype(int)
train_scores = []
val_scores = []

best_model_type = models['LightGBM']  # 通常LightGBMが最良
for train_size in train_sizes_abs:
    X_subset = X_train_scaled[:train_size]
    y_subset = y_train[:train_size]
    
    cv_scores = cross_val_score(best_model_type, X_subset, y_subset, 
                               cv=3, scoring='r2')
    train_scores.append(cv_scores.mean())
    
    # 検証スコアの推定
    best_model_type.fit(X_subset, y_subset)
    val_score = best_model_type.score(X_test_scaled, y_test)
    val_scores.append(val_score)

plt.plot(train_sizes_abs, train_scores, 'o-', label='Training score')
plt.plot(train_sizes_abs, val_scores, 'o-', label='Validation score')
plt.xlabel('Training Set Size')
plt.ylabel('R² Score')
plt.title('Learning Curves (LightGBM)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 詳細な結果をJSONに保存
final_results = {
    'data_info': {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features_used': numerical_features,
        'excluded_features': excluded_columns
    },
    'model_results': all_results,
    'best_model': {
        'name': best_model_name,
        'test_r2': sorted_models[0][1]['test_r2'],
        'cv_r2_mean': sorted_models[0][1]['cv_r2_mean'],
        'cv_r2_std': sorted_models[0][1]['cv_r2_std']
    },
    'pca_info': {
        'n_components': 5,
        'cumulative_variance_ratio': float(pca.explained_variance_ratio_.sum()),
        'variance_ratio': pca.explained_variance_ratio_.tolist()
    }
}

with open('comprehensive_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print("\n" + "="*60)
print("分析完了")
print("="*60)
print(f"\n結果ファイル:")
print("  - comprehensive_model_comparison.png: モデル比較の可視化")
print("  - comprehensive_results.json: 詳細な数値結果")
print(f"\n最終結論:")
print(f"  - 最良モデル: {best_model_name}")
print(f"  - テストデータR²スコア: {sorted_models[0][1]['test_r2']:.4f}")
print(f"  - 使用した特徴量数: {len(numerical_features)}")
print(f"  - データリークを避けるため、likes/comment_countは除外")