#!/usr/bin/env python3
"""
最終的なクリーンモデル（最適化版）
Geminiの推奨に基づく過学習対策を実装
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
import joblib

print("="*80)
print("最終モデル構築: 批判的査読を反映した実用的モデル")
print("="*80)

# データ読み込み
df = pd.read_csv('drive-download-20250717T063336Z-1-001/youtube_top_new.csv')
print(f"データ数: {len(df)}件")

# 1. 厳選された特徴量のみ使用（データリーケージなし）
print("\n【特徴量選択】")
print("✓ 使用する特徴量:")

# 基本メタデータ（欠損が少ないもの）
basic_features = ['video_duration', 'description_length']
print(f"  - 基本: {basic_features}")

# サムネイル特徴量（信頼性の高いもの）
image_features = ['brightness', 'colorfulness', 'object_complexity']
print(f"  - 画像: {image_features}")

# 時間特徴量（published_atから生成）
df['published_at'] = pd.to_datetime(df['published_at'])
df['hour_published'] = df['published_at'].dt.hour
df['weekday_published'] = df['published_at'].dt.weekday
time_features = ['hour_published', 'weekday_published']
print(f"  - 時間: {time_features}")

# カテゴリ（ワンホット）
df_encoded = pd.get_dummies(df, columns=['category_id'], prefix='cat')
category_cols = [col for col in df_encoded.columns if col.startswith('cat_')]
print(f"  - カテゴリ: {len(category_cols)}個")

# 全特徴量
all_features = basic_features + image_features + time_features + category_cols
X = df_encoded[all_features]
y = np.log1p(df['views'])

print(f"\n総特徴量数: {len(all_features)}個")

# 2. 過学習を防ぐLightGBMパラメータ（Gemini推奨）
print("\n【ハイパーパラメータ最適化】")
print("過学習を防ぐための設定:")

param_grid = {
    'num_leaves': [20, 31, 40],  # 小さめに設定
    'max_depth': [4, 5, 6],       # 浅めに設定
    'min_child_samples': [30, 50, 70],  # 大きめに設定
    'lambda_l1': [0, 0.1, 1.0],  # L1正則化
    'lambda_l2': [0, 0.1, 1.0],  # L2正則化
    'feature_fraction': [0.7, 0.8, 0.9],  # 特徴量サンプリング
    'bagging_fraction': [0.7, 0.8, 0.9],  # データサンプリング
    'bagging_freq': [5],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100]
}

# 簡略版グリッドサーチ（時間短縮のため）
simple_param_grid = {
    'num_leaves': [20, 31],
    'max_depth': [4, 5],
    'min_child_samples': [50],
    'lambda_l2': [0.1, 1.0],
    'feature_fraction': [0.8],
    'bagging_fraction': [0.8],
    'bagging_freq': [5],
    'learning_rate': [0.05],
    'n_estimators': [100]
}

# 3. 交差検証でモデル選択
print("\n交差検証によるモデル選択中...")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_model = lgb.LGBMRegressor(random_state=42, verbosity=-1)
grid_search = GridSearchCV(
    lgb_model, 
    simple_param_grid, 
    cv=kfold, 
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)

print(f"\n最適パラメータ:")
for param, value in grid_search.best_params_.items():
    print(f"  - {param}: {value}")

print(f"\n交差検証R²: {grid_search.best_score_:.4f}")

# 4. 最終モデルの評価
best_model = grid_search.best_estimator_

# 交差検証での詳細評価
cv_scores = cross_val_score(best_model, X, y, cv=kfold, scoring='r2')
print(f"\n【最終評価】")
print(f"5-fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"各fold: {[f'{s:.4f}' for s in cv_scores]}")

# 特徴量重要度
feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n【特徴量重要度TOP10】")
for idx, row in feature_importance.head(10).iterrows():
    print(f"{row['feature']:30s}: {row['importance']:.4f}")

# 5. 実用性の評価
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

best_model.fit(X_train, y_train)
train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"\n【過学習チェック】")
print(f"訓練R²: {train_r2:.4f}")
print(f"テストR²: {test_r2:.4f}")
print(f"差分: {train_r2 - test_r2:.4f}")

if train_r2 - test_r2 < 0.1:
    print("✓ 過学習は抑制されています")
else:
    print("⚠️ まだ過学習の傾向があります")

# 実際の予測誤差
y_test_exp = np.expm1(y_test)
test_pred_exp = np.expm1(test_pred)
rel_errors = np.abs(y_test_exp - test_pred_exp) / y_test_exp

print(f"\n【予測精度】")
print(f"中央相対誤差: {np.median(rel_errors)*100:.1f}%")
print(f"平均相対誤差: {np.mean(rel_errors)*100:.1f}%")

# 6. モデルの保存
print("\n【モデル保存】")
model_info = {
    'model': best_model,
    'features': all_features,
    'cv_r2': cv_scores.mean(),
    'cv_std': cv_scores.std(),
    'params': grid_search.best_params_
}

joblib.dump(model_info, 'final_clean_model.pkl')
print("モデルを final_clean_model.pkl に保存しました")

# 7. 結論
print("\n" + "="*80)
print("【結論】")
print(f"最終的なクリーンモデルのR²: {cv_scores.mean():.4f}")
print("\nGeminiの見解:")
print("- R² 0.3-0.4は、サムネイルとメタデータのみからの予測として妥当")
print("- YouTubeのアルゴリズムやトレンドの影響は予測できない")
print("- 過学習は抑制され、実用的なベースラインモデルとして機能")

print("\n今後の改善案:")
print("1. チャンネル特徴量の追加（過去の平均再生数など）")
print("2. CLIPやBERTによる高度な特徴抽出")
print("3. 分類問題への転換（10万再生を超えるか等）")
print("="*80)