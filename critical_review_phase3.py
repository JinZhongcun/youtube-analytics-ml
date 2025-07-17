#!/usr/bin/env python3
"""
批判的査読 Phase 3: モデル評価方法の検証
train/testの分割方法、評価指標、過学習の可能性を検証
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("批判的査読 Phase 3: モデル評価方法の徹底検証")
print("="*80)

# データ読み込み
df = pd.read_csv('drive-download-20250717T063336Z-1-001/youtube_top_new.csv')
print(f"データ数: {len(df)}件")

# クリーンな特徴量のみ使用
feature_cols = [
    'video_duration', 'tags_count', 'description_length',
    'object_complexity', 'element_complexity', 
    'brightness', 'colorfulness'
]

# published_atから特徴量生成
df['published_at'] = pd.to_datetime(df['published_at'])
df['days_since_publish'] = (pd.Timestamp.now(tz='UTC') - df['published_at']).dt.days
df['hour_published'] = df['published_at'].dt.hour
df['weekday_published'] = df['published_at'].dt.weekday

feature_cols.extend(['days_since_publish', 'hour_published', 'weekday_published'])

print(f"\n使用特徴量: {len(feature_cols)}個")

# データ準備
X = df[feature_cols]
y = np.log1p(df['views'])  # log変換

print("\n【1. データ分割の検証】")
# 複数のrandom_stateで安定性チェック
r2_scores = []
for seed in range(10):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    
    # シンプルなRFモデル
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

print(f"R²スコアの変動:")
print(f"  - 平均: {np.mean(r2_scores):.4f}")
print(f"  - 標準偏差: {np.std(r2_scores):.4f}")
print(f"  - 最小: {np.min(r2_scores):.4f}")
print(f"  - 最大: {np.max(r2_scores):.4f}")

if np.std(r2_scores) > 0.05:
    print("⚠️ 警告: 分割によるばらつきが大きい")

print("\n【2. 交差検証による評価】")
# 5-fold cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=kfold, scoring='r2')

print(f"5-fold CV結果:")
print(f"  - 平均R²: {cv_scores.mean():.4f}")
print(f"  - 標準偏差: {cv_scores.std():.4f}")
print(f"  - 各fold: {cv_scores}")

# train/test分割との差
split_mean = np.mean(r2_scores)
cv_mean = cv_scores.mean()
diff = abs(split_mean - cv_mean)
print(f"\n単純分割とCVの差: {diff:.4f}")
if diff > 0.05:
    print("⚠️ 警告: 評価方法により結果が大きく異なる")

print("\n【3. 過学習の検証】")
# 訓練データでの性能
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
rf.fit(X_train, y_train)

train_pred = rf.predict(X_train)
test_pred = rf.predict(X_test)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"訓練データR²: {train_r2:.4f}")
print(f"テストデータR²: {test_r2:.4f}")
print(f"差分: {train_r2 - test_r2:.4f}")

if train_r2 - test_r2 > 0.1:
    print("⚠️ 警告: 過学習の可能性")

print("\n【4. 予測値の分布検証】")
# 実際のviewsに戻す
y_test_exp = np.expm1(y_test)
test_pred_exp = np.expm1(test_pred)

# 予測誤差の分析
errors = y_test_exp - test_pred_exp
rel_errors = errors / y_test_exp

print(f"\n予測誤差統計:")
print(f"  - 平均絶対誤差: {np.mean(np.abs(errors)):,.0f} views")
print(f"  - 中央絶対誤差: {np.median(np.abs(errors)):,.0f} views")
print(f"  - 相対誤差中央値: {np.median(np.abs(rel_errors))*100:.1f}%")

# 極端な予測ミスをチェック
extreme_errors = np.abs(rel_errors) > 5  # 500%以上の誤差
print(f"\n極端な予測ミス（500%以上）: {extreme_errors.sum()}件 ({extreme_errors.sum()/len(y_test)*100:.1f}%)")

print("\n【5. データリーケージの最終確認】")
# 特徴量重要度
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n特徴量重要度TOP5:")
print(importances.head())

# days_since_publishが異常に高い場合は警告
if importances.iloc[0]['feature'] == 'days_since_publish' and importances.iloc[0]['importance'] > 0.3:
    print("\n⚠️ 警告: days_since_publishの重要度が異常に高い")
    print("時系列的な偏りがある可能性")

print("\n【6. サンプルサイズの妥当性】")
print(f"訓練データ: {len(X_train)}件")
print(f"テストデータ: {len(X_test)}件")
print(f"特徴量数: {len(feature_cols)}")
print(f"サンプル数/特徴量数: {len(X_train)/len(feature_cols):.1f}")

if len(X_train)/len(feature_cols) < 10:
    print("⚠️ 警告: サンプル数が少なすぎる可能性")

print("\n【最終評価】")
print(f"- 交差検証R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"- 過学習度: {train_r2 - test_r2:.4f}")
print(f"- 予測の実用性: 中央相対誤差 {np.median(np.abs(rel_errors))*100:.1f}%")

# 批判的結論
print("\n【批判的結論】")
issues = []

if cv_scores.std() > 0.05:
    issues.append("モデルの安定性に問題")
if train_r2 - test_r2 > 0.1:
    issues.append("過学習の傾向")
if np.median(np.abs(rel_errors)) > 1.0:
    issues.append("予測精度が実用レベルに達していない")
if importances.iloc[0]['importance'] > 0.3:
    issues.append("特定の特徴量に過度に依存")

if issues:
    print("問題点:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("評価方法は概ね適切")

# 予測vs実際のプロット
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test_exp, test_pred_exp, alpha=0.5)
plt.plot([y_test_exp.min(), y_test_exp.max()], 
         [y_test_exp.min(), y_test_exp.max()], 'r--')
plt.xlabel('Actual Views')
plt.ylabel('Predicted Views')
plt.title('Predictions vs Actual')
plt.xscale('log')
plt.yscale('log')

plt.subplot(1, 2, 2)
plt.hist(rel_errors[np.abs(rel_errors) < 5], bins=50)
plt.xlabel('Relative Error')
plt.ylabel('Count')
plt.title('Error Distribution (|error| < 500%)')
plt.tight_layout()
plt.savefig('model_evaluation_critical.png')
print("\nプロットを model_evaluation_critical.png に保存")