#!/usr/bin/env python3
"""
既存CSVのhas_faceカラムを使って単純比較
全4,817件での顔検出特徴量の効果分析
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
import lightgbm as lgb
import json

print("="*60)
print("has_faceカラム使用での全データ比較分析")
print("="*60)

# 既存の顔検出データ読み込み
df = pd.read_csv('youtube_complete_with_faces_full.csv')
print(f"データセット: {len(df)}件")

# has_faceカラムの状況確認
face_count = df['has_face'].sum()
no_face_count = len(df) - face_count
print(f"顔あり: {face_count}件 ({face_count/len(df)*100:.1f}%)")
print(f"顔なし: {no_face_count}件 ({no_face_count/len(df)*100:.1f}%)")

# 基本特徴量（顔検出なし）
basic_features = [
    'video_duration', 'tags_count', 'description_length',
    'object_complexity', 'element_complexity', 'brightness', 'colorfulness',
    'subscribers', 'log_subscribers', 'hour_published', 'days_since_publish'
]

# 顔検出を含む特徴量
face_features = basic_features + ['has_face']

# データ準備
X_basic = df[basic_features].fillna(0)
X_face = df[face_features].fillna(0)
y = df['log_views']

print(f"\n特徴量準備:")
print(f"  基本特徴量: {len(basic_features)}個")
print(f"  顔検出含む: {len(face_features)}個")
print(f"  全サンプル数: {len(X_basic)}件")

# 交差検証設定
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n=== モデル1: 基本特徴量のみ（{len(basic_features)}特徴量）===")
lgb_basic = lgb.LGBMRegressor(
    num_leaves=31, max_depth=6, min_child_samples=30,
    lambda_l2=0.1, feature_fraction=0.8, bagging_fraction=0.8,
    learning_rate=0.05, n_estimators=200, random_state=42, verbosity=-1
)

cv_scores_basic = cross_val_score(lgb_basic, X_basic, y, cv=kfold, scoring='r2')
print(f"CV R²: {cv_scores_basic.mean():.4f} ± {cv_scores_basic.std():.4f}")

# 特徴量重要度取得
lgb_basic.fit(X_basic, y)
importance_basic = pd.DataFrame({
    'feature': basic_features,
    'importance': lgb_basic.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n基本特徴量の重要度（Top 10）:")
total_importance_basic = importance_basic['importance'].sum()
for _, row in importance_basic.head(10).iterrows():
    pct = (row['importance'] / total_importance_basic) * 100
    print(f"  {row['feature']:<20} {row['importance']:>6.0f} ({pct:>5.1f}%)")

print(f"\n=== モデル2: 顔検出特徴量含む（{len(face_features)}特徴量）===")
lgb_face = lgb.LGBMRegressor(
    num_leaves=31, max_depth=6, min_child_samples=30,
    lambda_l2=0.1, feature_fraction=0.8, bagging_fraction=0.8,
    learning_rate=0.05, n_estimators=200, random_state=42, verbosity=-1
)

cv_scores_face = cross_val_score(lgb_face, X_face, y, cv=kfold, scoring='r2')
print(f"CV R²: {cv_scores_face.mean():.4f} ± {cv_scores_face.std():.4f}")

# 特徴量重要度取得
lgb_face.fit(X_face, y)
importance_face = pd.DataFrame({
    'feature': face_features,
    'importance': lgb_face.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n顔検出含む特徴量の重要度（全{len(face_features)}特徴量）:")
total_importance_face = importance_face['importance'].sum()
for i, (_, row) in enumerate(importance_face.iterrows()):
    pct = (row['importance'] / total_importance_face) * 100
    marker = " ← 👤顔検出" if row['feature'] == 'has_face' else ""
    rank = i + 1
    print(f"  {rank:>2}. {row['feature']:<18} {row['importance']:>6.0f} ({pct:>5.1f}%){marker}")

# 性能差分析
performance_diff = cv_scores_face.mean() - cv_scores_basic.mean()

print(f"\n=== 最終性能比較（全{len(df)}件データ）===")
print(f"基本モデル:     R² = {cv_scores_basic.mean():.4f} ± {cv_scores_basic.std():.4f}")
print(f"顔検出含む:     R² = {cv_scores_face.mean():.4f} ± {cv_scores_face.std():.4f}")
print(f"性能差:         {performance_diff:+.4f}")
print(f"改善率:         {(performance_diff/cv_scores_basic.mean())*100:+.2f}%")

# 顔検出特徴量の順位分析
has_face_row = importance_face[importance_face['feature'] == 'has_face']
has_face_rank = has_face_row.index[0] + 1
has_face_importance = has_face_row['importance'].iloc[0]
has_face_pct = (has_face_importance / total_importance_face) * 100

print(f"\n=== 顔検出特徴量の詳細分析 ===")
print(f"has_face順位:   {has_face_rank}位 / {len(face_features)}特徴量")
print(f"重要度:         {has_face_importance:.0f} ({has_face_pct:.2f}%)")
print(f"データ規模:     {len(df):,}件（全データ、サンプリングなし）")

# 統計的有意性の簡易チェック
improvement_significant = abs(performance_diff) > cv_scores_basic.std()
print(f"統計的意味:     {'有意' if improvement_significant else '誤差範囲内'}")

# 結果保存
results = {
    "dataset_info": {
        "total_samples": len(df),
        "face_count": int(face_count),
        "no_face_count": int(no_face_count),
        "face_ratio": float(face_count/len(df))
    },
    "model_performance": {
        "basic_model": {
            "cv_r2_mean": float(cv_scores_basic.mean()),
            "cv_r2_std": float(cv_scores_basic.std()),
            "features_count": len(basic_features)
        },
        "face_model": {
            "cv_r2_mean": float(cv_scores_face.mean()),
            "cv_r2_std": float(cv_scores_face.std()),
            "features_count": len(face_features)
        },
        "comparison": {
            "performance_diff": float(performance_diff),
            "improvement_pct": float((performance_diff/cv_scores_basic.mean())*100),
            "is_significant": improvement_significant
        }
    },
    "feature_importance": {
        "basic_model": [
            {
                "rank": i+1,
                "feature": row['feature'],
                "importance": float(row['importance']),
                "percentage": float((row['importance'] / total_importance_basic) * 100)
            }
            for i, (_, row) in enumerate(importance_basic.iterrows())
        ],
        "face_model": [
            {
                "rank": i+1,
                "feature": row['feature'],
                "importance": float(row['importance']),
                "percentage": float((row['importance'] / total_importance_face) * 100)
            }
            for i, (_, row) in enumerate(importance_face.iterrows())
        ]
    },
    "face_analysis": {
        "rank": int(has_face_rank),
        "importance": float(has_face_importance),
        "percentage": float(has_face_pct),
        "conclusion": "EFFECTIVE" if performance_diff > 0 and improvement_significant else "MINIMAL_IMPACT"
    }
}

with open('final_face_comparison_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n=== 最終結論 ===")
if performance_diff > 0 and improvement_significant:
    print(f"✅ 顔検出特徴量は性能を {performance_diff:.4f} 有意に改善")
    conclusion = "有効"
else:
    print(f"📊 顔検出特徴量の効果は微小または誤差範囲内")
    conclusion = "実質的に無効"

print(f"📊 順位: {has_face_rank}位 / {len(face_features)}特徴量")
print(f"📊 重要度: {has_face_pct:.2f}%")
print(f"📊 結論: 顔検出特徴量は{conclusion}")
print(f"📄 詳細: final_face_comparison_results.json")

print(f"\n" + "="*60)
print("全データ分析完了！")
print("="*60)