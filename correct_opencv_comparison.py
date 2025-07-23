#!/usr/bin/env python3
"""
正しいOpenCV特徴量比較分析
- OpenCV特徴量なし vs OpenCV特徴量あり（顔検出含む）の完全比較
- 全6,062件データでの分析
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
import lightgbm as lgb
import json

print("="*70)
print("正しいOpenCV特徴量比較分析 - 全6,062件データ")
print("="*70)

# 元データ（6,062件）読み込み
df_original = pd.read_csv('youtube_top_new_complete.csv')
print(f"元データセット: {len(df_original)}件")

# 顔検出データ読み込み
df_faces = pd.read_csv('youtube_complete_with_faces_full.csv')
print(f"顔検出データ: {len(df_faces)}件")

# has_faceカラムを元データにマージ（ない場合は0）
df_full = df_original.merge(
    df_faces[['video_id', 'has_face']], 
    on='video_id', 
    how='left'
)

# has_faceがNaNの場合は0（顔なし）として扱う
df_full['has_face'] = df_full['has_face'].fillna(0).astype(int)

print(f"マージ後データ: {len(df_full)}件")

# 時間関連特徴量の作成
df_full['published_at'] = pd.to_datetime(df_full['published_at']).dt.tz_localize(None)
df_full['hour_published'] = df_full['published_at'].dt.hour
df_full['days_since_publish'] = (pd.Timestamp.now() - df_full['published_at']).dt.days

# ログ変換
df_full['log_views'] = np.log10(df_full['views'] + 1)
df_full['log_subscribers'] = np.log1p(df_full['subscribers'])

# OpenCV特徴量なし（基本メタデータのみ）
basic_features = [
    'video_duration', 'tags_count', 'description_length',
    'subscribers', 'log_subscribers', 'hour_published', 'days_since_publish'
]

# OpenCV特徴量あり（画像解析特徴量を全て含む）
opencv_features = basic_features + [
    'object_complexity', 'element_complexity', 'brightness', 'colorfulness', 'has_face'
]

print(f"\n🔍 特徴量構成:")
print(f"  基本メタデータのみ: {len(basic_features)}個")
print(f"    {basic_features}")
print(f"  OpenCV画像解析含む: {len(opencv_features)}個")
print(f"    OpenCV特徴量: object_complexity, element_complexity, brightness, colorfulness, has_face")

# OpenCV特徴量の分布確認
opencv_only = ['object_complexity', 'element_complexity', 'brightness', 'colorfulness', 'has_face']
print(f"\n📊 OpenCV特徴量の基本統計:")
for feature in opencv_only:
    if feature in df_full.columns:
        values = df_full[feature].dropna()
        print(f"  {feature}: 平均={values.mean():.2f}, 標準偏差={values.std():.2f}")

# 顔検出の分布
face_count = df_full['has_face'].sum()
no_face_count = len(df_full) - face_count
print(f"\n👤 顔検出分布:")
print(f"  顔あり: {face_count}件 ({face_count/len(df_full)*100:.1f}%)")
print(f"  顔なし: {no_face_count}件 ({no_face_count/len(df_full)*100:.1f}%)")

# データ準備
X_basic = df_full[basic_features].fillna(0)
X_opencv = df_full[opencv_features].fillna(0)
y = df_full['log_views']

print(f"\n📋 分析設定:")
print(f"  サンプル数: {len(X_basic)}件")
print(f"  目的変数: log_views")
print(f"  交差検証: 5-fold CV")

# 交差検証設定
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n=== モデル1: 基本メタデータのみ（{len(basic_features)}特徴量）===")
lgb_basic = lgb.LGBMRegressor(
    num_leaves=31, max_depth=6, min_child_samples=30,
    lambda_l2=0.1, feature_fraction=0.8, bagging_fraction=0.8,
    learning_rate=0.05, n_estimators=200, random_state=42, verbosity=-1
)

print("交差検証実行中...")
cv_scores_basic = cross_val_score(lgb_basic, X_basic, y, cv=kfold, scoring='r2')
print(f"CV R²: {cv_scores_basic.mean():.4f} ± {cv_scores_basic.std():.4f}")

# 特徴量重要度取得
lgb_basic.fit(X_basic, y)
importance_basic = pd.DataFrame({
    'feature': basic_features,
    'importance': lgb_basic.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n基本メタデータ特徴量の重要度:")
total_importance_basic = importance_basic['importance'].sum()
for i, (_, row) in enumerate(importance_basic.iterrows()):
    pct = (row['importance'] / total_importance_basic) * 100
    print(f"  {i+1:>2}. {row['feature']:<20} {row['importance']:>6.0f} ({pct:>5.1f}%)")

print(f"\n=== モデル2: OpenCV画像解析特徴量含む（{len(opencv_features)}特徴量）===")
lgb_opencv = lgb.LGBMRegressor(
    num_leaves=31, max_depth=6, min_child_samples=30,
    lambda_l2=0.1, feature_fraction=0.8, bagging_fraction=0.8,
    learning_rate=0.05, n_estimators=200, random_state=42, verbosity=-1
)

print("交差検証実行中...")
cv_scores_opencv = cross_val_score(lgb_opencv, X_opencv, y, cv=kfold, scoring='r2')
print(f"CV R²: {cv_scores_opencv.mean():.4f} ± {cv_scores_opencv.std():.4f}")

# 特徴量重要度取得
lgb_opencv.fit(X_opencv, y)
importance_opencv = pd.DataFrame({
    'feature': opencv_features,
    'importance': lgb_opencv.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nOpenCV含む全特徴量の重要度:")
total_importance_opencv = importance_opencv['importance'].sum()
for i, (_, row) in enumerate(importance_opencv.iterrows()):
    pct = (row['importance'] / total_importance_opencv) * 100
    
    # OpenCV特徴量をマーク
    if row['feature'] in ['object_complexity', 'element_complexity', 'brightness', 'colorfulness']:
        marker = " ← 📷OpenCV画像"
    elif row['feature'] == 'has_face':
        marker = " ← 👤OpenCV顔"
    else:
        marker = " ← 📊基本データ"
    
    print(f"  {i+1:>2}. {row['feature']:<20} {row['importance']:>6.0f} ({pct:>5.1f}%){marker}")

# 性能差分析
performance_diff = cv_scores_opencv.mean() - cv_scores_basic.mean()

print(f"\n=== 最終性能比較（全{len(df_full)}件データ）===")
print(f"基本メタデータのみ: R² = {cv_scores_basic.mean():.4f} ± {cv_scores_basic.std():.4f}")
print(f"OpenCV特徴量含む:   R² = {cv_scores_opencv.mean():.4f} ± {cv_scores_opencv.std():.4f}")
print(f"性能差:             {performance_diff:+.4f}")
print(f"改善率:             {(performance_diff/cv_scores_basic.mean())*100:+.2f}%")

# OpenCV特徴量の貢献度分析
opencv_importance_sum = 0
for feature in ['object_complexity', 'element_complexity', 'brightness', 'colorfulness', 'has_face']:
    feature_row = importance_opencv[importance_opencv['feature'] == feature]
    if not feature_row.empty:
        opencv_importance_sum += feature_row['importance'].iloc[0]

opencv_contribution_pct = (opencv_importance_sum / total_importance_opencv) * 100

print(f"\n=== OpenCV特徴量の詳細分析 ===")
print(f"OpenCV特徴量の合計重要度: {opencv_importance_sum:.0f} ({opencv_contribution_pct:.1f}%)")

opencv_rankings = []
for feature in ['object_complexity', 'element_complexity', 'brightness', 'colorfulness', 'has_face']:
    feature_row = importance_opencv[importance_opencv['feature'] == feature]
    if not feature_row.empty:
        rank = importance_opencv[importance_opencv['feature'] == feature].index[0] + 1
        importance = feature_row['importance'].iloc[0]
        pct = (importance / total_importance_opencv) * 100
        opencv_rankings.append((feature, rank, importance, pct))

print(f"\nOpenCV各特徴量の順位:")
for feature, rank, importance, pct in sorted(opencv_rankings, key=lambda x: x[1]):
    print(f"  {rank:>2}位. {feature:<20} {importance:>6.0f} ({pct:>5.1f}%)")

# 統計的有意性の簡易チェック
improvement_significant = abs(performance_diff) > cv_scores_basic.std()
print(f"\n統計的意味: {'有意な改善' if improvement_significant and performance_diff > 0 else '誤差範囲内'}")

# 結果保存用データ
results = {
    "analysis_type": "OPENCV_FEATURES_COMPARISON",
    "dataset_info": {
        "total_samples": len(df_full),
        "face_count": int(face_count),
        "no_face_count": int(no_face_count),
        "face_ratio": float(face_count/len(df_full))
    },
    "model_performance": {
        "basic_metadata_only": {
            "cv_r2_mean": float(cv_scores_basic.mean()),
            "cv_r2_std": float(cv_scores_basic.std()),
            "features_count": len(basic_features),
            "features": basic_features
        },
        "opencv_features_included": {
            "cv_r2_mean": float(cv_scores_opencv.mean()),
            "cv_r2_std": float(cv_scores_opencv.std()),
            "features_count": len(opencv_features),
            "opencv_features": ['object_complexity', 'element_complexity', 'brightness', 'colorfulness', 'has_face']
        },
        "comparison": {
            "performance_diff": float(performance_diff),
            "improvement_pct": float((performance_diff/cv_scores_basic.mean())*100),
            "is_significant": bool(improvement_significant and performance_diff > 0)
        }
    },
    "opencv_analysis": {
        "total_contribution_pct": float(opencv_contribution_pct),
        "individual_rankings": [
            {
                "feature": feature,
                "rank": int(rank),
                "importance": float(importance), 
                "percentage": float(pct)
            }
            for feature, rank, importance, pct in opencv_rankings
        ]
    }
}

# JSON保存（boolエラー回避）
with open('correct_opencv_comparison_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n=== 最終結論 ===")
if performance_diff > 0 and improvement_significant:
    print(f"✅ OpenCV画像解析特徴量は性能を {performance_diff:.4f} 有意に改善")
    conclusion = "非常に有効"
elif performance_diff > 0:
    print(f"📊 OpenCV画像解析特徴量は性能を {performance_diff:.4f} 改善（誤差範囲内）")
    conclusion = "やや有効"
else:
    print(f"❌ OpenCV画像解析特徴量は性能を悪化")
    conclusion = "無効"

print(f"📊 OpenCV特徴量合計貢献度: {opencv_contribution_pct:.1f}%")
print(f"📊 結論: OpenCV画像解析特徴量は{conclusion}")
print(f"📄 詳細結果: correct_opencv_comparison_results.json")

print(f"\n" + "="*70)
print("正しいOpenCV特徴量比較分析完了！")
print("="*70)