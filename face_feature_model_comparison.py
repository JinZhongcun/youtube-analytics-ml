#!/usr/bin/env python3
"""
顔検出特徴量を含むモデル vs 含まないモデルの性能比較
特徴量重要度の変化を定量的に分析
"""
import pandas as pd
import numpy as np
import cv2
import os
import json
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("顔検出特徴量を含むモデル vs 含まないモデルの比較")
print("="*60)

# データ読み込み
df = pd.read_csv('youtube_top_new_complete.csv')
print(f"データセット: {len(df)}件")

# サムネイル画像の存在確認
thumbnails_dir = 'thumbnails'
df['thumbnail_exists'] = df['video_id'].apply(
    lambda x: os.path.exists(os.path.join(thumbnails_dir, f'{x}.jpg'))
)
print(f"サムネイル画像が存在: {df['thumbnail_exists'].sum()}/{len(df)}件")

# 画像が存在するデータのみ使用
df_with_images = df[df['thumbnail_exists']].copy()
print(f"分析対象: {len(df_with_images)}件")

def detect_face(video_id):
    """OpenCV Haar Cascadeで顔検出"""
    img_path = os.path.join(thumbnails_dir, f'{video_id}.jpg')
    
    try:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 顔検出
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        return int(len(faces) > 0)
    except Exception as e:
        return 0

# サンプルサイズを制限（高速化のため）
sample_size = min(1000, len(df_with_images))
df_sample = df_with_images.sample(n=sample_size, random_state=42).copy()
print(f"サンプルサイズ: {sample_size}件（高速化のため）")

# 顔検出実行
print(f"\n=== 顔検出実行中 ===")
face_results = []
for i, video_id in enumerate(df_sample['video_id']):
    if i % 100 == 0:
        print(f"  {i}/{sample_size}件処理済み ({i/sample_size*100:.1f}%)")
    
    has_face = detect_face(video_id)
    face_results.append(has_face)

df_sample['has_face'] = face_results

# 基本統計
face_count = sum(face_results)
no_face_count = len(face_results) - face_count
print(f"顔あり: {face_count}件, 顔なし: {no_face_count}件")

# 時間関連の特徴量作成
df_sample['published_at'] = pd.to_datetime(df_sample['published_at']).dt.tz_localize(None)
df_sample['hour_published'] = df_sample['published_at'].dt.hour
df_sample['days_since_publish'] = (pd.Timestamp.now() - df_sample['published_at']).dt.days

# ログ変換
df_sample['log_views'] = np.log10(df_sample['views'] + 1)
df_sample['log_subscribers'] = np.log1p(df_sample['subscribers'])

# 基本特徴量（顔検出なし）
basic_features = [
    'video_duration', 'tags_count', 'description_length',
    'object_complexity', 'element_complexity', 'brightness', 'colorfulness',
    'subscribers', 'log_subscribers', 'hour_published', 'days_since_publish'
]

# 顔検出を含む特徴量
face_features = basic_features + ['has_face']

# データ準備
X_basic = df_sample[basic_features].fillna(0)
X_face = df_sample[face_features].fillna(0)
y = df_sample['log_views']

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

print(f"\n基本特徴量の重要度（Top 8）:")
total_importance_basic = importance_basic['importance'].sum()
for _, row in importance_basic.head(8).iterrows():
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

print(f"\n顔検出含む特徴量の重要度（Top 9）:")
total_importance_face = importance_face['importance'].sum()
for _, row in importance_face.head(9).iterrows():
    pct = (row['importance'] / total_importance_face) * 100
    marker = " ← 顔検出" if row['feature'] == 'has_face' else ""
    print(f"  {row['feature']:<20} {row['importance']:>6.0f} ({pct:>5.1f}%){marker}")

# 性能差分析
performance_diff = cv_scores_face.mean() - cv_scores_basic.mean()
print(f"\n=== 性能比較 ===")
print(f"基本モデル:     R² = {cv_scores_basic.mean():.4f} ± {cv_scores_basic.std():.4f}")
print(f"顔検出含む:     R² = {cv_scores_face.mean():.4f} ± {cv_scores_face.std():.4f}")
print(f"性能差:         {performance_diff:+.4f}")
print(f"改善率:         {(performance_diff/cv_scores_basic.mean())*100:+.2f}%")

# 顔検出特徴量の順位
has_face_rank = importance_face[importance_face['feature'] == 'has_face'].index[0] + 1
has_face_importance = importance_face[importance_face['feature'] == 'has_face']['importance'].iloc[0]
has_face_pct = (has_face_importance / total_importance_face) * 100

print(f"\n=== 顔検出特徴量の分析 ===")
print(f"has_face順位:   {has_face_rank}位 / {len(face_features)}特徴量")
print(f"重要度:         {has_face_importance:.0f} ({has_face_pct:.1f}%)")

# 結果保存
results = {
    'dataset_info': {
        'original_data': len(df),
        'with_thumbnails': len(df_with_images), 
        'sample_size': sample_size,
        'face_count': int(face_count),
        'no_face_count': int(no_face_count)
    },
    'model_performance': {
        'basic_model': {
            'features_count': len(basic_features),
            'cv_r2_mean': float(cv_scores_basic.mean()),
            'cv_r2_std': float(cv_scores_basic.std())
        },
        'face_model': {
            'features_count': len(face_features),
            'cv_r2_mean': float(cv_scores_face.mean()),
            'cv_r2_std': float(cv_scores_face.std())
        },
        'comparison': {
            'performance_diff': float(performance_diff),
            'improvement_rate_pct': float((performance_diff/cv_scores_basic.mean())*100)
        }
    },
    'feature_importance': {
        'basic_model': importance_basic.head(8).to_dict('records'),
        'face_model': importance_face.head(9).to_dict('records')
    },
    'face_feature_analysis': {
        'rank': int(has_face_rank),
        'importance': float(has_face_importance),
        'percentage': float(has_face_pct)
    }
}

with open('face_feature_comparison_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n=== 最終結論 ===")
if performance_diff > 0:
    print(f"✅ 顔検出特徴量は性能を {performance_diff:.4f} 改善")
    print(f"✅ {has_face_rank}位の重要度 ({has_face_pct:.1f}%)")
    conclusion = "有効"
else:
    print(f"❌ 顔検出特徴量は性能を {performance_diff:.4f} 悪化")
    print(f"❌ {has_face_rank}位の重要度 ({has_face_pct:.1f}%)")
    conclusion = "無効"

print(f"📊 結論: 顔検出特徴量は予測性能に{conclusion}")
print(f"📄 詳細結果: face_feature_comparison_results.json")