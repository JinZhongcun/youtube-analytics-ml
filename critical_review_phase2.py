#!/usr/bin/env python3
"""
批判的査読 Phase 2: 特徴量エンジニアリングの妥当性検証
各特徴量が本当に予測に使えるかを徹底検証
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("批判的査読 Phase 2: 特徴量エンジニアリングの徹底検証")
print("="*80)

# データ読み込み
df = pd.read_csv('youtube_top_new_complete.csv')
print(f"データ数: {len(df)}件")

# 各特徴量を分類
print("\n【特徴量の分類と検証】")

print("\n=== 1. 基本メタデータ ===")
basic_features = ['video_duration', 'tags_count', 'description_length']
for feat in basic_features:
    print(f"\n{feat}:")
    print(f"  - 定義: {feat}")
    print(f"  - 範囲: {df[feat].min():.1f} ~ {df[feat].max():.1f}")
    print(f"  - 平均: {df[feat].mean():.1f}")
    print(f"  - 欠損: {df[feat].isna().sum()}")
    print(f"  - 判定: ✓ 安全（アップロード時に決定）")

print("\n=== 2. サムネイル特徴量 ===")
thumbnail_features = ['brightness', 'colorfulness', 'object_complexity', 'element_complexity']
for feat in thumbnail_features:
    print(f"\n{feat}:")
    print(f"  - 範囲: {df[feat].min():.1f} ~ {df[feat].max():.1f}")
    print(f"  - 標準偏差: {df[feat].std():.1f}")
    
    # データ型の問題をチェック
    if feat in ['object_complexity', 'element_complexity']:
        print(f"  - データ型: {df[feat].dtype}")
        print(f"  - ユニーク値: {df[feat].nunique()}")
        if df[feat].dtype == 'int64' and df[feat].nunique() < 20:
            print(f"  - 警告: 離散的すぎる可能性")
    
    print(f"  - 判定: ✓ 安全（アップロード時の画像から抽出）")

print("\n=== 3. 時間関連特徴量 ===")
print("\ntime_duration:")
print(f"  - 範囲: {df['time_duration'].min()} ~ {df['time_duration'].max()}")
print(f"  - 内容: おそらくデータ収集時点からの経過日数")
print(f"  - 判定: ⚠️ 要注意（データ収集時点に依存）")

print("\npublished_at:")
df['published_at_dt'] = pd.to_datetime(df['published_at'])
print(f"  - 最古: {df['published_at_dt'].min()}")
print(f"  - 最新: {df['published_at_dt'].max()}")
print(f"  - 判定: ✓ 安全（投稿日時は不変）")

print("\n=== 4. 派生特徴量の検証 ===")

# log変換の妥当性
print("\n【log変換の妥当性】")
for feat in ['video_duration', 'tags_count', 'description_length']:
    # 0の数をチェック
    zeros = (df[feat] == 0).sum()
    print(f"\n{feat}:")
    print(f"  - ゼロの数: {zeros} ({zeros/len(df)*100:.1f}%)")
    if zeros > 0:
        print(f"  - 警告: log(0+1) = 0 となるデータが存在")
    
    # 分布の歪度
    from scipy import stats
    skewness = stats.skew(df[feat].dropna())
    print(f"  - 歪度: {skewness:.2f}")
    if abs(skewness) > 2:
        print(f"  - 推奨: log変換が有効")
    else:
        print(f"  - 注意: log変換の効果は限定的")

# 比率特徴量の検証
print("\n【比率特徴量の妥当性】")
print("\ntags_per_second = tags_count / (video_duration + 1):")
df['tags_per_second'] = df['tags_count'] / (df['video_duration'] + 1)
print(f"  - 範囲: {df['tags_per_second'].min():.3f} ~ {df['tags_per_second'].max():.3f}")
print(f"  - 問題: video_duration=0の場合の処理")
print(f"  - 判定: △ 使用可能だが解釈に注意")

# カテゴリの分布
print("\n【カテゴリ特徴量】")
print(f"カテゴリ数: {df['category_id'].nunique()}")
print(f"最頻カテゴリ: {df['category_id'].mode()[0]} ({(df['category_id'] == df['category_id'].mode()[0]).sum()}件)")
print("判定: ✓ 安全（ワンホットエンコーディング推奨）")

# 相関分析
print("\n【特徴量間の相関】")
# 高相関ペアを探す
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

# views以外で高相関のペアを見つける
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            if col1 != 'views' and col2 != 'views':
                high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))

print("\n高相関ペア（|r| > 0.7）:")
for col1, col2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]:
    print(f"  - {col1} vs {col2}: {corr:.3f}")

# 最終判定
print("\n【最終判定：使用すべき特徴量】")
safe_features = [
    'video_duration',
    'tags_count',
    'description_length',
    'brightness',
    'colorfulness',
    'object_complexity',
    'element_complexity',
    'published_at（変換後）',
    'category_id（ダミー変数）'
]

print("\n✓ 安全な特徴量:")
for feat in safe_features:
    print(f"  - {feat}")

print("\n✗ 除外すべき特徴量:")
excluded = [
    'subscribers（時系列リーケージ）',
    'time_duration（データ収集時点依存）',
    'likes（結果変数）',
    'comment_count（結果変数）'
]
for feat in excluded:
    print(f"  - {feat}")

print("\n△ 注意して使う特徴量:")
caution = [
    'log変換（ゼロ値の扱い）',
    '比率特徴量（分母ゼロの処理）'
]
for feat in caution:
    print(f"  - {feat}")