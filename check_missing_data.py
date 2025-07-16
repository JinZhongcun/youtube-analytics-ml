#!/usr/bin/env python3
"""
不完全データの詳細調査
"""

import pandas as pd
import numpy as np

print("="*60)
print("データの欠損状況を詳細調査")
print("="*60)

# データ読み込み
df_old = pd.read_csv('youtube_top_jp.csv', skiprows=1)
df_new = pd.read_csv('youtube_top_new.csv')

print(f"旧データ（767件）のカラム数: {len(df_old.columns)}")
print(f"新データ（6,078件）のカラム数: {len(df_new.columns)}")

# カラムの比較
print("\n【カラムの違い】")
old_cols = set(df_old.columns)
new_cols = set(df_new.columns)

print("\n旧データのみにあるカラム:")
only_old = old_cols - new_cols
for col in sorted(only_old):
    print(f"  - {col}")

print("\n新データのみにあるカラム:")
only_new = new_cols - old_cols
for col in sorted(only_new):
    print(f"  - {col}")

# 重要な欠損データの確認
print("\n【重要な欠損データ】")
important_missing = ['subscribers', 'likes', 'comment_count', 'dislike_count']
for col in important_missing:
    if col in old_cols and col not in new_cols:
        print(f"❌ {col}: 新データに存在しない")

# データの中身を比較
print("\n【データ例の比較】")
print("\n旧データ（最初の1件）:")
print(df_old.iloc[0][['video_id', 'title', 'views', 'subscribers', 'likes', 'comment_count']].to_dict())

print("\n新データ（最初の1件）:")
available_cols = [col for col in ['video_id', 'title', 'views', 'subscribers', 'likes', 'comment_count'] if col in df_new.columns]
print(df_new.iloc[0][available_cols].to_dict())

# マージ後の状況
df_merged = pd.merge(df_new, df_old[['video_id', 'subscribers', 'likes', 'comment_count']], 
                     on='video_id', how='left')

print("\n【マージ後のデータ完全性】")
print(f"全データ数: {len(df_merged)}")
print(f"subscribersがある: {df_merged['subscribers'].notna().sum()} ({df_merged['subscribers'].notna().sum()/len(df_merged)*100:.1f}%)")
print(f"subscribersがない: {df_merged['subscribers'].isna().sum()} ({df_merged['subscribers'].isna().sum()/len(df_merged)*100:.1f}%)")

# viewsとsubscribersの相関を確認
complete_data = df_merged[df_merged['subscribers'].notna()]
if len(complete_data) > 0:
    correlation = np.corrcoef(np.log10(complete_data['views']+1), 
                             np.log10(complete_data['subscribers']+1))[0,1]
    print(f"\nviews と subscribers の相関係数: {correlation:.3f}")

# 欠損パターンの分析
print("\n【欠損パターン】")
missing_pattern = df_merged[['subscribers', 'likes', 'comment_count']].isnull()
pattern_counts = missing_pattern.value_counts()
print("\n欠損パターンの分布:")
for pattern, count in pattern_counts.items():
    print(f"  subscribers={pattern[0]}, likes={pattern[1]}, comment_count={pattern[2]}: {count}件")