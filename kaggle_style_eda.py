#!/usr/bin/env python3
"""
Kaggleスタイルの徹底的なEDA
データの中身を詳細に確認
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("Kaggleスタイル EDA - データを徹底的に理解する")
print("="*60)

# データ読み込み
df_old = pd.read_csv('youtube_top_new.csv')
df_new = pd.read_csv('youtube_top_new_complete.csv')

print("\n【1. データの基本情報】")
print(f"元データ: {df_old.shape}")
print(f"新データ: {df_new.shape}")

# カラムの詳細確認
print("\n【2. カラムの詳細】")
print("\n元データのカラム:")
for i, col in enumerate(df_old.columns):
    print(f"{i+1:2d}. {col:20s} - {df_old[col].dtype}")

print("\n新データのカラム:")
for i, col in enumerate(df_new.columns):
    print(f"{i+1:2d}. {col:20s} - {df_new[col].dtype}")

# 各カラムの統計情報
print("\n【3. 数値カラムの統計】")
numeric_cols = df_new.select_dtypes(include=[np.number]).columns
print(df_new[numeric_cols].describe())

# 欠損値チェック
print("\n【4. 欠損値の確認】")
missing = df_new.isnull().sum()
missing_pct = (missing / len(df_new) * 100).round(2)
missing_df = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
print(missing_df[missing_df['missing_count'] > 0])

# viewsの分布を確認
print("\n【5. viewsの分布】")
print(f"最小値: {df_new['views'].min():,}")
print(f"25%: {df_new['views'].quantile(0.25):,.0f}")
print(f"中央値: {df_new['views'].median():,.0f}")
print(f"75%: {df_new['views'].quantile(0.75):,.0f}")
print(f"最大値: {df_new['views'].max():,}")
print(f"平均: {df_new['views'].mean():,.0f}")
print(f"標準偏差: {df_new['views'].std():,.0f}")

# subscribersの分布を確認
print("\n【6. subscribersの分布】")
print(f"最小値: {df_new['subscribers'].min():,}")
print(f"25%: {df_new['subscribers'].quantile(0.25):,.0f}")
print(f"中央値: {df_new['subscribers'].median():,.0f}")
print(f"75%: {df_new['subscribers'].quantile(0.75):,.0f}")
print(f"最大値: {df_new['subscribers'].max():,}")
print(f"平均: {df_new['subscribers'].mean():,.0f}")
print(f"標準偏差: {df_new['subscribers'].std():,.0f}")

# カテゴリの分布
print("\n【7. カテゴリの分布】")
print(df_new['category_id'].value_counts().head(10))

# published_atの確認
print("\n【8. published_atの分析】")
df_new['published_at_dt'] = pd.to_datetime(df_new['published_at'])
print(f"最古の動画: {df_new['published_at_dt'].min()}")
print(f"最新の動画: {df_new['published_at_dt'].max()}")
print(f"期間: {(df_new['published_at_dt'].max() - df_new['published_at_dt'].min()).days}日")

# time_durationの確認
print("\n【9. time_durationの内容確認】")
print(f"time_durationのデータ型: {df_new['time_duration'].dtype}")
print(f"ユニークな値の数: {df_new['time_duration'].nunique()}")
print(f"サンプル（最初の10個）: {df_new['time_duration'].head(10).tolist()}")
print(f"最小値: {df_new['time_duration'].min()}")
print(f"最大値: {df_new['time_duration'].max()}")

# viewsとsubscribersの関係を詳しく見る
print("\n【10. views vs subscribers の関係】")
# 両方の対数を取って相関を見る
log_views = np.log10(df_new['views'] + 1)
log_subs = np.log10(df_new['subscribers'] + 1)
corr = np.corrcoef(log_views, log_subs)[0, 1]
print(f"log(views) vs log(subscribers)の相関: {corr:.4f}")

# 動画ごとのviews/subscribersの比率
df_new['views_per_subscriber'] = df_new['views'] / (df_new['subscribers'] + 1)
print(f"\nviews/subscribersの統計:")
print(f"平均: {df_new['views_per_subscriber'].mean():.2f}")
print(f"中央値: {df_new['views_per_subscriber'].median():.2f}")
print(f"最大: {df_new['views_per_subscriber'].max():.2f}")

# 異常値の確認
print("\n【11. 異常値の確認】")
# subscribersが0だが、viewsが多い動画
zero_subs = df_new[df_new['subscribers'] == 0]
print(f"subscribers=0の動画数: {len(zero_subs)}")
if len(zero_subs) > 0:
    print(f"その中でviewsの最大値: {zero_subs['views'].max():,}")

# viewsが極端に少ない動画
low_views = df_new[df_new['views'] < 10000]
print(f"\nviews < 10,000の動画数: {len(low_views)}")

# サンプルデータの確認
print("\n【12. データサンプル（最初の5行）】")
cols_to_show = ['video_id', 'views', 'subscribers', 'video_duration', 
                'tags_count', 'published_at', 'colorfulness']
print(df_new[cols_to_show].head())

# 元データと新データの同じvideo_idを比較
print("\n【13. 同じvideo_idのデータ比較】")
sample_id = df_new['video_id'].iloc[0]
print(f"\nvideo_id: {sample_id}")
print("\n元データ:")
if sample_id in df_old['video_id'].values:
    print(df_old[df_old['video_id'] == sample_id][['views', 'video_duration', 'tags_count']].iloc[0])
print("\n新データ:")
print(df_new[df_new['video_id'] == sample_id][['views', 'subscribers', 'video_duration', 'tags_count']].iloc[0])

# データの一貫性チェック
print("\n【14. データの一貫性チェック】")
# viewsが変わっているか確認
merged = pd.merge(df_old, df_new, on='video_id', suffixes=('_old', '_new'))
if len(merged) > 0:
    views_changed = merged[merged['views_old'] != merged['views_new']]
    print(f"viewsが変わった動画数: {len(views_changed)} / {len(merged)}")