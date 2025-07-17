#!/usr/bin/env python3
"""
R² = 0.99の原因調査
データリーケージの可能性をチェック
"""

import pandas as pd
import numpy as np

print("="*60)
print("データリーケージ調査")
print("="*60)

# データ読み込み
df_old = pd.read_csv('youtube_top_new.csv')  # 元のデータ
df_new = pd.read_csv('youtube_top_new_complete.csv')  # 新しいデータ

print(f"元データ: {len(df_old)}件")
print(f"新データ: {len(df_new)}件")

# カラムの比較
print("\n【カラムの違い】")
old_cols = set(df_old.columns)
new_cols = set(df_new.columns)
print(f"元データのカラム数: {len(old_cols)}")
print(f"新データのカラム数: {len(new_cols)}")
print(f"新しく追加されたカラム: {new_cols - old_cols}")

# データの中身をチェック
print("\n【データサンプル（最初の5件）】")
print("\n元データ:")
print(df_old[['video_id', 'views', 'video_duration', 'tags_count']].head())
print("\n新データ:")
print(df_new[['video_id', 'views', 'video_duration', 'tags_count', 'subscribers']].head())

# views と subscribers の相関をチェック
print("\n【相関分析】")
correlation = df_new[['views', 'subscribers', 'video_duration', 'colorfulness']].corr()
print(correlation)

# subscriber_per_view の分布を確認
df_new['subscriber_per_view'] = df_new['subscribers'] / (df_new['views'] + 1)
print("\n【subscriber_per_view の統計】")
print(f"平均: {df_new['subscriber_per_view'].mean():.4f}")
print(f"中央値: {df_new['subscriber_per_view'].median():.4f}")
print(f"標準偏差: {df_new['subscriber_per_view'].std():.4f}")
print(f"最大値: {df_new['subscriber_per_view'].max():.4f}")

# 同じ値のチェック
print("\n【重複チェック】")
print(f"video_id の重複: {df_new['video_id'].duplicated().sum()}")
print(f"views の重複: {df_new['views'].duplicated().sum()}")

# viewsとsubscribersの関係を詳しく見る
print("\n【views と subscribers の関係】")
# 線形関係をチェック
from sklearn.linear_model import LinearRegression
X = df_new[['subscribers']].values
y = np.log10(df_new['views'] + 1)
model = LinearRegression()
model.fit(X, y)
r2 = model.score(X, y)
print(f"subscribers → log(views) の R²: {r2:.4f}")

# 逆の関係もチェック
X2 = df_new[['views']].values
y2 = df_new['subscribers']
model2 = LinearRegression()
model2.fit(X2, y2)
r2_2 = model2.score(X2, y2)
print(f"views → subscribers の R²: {r2_2:.4f}")

# データの分散を確認
print("\n【データの分散】")
print(f"views の変動係数: {df_new['views'].std() / df_new['views'].mean():.4f}")
print(f"subscribers の変動係数: {df_new['subscribers'].std() / df_new['subscribers'].mean():.4f}")

# 異常値チェック
print("\n【異常値チェック】")
for col in ['views', 'subscribers', 'video_duration']:
    q1 = df_new[col].quantile(0.25)
    q3 = df_new[col].quantile(0.75)
    iqr = q3 - q1
    outliers = ((df_new[col] < q1 - 1.5*iqr) | (df_new[col] > q3 + 1.5*iqr)).sum()
    print(f"{col} の異常値: {outliers}件 ({outliers/len(df_new)*100:.1f}%)")