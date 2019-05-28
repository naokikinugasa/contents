# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # M データインポート

# ## T ライブラリのインポート

# ## itr

# ### ライブラリのインポート ※要検討　穴埋め
# Pythonではライブラリと呼ばれる便利なプログラムがあります。ライブラリを使えるように読み込むことをインポートと言います。データ分析で役立つライブラリとしては、pandas, numpy, matplotlibがあります。これらは使用頻度が高く、長い単語だと記述するのが大変ですので、ライブラリをインポートする際には合わせて”as”を使って、慣習的に短い名前で使えるようにします。
#
# import ライブラリ名 as 省略語
#
# と記述することで、ライブラリをインポートできます。
# ライブラリが大きい場合は全てをインポートせずにfromを使い、特定のものだけをインポートすることもできます。
#
# from ライブラリ名 import 必要なライブラリ as 省略語

# ## ope

# 1. pandasをpdと省略してインポートしてみよう
# 1. numpyをnpと省略してインポートしてみよう
# 1. matplotlibは大きなライブラリの為、中のpyplotというものだけをインポートする為に、fromを使い、pltと省略してインポートしてみよう

# ## src

import ______ as __ 
import _____ as __ 
from __________ import ______ as ___ 
# %matplotlib inline

# ## ans

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
# %matplotlib inline

# ## T データ読み込み

# ## itr

# ### データの読み込み　穴埋め
# それでは早速pandasライブラリを使って、データの読み込み方を学びましょう。pandasにはread_csvという便利なコマンド（関数と言います）があります。関数はコマンドの後に必ずカッコが必要となります。
#
# `pd.read_csv()`
#
# 一方、read_csvのcsvとはファイル形式のことで、カンマで区切られたデータのことを言います。下記のように()の中にファイル名を記述することで、csvデータを読み込むことができます。下記では変数と呼ばれるデータの入れ物に”data”と名前をつけて読み込みをしています。”=“は代入といい、データを変数に入れる際に利用します。
#
# `data = pd.read_csv(“filename.csv”)`
#
# また、tsvというタブで区切られたデータを読み込みたい場合は、()の中にオプションとしてsep=“\t”と記載します。このオプションは引数と呼ばれます。複数の引数を記述する際には”,”を記載し、その後に追加のオプションを記述します。
#
# `data = pd.read_csv(“filename.tsv”,sep=“\t”)`
#

# ## ope

# - trainという名前の変数にpandasでデータ読み込んだ結果のDataFrameを代入してください。
# - testという名前の変数にpandasでデータ読み込んだ結果のDataFrameを代入してください。

# ## src

train = __.________("_______________", ___="__")
test = __.________("______________", ___="__")

# ## ans

train = pd.read_csv("./Downloads/train_maker (1).tsv", sep="\t")
f = pd.value_counts(train['maker name']) <=10
index = f[f == True].index.values
def change(x):
    if x in index:
        x = 'other'
    return x
train['maker name'] = train['maker name'].apply(change)

sns.lmplot(data=train,x="weight", y="mpg", ci=0,fit_reg=True)

# +
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression

only_numeric_horsepower = train["horsepower"][train["horsepower"] != "?"]
avg = only_numeric_horsepower.astype("float64").mean()
train["horsepower"] = train["horsepower"].apply(lambda x : avg if x == "?" else x)
train["horsepower"] = train["horsepower"].astype("float64")

target = train['mpg']
maker_name = pd.get_dummies(train['maker name'],drop_first=True)
tmp = pd.concat([train,maker_name],axis=1)

dummy_train = tmp.drop(columns=['id','mpg','car name','maker name'])
no_dummy_train = new_train.drop(columns=['chevrolet','dodge','ford','other'])

# +
X_train,X_test,y_train,y_test = train_test_split(no_dummy_train, target, random_state = 1)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_train)
score = np.sqrt(MSE(y_train,y_pred))
print(score)
y_train_df = pd.DataFrame(y_train)
df = y_train_df.assign(pred=y_pred)
df['diff'] = df['mpg']-df['pred']

def make_category(x):
    if x >=5:
        x = 'toyota'
    elif 0 <= x < 5:
        x = 'nissan'
    elif -5 < x < 0:
        x = 'honda'
    elif x <= -5:
        x = 'mazuda'
    return x
df['maker'] = df['diff'].apply(make_category)

y_pred = lr.predict(X_test)
score = np.sqrt(MSE(y_test,y_pred))
print(score)

y_test_df = pd.DataFrame(y_test)
df2 = y_test_df.assign(pred=y_pred)
df2['diff'] = df2['mpg']-df2['pred']

def make_category(x):
    if x >=5:
        x = 'toyota'
    elif 0 <= x < 5:
        x = 'nissan'
    elif -5 < x < 0:
        x = 'honda'
    elif x <= -5:
        x = 'mazuda'
    return x
df2['maker'] = df2['diff'].apply(make_category)
tmp = pd.concat([df,df2])
new_df = tmp.drop(columns=['mpg','pred','diff'])
new_maker = pd.concat([train,new_df],axis=1)
new_maker

# +
only_numeric_horsepower = new_maker["horsepower"][new_maker["horsepower"] != "?"]
avg = only_numeric_horsepower.astype("float64").mean()
new_maker["horsepower"] = new_maker["horsepower"].apply(lambda x : avg if x == "?" else x)
new_maker["horsepower"] = new_maker["horsepower"].astype("float64")
new_maker = new_maker.drop(columns=['car name', 'maker name'])
target = new_maker['mpg']
maker_name = pd.get_dummies(new_maker,drop_first=True)
# tmp = pd.concat([new_maker,maker_name],axis=1)
maker_name

# dummy_train = tmp.drop(columns=['id','mpg','car name','maker name','maker'])
# no_dummy_train = dummy_train.drop(columns=['mazuda','nissan','toyota'])
# tmp_csv= new_maker.drop(columns=['car name', 'maker name'])
# tmp_csv.to_csv('train_add_maker.csv')

# +
X_train,X_test,y_train,y_test = train_test_split(dummy_train, target, random_state = 1)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_train)
score = np.sqrt(MSE(y_train,y_pred))
print(score)

y_pred = lr.predict(X_test)
score = np.sqrt(MSE(y_test,y_pred))
print(score)
# -

# ## T データ量・次元数の確認

# ## itr

# ### データ量の確認(追加) N択
# これでDataFrameとしてデータを読み込めました。<br>
# データの数と特徴量の数を確認してみましょう。<br>
# `DataFrame.shape`を用いることで、`(サンプル数、特徴数)`で表示されます。
#

# ## ope

# - このtrainデータのサンプル数、特徴数の組み合わせとして正しいものを選んでください。
#     1. サンプル数:10 特徴数:200
#     2. サンプル数:10 特徴数:199
#     3. サンプル数:200 特徴数:10
#     4. サンプル数:199 特徴数:10

# ## ans

# 4

# ## T テーブルの可視化

# ## itr

# ### DataFrameの表示　穴埋め
# 内容を確認するため、読み込んだデータを表示してみましょう。<br>
# `DataFrame.head()` を用いてください。デフォルトでは先頭から5行目まで表示します。<br>
# 引数に数字を指定するとその数字の行数分表示されます。<br>
#
# - DataFrameのhead()を用いてデータをちょうど10行表示してください

# ## src

 train.____(__)

# ## ans

train.head(10)

# ## itr

# ### 変数のデータ型
#
# 数学における数に小数や整数、有理数や実数等の数値の型があるように、一般にプログラム言語で扱う変数ではデータ型と呼ばれる型が存在します。このデータ型を把握することはプログラムにおいてもデータサイエンスにおいても大変重要です。主なデータ型は下記の通りです。
#
# int型：整数
#
# float型：小数点
#
# bool型：真偽値（True or False）
#
# str型：文字列（但し、pandasではobjectと表示）
#
# pandasでは、各カラムがどんなデータ型であるかは`DataFrame.info()`関数を使うことで確認をすることができます。

# 1. 変数trainの中で数値型でないカラムを全て選択してみよう（複数選択）
#
# [select]
# - id
# - mpg
# - cylinders
# - displacement
# - horsepower
# - weight
# - acceleration
# - model year
# - origin
# - car name

train.info()

# ## itr

# ### メモリサイズの確認
# 先ほどの`info()`で表示される項目に`memory usage`というものがあります。これはデータのメモリ使用量を表しており、データのサンプル数(行数)に比例して大きくなります。

# ## ope

# - trainのメモリ使用量を選択してください。<br>
#
#     1. 15.6 B
#     2. 15.6 KB
#     3. 7.8 MB
#     4. 7.8 GB

# # M ターゲットの確認

# ##  T 分布確認

# ##  T データ選択(オプション)

# ## itr

# ### データを選択しましょう　穴埋め
# データの一部分のみを選択して表示するために、DataFrameから必要な部分だけを選択しましょう。<br>
# `DataFrame["列名"]` を用います。<br>
# DataFrame名の後に[]を書き、その中に列名を書くとその列をDataFrameから抽出してくれます。
# DataFrameから列を選ぶと、帰ってくるデータ型はSeriesというものになります。<br>
# DataFrameが横と縦に広がっているエクセルのような２次元データを扱うのに対して、Seriesは１列のみの１次元データを扱うものになります。<br>
# 基本的にはDataFrameとSeriesは似ていますがDataFrameにあってSeriesにはないメソッド等があるなどの違いがあります。<br>
# ここでは、違いを気にする必要なありません。  

# ## ope

# - DataFrameからmpg列を選んで表示してください。
# - 選択した列を格納した変数の型を表示してください。

# ## src

mpg = train[____]
print(mpg)
____(mpg)

# ## ans

mpg = train["mpg"]
print(mpg.head())
type(mpg)

# ##  T ターゲットの分布確認

# ## itr

# ### データの可視方法　穴埋め
# データがどのような分布になっているかをヒストグラムを用いて可視化しましょう。<br>
# データの分布をみることで特徴量の偏りや特性を知ることができ、効率的なモデリングが可能になります。<br>
# ヒストグラムを描くには`Series.plot.hist()`を用います。

# ## ope

# - mpgのヒストグラムを表示してください

# ## src

mpg._______

# ## ans

mpg.plot.hist()

# ## itr

# ヒストグラムの形状には色々な種類があり、形状により読み取れることが異なります。
# [4種類のヒストグラム画像]

# ## ope

# - mpgのヒストグラムから読み取れることとして正しいものを選択してください。
#     1. 外れ値がある
#     2. 平均値と中央値が異なる
#     3. 左右対称である
#     4. 2グループに分割できる

# ## ans

# 2

# # 使用カラムの選択

# ## T 欠損処理(オプション)

# ## itr

# ### データの欠損の有無を確認しましょう。　穴埋め
# 機械学習モデルを作成する際に、データが欠けていると正しく学習できません。  
# データが欠けているかどうかを分析を始める前に確認しましょう。  
# - `DataFrame.isnull()`と`sum()`を使って各カラムごとの欠損値の合計値を算出しましょう。

# ## src

# + {"code_folding": []}
train_na_mask = train.____
train_na_mask.____
# -

# ## ans

train_na_mask = train.isnull()
train_na_mask.sum()

# ## itr

# ### 説明変数の選択(知識ベース変数選択? オプション)

# 学習に使用する説明変数を選択しましょう。<br>
# 前述しましたが、idはmpgとは関係ないと思われるので、削除します。<br>
# car nameは、使用するのであれば、文字の変数（カテゴリ変数）なので数値に変換する必要があります。<br>
# これはダミー変数化を行うことで可能となりますが、今回は省略するのでこちらも削除しましょう。<br><br>
# 目的変数mpgは、説明変数部分からは削除し、別の変数に代入しておきます。
#
#
# - trainからmpg, id, car nameのカラムを削除してtraindfに代入してください。
# - targetに目的変数mpgを代入してください。

# ## src

# +

traindf = train.drop(columns = ['___', '__', '________'])
target = train['___']
# -

# ## ans

# +

only_numeric_horsepower = train["horsepower"][train["horsepower"] != "?"]
avg = only_numeric_horsepower.astype("float64").mean()
train["horsepower"] = train["horsepower"].apply(lambda x : avg if x == "?" else x)
train["horsepower"] = train["horsepower"].astype("float64")
traindf = train.drop(columns = ['mpg', 'id', 'car name'])
target = train['mpg']
# -

# ## itr

# ### データの分割
# モデルの精度をより正しく評価するために、説明変数および目的変数を学習用データと評価用データに分割しましょう。<br>
# データの分割には、`scikit-learn`の`train_test_split()`を用います。使い方は下記のようになり、訓練で使う説明変数と目的変数、評価で使うための説明変数と目的変数というように４つの変数を得ることが出来ます。<br>
#
# ```
# X_train,X_test,y_train,y_test = train_test_split(X,y)
# ```
#
# -　Xとyに適切な変数名を代入してください。

# ## ope

# traindfとtargetを学習用データと評価表データに分割してください。

# ## src

traindf['maker name'].value_counts()

maker_name = pd.get_dummies(traindf['maker name'])
traindf = pd.concat([traindf,maker_name],axis=1)
traindf = traindf.drop(columns=['maker name','buick','mercury','honda','pontiac','fiat','oldsmobile','opel','renault','mazda','volvo','audi','peugeot','mercedes-benz','subaru','saab','bmw','triumph','capri','hi'])
traindf

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = ______________

# ## ans

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(traindf, target, random_state = 1)

# =======================================================================================================

# # mission モデリング

# ## task01 評価関数定義

# # itr

# RMSEは実測値と予測値の差の2乗の平均をとって、ルート化したもので、以下の式で現されます。
# 画像
#

# ### ope

# - RMSEの説明として正しいものを選んでください。
#     
#     1. 負の値になることがある
#     2. 値が小さい方が良い
#     3. 値が大きい方が良い
#     4. 0になることはない
#
# - 次の実測値と予測値の時のRMSEを選んでください。
#     実測値: 4, 6, 8
#     予測値: 2, 8,10
#     
#     1. 1.0
#     2. 2.0
#     3. 2.8
#     4. 3.2

# ### ans

# 2<br>
# 2

# # itr

# 今回は評価指標としてRMSEを用います。<br>
# SIGNATE上の同練習問題の評価指標としてこれを採用しているためです。<br><br>
# RMSEは0以上の値をとり、精度が良いほど小さな値となります。<br>
# scikit-learnにはRMSEを算出するメソッドは無いので、一旦MSEを算出してから平方根を取り算出します。(平方根の取り方は後ほど説明します)<br>MSEを算出するメソッドをインポートして使えるようにしておきます。<br>
# `from sklearn.metrics import mean_squared_error`　<br>

# ### ope 記述

# - 実際にmean_squared_errorをインポートしてください。また、その際に、mean_squared_errorでは名前が長すぎるので、MSEとしてインポートしてください。

# +
# 
# -

# # ans

from sklearn.metrics import mean_squared_error as MSE

# =======================================================================================================

# # T モデリング手法Aでの学習と評価

# # itr

# ### 重回帰分析をしてみよう
# 今回は重回帰分析を行ってみましょう。<br>
# scikit-learnから線形モデルをインポートします。<br>
# 線形モデルは`sklearn.linear_model`から`LinearRegression`という名前でインポートできます。<br><br>
#
# モデルを使えるようにするには、`LinearRegression`を一度変数に代入する必要があります。<br>
# `変数名 = LinearRegression()`<br><br>
# 次に、モデルを訓練します。<br>
# `変数名.fit(X, y)`とするだけで訓練は完了します。

# ### ope 穴埋め

# - LinearRegressionをインポートしてください。
# - lrという変数にLinearRegressionを代入してください。
# - モデルの訓練を行なってください。訓練データには、さきほど標準化して得られたX_train_s, y_trainを使用します。

# +
# from ____ import ____

# lr = ____
# lr.fit(____, ____)
# -

# # ans

# +
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
train_s = ss.fit_transform(X_train)
test_s = ss.transform(X_test)

X_train_s = pd.DataFrame(train_s,columns=X_train.columns.values)
X_test_s = pd.DataFrame(test_s, columns=X_test.columns.values)
# -

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# # itr

# ### 訓練済みモデルの精度を評価しよう
#
# なんとかして線形回帰モデルを訓練することが出来ました。<br>
# では作ったモデルは良いものだったのでしょうか？ それとも悪いモデルだったのでしょうか？<br>
# それを判断するためにモデルがどれくらいの精度で正解を当てることが出来ているのかを調べていきます。<br>
# `y_pred = lr.predict(X)` とするとy_predに予測結果が代入されます。<br><br>
# 評価には、モデリングの冒頭でインポートしたMSEを使用します。<br>
# 最終的にはRMSE(MSEをルート化した値)を求めたいので、MSEを算出したのち、numpyのsqrtでルート化します。<br>
# `np.sqrt(MSE(実際のy,予測したy))`とすることでRMSEを算出することができます。<br><br>
# まずは、訓練データに対する予測精度を評価しましょう。

# ### ope 穴埋め

# - 訓練に使用したデータに対して予測するように空欄を埋めてください。
# - 実際のy(訓練に使用した目的変数)と予測結果RMSEを算出するように空欄を埋めてください。

# +
# y_pred = lr.predict(____)

# score = np.sqrt(MSE(____, ____))
# print(score)
# -

# # ans

y_pred = lr.predict(X_train)
score = np.sqrt(MSE(y_train,y_pred))
print(score)
# result = pd.DataFrame([y_train.values,y_pred]).T
# result.columns = ['train','pred']
# result.index = y_train.index
# result['diff'] = result['train']-result['pred']
# result['square'] = result['diff']**2
# result

# =======================================================================================================

# ## T 精度評価

# # itr

# ### 汎化性能を評価しよう
# 私達は今、モデルを訓練する際に用いたデータと同じデータでモデルの良し悪しを測っているので将来の未知のデータが来た際にどれだけモデルが対応できるかということが分かりません。<br>
# ですので、モデルの訓練には使用していない、モデル評価用のデータ（X_test_s, y_test）を用いてモデルの評価を行う必要があります。<br>
# このような未知のデータ（訓練には使用していないデータ）に対する予測性能を汎化性能といいます。

# ### ope 穴埋め

# - 評価用のデータに対する予測および精度評価を行なってください。コードとしては、訓練データに対する予測評価を行なったときとほぼ同じです。

# +
# y_pred = ____
# score = ____
# print(score)
# -

# # ans

y_pred = lr.predict(X_test)
score = np.sqrt(MSE(y_test,y_pred))
print(score)

# =======================================================================================================
