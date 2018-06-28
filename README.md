# 2018NSYSUBigData
## Project
* #### **題目**
> ###### 檢視SVM、LSTM及XGboost在預測台灣20家公司股價漲跌的表現
[Stock Trend Prediction: Based on Machine Learning Methods](https://cloudfront.escholarship.org/dist/prd/content/qt0cp1x8th/qt0cp1x8th.pdf?t=p63wi3)
###### -
* #### **組員**
> ###### 林羿豪、羅宸旻、林惠雯
###### -
* #### **動機**
> ###### 　　在變化多端的股票市場，我們想了解究竟是否能利用過去的歷史資訊，透過機器學習技術達到對未來股價有效的預測，並探究何種參數、機器起學習模型能有最有效的預測力，而在尋找參考資料的同時我們找到一篇相關論文是出自Song, Yuan學者，當中他取20支美股之各參數，運用了LSTM、GRU、XGBoost、SVM四種機器學習模型做預測股價，我們試著將此篇論文的想法運用在台股上探究其成效為何。
###### -
* #### **計畫摘要**
> ###### <table><tr><td bgcolor=#7FFFD4>1.選擇20支台股股票，其中包含</td></tr></table>
> ###### 　　台積電(2330)、鴻海(2317)、大立光(3008)、台塑(1301)、中華電(2412)
> ###### 　　南亞(1303)、台化(1326)、國泰金(2882)、台達電(2308)、聯發科(2454)
> ###### 　　中信金(2891)、富邦金(2881)、中鋼(2002)、統一(1216)、兆豐金(2886)
> ###### 　　可成(2474)、華碩(2357)、第一金(2892)、台灣大(3045)、南亞科(2408)
> ###### <table><tr><td bgcolor=#7FFFD4>2.取用論文中的參數包含</td></tr></table>
> ###### 　　開盤價、最高價、最低價、收盤價、成交量、收盤價(調整)、日報酬率(調整)、RSI、ADX、SAR
> ###### 　　以上參數皆用過去五日來預測當日漲跌，例如:第6天漲跌是透過過去5日之上述全部參數所預測
> ###### <table><tr><td bgcolor=#7FFFD4>3.採用用論文中的機器學習模型包含</td></tr></table>
> ###### 　　LSTM、GRU、XGBoost、SVM
> ###### <table><tr><td bgcolor=#7FFFD4>4.總結出各預測力與原論文之結果比較</td></tr></table>
> ###### 　　　＜以下為原論文之預測力＞　　　　　　　　　　＜以下為實作台股之預測力＞
> ###### 　　![img](https://i.imgur.com/YVPFhYT.png) 　　![img](https://i.imgur.com/fObOwiN.png)
* #### **模型介紹**
> ###### *XGBoost*
> ###### XGBoost的全稱為eXtreme Gradient Boosting，是GBDT的一種高效實現，XGBoost中的基學習器除了可以是CART（gbtree）也可以是線性分類器（gblinear）。
> ###### 什麼是GBDT？
> ###### GBDT(Gradient Boosting Decision Tree) 又叫 MART（Multiple Additive Regression Tree)，是一種疊代的決策樹算法，該算法由多棵決策樹組成，所有樹的結論累加起來做最終答案。它在被提出之初就和SVM一起被認為是泛化能力（generalization)較強的算法。GBDT的核心在於，每一棵樹學的是之前所有樹結論和的殘差，這個殘差就是一個加預測值後能得真實值的累加量。與隨機森林不同，隨機森林採用多數投票輸出結果；而GBDT則是將所有結果累加起來，或者加權累加起來。
> ###### XGBoost對GBDT的改進
> ###### 　　　1 . 避免過擬合
> ###### 　　　目標函數之外加上了正則化項整體求最優解，用以權衡目標函數的下降和模型的複雜程度，避免過擬合。基學習為CART時，正則化項與樹的葉子節點的> ###### 　　　數量T和葉子節點的值有關。
> ###### 　　
> ###### 　　　2 . 二階的泰勒展開，精度更高
> ###### 　　　不同於傳統的GBDT只利用了一階的導數信息的方式，XGBoost對損失函數做了二階的泰勒展開，精度更高。
> ###### 　　　第t次的損失函數：
> ###### 　　　對上式做二階泰勒展開( g為一階導數，h為二階導數)：
> ###### 　　
> ###### 　　　3 . 樹節點分裂優化
> ###### 　　　選擇候選分割點針對GBDT進行了多個優化。正常的樹節點分裂時公式如下：
> ###### 　　　XGBoost樹節點分裂時，雖然也是通過計算分裂後的某種值減去分裂前的某種值，從而得到增益。但是相比GBDT，它做了如下改進：
> ###### 　　　通過添加閾值gamma進行了剪枝來限制樹的生成
> ###### 　　　通過添加係數lambda對葉子節點的值做了平滑，防止過擬合。
> ###### 　　　在尋找最佳分割點時，考慮傳統的枚舉每個特徵的所有可能分割點的貪心法效率太低，XGBoost實現了一種近似的算法，即：根據百分位法列舉幾個可> 
> ###### 　　　能成為
> ###### 　　　分割點的候選者，然後從候選者中根據上面求分割點的公式計算找出最佳的分割點。
> ###### 　　　特徵列排序後以塊的形式存儲在內存中，在疊代中可以重複使用；雖然boosting算法疊代必須串行，但是在處理每個特徵列時可以做到並行。
> ###### *LSTM*
> ###### Long Short Term Memory 網路，通常稱為 LSTMs，是一個特殊的RNN，能夠學習 Long-term 依賴問題。由 Hochreiter 和 Schmidhuber 在 1997年首先提出，近幾年被很多學者優化，並廣泛應用。 在理解LSTM網路的工作原理前，我們先看下標準的RNN是如何工作的，如下圖所示，它首先是結構重複的單元組成，每個單元僅有一層 tanh層組成，將xt和ht-1聯合加權並經過tanh啟用函式輸出到下一個時序，並且每個單元的隱藏狀態Ct 與 ht相等。
> ###### 　　![img](https://i.imgur.com/jVHOMy7.png)
* #### **程式碼**
> ###### XGBoost
```python
importimport  osos
mingw_pathmingw_pat  = 'C:\\Program Files\\mingw-w64\\x86_64-7.3.0-posix-seh-rt_v5-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
```
```python
import xgboost as xgb
import pandas as pd
import numpy as np
from datetime import datetime
import talib
```
```python
讀資料、加變數
df = pd.read_csv("stockdata.csv",encoding = 'big5')
df.columns=['code','date','open','high','low','close','volume',"adjclose", "return"]
df["return"] = df["return"]/100
df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d').dt.strftime("%Y-%m-%d");
df2330 = df[df.loc[:, "code"] == 2330]
df2330.loc[:,'RSI'] = talib.RSI(df2330['close'].values.astype('float64'))
df2330["ADX"] = talib.ADX(df2330['high'].values, df2330['low'].values, df2330['close'].values, timeperiod = 14)
df2330["SAR"] = talib.SAR(df2330['high'].values, df2330['low'].values, acceleration=0.2)
```python
加入變數
for i in range(5,0,-1):
    df2330["open"+str(i)] = df2330["open"].shift(i)
    df2330["high"+str(i)] = df2330["high"].shift(i)
    df2330['low'+str(i)] = df2330['low'].shift(i)
    df2330["close"+str(i)] = df2330["close"].shift(i)
    df2330["volume"+str(i)] = df2330["volume"].shift(i)
    df2330["adjclose"+str(i)] = df2330["adjclose"].shift(i)
    df2330["return"+str(i)] = df2330["return"].shift(i)
    df2330["RSI"+str(i)] = df2330["RSI"].shift(i)
    df2330["ADX"+str(i)] = df2330["ADX"].shift(i)
    df2330["SAR"+str(i)] = df2330["SAR"].shift(i)
```
```python
df2330.head()
df2330 = df2330.dropna()
from sklearn import model_selection, ensemble, preprocessing, metrics
from xgboost import XGBClassifier
xgbc = XGBClassifier()
df2330['label'] = (df2330.close - df2330.close.shift(1)) > 0
X = df2330[['open1', 'high1', 'low1', 'close1', 'volume1', "RSI1", "ADX1", "SAR1", "adjclose1", "return1",
               'open2', 'high2', 'low2', 'close2', 'volume2', "RSI2", "ADX2", "SAR2", "adjclose2", "return2",
               'open3', 'high3', 'low3', 'close3', 'volume3', "RSI3", "ADX3", "SAR3", "adjclose3", "return3",
               'open4', 'high4', 'low4', 'close4', 'volume4', "RSI4", "ADX4", "SAR4", "adjclose4", "return4",
               'open5', 'high5', 'low5', 'close5', 'volume5', "RSI5", "ADX5", "SAR5", "adjclose5", "return5"]]
y = df2330['label']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
xgbc.fit(X_train, y_train)
print('The accuracy of eXtreme Gradient Boosting Classifier on testing set', xgbc.score(X_test, y_test))
```
> ###### LSTM
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
import scipy
```
```python
df2330 = df[df.loc[:, "code"] == 2330]
df2330.loc[:,'RSI'] = talib.RSI(df2330['close'].values.astype('float64'))
df2330["ADX"] = talib.ADX(df2330['high'].values, df2330['low'].values, df2330['close'].values, timeperiod = 14)
df2330["SAR"] = talib.SAR(df2330['high'].values, df2330['low'].values, acceleration=0.2)

for i in range(5,0,-1):
    df2330["open"+str(i)] = df2330["open"].shift(i)
    df2330["high"+str(i)] = df2330["high"].shift(i)
    df2330['low'+str(i)] = df2330['low'].shift(i)
    df2330["close"+str(i)] = df2330["close"].shift(i)
    df2330["volume"+str(i)] = df2330["volume"].shift(i)
    df2330["adjclose"+str(i)] = df2330["adjclose"].shift(i)
    df2330["return"+str(i)] = df2330["return"].shift(i)
    df2330["RSI"+str(i)] = df2330["RSI"].shift(i)
    df2330["ADX"+str(i)] = df2330["ADX"].shift(i)
    df2330["SAR"+str(i)] = df2330["SAR"].shift(i)
    
df2330['label'] = (df2330.close - df2330.close.shift(1)) > 0
```
```python
df2330.drop(df.columns[[0,1]], axis=1, inplace=True)
df2330.head()
df2330 = df2330.dropna()
```
```python
讀資料
def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix() #pd.DataFrame(stock)
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [x_train, y_train, x_test, y_test]
```
```python
建立LSTM
def build_model2(layers):
        d = 0.2
        model = Sequential()
        model.add(LSTM(50, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(Dropout(d))
        model.add(LSTM(80, input_shape=(layers[1], layers[0]), return_sequences=False))
        model.add(Dropout(d))
        model.add(Dense(50,activation='relu'))
        model.add(Dense(1,init='uniform',activation='sigmoid'))
        model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
        return model
```
```python
window = 1
X_train, y_train, X_test, y_test = load_data(df2330[::-1], window)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)
```
```python
model = build_model2([61,window,1])
model.fit(
    X_train,
    y_train,
    batch_size=80,
    nb_epoch=50,
    validation_split=0.2,
    verbose=0)
score = model.evaluate(X_test,y_test,verbose=2)
print('The accuracy of eXtreme Gradient Boosting Classifier on testing set:')
score[1]
ratio=[]
p = model.predict_classes(X_test)
for u in range(len(y_test)):
    pr = p[u][0]
    ratio.append((y_test[u]/pr)-1)
    diff.append(abs(y_test[u]- pr))
import matplotlib.pyplot as plt2
plt2.plot(p,color='red', label='prediction')
plt2.plot(y_test,color='blue', label='real')
plt2.legend(loc='upper left')
plt2.show()
```
![img](https://i.imgur.com/5k0T0Cu.png)
> ###### SVM
```python
import numpy as np
import pandas as pd
import talib as ta
import pandas_datareader as web
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
```
```python
stockdata_df = pd.read_csv("stockdata.csv")
stockdata_df.columns = ['Code', 'Date', 'Open', 'High', 'Low', 'Close' ,'Volume', 'CloseA', 'ReturnA']
stock2330_df = stockdata_df.query('Code == 2')
stock2330_dfstock2330 ['RSI'] = ta.RSI(stock2330_df['Close'].values.astype('float64'), 14)
stock2330_df['ADX'] = ta.ADX(stock2330_df['High'].values.astype('float64'), stock2330_df['Low'].values.astype('float64'), stock2330_df['Close'].values.astype('float64'), timeperiod = 10 )
stock2330_df['SAR'] = ta.SAR(stock2330_df['High'].values.astype('float64'), stock2330_df['Low'].values.astype('float64'))
stock2330_df['label'] = (stock2330_df.Close - stock2330_df.Close.shift(1)) > 0
```
```python
stock2330_df['Open(-1)'] = stock2330_df['Open'].shift(+1)
stock2330_df['High(-1)'] = stock2330_df['High'].shift(+1)
stock2330_df['Low(-1)'] = stock2330_df['Low'].shift(+1)
stock2330_df['Close(-1)'] = stock2330_df['Close'].shift(+1)
stock2330_df['Volume(-1)'] = stock2330_df['Volume'].shift(+1)
stock2330_df['CloseA(-1)'] = stock2330_df['CloseA'].shift(+1)
stock2330_df['ReturnA(-1)'] = stock2330_df['ReturnA'].shift(+1)
stock2330_df['RSI(-1)'] = stock2330_df['RSI'].shift(+1)
stock2330_df['ADX(-1)'] = stock2330_df['ADX'].shift(+1)
stock2330_df['SAR(-1)'] = stock2330_df['SAR'].shift(+1)

stock2330_df['Open(-2)'] = stock2330_df['Open'].shift(+2)
stock2330_df['High(-2)'] = stock2330_df['High'].shift(+2)
stock2330_df['Low(-2)'] = stock2330_df['Low'].shift(+2)
stock2330_df['Close(-2)'] = stock2330_df['Close'].shift(+2)
stock2330_df['Volume(-2)'] = stock2330_df['Volume'].shift(+2)
stock2330_df['CloseA(-2)'] = stock2330_df['CloseA'].shift(+2)
stock2330_df['ReturnA(-2)'] = stock2330_df['ReturnA'].shift(+2)
stock2330_df['RSI(-2)'] = stock2330_df['RSI'].shift(+2)
stock2330_df['ADX(-2)'] = stock2330_df['ADX'].shift(+2)
stock2330_df['SAR(-2)'] = stock2330_df['SAR'].shift(+2)

stock2330_df['Open(-3)'] = stock2330_df['Open'].shift(+3)
stock2330_df['High(-3)'] = stock2330_df['High'].shift(+3)
stock2330_df['Low(-3)'] = stock2330_df['Low'].shift(+3)
stock2330_df['Close(-3)'] = stock2330_df['Close'].shift(+3)
stock2330_df['Volume(-3)'] = stock2330_df['Volume'].shift(+3)
stock2330_df['CloseA(-3)'] = stock2330_df['CloseA'].shift(+3)
stock2330_df['ReturnA(-3)'] = stock2330_df['ReturnA'].shift(+3)
stock2330_df['RSI(-3)'] = stock2330_df['RSI'].shift(+3)
stock2330_df['ADX(-3)'] = stock2330_df['ADX'].shift(+3)
stock2330_df['SAR(-3)'] = stock2330_df['SAR'].shift(+3)

stock2330_df['Open(-4)'] = stock2330_df['Open'].shift(+4)
stock2330_df['High(-4)'] = stock2330_df['High'].shift(+4)
stock2330_df['Low(-4)'] = stock2330_df['Low'].shift(+4)
stock2330_df['Close(-4)'] = stock2330_df['Close'].shift(+4)
stock2330_df['Volume(-4)'] = stock2330_df['Volume'].shift(+4)
stock2330_df['CloseA(-4)'] = stock2330_df['CloseA'].shift(+4)
stock2330_df['ReturnA(-4)'] = stock2330_df['ReturnA'].shift(+4)
stock2330_df['RSI(-4)'] = stock2330_df['RSI'].shift(+4)
stock2330_df['ADX(-4)'] = stock2330_df['ADX'].shift(+4)
stock2330_df['SAR(-4)'] = stock2330_df['SAR'].shift(+4)

stock2330_df['Open(-5)'] = stock2330_df['Open'].shift(+5)
stock2330_df['High(-5)'] = stock2330_df['High'].shift(+5)
stock2330_df['Low(-5)'] = stock2330_df['Low'].shift(+5)
stock2330_df['Close(-5)'] = stock2330_df['Close'].shift(+5)
stock2330_df['Volume(-5)'] = stock2330_df['Volume'].shift(+5)
stock2330_df['CloseA(-5)'] = stock2330_df['CloseA'].shift(+5)
stock2330_df['ReturnA(-5)'] = stock2330_df['ReturnA'].shift(+5)
stock2330_df['RSI(-5)'] = stock2330_df['RSI'].shift(+5)
stock2330_df['ADX(-5)'] = stock2330_df['ADX'].shift(+5)
stock2330_df['SAR(-5)'] = stock2330_df['SAR'].shift(+5)
```
```python
stock2330_df.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(stock2330_df[['Open(-1)', 'High(-1)', 'Low(-1)', 'Close(-1)' ,'Volume(-1)', 'CloseA(-1)', 'ReturnA(-1)', 'RSI(-1)', 'ADX(-1)', 'SAR(-1)', 'Open(-2)', 'High(-2)', 'Low(-2)', 'Close(-2)' ,'Volume(-2)', 'CloseA(-2)', 'ReturnA(-2)', 'RSI(-2)', 'ADX(-2)', 'SAR(-2)', 'Open(-3)', 'High(-3)', 'Low(-3)', 'Close(-3)' ,'Volume(-3)', 'CloseA(-3)', 'ReturnA(-3)', 'RSI(-3)', 'ADX(-3)', 'SAR(-3)', 'Open(-4)', 'High(-4)', 'Low(-4)', 'Close(-4)' ,'Volume(-4)', 'CloseA(-4)', 'ReturnA(-4)', 'RSI(-4)', 'ADX(-4)', 'SAR(-4)', 'Open(-5)', 'High(-5)', 'Low(-5)', 'Close(-5)' ,'Volume(-5)', 'CloseA(-5)', 'ReturnA(-5)', 'RSI(-5)', 'ADX(-5)', 'SAR(-5)']], stock2330_df[['label']], test_size=0.3, random_state=0)
```
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
from sklearn.svm import SVC
clf = SVC()
clf.fit(X_train, y_train)
survived_predict = clf.predict(X_test)
from sklearn import model_selection, ensemble, preprocessing, metrics
accuracy = metrics.accuracy_score(y_test, survived_predict)
#print(accuracy)
fpr, tpr, thresholds = metrics.roc_curve(y_test, survived_predict)
auc = metrics.auc(fpr, tpr)
#print(auc)
print('準確率: {}'.format(auc))
print('AUC值: {}'.format(accuracy))
```
* #### **結論**
> ###### 從我們的結果來看，LSTM如同論文裡為表現最好的方法，但就三種方法來說，整體表現都不太理想，我們認為可能的原因:
> ###### 　　(1)不同市場的差異
> ###### 　　(2)因只挑選20家公司，可能會有偏誤
> ###### 　　(3)模型參數調整不一致
> ###### 　　(4)變數處理過程出問題
* #### **待改進**
> ###### 　　(1)依產業分類，分析是否不同產業有其較適合的方法與參數
> ###### 　　(2)更深入了解模型參數的意義
> ###### 　　(3)由20間公司擴增為全市場的股票
###### -
* #### **參考資訊**
> ###### [台積電隔日股價走勢線上預測](https://github.com/ChenHandsomeboy/Team_Project/tree/master)
> ###### [DEEP STOCK REPRESENTATION LEARNING: FROM CANDLESTICK CHARTS TO INVESTMENT DECISIONS](https://arxiv.org/pdf/1709.03803.pdf)
> ###### [量化投資精選](https://community.bigquant.com/t/%E9%87%8F%E5%8C%96%E7%A0%94%E7%A9%B6%E6%AF%8F%E5%91%A8%E7%B2%BE%E9%80%89-20170929/2821)
> ###### [Stock Trend Prediction: Based on Machine Learning Methods](https://cloudfront.escholarship.org/dist/prd/content/qt0cp1x8th/qt0cp1x8th.pdf?t=p63wi3)ㄒ
