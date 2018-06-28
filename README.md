# 2018NSYSUBigData
## Project
* #### **題目**
> ###### 檢視SVM、LSTM及XGboost在預測台灣20家公司股價漲跌的表現
[Stock Trend Prediction: Based on Machine Learning Methods]
(https://cloudfront.escholarship.org/dist/prd/content/qt0cp1x8th/qt0cp1x8th.pdf?t=p63wi3)
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
> ###### [Stock Trend Prediction: Based on Machine Learning Methods](https://cloudfront.escholarship.org/dist/prd/content/qt0cp1x8th/qt0cp1x8th.pdf?t=p63wi3)
