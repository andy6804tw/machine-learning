## 前言
Boosting 是希望能夠由後面生成的樹，來修正前面數學的不好的地方，這裏使用 Gradient 來修正錯誤的方式。

Gradient Boosting 的概念就是，我們先有一些訓練資料，接下來用這些資料建立一些比較簡單的決策樹，建出來之後會得到一個分數，接下來做第二顆樹的時候我們把表現不好的樣本的權重放大，並再用一樣的決策樹進行決策。最後會執行n棵樹後，每一棵樹都有各自的分數，我們將每一個分數乘上學習速率，最後將結果加總起來。這就是所謂的 Gradient Boosting。

### Bagging vs. Boosting
一般來說 Boosting 的模型會比 Bagging 來的精準。

- Bagging 透過抽樣的方式生成樹，每棵樹彼此獨立
- Boosting 透過序列的方式生成樹，後面生成的樹會與前一棵樹相關

 ## XGboost
XGboost 全名為eXtreme Gradient Boosting，是以 Gradient Boosting 為基礎下去實作，並添加一些新功能。可以說是結合 Bagging 和 Boosting 的優點，XGboost 保有 Gradient Boosting 的做法，每一棵樹是互相關聯的希望後面生成的樹能夠修正前面的樹犯錯的地方。此外 XGboost 是採用 Features sampling，和隨機森林一樣在生成每一棵樹的時候隨機採樣特徵並不會每次都拿全部的特徵下去做學習。此外為了讓模型不要變得太複雜，在目標函數加上正規化。因為在訓練時為了擬合模型，會產生很多高次項，但反而容易被雜訊干擾導致過度擬合，因此L1/L2 Regularization為讓Loss Function更平滑，抗雜訊干擾能力越大。最後 XGboost 還用到了一階導數和二階導數來生成下一棵樹，Gradient 就是所謂的一階導數，而 Hessian 即為二階導數。




- [How Gradient Boosting work](https://bradleyboehmke.github.io/HOML/gbm.html)