# Ensemble learning
Ensemble learning試圖以一個系統化的方式將好幾個監督式學習的模型結合在一起，目的是希望結合眾多的模型產生一個更強大的模型。在許多科學競賽中Ensemble learning在實務上是非常有效的提升預測準確率。

為什麼Ensemble learning強大？假如我們有好幾個彼此獨立而且歧異性夠高的模型，這些模型我們稱為base learner。他們統合之後的結果通常比較精準，原因是因為每一個base learner有各自的偏見但是也有各自合理的地方。合理的地方會互相加強，各自偏見的地方會互相抵銷掉。

更具體的舉例來說假設我們有三個獨立且歧異性夠高的二元分類得演算法，每一個準確率都0.6。假設我們想將這三個模型整合在一起，整合的方式是三個裡面果有兩個以上都認為是某個類別，最後預測的結果就是該類別。此種方法是採多數決的方式。

依照Ensemble的方式，我們可以分為三類。第一類為Bagging，第二類為Boosting，第三類為Stacking。Bagging指的是我們把訓練資料重新採樣產生不同組的訓練資料，根據不同組的訓練資料即使我們用同一種演算法我們也會得到不一樣的模型，他的樹是各自獨立因此可以平行化處理。Boosting則會根據每一筆訓練資料的難或簡單給予不同的權重，首先我們會訓練一個base learner然後根據base learner預測的結果對或錯來分辨該筆資料是一個簡單還是困難的資料。對於難的資料我們加強他的權重在訓練一個新的分類器或回歸器，我們目標是希望再訓練後新的模型在這些難的資料能夠表現得更好。我們不斷重複這些步驟，不斷地加入新的base learner，且新的base learner把過去表現不好的地方改善，這就是Boosting精神。因此Boosting的每一棵樹是乎相有關聯性的做完第一棵樹可能進行下一棵樹的生成。最後一個是Stacking，假設我們有好幾個base learner，每一個base learner都有判斷解決問題的能力。Stacking會把這些base learner的輸出當成是一個新的模型的輸入，最後再重新訓練一個模型來預測最終結果。

- Bagging
- Boosting
- Stacking

### 1. Bagging (resample training data)
Bagging 是 Bootstrap aggregating 的縮寫，Bootstrap 指的是假設我們有n筆資料我們從中抽取n’筆資料出來，這n’筆資料是可以被重複抽取的。假設我們有一萬筆資料我們要從中抽取100筆資料出來，這100筆資料裡面可能會有重複的數據，這意味著同一筆資料可能會重複抽到好幾次，這種概念即為 Bootstrap。Bootstrap aggregating 就是重複 Bootstrap 步驟重複 m 次，做完之後我們會有 m 組的訓練資料，每一組訓練資料內都有 n’ 筆資料。最後我們會根據這 m 組的訓練資料，每一組都會去訓練出一個模型，就算我們使用同一個訓練的演算法，因為訓練資料不同因此我們會得到 m 個不同的模型。這 m 個模型每一個都是微弱的learner，最後合併在一起就會形成一個比較強的learner了
#### 1.1 Random forest
隨機森林其實就是進階版的決策樹，這句話簡單說明，就是很多哭棵樹加起來可以變成一座森林。隨機森林是使用 Bagging + randomizes freature set 的技術所產生出來的 Ensemble learning 演算法。隨機森林就是由很多的樹集合在一起，每一棵樹都是各自的決策樹。然而每一棵樹的訓練資料是經由 Bagging + randomizes freature set 所產生的。所謂的 Bagging 是每一棵樹的從原始資料中(random sample with replacement)取用放回的方式隨機的產生訓練資料。除此之外我們要求每一棵樹只能看見部分的特徵，換句話說假設我們有一萬筆資料，每筆資料有100個特徵，每一棵樹只能從這一萬筆資料透過 Bagging 的方式隨機拿出n筆資料。除此之外這n筆資料可能只隨機挑選k個特徵做樣本。因此在隨機森林每一棵樹的特徵數量可能都不同，所以最後決策出來的結果都會不一樣。最後再根據任務的不同來做回歸或是分類的問題，如果是回歸問題我們就將這些決策數的輸出做平均得到最後答案，若是分類問題我們則用投標採多數決的方式來整合所有樹預測的結果。

每一棵樹在生成的過程中，可能用到不同的訓練資料和不一樣數量的特徵。會用到哪些訓練資料及特徵都是由隨機決定。

因為我們要訓練很多樹，因此不管在訓練或是預測的時候都會要更多的運算資源，不過因為在隨機森林每一棵樹都是獨立的，所以不管是在訓練或是預測的階段每一棵樹都能平行化的運行。

![](https://i.imgur.com/v2Sm3rB.png)

#### 1.2 Bagging 小結
Bagging 是在訓練資料上動手腳，即使我們用同一個監督式學習的演算法，會因為訓練集的不同產生不一樣的模型。那如何產生不一樣的訓練資料？我們是用 Bagging 的方式 random sample with replacement 產生好幾組不一樣的訓練資料。每一組訓練資料會產生出一個模型，最後再將好幾個決策樹的模型整合在一起得到最終的預測。


### 2. Boosting (reweight training data + weighted models)
相較於 Bagging 而 Boosting 有下面幾個特色。第一個是每個的模型重要程度是不一樣的，當我們要整合最後的模型時每個模型的權重是不一樣的。第二個在訓練的時候每一筆的資料權重也是不一樣的，有些資料可能在某一迭代的權重是比較高的，而在其他迭代中該筆資料權重可能是比較低的。

![](https://i.imgur.com/kZPcFvJ.png)

#### 2.1 AdaBoost
AdaBoost 是 Adaptive Boosting 的縮寫。需要人工的給定 hyperparameter (K)，也就是希望能產生出K個 base learners。當某筆資料在這一回合被猜錯了，代表下一回合他的權重比較大會較重要，想辦法讓機器學會。這邊有兩個權重第一個為模型權重代表他預測有多準(a)，另一個是筆一筆訓練資料他在這一回合有多重要(w)。

```
yi=a1f1(xi)+a2f2(xi)+...+akfk(xi)

wi代表某筆資料在該回合的重要程度，若猜錯wi重要程度會比較大代表下一回合 要讓他學會
a代表此模型的權重 err越大a越小代表此模型參考價值不高

後來的learner會針對前面learner做得不好的部分去做補強
```

#### 2.2 Gradient Boosting
Gradient Boosting 是 AdaBoost 的延伸。Gradient Boosting 一樣也有一個 hyperparameter (K)，也就是 base learners 的數量。最後的預測即為將這k個learner的預測加總在一起。在Gradient Boosting中後來的learner會針對前面learner做得不好的部分去做補強，但並不像 AdaBoost 一樣增加那些先前預測不對的樣本權重。

Gradient Boosting 的做法是先訓練出一個模型F=f1接著在訓練一個f2模型，這個f2模型是要預測的目標是實際上的y減去上一個f1(x)預測的結果[y-f(x)]，也就是說第一個模型預測的結果跟實際的答案中間的誤差，第二個模型就是要來預測這個誤差有多少。接下來F=f1+f2，由於第二個模型是是預測第一個模型與實際答案的差距，因此f1+f2的結果應該會比f1的結果更接近真實答案。最後重複上述步驟形成f3、f4...最後的結果即爲將這些learner相加一起為最後的答案。每一個f都是簡單很淺的決策樹因此不會有過度擬合問題

```
F(xi)=f1(xi)+f2(xi)+...
```

#### 2.3 Gradient Boosting vs decision tree
決策樹通常為一個複雜的樹，而在 Gradient Boosting 是產生非常多棵的樹但是每一哭棵的樹都很簡單，希望新的樹可以針對舊的樹預測不太好的部分做一些補強。所以最終我們要把這麼多簡單的樹合再一起才能當最後的預測，所以他是一個 ensemble of the tree

#### 2.4 Gradient Boosting vs random forest
隨機森林也是產生很多棵樹，每一棵也不是很複雜，最後也是將所有的樹 ensemble 在一起，兩者差別在於隨機森林彼此的樹是獨立的並無明確的關係。但是Boosting後面的樹都是希望彌補前面的樹不足的部分，所以這些樹一棵一棵的產生的，我們必須先產生第一棵樹然後第二棵樹在針對第一棵樹不足的地方去做一些補強，第三棵樹再根據前兩棵樹不好的地方在做補強。因為隨機森林的每一棵樹是獨立的因此在產生樹的過程是可以平行化的，預測時也可以各自做各自的最後再合併一起。Gradient Boosting在訓練時由於後面的樹會根據前面的樹不好的地方做補強所以我們只能一棵樹一棵樹單獨的產生。

### 3. Stacking
首先產生出m個 base learners(模型)彼此間並互相無關連，例如第一個learner為 logistic regression第二個為決策樹。訓練完m個模型後，我們要把這m個模型合併在一起。合併的方式是我們另外再訓練一個模型，這個模型把ｍ個base learner的輸出當成新的模型的輸入因此我們會根據這m個特徵利用集成式學習其中的演算法來學習一個模型並預測最終結果。最後的 Ensemble model 可以使用線性回歸或是深度神經網路來做學習。總之 Ensemble model 的目的是把每一個 base learners 的輸出當成線索，並把這些線索想辦法做整合來得到最終的答案。另外在進行 Stacking 的時候要注意的是在訓練 m 個 base learners 的訓練資料和訓練 Ensemble model 的資料兩者的訓練資料要不同。


- blending weak learners

## Ensemble learning 小結
Ensemble learning 通常可以讓 base learner 的表現變好，這一點在許多實戰上被驗證了。Ensemble learning有三種類型第一種為 Bagging，第二種為 Boosting，第三種為 Stacking。Bagging 的技術是將訓練資料重新採樣，代表的方法是隨機森林。隨機森林除了 Bagging 之外，還有另一個隨機的因素是每一棵樹都只能看到一部分的特徵，這些特徵是由隨機決定的。Boosting 是一次產生一個新的模型，新的模型的目標是要補強舊的模型表現不好的部分，代表的方法有 AdaBoost 與 Gradient Boosting。 Stacking 指的是我們另外利用一個機器學習的模型來決定各個 base learns 要用什麼的方式做融合。