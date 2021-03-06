# Unsupervised Learning
只給feature x 想辦法會從中找出pattern，例如從feature中找出dimensions。Dimension Reduction，顧名思義，就是原本的Data寫在一個比較高的維度作標上，我們希望找到一個低維度的作標來描述它，但又不能失去Data本身的特質。如何找出最具有代表性的feature或者是具有代表性的feature做完transform之後的結果。目標是根據feature就把資料樣本分為幾群這是一個clustering的問題。非監督式學習是透過feature做出判斷，或是pattern的萃取，這一類的問題都屬於非監督式學習的範疇。

## Dimension Reduction
假設一開始有d個特徵，經過降為後有k個特徵。如果降維後的特徵可以大部分的表示原來d個維度的特徵這就是一個好的降維。

## 為什麼要做降維?
如果我們能把一些資料做壓縮，又能夠保持資料原來的的性質。因此我們可以用比較少的空間，或是計算時用比較少的資源，就可以得到跟沒有做資料壓縮之前得到相似的結果。
此外做資料降維可以幫助資料視覺化，二維可以用xy平面圖表示，三維可以用xyz立體圖作表示，而大於三維的空間難以視覺化做呈現。這時如果我們可以將多維的資料投影到一個二維的平面上，而且投影的結果是有保持原本資料點跟點資料的關西這就是一個成功的降維。我們用比較少的空間去儲存大部分的資訊，當然會有一些損失的問題。

## Principal component analysis(PCA)
PCA的目的是把高維的點頭影到低維的空間上，並且低維度的空間保有高維空間中大部分的性質。性質指的是點與點之間的關係。另外從高維轉低維的過程中PCA只允許做線性的轉換。

降維投影方式有很多種，我們希望在投影後的variance是最大的。也就是說分部是最散的。variance大的意思是，在低維空間當中每一個的點與其他點之間關西還是能夠輕易的辨認出來不會全部擠在一塊。

### PCA的前置步驟
1. 先求出所有資料點中心µ，也就是將每一個資料點做平均
2. 將每一個資料點減去µ，這步驟是將資料點平移，平移後原點是所有點的中心(1`2步是讓mean=0)
3. 計算每一個feature的variance
4. 把每一個值都除以variance(重新把每一個feature做scaleing使得大家都共有unit variance)

### Finding u1
接著要找一個向量u1，我們希望把原始的x投影到u1之後variance是最大的，xi點投影到u1方向後其座標。因此所有投影過的點平方做平均及為variance

## T-Distributed Stochastic Neighbor Embedding (t-SNE)
目標跟PCA是一樣的，希望把高維的資料投影到低維中，並且保留高維中的點與點之間的關係與特性。兩者不同的點在於t-SNE允許非線性的專換，可能一維是平方除以2另一維是開根號除以10。t-SNE後原本相近的點依然相近，反之原本距離遠的投影後依然保持遠的距離。原本PCA只允許線性的轉換，因此會映射到一條線使得varience最大，所以本來應該要分開的兩群就混雜再一起了。相對的t-SNE允許非線性的轉換，因此有機會在原本分開的三群在做完投影後依然是開的。

![](https://i.imgur.com/nBNzGd7.png)

評估方式使得PQ的kl divergence要最小，kl divergence是衡量兩個之間分佈的距離，降維前後的任兩點距離的probability distribution越相近越好。

![](https://i.imgur.com/kiXNBq2.png)


高維和低維使用不同的 similarity measures，其原因是在高維若距離較遠時若使用相同的similarity measures話最後的低維的投影結果反而會跛較近，換句話說在高維分部看起來分佈在不同群的點投影到低維之後有可能會混雜在一起。(crowding problem)因此分群的效果會比較明顯。

![](https://i.imgur.com/rIklBR8.png)

## 總結
PCA和t-SNE是兩個不同降維的方法，PCA的優點在於簡單若新的點要映射時直接代入公式即可得出降維後的點。若t-SNE有新的點近來時我們沒有去計算新的點和舊的點之間的關係因此我們無法將新的點投影下去。t-SNE的優點是可以保留原本高維距離較遠的點降維後依然保持遠的距離，因此這些群降維後依然保持群的特性。
- PCA允許線性的轉換
- t-SNE允許非線性的轉換