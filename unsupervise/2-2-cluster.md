# Clustering
Clustering指的是只給定資料點的的特徵，我們要設法把資料點分成幾群。最基本的想法是把附近的點都放在一起或是把相似的點放在一起。分為幾群是要依靠領域知識或是任務需要。

我們要把相近的點分成同一群，因此定義遠近是很重要的事情。最常用的方式是euclidean distance(歐幾里得距離)，如果小就代表是相近的點，反之。兩個set的距離計算使用(Jaccard distance)。兩個vector之間的距離計算使用(cosine distance)。

一但我們定義好兩點之間的距離後，我們就可以用Cluster演算法來做分類。

## K-means 
1. 定義群的個數k
2. 隨機產生出k個點作為k個群的中心
3. 看過所有資料點並且找出每個資料點與哪一個中心最近
4. 一但知道每個資料點屬於哪一個集群之後，我們可以知道屬於同個群的中心應該在哪
5. 跳回3直到迴圈結束或收斂

![](https://i.imgur.com/TOFkBpR.png)

> example: k-means for segmentation

## Hierarchical clustering
階層式集群 (Hierarchical clustering)可以輕鬆的調整群的數量。

-  Agglomerative (bottom-up)
根據點與點間的距離找出哪兩個是相近的把發分為同一群，直到所有點都變成同一群為止
- Divisive (top-down)
假設一開始所有點都為同一群，每次都分為兩群直到每一點都屬於一群(依賴其他分群演算法)


## Clustering vs classification
兩者常被搞混，Clustering是一個非監督式學習的演算法，只根據特徵就把資料分群。
classification是監督式學習的其中一種方法，每一個特徵都有相對應的target標籤，所以是根據特徵和標籤支籤的關西，想辦法創造出一個model未來透過輸入特徵就能馬上分辨出種類。