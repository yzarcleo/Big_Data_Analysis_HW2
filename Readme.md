巨量資料分析第二次作業
請以XGBoost與Gradient Boosting對於Microsoft Malware 2015 Dataset進行分析，透過參數組合分析，選擇最適合的參數組合.

經測試較佳參數選擇如下，正確率為99.41%
 n_estimators=1000,
 max_depth=5,
 min_child_weight=5,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27

本篇時做主要參考http://blog.csdn.net/u010657489/article/details/51952785進行實做。
文章提到booster参数有以下可以進行調教:
1、eta[默認0.3]典型值為0.01-0.2。
2、min_child_weight[默認1]，決定最小葉子節點樣本權重和。
3、max_depth[默認6]，max_depth越大，模型會學到更具體更局部的樣本。典型值：3-10
4、max_leaf_nodes，樹上最大的節點或葉子的數量。
5、gamma[默認0]。Gamma指定了節點分裂所需的最小損失函數下降值。
6、max_delta_step[默認0]，這參數限制每棵樹權重改變的最大步長。這個參數一般用不到。
7、subsample[默認1]這個參數控制對於每棵樹，隨機採樣的比例。如果這個值設置得過小，它可能會導致欠擬合。典型值：0.5-1
8、colsample_bytree[默認1]用來控制每棵隨機採樣的列數的占比(每一列是一個特徵)。
典型值：0.5-1
9、colsample_bylevel[默認1]用來控制樹的每一級的每一次分裂，對列數的採樣的占比。
10、lambda[默認1]權重的L2正則化項。這個參數是用來控制XGBoost的正則化部分的。
11、alpha[默認1]權重的L1正則化項。(和Lasso regression類似)。可以應用在很高維度的情況下，使得演算法的速度更快。
12、scale_pos_weight[默認1]在各類別樣本十分不平衡時，把這個參數設定為一個正值，可以使演算法更快收斂。

以下為我嘗試測試過程與代碼
測試:使用預設XGBClassifier()針對資料集進行測試，正確率達99.39%

# use feature importance for feature selection
from numpy import loadtxt
from numpy import sort
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

# load data
Info=pd.read_csv("G:\\LargeTrain.csv")
# split data into X and y
Y = Info.pop('Class')
X = Info
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
# fit model on all training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data and evaluate
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Fit model using each importance as a threshold
thresholds = sort(model.feature_importances_)
Accuracy: 99.39%


依據文章測試過程可區分以下進行參數測試調教
第一步：確定學習速率和tree_based 參數調優的估計器數目
第二步： max_depth 和 min_weight 參數調優
第三步：gamma參數調優
第四步：調整subsample 和 colsample_bytree 參數
第五步：正則化參數調優
第6步：降低學習速率

為了開始進行boosting 參數效能調教，先給參數一個初始值。依照文章建議如下方法取值： 
1、max_depth = 5 :這個參數的取值最好在3-10之間。先選的起始值為5(4-6之間可)。 
2、min_child_weight = 1。 
3、gamma = 0: 起始值也可以選其它比較小的值，在0.1到0.2之間就可以。
4、subsample,colsample_bytree = 0.8(典型值的範圍在0.5-0.9之間)。 
5、scale_pos_weight = 1: 這個值是因為類別十分不平衡。 
以上只是初始值，因為後續需要調優。這裡把學習速率就設成預設的0.1。然後用xgboost中的cv函數來確定最佳的決策樹數量。


1.使用初始值進行測試，正確率為99.33%
model = XGBClassifier(learning_rate =0.1,
 n_estimators=1000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
 model.fit(X_train, y_train)



2.調整max_depth=5, min_child_weight=5後正確為99.41%
 model = XGBClassifier(learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=5,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
 model.fit(X_train, y_train)
 
3.調整n_estimators參數到5000，準確率為99.39%
 model = XGBClassifier(learning_rate =0.01,
 n_estimators=5000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
 model.fit(X_train, y_train)

4.調整subsample 和 colsample_bytree 參數均為0.6，準確率為99.28%
 model = XGBClassifier(learning_rate =0.1,
 n_estimators=1000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.6,
 colsample_bytree=0.6,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)



5.調整gamma參數到5，程式出現error，失敗
C:\ProgramData\Anaconda3\lib\site-packages\pandas\core\ops.py:792: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
  result = getattr(x, name)(y)
