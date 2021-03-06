from sklearn import neighbors
import pandas

filename = 'iris.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(filename)
print data.head()
featured_cols = ['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']
x = data.loc[:,featured_cols]
y = data.Species
print(y.shape)

knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn.fit(x,y)
x_pred = [3, 5, 4, 2]
res = knn.predict([x_pred, ])
print "Printing Res"
print res

#print data.Species
print(knn.predict_proba([x_pred, ]))
