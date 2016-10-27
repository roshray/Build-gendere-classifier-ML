
#Predicting Best model among DTS,NaiveBayes,KNN and SVM

from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


#[height, weight, shoe_size]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)
prediction1 = clf.predict(X)

clf1 = GaussianNB()
clf1= clf1.fit(X,Y)
prediction2 = clf1.predict(X)


clf2 = KNeighborsClassifier()
clf2 = clf2.fit(X,Y)
prediction3 = clf2.predict(X)


clf3 = SVC()
clf3 = clf3.fit(X,Y)
prediction4 = clf3.predict(X)

#prepare models 

Y_true = Y

result = accuracy_score(Y_true , prediction1)
result1 = accuracy_score(Y_true , prediction2)
result2 = accuracy_score(Y_true , prediction3)
result3 = accuracy_score(Y_true , prediction4)


best_one = max(result,result1, result2, result3)
print(best_one)

classifier_names = {'tree':result, 'Gaussian':result1, 'KNeighborsClassifier':result2, 'SVC':result3}  

best_classifiers = [name for name in classifier_names if classifier_names[name] == best_one]


print("Best one:")
for best_one_model in best_classifiers: print(" "+best_one_model)








