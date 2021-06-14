import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
df=pd.read_csv('heart.csv')
df.head()
df.info()
scaler = StandardScaler()
scaler.fit(df)
df
from sklearn.model_selection import train_test_split
y=df['target']
x=df.drop('target',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,test_size=0.2)
x_train.head()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score,precision_score,classification_report,roc_auc_score,roc_curve
# search for optimun parameters using gridsearch
params = {'penalty':['l1','l2'],
         'C':[0.01,0.1,1,10,100],
         'class_weight':['balanced',None]}
logistic_clf = GridSearchCV(LogisticRegression(),param_grid=params,cv=10)
#train the classifier
logistic_clf.fit(x_train,y_train)
logistic_clf.best_params_
import matplotlib.pyplot as plt
cm=confusion_matrix(y_test,logistic_predict)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
#knn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
# search for optimun parameters using gridsearch
params= {'n_neighbors': np.arange(1, 10)}
grid_search = GridSearchCV(estimator = KNeighborsClassifier(), param_grid = params,
                           scoring = 'accuracy', cv = 10, n_jobs = -1)
knn_clf = GridSearchCV(KNeighborsClassifier(),params,cv=3, n_jobs=-1)
# train the model
knn_clf.fit(x_train,y_train)
knn_clf.best_params_
# predictions
knn_predict = knn_clf.predict(x_test)
#accuracy
knn_accuracy = accuracy_score(y_test,knn_predict)
print(f"Using k-nearest neighbours we get an accuracy of {round(knn_accuracy*100,2)}%")
cm=confusion_matrix(y_test,knn_predict)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
#decision tree
from sklearn.tree import DecisionTreeClassifier
dtree= DecisionTreeClassifier(random_state=7)
# grid search for optimum parameters
params = {'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15],
          'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11]}
tree_clf = GridSearchCV(dtree, param_grid=params, n_jobs=-1)
# train the model
tree_clf.fit(x_train,y_train)
tree_clf.best_params_
# predictions
tree_predict = tree_clf.predict(x_test)
#accuracy
tree_accuracy = accuracy_score(y_test,tree_predict)
print(f"Using Decision Trees we get an accuracy of {round(tree_accuracy*100,2)}%")
cm=confusion_matrix(y_test,tree_predict)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
#svm

from sklearn.svm import SVC

#grid search for optimum parameters
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
svm_clf = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=10)

# train the model
svm_clf.fit(x_train,y_train)
svm_clf.best_params_

# predictions
svm_predict = svm_clf.predict(x_test)
#accuracy
svm_accuracy = accuracy_score(y_test,svm_predict)
print(f"Using SVM we get an accuracy of {round(svm_accuracy*100,2)}%")
cm=confusion_matrix(y_test,svm_predict)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
#rf classifier
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(max_depth=20,max_features=7)
rfc.fit(x_train,y_train)
rfc_predict=rfc.predict(x_test)
rfc_accuracy = accuracy_score(y_test,rfc_predict)
print(f"Using rfc we get an accuracy of {round(rfc_accuracy*100,2)}%")
cm=confusion_matrix(y_test,svm_predict)
conf_matrix=pd.DataFrame(data=cm,columns=['NO attack','Attack'],index=['NO attack','Attack'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
#nb classifier
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x_train,y_train)
clf_predict=clf.predict(x_test)
clf_accuracy = accuracy_score(y_test,clf_predict)
print(f"Using gnb we get an accuracy of {round(clf_accuracy*100,2)}%")
cm=confusion_matrix(y_test,svm_predict)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
svm_percision=precision_score(y_test,svm_predict)
logistic_percision=precision_score(y_test,logistic_predict)
knn_percision=precision_score(y_test,knn_predict)
tree_percision=precision_score(y_test,tree_predict)
clf_percision=precision_score(y_test,clf_predict)
rfc_percision=precision_score(y_test,rfc_predict)
svm_recall=recall_score(y_test,svm_predict)
logistic_recall=recall_score(y_test,logistic_predict)
knn_recall=recall_score(y_test,knn_predict)
tree_recall=recall_score(y_test,tree_predict)
clf_recall=recall_score(y_test,clf_predict)
rfc_recall=recall_score(y_test,rfc_predict)
svm_f1 = f1_score(y_test, svm_predict)
print(f'The f1 score for SVM is {round(svm_f1*100,2)}%')
logistic_f1 = f1_score(y_test, logistic_predict)
print(f'The f1 score for Logistic is {round(logistic_f1*100,2)}%')
knn_f1 = f1_score(y_test, knn_predict)
print(f'The f1 score for KNN is {round(knn_f1*100,2)}%')
tree_f1 = f1_score(y_test, tree_predict)
print(f'The f1 score for tree is {round(tree_f1*100,2)}%')
clf_f1 = f1_score(y_test, clf_predict)
print(f'The f1 score for rfc is {round(clf_f1*100,2)}%')
rfc_f1 = f1_score(y_test, rfc_predict)
print(f'The f1 score for SVM is {round(rfc_f1*100,2)}%')
comparison = pd.DataFrame({
    "Logistic regression":{'Accuracy':log_accuracy, 'percision':logistic_percision,'recall':logistic_recall,'F1 score':logistic_f1},
    "K-nearest neighbours":{'Accuracy':knn_accuracy,'percision':knn_percision,'recall':knn_recall,'F1 score':knn_f1},
    "Decision trees":{'Accuracy':tree_accuracy,'percision':tree_percision,'recall':tree_recall,'F1 score':tree_f1},
    "Support vector machine":{'Accuracy':svm_accuracy,'percision':svm_percision,'recall':svm_recall,'F1 score':svm_f1},
    "Gaussian NB":{'Accuracy':clf_accuracy,'percision':clf_percision,'recall':clf_recall,'F1 score':clf_f1},
    "Random Forest":{'Accuracy':rfc_accuracy,'percision':rfc_percision,'recall':rfc_recall,'F1 score':rfc_f1}
}).T
fig = plt.gcf()
fig.set_size_inches(15, 15)
titles = ['Accuracy','percision','recall','F1 score']
for title,label in enumerate(comparison.columns):
    plt.subplot(2,2,title+1)
    sns.barplot(x=comparison.index, y = comparison[label], data=comparison)
    plt.xticks(fontsize=10)
    plt.title(titles[title])
plt.show()
comparison = pd.DataFrame({
    "Logistic regression":{'Accuracy':log_accuracy, 'percision':logistic_percision,'recall':logistic_recall,'F1 score':logistic_f1},
    "K-nearest neighbours":{'Accuracy':knn_accuracy,'percision':knn_percision,'recall':knn_recall,'F1 score':knn_f1},
    "Decision trees":{'Accuracy':tree_accuracy,'percision':tree_percision,'recall':tree_recall,'F1 score':tree_f1},
    "Support vector machine":{'Accuracy':svm_accuracy,'percision':svm_percision,'recall':svm_recall,'F1 score':svm_f1},
    "Gaussian NB":{'Accuracy':clf_accuracy,'percision':clf_percision,'recall':clf_recall,'F1 score':clf_f1},
    "Random Forest":{'Accuracy':rfc_accuracy,'percision':rfc_percision,'recall':rfc_recall,'F1 score':rfc_f1}
})
comparison.head()
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
#plt.bar(['Logistic Regression','Decision Tree','SVM','Gaussian NB','Boosted Random Forest'],[f1_lr,f1_dtc,f1_svm,f1_gnb,f1],color=['red','green','purple','orange','Blue'])
plt.plot(['Logistic Regression','Decision Tree','SVM','Gaussian NB','KNN','RFC'],[logistic_f1,tree_f1,svm_f1,clf_f1,knn_f1,rfc_f1],color='purple',marker='D')
plt.plot(['Logistic Regression','Decision Tree','SVM','Gaussian NB','KNN','RFC'],[log_accuracy,tree_accuracy,svm_accuracy,clf_accuracy,knn_accuracy,rfc_accuracy],color='red',marker='^')
plt.plot(['Logistic Regression','Decision Tree','SVM','Gaussian NB','KNN','RFC'],[logistic_percision,tree_percision,svm_percision,clf_percision,knn_percision,rfc_percision],color='blue',marker='s')
plt.plot(['Logistic Regression','Decision Tree','SVM','Gaussian NB','KNN','RFC'],[logistic_recall,tree_recall,svm_recall,clf_recall,knn_recall,rfc_recall],color='green',marker='P')
plt.legend(('F1 Score','Accuracy','Precision','Recall'))
plt.title('Comparison of various models\' performance')

plt.show(fig)
