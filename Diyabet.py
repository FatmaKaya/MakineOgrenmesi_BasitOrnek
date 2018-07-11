
# Kütüphaneleri yüklüyoruz 
import pandas as pd
from sklearn import model_selection
# Sınıflandırma Modellerine Ait Kütüphaneler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Dataseti yüklüyoruz
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe=pd.read_csv(url,names=names)

# Dataset içeriğindeki veri(satır) ve öznitelik(sütun)
print("Dataset veri(satır) ve öznitelik(sütun):"+str(dataframe.shape))

# Verilerin ilk 10 satırını görmek için 	
print("Dataset içeriği:\n",dataframe.head(10))

# Datasetin eğitim ve test verisi olarak ayrıştırılması 
array=dataframe.values
X=array[:,0:8]
Y=array[:,8]
test_size=0.33
seed=7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)


models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('K-NN', KNeighborsClassifier()))
models.append(('Decision Tree (CART)', DecisionTreeClassifier()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('SVM', SVC()))

# Modelleri test edelim
max_model=0
print("\nHer modelin ACC (Accuracy / Doğruluk) ölçütüne göre başarıları:")
for name, model in models:
    model=model.fit(X_train, Y_train)
    Y_pred= model.predict(X_test)
    from sklearn import metrics
    m_accuracy_score=(metrics.accuracy_score(Y_test, Y_pred)*100)
    print("Model -> %s -> ACC: %%%.2f" % (name,m_accuracy_score))
    if m_accuracy_score > max_model:
        max_model=metrics.accuracy_score(Y_test, Y_pred)*100
        max_model_name=name
    
#En iyi sonucu veren model
print("\nEn iyi ACC sonucunu veren model : " )
print("Model -> %s -> ACC: %%%.2f" % ( max_model_name, max_model))