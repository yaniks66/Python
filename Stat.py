import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split

file = pd.read_csv('auto_ru.csv', delimiter = ',',engine='python' )   
file = file.rename(columns = {'Марка':'Mark','Модель':'Model','Мощность':'Power',
                              'Топливо':'Fuel', 'Цена':'Price','Год производства':'Year',
                              'Пробег':'Mileage', 'Коробка':'Transmission',
                              'Привод':'Drive', 'Кузов':'Body', 'Цвет':'Color'})
file = file.dropna(axis='index', how='any', subset=['Color'])
file = file.loc[file.Price > 10000]
file.Power, file.Price, file.Mileage, file.Year = file.Power.astype(int),file.Price.astype(int), file.Mileage.astype(int), file.Year.astype(int)

x = pd.get_dummies(file.drop(columns = 'Price'))
y = file.Price

#Функция разделения выборки не тест и трейн 30/70%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)
clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 20)
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print('Accuracy on trenning set: {:.3f}'.format(clf.score(x_train, y_train)))
print('Accuracy on test set: {:.3f}'.format(clf.score(x_test, y_test)))
