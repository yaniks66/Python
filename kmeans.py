from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
import pandas as pd
from sklearn.svm import LinearSVR
from sklearn import metrics
from sklearn import preprocessing
import seaborn as sns
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

n_neighbors_value = range(1,7,1)
scores_data = pd.DataFrame()
for n in n_neighbors_value:
    clf = KNeighborsRegressor(n_neighbors = n)
    clf = clf.fit(x_train,y_train)
    temp_score_data = pd.DataFrame({'n_neighbors': [n],
                                    'train_score': [clf.score(x_train,y_train)],
                                    'test_score': [clf.score(x_test,y_test)]})
    scores_data = scores_data.append(temp_score_data)
ax=sns.lineplot(x = 'n_neighbors', y = 'train_score',data = scores_data)
ax=sns.lineplot(x = 'n_neighbors', y = 'test_score',data = scores_data)

print(scores_data.loc[scores_data.test_score == max(scores_data.test_score)])

#clf = KNeighborsRegressor(n_neighbors = 5)
#clf = clf.fit(x_train,y_train)

#print('MSE on trenning set: {:.3f}'.format(clf.score(x_train, y_train)))
#print('MSE on test set: {:.3f}'.format(clf.score(x_test, y_test)))
