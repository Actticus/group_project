import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt

def create_graph(ax, x, y, i, title):
    pl = ax[i // 2, i % 2]
    pl.set_title(title)
    if not i:
        pl.scatter(x, y, color="#FACA78", label="Входные данные")
        pl.plot(x, linear_predict, color="#4270AD", label="Линейное предсказание")
    else:
        pl.scatter(x, y, color="#FACA78")
        pl.plot(x, linear_predict, color="#4270AD")
    pl.tick_params(
        axis = 'both',
        which = 'major',
        direction = 'inout',
        length = 10,
        width = 1.5,
        pad = 10,
        bottom = True,
        left = True,
        labelbottom = True,
        labelleft = True,
    )
    pl.minorticks_on()
    pl.grid(which='both', color = '#828282', linestyle = '--')
    pl.set_xlabel("Доллар")
    pl.set_ylabel("Цена")
    pl.set_xscale("linear")



dt = "C:/Users/salty/Desktop"
dollar = np.array(pd.read_excel(f"{dt}/2.xls").values)
goods = np.array(pd.read_excel(f"{dt}/cen-god.xls").values)
poly = PolynomialFeatures(degree=7, include_bias=False)
years = [i for i in range(2001, 2021)]
f = '{:,.3f}'


for i, rows in enumerate(goods):
    if i % 4 == 0:
        fig, ax = plt.subplots(2, 2)
    model = linear_model.LinearRegression()
    model.fit(dollar, goods[i, 1:])
    x = dollar.reshape((1, -1)).tolist()[0]
    r_sq = model.score(dollar, goods[i, 1:])
    linear_predict = model.predict(dollar)
    print(f"{f.format(model.coef_[0])}*x{'+' if model.intercept_ > 0 else ''}{f.format(model.intercept_)}")
    # rmse = np.sqrt(mean_squared_error(goods[i, 1:],linear_predict))    
    # print(rmse)
    
    create_graph(ax, dollar, goods[i, 1:], i % 4, goods[i, 0])

plt.show()
