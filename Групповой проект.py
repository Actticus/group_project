import pandas as pd
import scipy
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
        pad = 2,
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


dollar = np.array(pd.read_excel(f"./1.xls").values)
goods = np.array(pd.read_excel(f"./2.xls").values)
poly = PolynomialFeatures(degree=7, include_bias=False)
f = '{:,.2f}'

name_list = [] 
eq_list = []
fisher_f_list = []
fisher_f_krit_list = []
rmse_list = []
r_sq_list = []

for i, rows in enumerate(goods):
    if i % 4 == 0:
        fig, ax = plt.subplots(2, 2)
    name = goods[i, 0]
    data = goods[i, 1:]
    model = linear_model.LinearRegression()
    model.fit(dollar, goods[i, 1:])
    x = dollar.reshape((1, -1)).tolist()[0]
    r_sq = model.score(dollar, goods[i, 1:])
    linear_predict = model.predict(dollar)
    eq = f"{f.format(model.coef_[0])}*x{'+' if model.intercept_ > 0 else ''}{f.format(model.intercept_)}"
    fisher_f = (r_sq / (1 - r_sq)) * (20 - 2)
    fisher_f_krit = scipy.stats.f.ppf(q = 1 - 0.05, dfn = 1, dfd = 20)
    rmse = np.sqrt(mean_squared_error(goods[i, 1:],linear_predict)) / (max(goods[i, 1:]) - min(goods[i, 1:])) * 100

    create_graph(ax, dollar, data, i % 4, name)
    name_list.append(name)
    eq_list.append(eq)
    fisher_f_list.append(fisher_f)
    fisher_f_krit_list.append(fisher_f_krit)
    rmse_list.append(rmse)
    r_sq_list.append(r_sq)
    fig.tight_layout()
    fig.legend()

df = pd.DataFrame({
        "Имя": name_list,
        "Уравнение": eq_list,
        "Наблюдаемое значение критерия Фишера": fisher_f_list,
        "Критическое значение критерия Фишера": fisher_f_krit_list,
        "Коэффициент детерминации": r_sq_list,
        "Остаточная дисперсия": rmse_list,
    })
print(df)
df.to_excel("Вывод.xls")
plt.show()
