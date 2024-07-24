import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv("homeprices.csv")
print(df)

# Here we can see that at 2 there is NaN in bedrooms. to fix that we take the median of all and put that into that places for solve this problem.

df.bedrooms.fillna(df.bedrooms.median(), inplace=True)
print(df)

reg = linear_model.LinearRegression()
reg.fit(df.drop('price',axis='columns'),df.price)

print(reg.coef_)
print(reg.intercept_)

print(reg.predict([[3000, 3, 40]]))
print(reg.predict([[2500, 4, 5]]))
print(reg.predict([[3000, 3, 3]]))