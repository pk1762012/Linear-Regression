# Execution of linear regression function will be called from here
import pandas as pd

from LinearReg import LinearRegOls

IceCream = pd.read_csv("IceCreamData.csv")

y = IceCream['Revenue']
X = IceCream[['Temperature', 'Wind']]

#y = [1,3,4,5,2,3,4,8]
#X = [(1,2,3,4,5,6,7,8), (2,3,4,5,7,8,9,12)]


regols = LinearRegOls(X, y, constant= False)
regols.runall()