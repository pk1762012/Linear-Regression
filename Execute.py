# Execution of linear regression function will be called from here
import pandas as pd

from LinearReg import LinearRegOls

y = [1,3,4,5,2,3,4]
X = [1,2,3,4,5,6,7]


regols = LinearRegOls(X, y)
regols.runall()