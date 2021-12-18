# Linear regression class
from turtle import heading

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import cdb
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smd
from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import coint
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from termcolor import colored
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings('ignore')


class LinearRegOls:

    def __init__(self, X, y, constant = True, HAC = True, OOS_begin = None, OOS_end = None):
        self.X = X
        self.y = y
        self.constant = constant
        self.HAC = HAC
        self.model = None
        self.exog = None
        self.model_OOS = None
        self.OOS_begin = OOS_begin
        self.OOS_end = OOS_end

    def regressionexecute(self):

        if self.constant:
            self.exog = sm.add_constant(self.X)
        else:
            self.exog = self.X
        if not self.HAC:
            self.model = sm.OLS(self.y, self.exog).fit()
        else:
            self.model = sm.OLS(self.y, self.exog).fit(cov_type='HAC',cov_kwds={'maxlags':6})
        pass

    def regressionsummary(self):
        #heading('Summary of Regression')
        self.regressionexecute()
        print(self.model.summary())

    def adftests(self):
        self.regressionexecute()
        #heading()
        #print(pd.DataFrame(self.model.resid).describe())
        #print(self.model.resid)
        #heading('Residual plot')

        pd.DataFrame(self.model.resid).plot()
        plt.show()
        result = adfuller(self.model.resid, autolag='AIC')
        print(f'ADF Statistic: {result[0]}')
        print(f'n_lags: {result[1]}')
        print(f'p-value: {result[1]}')
        for key, value in result[4].items():
            print('Critical Values:')
            print(f'   {key}, {value}')


    def residualsummary(self):
        self.regressionexecute()
        #heading('Summary statistics of residuals')
        #pd.DataFrame(self.model.resid).describe()
        #heading('Residual plot')
        #print()
        pd.DataFrame(self.model.resid).plot()
        plt.show()
        sm.graphics.tsa.plot_acf(self.model.resid, lags = 3)
        sm.graphics.tsa.plot_pacf(self.model.resid, lags = 3)
        plt.show()

    def bptest(self):
        self.regressionexecute()
        #heading('Breusch-Pagan Test')
        print(smd.het_breuschpagan(self.model.resid, self.model.model.exog))

    def vif(self):
        self.regressionexecute()
        #heading('Test results of VIF')
        vif = pd.DataFrame()
        if not self.constant:
            vif_X = sm.add_constant(self.X)
            vif['Variables'] = vif_X.columns
            vif['VIF'] = [variance_inflation_factor(vif_X.values, i) for i in range(vif_X.shape[1])]
        else:
            vif['Variables'] = self.exog.columns
            vif['VIF'] = [variance_inflation_factor(self.exog.values, i) for i in range(self.exog.shape[1])]
        print(vif)

    def insample(self):
        self.regressionexecute()
        #heading('In sample fit')
        ActvsPred = pd.DataFrame({'Actual': self.y.squeeze(), 'Predicted': self.model.predict(self.exog)})
        ActvsPred[['Actual', 'Predicted']].plot(figsize = (10,5))
        plt.show()
        InSampleRMSE = np.sqrt(((ActvsPred['Actual'] - ActvsPred['Predicted'])**2).mean())
        print(colored(f'In-sample RMSE is {InSampleRMSE:.2f}.', attrs=['bold']))
        print()

    def outsample(self):
        #heading(f'Out of Sample test: {self.OOS_begin} to {self.OOS_end}')
        self.regressionexecute()
        if (self.OOS_begin is None or self.OOS_end is None):
            X_Train = self.exog.iloc[:-12]
            X_test = self.exog.iloc[12:]
            y_train = self.y.iloc[:-12]
            y_test = self.y.iloc[-12:]
        else:
            X_Train = self.exog.iloc[:self.OOS_begin]
            X_test = self.exog.iloc[self.OOS_begin:self.OOS_end]
            y_train = self.y.iloc[:self.OOS_begin]
            y_test = self.y.iloc[self.OOS_begin:self.OOS_end]

        if not self.HAC:
            self.model_OOS = sm.OLS(y_train,X_Train).fit()
        else:
            self.model_OOS = sm.OLS(y_train, X_Train).fit(cov_type='HAC',cov_kwds={'maxlags':6})

        #heading('Out sample fit')


        ActvsPred = pd.DataFrame({'Actual':y_test.squeeze(),'Predicted':self.model.predict(X_test)})
        ActvsPred[['Actual','Predicted']].plot(figsize = (10,5))
        plt.show()
        OutSampleRMSE = np.sqrt(((ActvsPred['Actual'] - ActvsPred['Predicted'])**2).mean())
        print(colored(f'Out sample RMSE is {OutSampleRMSE: .2f}.',attrs=['bold']))
        print()

    def runall(self):
        self.regressionsummary()
        self.adftests()
        self.residualsummary()
        self.bptest()
        self.vif()
        self.insample()
        self.outsample()
        pass

    def Forecast(self, data):

        '''
        :param data: Dataframe containing the regressors
        :return: Series containing projections using the input data
        '''

        self.regressionexecute()
        if self.constant:
            data = sm.add_constant(data)
        forecast = self.model.predict(data)
        return forecast










