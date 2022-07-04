from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

""""
========================== categoricalEncoderOperator ======================
"""
class categoricalEncoderOperator(TransformerMixin, BaseEstimator):
    """
    Auto: Preng Biba
    Version: 1.0.0
    Descripción: Operador de ingeniería de caracteristicas para codificar las variables categóricas.
    """

    def __init__(self, varNames):
        self.varNames = varNames
    
    def fit(self, X, y=None):
        self.mapper = {}
        for varname in self.varNames:
            self.mapper[varname] = X[varname].value_counts().to_dict()
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for varname in self.varNames:
            X[varname] = X[varname].map(self.mapper[varname])
        return X

""""
========================== targetEncoderOperator ======================
"""
class targetEncoderOperator(TransformerMixin, BaseEstimator):
 
    def __init__(self, targetName):
        self.targetName = targetName

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y):
        y = pd.get_dummies(y, drop_first=True)
        return y

    