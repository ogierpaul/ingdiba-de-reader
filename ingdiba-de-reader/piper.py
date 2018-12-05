from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class ColumnSelector(TransformerMixin):
    def transform(self, X, y=None):
        usecols = ['VISA', 'amount', 'ispos', 'isround']
        X2 = X[usecols].astype(float).fillna(0)
        return X2

    def fit(self, X=None, y=None):
        return self


class OneHotCategory(TransformerMixin):
    def __init__(self, on):
        TransformerMixin.__init__(self)
        self.on = on
        self.le = LabelEncoder()
        self.oh = OneHotEncoder()

    def fit(self, X, y=None):
        a = X[self.on].fillna('nan')
        self.le = self.le.fit(a)
        b = self.le.transform(a).reshape(-1, 1)
        self.le.oh = self.oh.fit(b)
        return self

    def transform(self, X, y=None):
        a = self.oh.transform(self.le.transform(X[self.on].fillna('nan')).reshape(-1, 1))
        return a