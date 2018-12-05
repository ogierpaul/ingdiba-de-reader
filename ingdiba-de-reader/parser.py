import re
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_union, make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


def _extract_vendor(field):
    """
    extract valuable information from the vendor field of ING DIBA
    Args:
        field (str): content
    Returns:
        dict: {'VISA':bool, 'vendornew':str}
    Examples:
        >> extract_vendor('VISA Spotify Ltd')
        {'VISA': True, 'vendornew': 'Spotify Ltd'}
    """
    features = {}
    if field is not None:
        res = re.search(pattern=r"^\b{}\b".format('VISA'), string=field)
        if res is not None:
            features['VISA'] = True
            field = field[4:].strip()
        features['vendornew'] = field
    return features


def _extract_description(field, creditcards=None):
    """
    Extract information from the description field
    Args:
        field (str): field description
        creditcards: list of credit cards ['NRXXXXXXX', ..]
    Returns:
        {
            'creditcard': 'NRXXXXXXX',
            'lastschrift_type': 'KAUFUMSATZ'|'BARGELDAUSZAHLUNG' | 'WECHSELKURSGEBUEHR' | 'Rechnung' |'Beitrag'
            'location': location as appears on ING diba
            'transaction_day': 04
            'transaction_month': 11
            'timeofday': 115534
            'desc_rest': rest
        }
    """
    features = {}
    if pd.isnull(field) or field == 'nan':
        return features
    if creditcards is not None:
        for cd in creditcards:
            res = re.search(pattern=r"^\b{}\b".format(cd), string=field)
            if res is not None:
                features['creditcard'] = field[res.start():res.end()].strip()
                field = re.sub(pattern=r"^\b{}\b".format(cd), repl='', string=field)
    res = re.search(
        pattern=r"\b{}\b |\b{}\b | \b{}\b | \b{}\b".format(
            'KAUFUMSATZ', 'BARGELDAUSZAHLUNG', 'WECHSELKURSGEBUEHR', 'Rechnung', 'Beitrag'
        ),
        string=field,
        flags=re.IGNORECASE)
    if res is not None:
        features['lastschrift_type'] = field[res.start(): res.end()]
        if features['lastschrift_type'] in ['KAUFUMSATZ', 'BARGELDAUSZAHLUNG']:
            features['location'] = field[:res.start()].lstrip().rstrip()
            field = field[res.end():].lstrip().rstrip()
            if len(field.split(' ')[0]) == 5:
                features['transaction_day'], features['transaction_month'] = field.split(' ')[0].split('.')
                field = field[5:].strip()
                features['timeofday'] = field[:6]
                field = field[6:].strip()
    features['desc_rest'] = field
    return features


def _extract_amount(r):
    features = {}
    features['ispos'] = r > 0
    features['isround'] = (r % 10 == 0)
    return features


def _preparedf(df):
    ve = df['vendor'].apply(_extract_vendor).apply(pd.Series)
    de = df['description'].apply(_extract_description).apply(pd.Series)
    ae = df['amount'].apply(_extract_amount).apply(pd.Series)
    df2 = df.join(ve).join(de).join(ae)
    return df2


class _ColumnSelector(TransformerMixin):
    def transform(self, X, y=None):
        usecols = ['VISA', 'amount', 'ispos', 'isround']
        X2 = X[usecols].astype(float).fillna(0)
        return X2

    def fit(self, X=None, y=None):
        return self


class _OneHotCategory(TransformerMixin):
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


class INGParser(TransformerMixin):
    def transform(self, X, y=None):
        return _preparedf(X)

    def fit(self, X=None, y=None):
        return self


tfcols = ['vendornew', 'desc_rest']
streams = []
for c in tfcols:
    pipe = Pipeline(
        [
            ('col_' + c, FunctionTransformer(lambda X: X[c].fillna('nan').values, validate=False)),
            ('tf_' + c, TfidfVectorizer())
        ]
    )
    streams.append(pipe)
pipe = Pipeline(
    [
        ('usecols',
         _ColumnSelector())
    ]
)
streams.append(pipe)
for c in ['operation_type', 'lastschrift_type']:
    streams.append(_OneHotCategory(on=c))

scorer = make_pipeline(
    *[
        INGParser(),
        make_union(*streams)
    ]
)
