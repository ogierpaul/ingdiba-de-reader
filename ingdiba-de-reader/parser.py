import re

import pandas as pd
from sklearn.base import TransformerMixin


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


class ING_Parser(TransformerMixin):
    def transform(self, X, y=None):
        return _preparedf(X)

    def fit(self, X=None, y=None):
        return self