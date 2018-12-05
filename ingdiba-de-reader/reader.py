import pandas as pd
from hashlib import md5


def _uuid(r):
    """
    creates a unique id for each ing diba transaction based on:
        ['booking_date', 'amount', 'vendor', 'description', 'saldo']
    Args:
        r (pd.Series): row of a dataframe containing the above columns

    Returns:
        str: 32 char md5 hexdigest of concatenation of r's columns
    """

    h = md5(
        '_'.join(
            [str(r[c]).replace(' ', '') for c in ['booking_date', 'amount', 'vendor', 'description', 'saldo']]
        ).encode()
    ).hexdigest()
    return h


def read_csv_diba(filepath):
    """
    Args:
        filepath (str): path of the csv file to pass on to pd.read_csv
    Returns:
        pd.DataFrame
    Raises:
        ValueError if unable to find headers
        KeyError if Saldo not in columns
    """
    found = False
    for i in range(0, 20):
        df = pd.read_csv(filepath, sep=';', skiprows=i, encoding='cp1252', dtype=str, nrows=1)
        if df.columns.tolist()[0] == 'Buchung':
            found = True
            df = pd.read_csv(filepath, sep=';', skiprows=i, encoding='cp1252', dtype=str)
            break
    if found is False:
        raise ValueError(
            'unable to find headers for file {} - Buchung should be in the first column of the first 20 lines')
    if 'Saldo' not in df.columns:
        raise KeyError('Saldo should be in csv extract - in order to identify duplicates')
    df['file'] = filepath
    df = df[
        ['Buchung', 'Valuta', 'Auftraggeber/Empfänger', 'Buchungstext', 'Verwendungszweck', 'Betrag', 'Währung', 'file',
         'Saldo']]
    df = df.rename(
        columns={
            'Buchung': 'booking_date',
            'Valuta': 'valuta_date',
            'Auftraggeber/Empfänger': 'vendor',
            'Buchungstext': 'operation_type',
            'Verwendungszweck': 'description',
            'Betrag': 'amount',
            'Währung': 'currency',
            'Saldo': 'saldo'
        }
    )
    for c in ['booking_date', 'valuta_date']:
        df[c] = pd.to_datetime(df[c], dayfirst=True, yearfirst=False)
        df.sort_values(by=['booking_date', 'valuta_date'], inplace=True, ascending=True)
    df['amount'] = df['amount'].str.replace('.', '').str.replace(',', '.').astype(float)
    df['uid'] = df.apply(_uuid, axis=1)
    df.set_index(['uid'], inplace=True, drop=True)
    return df
