import pandas as pd


"""
Fonction qui sort un dataframe du pourcentage de nul par colonne
input = dataframe
output = dataframe
"""
def preproc_null(data):
    # NaN count for each column 
    data.isnull().sum().sort_values(ascending=False) 
    # DataFrame 
    pd.DataFrame(data.isnull().sum().sort_values(ascending=False), columns=["is_null"]) 
    # NaN percentage for each column data.isnull().sum().sort_values(ascending=False)/len(data) 
    return pd.DataFrame([ data.isnull().sum(), data.isnull().sum()/len(data) ], index=["null_count", "null_share"]).T.sort_values(by="null_share", ascending=False)


