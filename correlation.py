import pandas as pd


# Computing the correlation matrix
def correlation_matrix(dataframe):
    cor_matrix = dataframe.corr()
    print(cor_matrix)
    return cor_matrix

