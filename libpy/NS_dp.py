# Normalize_Standardize_data_preprocessing.py

import pandas, seaborn, matplotlib
import os, sys
import numpy
from sklearn import preprocessing


def Standardize_Values_Null(df, columns):

	df = df.drop( columns = columns )
	df.dropna(inplace=True)
	df.reset_index( drop=True, inplace=True )

	return df

# Standardize values (Continuous)
def Standardize_Values_Continuous(df, columns):
	transformer = preprocessing.MinMaxScaler()  #MaxAbsScaler

	temp_df = df[ columns ]

	df.drop( columns=columns, inplace=True )
	df = pandas.concat( [ df, pandas.DataFrame( transformer.fit_transform( temp_df ), columns=columns ) ], axis=1 )

	return df

# Pandas get_dummies (Nominal)
def Standardize_Values_Nominal(df, columns):

	temp_df = df[ columns ]

	df.drop( columns=columns, inplace=True )
	dummy_df = pandas.get_dummies( temp_df )

	df = pandas.concat( [ df, dummy_df ], axis = 1)

	return df


# Ordinal Values
def Standardize_Values_Ordinal(df, columns):

	ordinalEncoder = preprocessing.OrdinalEncoder()

	temp_df = df[ columns ]

	df.drop( columns=columns, inplace=True )

	df = pandas.concat( [ df, pandas.DataFrame(
	    ordinalEncoder.fit_transform( temp_df ), columns=columns 
	) ], axis = 1)

	df.dropna(inplace=True)
	df.reset_index( drop=True, inplace=True )

	return df


def clean_Ames_Housing(df):

	columns = [ "Lot Frontage", "Fireplace Qu", "Fence", "Alley", "Misc Feature", "Pool QC" ]
	df = Standardize_Values_Null(df, columns)

	columns = [ "Lot Area", "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF", 
	               "1st Flr SF", "2nd Flr SF", "Low Qual Fin SF", "Gr Liv Area", "Garage Area", "Wood Deck SF", "Open Porch SF",
	               "Enclosed Porch", "Screen Porch", "Pool Area", "Misc Val",  ]
	df = Standardize_Values_Continuous(df, columns)


	columns = [ "MS SubClass", "MS Zoning", "Street", "Land Contour", "Lot Config", "Neighborhood", "Condition 1", "Condition 2",
	              "Bldg Type", "House Style", "Roof Style", "Roof Matl", "Mas Vnr Type", "Foundation", "Exterior 1st", 
	               "Exterior 2nd", "Heating", "Central Air", "Garage Type", "Sale Type", "Sale Condition", ]
	df = Standardize_Values_Nominal(df, columns)


	columns = [ "Lot Shape", "Utilities", "Land Slope", "Overall Qual", "Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Exposure", 
	              "BsmtFin Type 1", "BsmtFin Type 2", "Electrical", "Functional", "Kitchen Qual", "Heating QC", "Bsmt Cond",
	              "Garage Finish", "Garage Qual", "Garage Cond", "Paved Drive", ]
	df = Standardize_Values_Ordinal(df, columns)

	return df
