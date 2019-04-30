# Feature_Selection

import pandas

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from sklearn import preprocessing


def train_test(X, y, test_size=0.2):

	# test 
	X_test, y_test = X[ : int( len(X) * test_size ) ] , y[ : int( len(X) * test_size ) ]

	# Train
	X_train, y_train = X[ int( len(X) * test_size ) : ] , y[ int( len(X) * test_size ) : ]

	return X_train, y_train, X_test, y_test

def feature_selector(model_X, model_y, k=10):

	selector = SelectKBest(chi2, k=k)
	selector.fit( model_X, model_y )

	# Get idxs of columns to keep
	idxs_selected = selector.get_support(indices=True)

	# Create new dataframe with only desired columns, or overwrite existing
	columns = model_X.columns[ idxs_selected ]
	 
	model_X = model_X[ columns ]


	return model_X, columns

def feature_select(df, test_size=0.4, k=10):
	
	data_frame = shuffle(df)
	X, y = data_frame.iloc[ : , 2: ].drop( columns=[ "SalePrice" ] ), data_frame[ "SalePrice" ]

	X_train, y_train, X_test, y_test = train_test(X, y, test_size)

	# # preprocessing.StandardScaler() preprocessing.RobustScaler()
	# scaler = preprocessing.Normalizer()
	# columns = X_train.columns

	#  # Don't cheat - fit only on training data
	# scaler.fit(X_train, y_train)  
	# X_train = scaler.transform(X_train)
	# X_train = pandas.DataFrame( X_train, columns=columns )
	
	# # apply same transformation to test data
	# X_test = scaler.transform(X_test)
	# X_test = pandas.DataFrame( X_test, columns=columns )

	X_train, columns = feature_selector(X_train, y_train, k)
	X_test = X_test[ columns ]

	return X_train, y_train, X_test, y_test