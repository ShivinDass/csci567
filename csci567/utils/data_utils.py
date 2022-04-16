import os
import pandas as pd

def load_csv(filename):
	'''
		load file from DATA_DIR/filename
	'''

	data_path = os.environ["DATA_DIR"]
	path = os.path.join(data_path, filename)
	
	if os.path.exists(path):
		return pd.read_csv(path)
	else:
		raise Exception(
				"Incorrect filepath: " + path
			)

def get_train_data():
	'''
		generate training data by only returning the transactions that took place on or before 2020-09-15
	'''

	train = load_csv("transactions_train.csv")

	train.t_dat = pd.to_datetime(train.t_dat)
	return train.loc[train.t_dat <= pd.to_datetime('2020-09-15')]

def get_val_data():
	'''
		generate validation data by only returning the articles purchased in the week 2020-09-16 to 2020-09-22
	'''

	path = os.path.join(os.environ["DATA_DIR"], "validation.csv")
	
	if not os.path.exists(path):
		val = load_csv("transactions_train.csv")
		
		val.t_dat = pd.to_datetime(val.t_dat)
		val = val.loc[val.t_dat >= pd.to_datetime('2020-09-16')]

		val = val.groupby('customer_id').article_id.apply(list).reset_index()
		val = val.rename({'article_id' : 'prediction'}, axis = 1)

		val['prediction'] = val.prediction.apply(lambda x: ' '.join(['0'+str(k) for k in x]))
		
		val.to_csv(path)		

	val = load_csv("validation.csv")

	return val
