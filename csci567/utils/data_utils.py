import os
import pandas as pd

def load_csv(filename, dtype=str):
	'''
		load file from DATA_DIR/filename
	'''

	data_path = os.environ["DATA_DIR"]
	path = os.path.join(data_path, filename)
	
	if os.path.exists(path):
		return pd.read_csv(path, dtype=dtype)
	else:
		raise Exception(
				"Incorrect filepath: " + path
			)

def save_csv(data, filename):
	'''
		save data to DAT_DIR/filename
	'''

	data_path = os.environ["DATA_DIR"]
	path = os.path.join(data_path, filename)

	data.to_csv(path, index = False)

def get_train_data(cutoff_date = '2020-09-15'):
	'''
		generate training data by only returning the transactions that took place on or before 2020-09-15
	'''

	train = load_csv("transactions_train.csv")

	train.t_dat = pd.to_datetime(train.t_dat)
	if cutoff_date:
		return train.loc[train.t_dat <= pd.to_datetime(cutoff_date)]
	return train

def get_val_data(cutoff_date = '2020-09-16'):
	'''
		generate validation data by only returning the articles purchased in the week 2020-09-16 to 2020-09-22
	'''

	path = os.path.join(os.environ["DATA_DIR"], "validation_{}.csv".format(cutoff_date))
	
	if not os.path.exists(path):
		val = load_csv("transactions_train.csv")
		
		val.t_dat = pd.to_datetime(val.t_dat)
		val = val.loc[val.t_dat >= pd.to_datetime(cutoff_date)]

		val = val.groupby('customer_id').article_id.apply(list).reset_index()
		val = val.rename({'article_id' : 'prediction'}, axis = 1)

		val['prediction'] = val.prediction.apply(lambda x: ' '.join([str(k) for k in x]))
		
		val.rename(columns={"Unnamed: 0": "customer_id"})
		val.to_csv(path)

	val = load_csv("validation_{}.csv".format(cutoff_date))

	return val
