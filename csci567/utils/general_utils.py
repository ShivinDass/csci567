from csci567.metrics.mapk import mapk
from csci567.utils.data_utils import load_csv

def get_val_score(val_file = "validation.csv", sub_file = "submission.csv"):
	'''
		Compute the validation score between val_file and sub_file data

	'''
	val = load_csv(val_file)

	sub = load_csv(sub_file)
	sub.set_index('customer_id').loc[val.customer_id].reset_index()

	return mapk(val.prediction.str.split(), sub.prediction.str.split(), k=12)