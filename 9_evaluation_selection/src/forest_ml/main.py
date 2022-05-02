from pathlib import Path
# from .search_logistic import get_rfc_opt_hyperparameters
import search_logistic

import train_logistic
import train_rfc
import pipeline as p


# dataset_path = Path('/Users/yuraaslamov/Documents/GitHub/ml-python/9_evaluation_selection/data/train.csv')
# model_path = Path('/Users/yuraaslamov/Documents/GitHub/ml-python/9_evaluation_selection/results/model')
dataset_path = Path('../../data/train.csv')
model_path = Path('../../data/model.joblib')
# pd = get_dataset(csv_path=path, random_state=42, test_split_ratio=0.2)

# tr.train_cl(
#     dataset_path=dataset_path,
#     save_model_path=model_path,
#     random_state=42,
#     test_split_ratio=0.2,
#     use_scaler=True,
#     max_iter=1000,
#     logreg_c=1,
# )

# train_logistic.train()
# train_RFC.train()


# model = p.create_RFC_pipeline()
# print(model.get_params().keys())

# search_logistic.\
# search_logistic.get_rfc_opt_hyperparameters(dataset_path)
# search_logistic.get_rfc_opt_hyperparameters()
# train_rfc.train_with_opt_hyperparameters()
train_logistic.train_with_opt_hyperparameters()