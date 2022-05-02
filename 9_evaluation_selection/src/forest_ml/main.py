from pathlib import Path

import train_logistic
import train_rfc

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
train_RFC.train()