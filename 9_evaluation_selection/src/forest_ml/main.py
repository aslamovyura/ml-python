from pathlib import Path
# from .data import get_dataset
import train as tr

dataset_path = Path('/Users/yuraaslamov/Documents/GitHub/ml-python/9_evaluation_selection/data/train.csv')
model_path = Path('/Users/yuraaslamov/Documents/GitHub/ml-python/9_evaluation_selection/results/model')
# pd = get_dataset(csv_path=path, random_state=42, test_split_ratio=0.2)

acc = tr.train_cl(
    dataset_path=dataset_path,
    save_model_path=model_path,
    random_state=42,
    test_split_ratio=0.2,
    use_scaler=True,
    max_iter=1000,
    logreg_c=1,
)

print(f'Accuracy={acc}')