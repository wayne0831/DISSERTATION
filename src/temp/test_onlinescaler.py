from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from src.util.preprocess import OnlineStandardScaler
import numpy as np

# 데이터 생성
X, y = make_classification(n_samples=1000, n_features=5, random_state=42)

# 초기화
scaler = OnlineStandardScaler()
ml_mdl = GaussianNB()

y_pred_list = []

# 온라인 학습 루프
batch_size = 12  # 또는 mini-batch 크기
num_data   = len(X)
for idx in range(0, num_data, batch_size):
    print(f'start_idx: {idx} / end_idx: {idx + batch_size}')
    start_idx, end_idx = idx, idx + batch_size
    X_idx = X[start_idx:end_idx]
    y_idx = y[start_idx:end_idx]

    # partially scale dataset
    scaler.partial_fit(X_idx)
    X_scl = scaler.transform(X_idx)

    # partially fit the ml model
    ml_mdl.partial_fit(X_scl, y_idx, classes=np.unique(y)) if start_idx == 0 else ml_mdl.partial_fit(X_scl, y_idx)

    # predict the dataset
    y_pred = ml_mdl.predict(X_scl)
    y_pred_list.extend([int(x) for x in y_pred])

    print(y_pred)
# end for


print(222222222222222222)
# print(len(y), len(y_pred_list))
print(y_pred_list)
# print(33333333333333333333)
# print(y_pred)