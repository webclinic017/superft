# from catboost.datasets import titanic
# from catboost import CatBoostClassifier, Pool, metrics, cv
# from pathlib import Path
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# import numpy as np
# import os
# while "freqtrade" not in os.listdir():
#     os.chdir("..")

# train_df, test_df = titanic()
# null_value_stats = train_df.isnull().sum(axis=0)
# null_value_stats
# train_df.fillna(-999, inplace=True)
# test_df.fillna(-999, inplace=True)
# X = train_df.drop('Survived', axis=1)
# y = train_df.Survived
# categorical_features_indices = np.where(X.dtypes != float)[0]

# X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=42)
# X_test = test_df

# model = CatBoostClassifier(
#     learning_rate=0.1,
#     custom_loss=[metrics.Accuracy()],
#     random_seed=42,
#     task_type="GPU",
#     iterations=10000
# )

# print("Fitting...")

# model.fit(
#     X_train, y_train,
#     cat_features=categorical_features_indices,
#     eval_set=(X_validation, y_validation),
#     verbose=10,
# )

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)


import pandas as pd

df_onepair = pd.DataFrame({})