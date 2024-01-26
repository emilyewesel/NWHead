import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import nwhead

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


# def load_model(model_name, num_classes):
#     if model_name == 'resnet18':
#         model = models.resnet18(pretrained=True)
#         # Modify the classifier layer for the specified number of classes
#         model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
#         return model
#     else:
#         raise NotImplementedError(f"Model {model_name} not supported")

# class NWHead(nn.Module):
#     def forward(self,
#                 query_feats,
#                 support_feats,
#                 support_labels):
#         """
#         Computes Nadaraya-Watson prediction.
#         Returns (softmaxed) predicted probabilities.
#         Args:
#             query_feats: (b, embed_dim)
#             support_feats: (b, num_support, embed_dim)
#             support_labels: (b, num_support, num_classes)
#         """
#         query_feats = query_feats.unsqueeze(1)

#         scores = -torch.cdist(query_feats, support_feats)
#         probs = F.softmax(scores, dim=-1)
#         return torch.bmm(probs, support_labels).squeeze(1)

# Load the CSV file into a DataFrame
df = pd.read_csv('/dataNAS/people/paschali/datasets/chexpert-public/chexpert-public/train.csv')
print(df.head())
df.dropna(subset=['Support Devices'], inplace=True)
baase = "/dataNAS/people/paschali/datasets/chexpert-public/chexpert-public/train/"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Data Preprocessing
# Assuming you have a feature 'Path' and a target variable 'Support Devices'
# You may need to modify this based on your actual dataset structure
# X = df[['Sex', 'Age']]  # Add other features as needed
# y = df['Support Devices']


# # Handle missing values
# imputer = SimpleImputer(strategy='mean')
# X[['Age']] = imputer.fit_transform(X[['Age']])

# # Encode categorical variables
# label_encoder = LabelEncoder()
# X['Sex'] = label_encoder.fit_transform(X['Sex'])

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Model Selection and Training
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Model Evaluation
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# print(f"Accuracy: {accuracy}")
# print("Classification Report:")
# print(report)

# # Prediction on new data
# # Assuming you have a new data point stored in a variable 'new_data'
# new_data = pd.DataFrame({'Sex': ['Male'], 'Age': [45]})  # Add other features as needed
# new_data['Sex'] = label_encoder.transform(new_data['Sex'])
# prediction = model.predict(new_data)

# print("Prediction for new data:", prediction)


X = df[['Path', 'Sex', 'Age']]  # Add other features as needed
y = df['Support Devices']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X[['Age']] = imputer.fit_transform(X[['Age']])

# Encode categorical variables
label_encoder = LabelEncoder()
X['Sex'] = label_encoder.fit_transform(X['Sex'])

# Load the VGG16 model pre-trained on ImageNet data
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Function to load and preprocess images
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array.reshape(1, 224, 224, 3))
    return img_array

# Extract features from images
X_images = []
for img_path in X['Path']:
    img_path = baase + img_path
    img_array = load_and_preprocess_image(img_path)
    features = vgg16_model.predict(img_array)
    X_images.append(features.flatten())

X_images = pd.DataFrame(X_images)

# Combine image features with other features
X_combined = pd.concat([X.drop(columns=['Path']), X_images], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Model Selection and Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)