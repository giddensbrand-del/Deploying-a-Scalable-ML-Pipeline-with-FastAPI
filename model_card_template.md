# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This project uses a RandomForestClassifier model from scikit-learn to predict if an individual's income is >$50K per year based on census data. Trained as part of Udacity's "Deploying a Machine Learning Model with FastAPI".

The machine learning pipeline uses preprocessing with OneHotEncoder for categorical features and LabelBinarizer for the target label. The trained model, encoder, and label binarizer are serialized using pickle.

## Intended Use

The intended use is educational. The model demonstrates a complete machine learning deployment workflow:
preprocessing, training, evaluation, inference, testing, and API deployment.

The model predicts salary classification using demographic and employment-related information from census data.

This model should not be used in real-world hiring, lending, insurance, or legal decision-making systems.

## Training Data

The model was trained on the U.S. Census Income dataset.

Features:
- age
- workclass
- education
- marital-status
- occupation
- relationship
- race
- sex
- native-country
- hours-per-week
- capital-gain
- capital-loss

Target:
- salary

Salary is classified into:
- <=50K
- >50K

Categorical features were transformed using one-hot encoding before model training.

## Evaluation Data

The dataset was split into training and testing datasets. (80/20 train-test split, random seed of 42)

Model performance was evaluated on the test dataset after preprocessing using the same encoder and label binarizer created during training.


## Metrics

Precision measures how often positive predictions are correct. 
Recall measures how many actual positive cases are identified. 
The F1 score provides a balance between precision and recall.

Precision: 0.7419
Recall: 0.6384
F1 Score: 0.6863

## Ethical Considerations

The dataset contains demographic information which may introduce bias, such as race, sex.
Because the model learns patterns from historical data, it may reinforce existing societal/demographic biases.
The model should not be used in high-stakes decision-making systems and educational purposes only.


## Caveats and Recommendations

The model performance could potentially be improved through:
- hyperparameter tuning
- additional feature engineering
- cross-validation
- fairness evaluation