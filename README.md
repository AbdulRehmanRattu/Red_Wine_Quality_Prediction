
# Red Wine Quality Prediction

This repository contains a study leveraging the Wine Quality Dataset from the UC Irvine Repository to predict red wine quality using Random Forest and SVM algorithms. The project demonstrates the effectiveness of machine learning in classifying wine quality based on 11 physiochemical properties.

## Project Overview

The study focuses on predicting the quality of red wine based on its chemical composition. Using advanced machine learning techniques, the project aims to classify wine quality and provide insights into the factors influencing it.

## Dataset

- **Source:** UC Irvine Machine Learning Repository
- **Features:** 11 chemical properties
- **Target Variable:** Quality (scale from 3 to 8)

## Machine Learning Models

- **Random Forest**
- **Support Vector Machine (SVM)**

## Key Findings

- Random Forest outperforms SVM in predicting wine quality.
- Feature importance analysis highlights alcohol as the most significant predictor.
- Techniques like SMOTE and hyperparameter tuning can improve model performance.

## Repository Structure

- `data/` - Contains the Wine Quality Dataset.
- `notebooks/` - Jupyter notebooks for data exploration and model training.
- `src/` - Source code for data preprocessing, model training, and evaluation.
- `results/` - Visualizations and model evaluation results.
- `README.md` - Project overview and instructions.

## Installation

To run this project locally, please follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/AbdulRehmanRattu/Red_Wine_Quality_Prediction.git
    cd Red_Wine_Quality_Prediction
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv env
    ```

3. **Activate the virtual environment:**

    - For Windows:
      ```bash
      .\env\Scripts\activate
      ```
    - For macOS and Linux:
      ```bash
      source env/bin/activate
      ```

4. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the project, follow the steps outlined in the Jupyter notebooks located in the `notebooks/` directory.

1. **Data Exploration and Preprocessing:**
   - Open `data_exploration.ipynb` to explore the dataset and preprocess the data.

2. **Model Training and Evaluation:**
   - Open `model_training.ipynb` to train and evaluate the Random Forest and SVM models.

3. **Results and Analysis:**
   - Open `results_analysis.ipynb` to view the visualizations and analysis of the model performance.

## Visualizations

- **Feature Importance Plot:** Shows the importance of each feature in predicting wine quality.
- **Confusion Matrix:** Visualizes the performance of the models in classifying each quality class.
- **Precision-Recall Curve:** Evaluates the trade-off between precision and recall for the models.

## Limitations

1. **Class Imbalance:** Significant imbalance in the dataset's class distribution impacted the models' ability to predict minority classes effectively.
2. **Performance on Minority Classes:** Both models struggled with accurate predictions for less represented quality classes.
3. **Feature Dynamics:** The specific influence of each chemical attribute on different quality levels was not deeply explored.
4. **Risk of Overfitting:** Complexity of the models, especially Random Forest, poses a risk of overfitting.

## Improvement Strategies

1. **Handling Class Imbalance:** Implementing techniques like SMOTE or further adjusting class weights.
2. **Enhanced Feature Engineering:** Revealing additional predictive relationships within the data.
3. **Broader Model Exploration:** Experimenting with different models or ensemble techniques.
4. **Enhanced Validation Techniques:** Employing rigorous cross-validation and regularization methods.

## Lessons Learned

1. **Complex Nature of Wine Quality Prediction:** Emphasizes the complexity inherent in such tasks.
2. **Critical Role of Data Preprocessing:** Importance of thorough data preprocessing.
3. **Importance of Model Selection and Tuning:** Need for careful model selection and hyperparameter tuning.
4. **Relevance of Evaluation Metrics:** Importance of selecting appropriate evaluation metrics.

## Conclusion

The analysis of the wine quality prediction models reveals that Random Forest models, particularly when trained with balanced class weights, outperform Support Vector Machines in predicting wine quality based on various chemical attributes. Feature engineering, along with hyperparameter tuning, moderately enhances model performance, although challenges persist in accurately classifying certain wine quality classes.

## References

1. UCI Machine Learning Repository. "Wine Quality Data Set." [Online]. Available: [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
2. L. Breiman, "Random Forests," Machine Learning, 2001.
3. C. Cortes and V. Vapnik, "Support-Vector Networks," Machine Learning, 1995.
4. A. C. Müller and S. Guido, "Introduction to Machine Learning with Python," O'Reilly Media, 2017.

## Contact

If you face any difficulty running this repository, feel free to contact me at:

- **Email:** rattu786.ar@gmail.com
- **LinkedIn:** [Abdul Rehman Rattu](https://www.linkedin.com/in/abdul-rehman-rattu-395bba237)
