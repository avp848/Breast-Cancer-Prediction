# Breast-Cancer-Prediction
### Problem Statement
The key challenge was to efficiently classify and diagnose tumors as malignant or benign.

### Data Import and Wrangling
The original dataset is from University of Wisconsin Hospital. However, the data provided for this study wasn’t clean, and we had to drop the id and unnamed columns so as to remove any NA values and be able to operate on a fully functional dataset.

### Methodology
I built a machine learning classifier to predict if the tumor was cancerous or not. To do so, I incorporated logistic regression, classification, and class imbalance to analyze tumor features and create the most efficient model.

### Algorithms Used
 - KKNN, KKNN-balanced, SVM, GLM, RPART
 - Classified malignant and benign tumors into numeric values of 1 and 0 respectively

### Challenges
- Varying accuracy 
- Cleaning data and getting constant errors because of data loss
- Being able to use best set of features without much domain knowledge

### Significance
ML models improve accuracy and are useful tools to make the diagnosis more efficient. Most oncologists are able to diagnose at 75% accuracy while ML can give you around 92-95% accuracy. This improves the detection of cancer in early stages, which is key to prevent metastasis and death. The final model was built using SVM as it gave the best accuracy of 97.05%.
