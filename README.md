# Ames Housing Regression Analysis

## Project Overview
This project analyzes the Ames Housing dataset to predict house sale prices using multiple linear regression. Two models are implemented:

- **Standard Linear Regression** using `Overall Qual` and `Gr Liv Area`.
- **Log-Transformed Linear Regression** with additional feature `Year Built` to improve model assumptions.

**Key goals:**
- Explore relationships between home features and sale price
- Build predictive regression models
- Validate regression assumptions (normality, linearity, heteroscedasticity, multicollinearity)
- Provide interpretable insights for house pricing

## Dataset
The dataset comes from the [Ames Housing dataset](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset).  
It contains 2,930 observations and 82 features describing residential homes in Ames, Iowa.

**Included in repo:**  
`data/AmesHousing.csv`

## Project Structure
```text
ames-housing-regression/
│
├── data/                   # Dataset CSV
│   └── AmesHousing.csv
├── notebooks/              # Jupyter notebooks
│   └── ames_regression.ipynb
├── src/                    # Reusable Python functions/modules
│   └── ames_models.py
├── README.md               # This file
├── requirements.txt        # Required Python packages
└── .gitignore
