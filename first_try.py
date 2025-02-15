import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the Dataset
df = pd.read_csv('Housing.csv')

# Quick check of the DataFrame
# print(df.head())

# 2. Visualisierung der Preisverteilung (Histogramm + Dichtekurve)
plt.figure(figsize=(8, 5))
sns.histplot(df['price'], kde=True, color='blue', bins=30)

# Methode A: Wissenschaftliche Notation ausschalten
plt.ticklabel_format(style='plain', axis='x')

# ODER Methode B: Eigene Formatierungsfunktion für die x-Achse
# (Auskommentieren und statt dessen verwenden, wenn du möchtest)
# formatter = FuncFormatter(lambda x, pos: f'{x:,.0f}')
# plt.gca().xaxis.set_major_formatter(formatter)

plt.title('Verteilung der Hauspreise')
plt.xlabel('Preis')
plt.ylabel('Häufigkeit')
plt.tight_layout()
#plt.show()

# 2. Preprocessing
# Identify numeric columns to add noise (e.g., area, bedrooms, bathrooms, stories, parking, price)
numeric_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']

# Number of times to duplicate
dup_factor = 5

augmented_rows = []
for idx, row in df.iterrows():
    for _ in range(dup_factor):
        new_row = row.copy()
        for col in numeric_cols:
            # For each numeric column, add small Gaussian noise (1% of its value)
            noise = np.random.normal(0, 0.01 * row[col])
            new_row[col] = row[col] + noise
        augmented_rows.append(new_row)

augmented_df = pd.DataFrame(augmented_rows, columns=df.columns)

# Now combine with original data
df_augmented = pd.concat([df, augmented_df], ignore_index=True)
#df_augmented = df
# Make sure to fix integer or categorical columns if they must stay integer (round them, clip them, etc.)
df_augmented['bedrooms'] = df_augmented['bedrooms'].round().clip(lower=0)

df = df_augmented
print("Original dataset size:", len(df))
print("Augmented dataset size:", len(df_augmented))


## 2.1 Convert 'yes'/ 'no' columns to 1/0
yes_no_cols = ['mainroad', 'guestroom', 'basement',
               'hotwaterheating', 'airconditioning', 'prefarea']

for col in yes_no_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

#df = df.drop('hotwaterheating', axis=1)
df = df.drop('guestroom', axis=1)

## 2.2 Handle the 'furnishingstatus' column (with three categories: furnished, semi-furnished, unfurnished)
# One option: create dummy variables
furnishing_dummies = pd.get_dummies(df['furnishingstatus'], prefix='furnish')
print(furnishing_dummies)
df = pd.concat([df, furnishing_dummies], axis=1)

# Drop the original furnishingstatus column
df.drop('furnishingstatus', axis=1, inplace=True)

# 3. Separate Features (X) and Target (y)
X = df.drop('price', axis=1)  # all columns except 'price'
y = df['price']               # the target is 'price'

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train a Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluate the Model
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)



print("Evaluation on Test Set:")
print(f"  - MSE: {mse:,.2f}")
print(f"  - RMSE: {rmse:,.2f}")
print(f"  - R^2:  {r2:.4f}")

# 7. Making a Single Prediction
# Example: Suppose we want to predict the price for a new house with the following attributes
# (Area=7420, Bedrooms=4, Bathrooms=2, Stories=3, mainroad=yes, guestroom=no,
#  basement=no, hotwaterheating=no, airconditioning=yes, parking=2, prefarea=yes,
#  furnishingstatus=furnished)
# We need to prepare it exactly like our training data after preprocessing.

new_house = {
    'area': [7420],
    'bedrooms': [4],
    'bathrooms': [2],
    'stories': [3],
    'mainroad': [1],         # 'yes' -> 1
    #'guestroom': [0],        # 'no' -> 0
    'basement': [0],         # 'no' -> 0
    'hotwaterheating': [0],  # 'no' -> 0
    'airconditioning': [1],  # 'yes' -> 1
    'parking': [2],
    'prefarea': [1],         # 'yes' -> 1
    # For furnishingstatus = 'furnished':
    'furnish_furnished': [1],
    'furnish_semi-furnished': [0],
    'furnish_unfurnished': [0]
}

new_house_df = pd.DataFrame(new_house)

predicted_price = model.predict(new_house_df)
print("\nPredicted price for the new house:", predicted_price[0])
