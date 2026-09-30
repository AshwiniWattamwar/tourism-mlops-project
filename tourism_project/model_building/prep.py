
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(
    "tourism_project/data/tourism.csv"
)

# Remove unnecessary columns
drop_cols = ["CustomerID"]

if "Unnamed: 0" in df.columns:
    drop_cols.append("Unnamed: 0")

df.drop(drop_cols, axis=1, inplace=True)

# Identify numerical and categorical columns
num_cols = df.select_dtypes(
    exclude="object"
).columns

cat_cols = df.select_dtypes(
    include="object"
).columns

# Handle missing values

for col in num_cols:
    df[col] = df[col].fillna(
        df[col].median()
    )

for col in cat_cols:
    df[col] = df[col].fillna(
        df[col].mode()[0]
    )

# Split features and target

X = df.drop(
    "ProdTaken",
    axis=1
)

y = df["ProdTaken"]

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Save datasets

X_train.to_csv(
    "Xtrain.csv",
    index=False
)

X_test.to_csv(
    "Xtest.csv",
    index=False
)

y_train.to_csv(
    "ytrain.csv",
    index=False
)

y_test.to_csv(
    "ytest.csv",
    index=False
)

print("Train Test Split Saved Successfully")
