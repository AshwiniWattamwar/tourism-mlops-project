
import pandas as pd

df = pd.read_csv(
    "tourism_project/data/tourism.csv"
)

expected_columns = [
    'CustomerID',
    'ProdTaken',
    'Age',
    'TypeofContact',
    'CityTier',
    'Occupation',
    'Gender',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'MaritalStatus',
    'NumberOfTrips',
    'Passport',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'Designation',
    'MonthlyIncome',
    'PitchSatisfactionScore',
    'ProductPitched',
    'NumberOfFollowups',
    'DurationOfPitch'
]

missing_cols = [
    col for col in expected_columns
    if col not in df.columns
]

if len(missing_cols)==0:
    print("Dataset Validation Passed")
else:
    print("Missing Columns")
    print(missing_cols)

print("\nShape")
print(df.shape)

print("\nSummary")
print(df.describe(include="all"))
