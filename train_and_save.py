import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

print("Loading dataset...")
california = fetch_california_housing(as_frame=True)
df = california.frame

# Features and Target
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a pickle file
filename = "model.pkl"
with open(filename, "wb") as file:
    pickle.dump((model, list(X.columns)), file)

print(f"Model successfully saved to {filename}")