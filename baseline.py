import yfinance as yf
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
# Scale data for ANN
from sklearn.ensemble import RandomForestClassifier
scaler = StandardScaler()

# Function to calculate MACD and Signal Line Indicators
def calculate_macd(data, slow=26, fast=12, smooth=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=smooth, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def rsi(data, period=14):
  """

  """
  delta = data['Close'].diff()
  delta = delta.dropna()
  up = delta[delta > 0]
  down = delta[delta < 0]
  avg_gain = up.ewm(alpha=1/period, min_periods=period).mean()
  avg_loss = abs(down.ewm(alpha=1/period, min_periods=period).mean())
  rs = avg_gain / avg_loss
  rsi = 100 - 100 / (1 + rs)
  return rsi

# Example usage
data = pd.DataFrame({'Close': [10, 12, 11, 13, 15, 14]})
rsi_values = rsi(data)

print(rsi_values)


# Retrieve Apple's stock data from Yahoo Finance
data = yf.download('AAPL', start='1999-08-01', end='2024-02-20')

# Define the ticker for Brent Crude Oil futures and the time period
ticker = 'BZ=F'  # Ensure this ticker is current via Yahoo Finance
start_date = '1999-08-01'
end_date = '2024-02-20'

# Download the data
#data = yf.download(ticker, start=start_date, end=end_date)

# Check the data
print(data.head())



# Calculate MACD and Signal line indicators
macd, signal, hist = calculate_macd(data)

# Plotting Moving Averages and Exponentially moving averages
data['MA5'] = data['Close'].rolling(window=5).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

data['EMA5'] = data['Close'].ewm(span=5, adjust=False).mean()
data['EMA200'] = data['Close'].rolling(window=200).mean()
data['Volume']=data['Volume'].rolling(window=20).mean()
# Replace zero values in 'Volume' if they exist, assuming a small value like 1
data['Volume'] = data['Volume'].mean()
# Add MACD, Signal, and Histogram to the DataFrame
data['MACD'] = macd
data['Signal'] = signal
data['MACD_Histogram'] = hist

# Print the DataFrame to see the new columns
print(data.head())
# Apply the logarithmic transformation
data['Log_Volume'] = np.log(data['Volume'])
horizon=1

# Calculate the percentage change
data['Pct_Change'] = data['Close'].pct_change(periods=horizon)

# Define the indicator function
data['Y'] = np.where(data['Pct_Change'] > 0, 1, 0)
data = data.dropna()

# Separate the features and the target variable
X = data[['MA5', 'MA200','MACD','MACD_Histogram']]
y = data['Y']

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a logistic regression model
model = LogisticRegression()

# Fit the model
model.fit(X_train, y_train)

# Predict the target variable
y_pred = model.predict(X_test)

# Calculate the accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print accuracy and confusion matrix
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)


# Calculate the accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print accuracy and confusion matrix
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# Plotting the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test.index, y_test, color='blue', label='Actual', alpha=0.5, s=100)
plt.scatter(y_test.index, y_pred, color='red', label='Predicted', alpha=0.5, marker='x', s=100)
plt.title('Actual vs Predicted Values')
plt.xlabel('Date')
plt.ylabel('Class')
plt.yticks([0, 1], ['Negative', 'Positive'])
plt.legend()
plt.grid(True)
plt.show()


lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Artificial Neural Network
ann_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
ann_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train_scaled, y_train, epochs=50, verbose=0)
ann_loss, ann_accuracy = ann_model.evaluate(X_test_scaled, y_test, verbose=0)

# Create a DataFrame to display results
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'ANN'],
    'Accuracy': [lr_accuracy, rf_accuracy, ann_accuracy]
})

print(results)

importances = rf_model.feature_importances_

# Organize them into a DataFrame
features = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display the feature importance
print(features)

# Plotting feature importances
plt.figure(figsize=(10, 6))
plt.barh(features['Feature'], features['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance in Random Forest Classifier')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important at the top
plt.show()


# Calculate the correlation matrix
corr_matrix = X_train.corr()

# Display the correlation matrix
print(corr_matrix)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix of Training Features')
plt.show()