import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import re  # Import the regular expression module
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Function to extract only the numeric part of the string
def extract_numeric(value):
    match = re.match(r"([-+]?\d*\.?\d+)", value)
    return match.group(0) if match else None

def plot_comparision(predictions_file,actuals_file):
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)

    with open(actuals_file, 'r', encoding='utf-8') as f:
        actuals_data = json.load(f)

    # Assuming the structure of both JSON files is the same and the 'output' field is present in both
    predicted_values = []
    actual_values = []

    i=0
    for pred, act in zip(predictions_data, actuals_data):
        # Use the extract_numeric function to only get numeric part
        i=i+1
        list_0=[float(extract_numeric(value)) for value in pred['output'].split(',') if extract_numeric(value)]
        if len(list_0) < 48:
            print(f"iteration {i}: ",len(list_0))
            list_0=list_0+[list_0[-1]]*(48-len(list_0))
            #index1.append(i)
            predicted_values.extend(list_0)
            actual_values.extend([float(extract_numeric(value)) for value in act['output'].split(',') if extract_numeric(value)][:48])
            continue
        predicted_values.extend([float(extract_numeric(value)) for value in pred['output'].split(',') if extract_numeric(value)][:48])
        actual_values.extend([float(extract_numeric(value)) for value in act['output'].split(',') if extract_numeric(value)][:48])

    actual_values=actual_values
    predicted_values=predicted_values
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_values, predicted_values)
    mape = np.mean(np.abs((np.array(actual_values) - np.array(predicted_values)) / np.array(actual_values))) * 100
    r2 = r2_score(actual_values, predicted_values)
    explained_variance = explained_variance_score(actual_values, predicted_values)

    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')
    print(f'MAPE: {mape}%')
    print(f'R^2: {r2}')
    print(f'Explained Variance: {explained_variance}')

    hits = np.sum(np.sign(np.array(actual_values[1:]) - np.array(actual_values[:-1])) == np.sign(np.array(predicted_values[1:]) - np.array(predicted_values[:-1])))
    hit_rate = hits / (len(actual_values) - 1)
    print(f'Hit Rate: {hit_rate}')



    # Plotting
    plt.figure(figsize=(15, 7))
    plt.plot(predicted_values[:5*48], 'r-', label='Predicted Values')
    plt.plot(actual_values[:5*48], 'b-', label='Actual Values')

    plt.title('Comparison of Predicted and Actual Values')
    plt.xlabel('Time Index')  # Change to 'Timestamp' if actual timestamps are used
    plt.ylabel('Values')
    plt.legend()
    plt.show()