import json
import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import re  
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import openai
from dateutil import parser


# Function to extract only the numeric part of the string
def extract_numeric(value):
    match = re.match(r"([-+]?\d*\.?\d+)", value)
    return match.group(0) if match else None

def try_parse(datetime_string):
    formats = ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%S.%f%z']
    for fmt in formats:
        try:
            return pd.to_datetime(datetime_string, format=fmt)
        except ValueError:
            continue
    raise ValueError(f"Date format for '{datetime_string}' is not supported.")

def validation_with_evaluation_agent (num,predictions_file, actuals_file,all_news_file):
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
            #print(f"iteration {i}: ",len(list_0))
            list_0=list_0+[list_0[-1]]*(48-len(list_0))
            #index1.append(i)
            predicted_values.extend(list_0)
            actual_values.extend([float(extract_numeric(value)) for value in act['output'].split(',') if extract_numeric(value)][:48])
            continue
        predicted_values.extend([float(extract_numeric(value)) for value in pred['output'].split(',') if extract_numeric(value)][:48])
        actual_values.extend([float(extract_numeric(value)) for value in act['output'].split(',') if extract_numeric(value)][:48])

    actual_values=actual_values
    predicted_values=predicted_values

    predicted = pd.array(predicted_values)
    actual = pd.array(actual_values)
    errors = predicted - actual
    
    errors = errors [num*48:(num+1)*48]
    actual = actual [num*48:(num+1)*48]
    
    background = actuals_data[num]["input"][actuals_data[num]["input"].find("The region for prediction"):actuals_data[num]["input"].find(" Weather of the start date:")]
    print("background: ", background)
    
    
    historical_time = actuals_data[num]["input"][actuals_data[num]["input"].find("The start date of historical data was on ")+41:actuals_data[num]["input"].find(" that is")]
    print("historical_time: ", historical_time)
    
    predictions_time = datetime.strptime(historical_time, '%Y-%m-%d')+ pd.Timedelta(days=1)
    predictions_time = predictions_time.strftime('%Y-%m-%d')
    print("predictions_time: ", predictions_time)
    
    
    selected_news = actuals_data[num]["input"][actuals_data[num]["input"].find("Weather of the prediction date: "):]
    
    news_df = pd.read_json(all_news_file)
    # Define a lambda function that uses dateutil to parse dates
    #parse_func = lambda x: parser.parse(x)

    # Apply this function to each element in the publication_time column
    #news_df['publication_time'] = news_df['publication_time'].apply(parse_func)
    news_df['publication_time'] = pd.to_datetime(news_df['publication_time'], utc=True)
    
    specified_date = predictions_time
    specified_datetime = pd.to_datetime(specified_date)

    # Create conditions for the same day and the day before
    condition_today = news_df['publication_time'].dt.date == specified_datetime.date()
    condition_yesterday = news_df['publication_time'].dt.date == (specified_datetime - timedelta(days=1)).date()
    all_news = news_df[(condition_today | condition_yesterday)][["publication_time", "category", "summary"]]#包含预测前和预测当天新闻
   
    all_news['publication_time'] = all_news['publication_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    all_news = all_news.to_json(orient='records',lines=False)
    #print("all_news: ", all_news)   
    
    return actual,errors,background,historical_time,predictions_time,selected_news,all_news

def gpt_chain_of_thoughts(background, selected_news, all_news, predictions_time, actual, errors,selection_news_logic_format,selection_news_logic_latest):
    # Set your API key here
    openai.api_base = #enter your api base 
    openai.api_key = #enter your api key
    
    prompt1 = "Based on historical data and relevant news, I have predicted the future electricity load in the specific region. I will provide you with our predicted values and actual values, as well as the news references. Please assess the accuracy of the predictions and analyze whether any important news has been overlooked."
    
    prompt2 = f"This is the basic condition: {background}"
    
    prompt3 = f"This is the news we used for the prediction:{selected_news}; Here are all news including the day-ahead news and today's news in json format: {all_news}. Please determine whether there was any news that should have been considered in the prediction but was not included."
    
    prompt4 = f"The actual value is {actual}. The results that predicted values minus actual values are as follows, starting from {predictions_time} at 00:00, with each point representing a half-hour interval: {errors}."

    prompt5 = "According to the news and errors at each specific time, determine if any news has been missed. The output format should be: The missed news is xxx, occurred at xxxx, the possible reasoning is xxxx."
    
    prompt6 = f"According to the outlooked news, please directly rephrase the latest prediction logic and output the adjusted new logic. Compare with the basic output format, You can add new bullet points and enrich the explanation of each bullet point as possible, and give some detailed examples (e.g. some specific event or issues). The basic output format is: {selection_news_logic_format}. This is the latest prediction logic that you need to adjust and improve:{selection_news_logic_latest}"
    try:
        
        # Initialize the chat completion with the first prompt
        response1 = openai.ChatCompletion.create(
            model="gpt-4-turbo-2024-04-09",  # Choose the appropriate model
            messages=[
                {"role": "system", "content": "You are a helpful assistant analyzing electricity load predictions."},
                {"role": "user", "content": prompt1+prompt2}
            ]
        )
        
        # Extract the chat continuation from the first response
        chat_log1 = response1["choices"][0]["message"]["content"]

        # Follow up with the second prompt
        response2 = openai.ChatCompletion.create(
            model="gpt-4-turbo-2024-04-09",
            messages=[
                {"role": "system", "content": "You are a helpful assistant analyzing electricity load predictions."},
                {"role": "user", "content": chat_log1},
                {"role": "user", "content": prompt3+prompt4+prompt5}
            ]
        )
        
        # Continue the chat with the third prompt
        chat_log2 = response2["choices"][0]["message"]["content"].replace("**", " ").replace("###", " ")
        print(chat_log2)
        response3 = openai.ChatCompletion.create(
            model="gpt-4-turbo-2024-04-09",
            messages=[
                {"role": "system", "content": "You are a helpful assistant analyzing electricity load predictions."},
                {"role": "user", "content": chat_log2},
                {"role": "user", "content": prompt6}
            ]
        )
        
        final_output = response3["choices"][0]["message"]["content"]
        return final_output

    except Exception as e:
        return str(e)
