import pandas as pd
import random
from datetime import timedelta
from datetime import datetime
import holidays
import json
import random
import numpy as np
import re


def check_holiday_or_not(date, region):
    # Select Australian state, e.g., New South Wales
    aus_holidays = holidays.Australia(prov=region)
    # Set the date to check
    date_to_check = date  # example: '2023-12-25'

    # Check if the date is a public holiday
    if date_to_check in aus_holidays:
        return f"a public holiday: {aus_holidays.get(date_to_check)}."
    else:
        return f"not a public holiday."


def check_weekday_or_weekend(date):
    # Ensure the input is a datetime object
    if not isinstance(date, datetime):
        raise ValueError("The date should be a datetime object")

    # Check if the date is a weekday or weekend
    if date.weekday() < 5:
        return 'Weekday'
    else:
        return 'Weekend'


def categorize_state(region):
    """Categorizes a region entry into Australian state(s) based on known names, abbreviations, and cities."""
    state_patterns = {
        'NSW': r'\b(NSW|New South Wales|Sydney|Newcastle|Wollongong|East Coast|Eastern Australia|East Coast of Australia)\b',
        'VIC': r'\b(VIC|Victoria|Melbourne|East Coast|Eastern Australia|East Coast of Australia)\b',
        'QLD': r'\b(QLD|Queensland|Brisbane|Gold Coast|Sunshine Coast|Eastern Australia|East Coast of Australia)\b',
        'SA': r'\b(SA|South Australia|Adelaide|Southern States)\b',
        'TAS': r'\b(TAS|Tasmania|Hobart)\b',
        'WA': r'\b(WA|Western Australia|Perth)\b',
        'NT': r'\b(NT|Northern Territory|Darwin)\b',
        'ACT': r'\b(ACT|Australian Capital Territory|Canberra)\b'
    }
    matched_states = ""

    # Convert region to string to safely perform regex and substring checks
    region_str = str(region)

    i = 0
    for state, pattern in state_patterns.items():
        if re.search(pattern, region_str, re.IGNORECASE):
            if i == 0:
                matched_states += state
            else:
                matched_states += ',' + state
            i = i + 1

    if not matched_states:
        if "AUS" in region_str or 'Australia' in region_str or 'National' in region_str or 'General' in region_str or 'multiple' in region_str or "Various" in region_str:
            return 'National'
        elif 'Global' in region_str or 'International' in region_str or 'Asia Pacific' in region_str or "US" in region_str or 'NZ' in region_str or 'New Zealand' in region_str:
            return 'Global'
        else:
            return 'Unknown'
    return matched_states


def format_news(parsed_data, date, region):
    """Helper function to extract and format news from the parsed data."""
    parsed_data['categorized_region'] = parsed_data['region'].apply(categorize_state)
    parsed_data['date'] = pd.to_datetime(parsed_data['date'])
    # Format the date string to match the dataframe format
    date_str = date.strftime('%Y-%m-%d')

    # Filter news entries based on the date and the region
    filtered_news = parsed_data[
        (parsed_data['date'] == pd.to_datetime(date_str)) &
        ((parsed_data['region'].str.contains(region)) |
         (parsed_data['categorized_region'].isin(['Global', 'National', 'Unknown'])))
    ]

    if not filtered_news.empty:
        news_texts = []
        for _, news_entry in filtered_news.iterrows():
            news_text = news_entry['news'].replace("..", ".")
            rationality = news_entry['rationality']
            time = news_entry['date'].strftime('%Y-%m-%d')
            news_texts.append(f"On {time}, in the state of {news_entry['categorized_region']}, the news was: '{news_text}'. Rationality behind it: {rationality}")
        return " ".join(news_texts)
    else:
        return f"No relevant news available for {date_str}."


def parse_news_TS_final(weather_data_file, news_data_file, ts_file, save_file):
    all_weather_data = pd.read_csv(weather_data_file, encoding='utf-8')
    parsed_data = pd.read_csv(news_data_file, encoding='utf-8')

    # Ensure SETTLEMENTDATE column is of datetime type
    all_data = pd.read_csv(ts_file)
    all_data['SETTLEMENTDATE'] = pd.to_datetime(all_data['SETTLEMENTDATE'])

    # Initialize result list
    result_list = []

    # Get all unique REGION values
    unique_regions = all_data['REGION'].unique()

    # Define time ranges (in days)
    time_ranges = {'day': 1}

    # Iterate through each region
    i = 0
    for region in unique_regions:
        # Find and sort data for the current region
        region_data = all_data[all_data['REGION'] == region].sort_values(by='SETTLEMENTDATE')

        # Get possible start dates
        start_dates = pd.to_datetime(region_data['SETTLEMENTDATE'].dt.date.unique())

        # For each possible start date, randomly select a time range
        for start_date in start_dates:
            if start_date < pd.to_datetime("2019-1-1"):
                continue

            if start_date == pd.to_datetime("2021-1-1"):
                break

            # Randomly select a time range
            chosen_time_range = random.choice(list(time_ranges.values()))

            # Get the end date of the input period
            input_end_date = start_date + timedelta(days=chosen_time_range)

            if input_end_date.strftime('%Y-%m-%d') == "2023-11-30":
                print(input_end_date, " removed")
                continue

            # Get weather data for the input period
            input_weather_data = all_weather_data[all_weather_data['State'] == region[:-1]][
                all_weather_data['Date'] == start_date.strftime('%Y-%m-%d')]
            input_min_temp = str(input_weather_data["Min Temp (K)"].tolist()[0])
            input_max_temp = str(input_weather_data["Max Temp (K)"].tolist()[0])
            input_humidity = str(input_weather_data["Afternoon Humidity"].tolist()[0])
            input_pressure = str(input_weather_data["Afternoon Pressure"].tolist()[0])

            # Get power data for the input period
            input_data = region_data[
                (region_data['SETTLEMENTDATE'] >= start_date) &
                (region_data['SETTLEMENTDATE'] < input_end_date)]

            if len(input_data['TOTALDEMAND'].tolist()) > chosen_time_range * 48:
                input_data.set_index('SETTLEMENTDATE', inplace=True)
                input_data = input_data['TOTALDEMAND'].resample('30T').mean().reset_index()

            # Get power data for the output period (next day)
            output_data = region_data[
                (region_data['SETTLEMENTDATE'] >= input_end_date) &
                (region_data['SETTLEMENTDATE'] < input_end_date + timedelta(days=1))]

            if len(output_data['TOTALDEMAND'].tolist()) > 48:
                output_data.set_index('SETTLEMENTDATE', inplace=True)
                output_data = output_data['TOTALDEMAND'].resample('30T').mean().reset_index()

            # If output_data is empty, skip this date
            if output_data.empty:
                continue

            # Get news data for the input and output periods
            new_parsed_data = parsed_data.copy()
            new_parsed_data["date"] = pd.to_datetime(new_parsed_data["date"])
            date_list = new_parsed_data["date"].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()

            if start_date.strftime('%Y-%m-%d') not in date_list:
                formatted_news_1 = ""
            else:
                formatted_news_1 = format_news(parsed_data, start_date, region[:-1])

            if input_end_date.strftime('%Y-%m-%d') not in date_list:
                formatted_news_2 = ""
            else:
                formatted_news_2 = format_news(parsed_data, input_end_date, region[:-1])

            # Get weather data for the output period
            output_weather_data = all_weather_data[all_weather_data['State'] == region[:-1]][
                all_weather_data['Date'] == input_end_date.strftime('%Y-%m-%d')]
            output_min_temp = str(output_weather_data["Min Temp (K)"].tolist()[0])
            output_max_temp = str(output_weather_data["Max Temp (K)"].tolist()[0])
            output_humidity = str(output_weather_data["Afternoon Humidity"].tolist()[0])
            output_pressure = str(output_weather_data["Afternoon Pressure"].tolist()[0])

            # Format instruction and output, keeping one decimal point
            formatted_instruction = ",".join(f"{demand:.1f}" for demand in input_data['TOTALDEMAND'])
            formatted_output = ",".join(f"{demand:.1f}" for demand in output_data['TOTALDEMAND'])

            # Format input dates, removing leading zeros from months and days
            formatted_input_dates = ",".join(date.strftime('%m/%d %H:%M:%S').lstrip("0").replace('/0', '/') for date in input_data['SETTLEMENTDATE'])

            # Construct result dictionary
            result_dict = {
                "instruction": "The historical load data is: " + formatted_instruction,
                "input": "Based on the historical load data, please predict the load consumption in the next day. " +
                         "The region for prediction is " + region[:-1] + ". The start date of historical data was on " +
                         start_date.strftime('%Y-%m-%d').replace('-0', '-') + " that is " + check_weekday_or_weekend(
                    start_date) + ", and it is " + check_holiday_or_not(start_date.strftime('%Y-%m-%d'), region[:-1]) +
                         " The data frequency is 30 minutes per point." + " Historical data covers " + str(
                    chosen_time_range) + " day." +
                         " The date of prediction is on " + input_end_date.strftime('%Y-%m-%d').replace('-0', '-') +
                         " that is " + check_weekday_or_weekend(input_end_date) + ", and it is " + check_holiday_or_not(
                    input_end_date.strftime('%Y-%m-%d'), region[:-1]) +
                         " Weather of the start date: the minimum temperature is " + input_min_temp + "; the maximum temperature is " +
                         input_max_temp + "; the humidity is " + input_humidity + "; the pressure is " + input_pressure + ". " +
                         " Weather of the prediction date: the minimum temperature is " + output_min_temp + "; the maximum temperature is " +
                         output_max_temp + "; the humidity is " + output_humidity + "; the pressure is " + output_pressure + ". " +
                         formatted_news_1 + formatted_news_2,
                "output": formatted_output
            }
            print(result_dict)

            # Add to result list
            result_list.append(result_dict)
        i = i + 1
        print(f"region{i}: {region} is completed")

    # Display or save results
    with open(save_file, 'w') as f:
        json.dump(result_list, f)
    return result_list



def train_validation_split(json_file_path,train_save_file,val_save_file,val_num):
    # Reading the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:  # Use UTF-8 for reading
        result_list = json.load(file)
    print(result_list[-5:])
    
    #indices = np.load(random_indices_file).tolist()
    #test_dataset = [result_list[index] for index in indices]
    
    
    # Randomly selecting 2000 data points for the validation set
    #result_list = [data for data in result_list if data not in test_dataset]
    validation_set = random.sample(result_list, val_num)
    train_dataset = [data for data in result_list if data not in validation_set]

    #print(f"Test Set Size: {len(test_dataset)}")
    print(f"Validation Set Size: {len(validation_set)}")
    print(f"Train Dataset Size: {len(train_dataset)}")

    # Saving to files
    #with open(test_save_file, 'w', encoding='utf-8') as f:
        #json.dump(test_dataset, f, ensure_ascii=False, indent=4)

    with open(train_save_file, 'w', encoding='utf-8') as f_train:
        json.dump(train_dataset, f_train, ensure_ascii=False, indent=4)
        
    with open(val_save_file, 'w', encoding='utf-8') as f_val:
        json.dump(validation_set, f_val, ensure_ascii=False, indent=4)