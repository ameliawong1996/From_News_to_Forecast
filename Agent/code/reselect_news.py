import openai
import pandas as pd
import os
import random
from datetime import timedelta
import json

def fetch_news(cutoff_date, day_num, df):

    cutoff_date = pd.to_datetime(cutoff_date)
    
    news_before = cutoff_date - pd.Timedelta(days=day_num)
    news_after = cutoff_date + pd.Timedelta(days=1)

    selected_df_before = df[news_before <= df['publication_time']][df['publication_time'] < cutoff_date]
    if selected_df_before.empty:
        print("No news found before the prediction date: ",cutoff_date)
        formatted_statements_before = " No news found before the prediction date."
    else:
        formatted_statements_before = selected_df_before.apply(lambda row: f"In {row['publication_time']}, the news was published that {row['summary']}", axis=1)
        formatted_statements_before = "All news before the prediction include: "+' '.join(formatted_statements_before.tolist())

    selected_df_after = df[cutoff_date <= df['publication_time']][df['publication_time'] < news_after]
    if selected_df_after.empty:
        print("No news found on the prediction date: ",cutoff_date)
        formatted_statements_after = " No news found on the prediction date."
    else:
        formatted_statements_after = selected_df_after.apply(lambda row: f"In {row['publication_time']}, the news was published that {row['summary']}", axis=1)
        formatted_statements_after = "All news for the prediction include: "+' '.join(formatted_statements_after.tolist())

    return formatted_statements_before,formatted_statements_after


def gpt_reselect_news(prompt):
    # Set your API key here
    openai.api_base = #enter your api base 
    openai.api_key = #enter your api key

    try:
        # Call the OpenAI GPT API for chat completions
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo-2024-04-09", # or another model
            messages=[
                {"role": "system", "content": "You are a helpful assistant analyzing electricity load predictions."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return str(e)

def reselect_news_procedure(raw_news_file_path,csv_file_path,initial_reasoning):
    selected_news = pd.DataFrame(columns=['time', 'news'])
    prompt2 = "If I give you all news before the prediction, based on the above positive & negative effect analysis, 1) please choose all news that may have a long-term affect on future load consumption; 2) please choose all news that may have a short-term effect on today's load consumption.  3) please choose all news that may have a real-time direct effect on today's load consumption. if there is no suitable news, please say no. Also, please include the region (NSW/VIC/TSA/QLD/SA/WA) and time information of these news. If there are multiple relavant news, please ensure that you include all relavant news. Organize the paragraph in this format: Long-Term Effect on Future Load Consumption: news is xxx; region is xxx; time is xxxx; the rationality is that xxx."

    format_output2 = """
Remember to only give the json output including all relavant news and make it the valid json format.  Format is {
"Long-Term Effect on Future Load Consumption": [
        {
            "news": "Work on WA’s latest $1b lithium plant will start within days as US resources giant Albemarle begins building a major processing facility outside Bunbury, creating hundreds of jobs.",
            "region": "WA",
            "time": "2019-01-03 16:40:00",
            "rationality": "The construction and operation of a major lithium processing facility will likely influence long-term electricity demand through increased industrial activity and potential population growth in the area due to new job opportunities."
        },
        {
            "news": "Another major renewable energy project was initiated in WA, expected to supply significant power by 2022.",
            "region": "WA",
            "time": "2019-03-15 11:30:00",
            "rationality": "Long-term electricity load will be impacted by the integration of renewable energy sources, which are expected to offset dependence on traditional fossil fuels."
        }
    ],
    "Short-Term Effect on Today's Load Consumption": [
        {
            "news": "SA just sweltered through a very warm night, after a day of extreme heat where some regional areas reached nearly 48C.",
            "region": "SA",
            "time": "2019-01-03 17:57:00",
            "rationality": "Extreme weather conditions, particularly the intense heat, will lead to higher electricity consumption in the short term as residents and businesses increase the use of air conditioning and cooling systems to manage temperatures."
        },
        {
            "news": "A sudden cold snap in Victoria leads to a spike in electric heating usage.",
            "region": "VIC",
            "time": "2019-01-04 05:22:00",
            "rationality": "Short-term electricity load spikes are often caused by unexpected weather events that drive up heating or cooling demand."
        }
    ],
    "Real-Time Direct Effect on Today's Load Consumption": [
        {
            "news": "An unseasonal downpour has wreaked havoc on Perth’s electricity network this morning.",
            "region": "WA",
            "time": "2019-01-03 10:11:00",
            "rationality": "The sudden weather event causing disruptions to the electricity network can have an immediate impact on load consumption due to power outages, infrastructure damage, or emergency response measures."
        },
        {
            "news": "Lightning strike at a major substation causes widespread outages in Sydney.",
            "region": "NSW",
            "time": "2019-01-03 19:45:00",
            "rationality": "Direct effects on load consumption include sudden drops in power supply, triggering emergency measures to restore stability in the network."
        }
    ]}"""
    
    
    with open(raw_news_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    news_df = pd.DataFrame(data)
    news_df['publication_time'] = pd.to_datetime(news_df['publication_time'])
    
    dates_range= pd.date_range(start=f"2019-01-01", end=f"2021-01-01")
    for date in dates_range:
        formatted_date = date.strftime('%Y-%m-%d')
        news_before,news_after=fetch_news(date, 1, news_df)
        prompt1 = f"The prediction date is {formatted_date}."
        prompt3 = f"The news happened before the prediction include:{news_after}"
 
        prompt = initial_reasoning + prompt1 + prompt2 + prompt3 + format_output2
        response = gpt_reselect_news(prompt)
        response = response[response.find("{"):response.rfind("}") + 1].replace("\n", "")
        print(response)

        try:
            response_json = json.loads(response)
            print("The response is in JSON format.")
        except json.JSONDecodeError:
            print("The response is not in JSON format.")
        news_string = response #response
        df_extended = pd.DataFrame({'time': [formatted_date], 'news': [news_string]})
        selected_news = pd.concat([selected_news, df_extended], ignore_index=True)

        selected_news.to_csv(csv_file_path, index=False, encoding='utf-8')  