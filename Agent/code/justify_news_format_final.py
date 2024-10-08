import json
import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score


def replace_in_text(text):
    """ Replace unwanted characters in the given text. """
    replacements = {
        "```": "",
        "json": "",
        "**": "",
        ': no': ':"no"'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def process_news_section(news_json, section_key):
    """ Process each news section in the JSON and return a list of news details. """
    news_section = news_json.get(section_key, [])
    results = []
    if not news_section:
        results.append({'time': None, 'news': "there is no suitable news.", 'region': None, 'rationality': "there is no suitable news."})
    else:
        for item in news_section:
            news = item.get("news", "there is no suitable news.")
            rationality = item.get("rationality", "there is no suitable news.")
            time = item.get("time", None)
            region = item.get("region", None)
            results.append({'time': time, 'news': news, 'region': region, 'rationality': rationality})
    return results

def justify_news_format_final(news_csv, save_file):
    result_data = []
    news_csv["time"] = pd.to_datetime(news_csv["time"])
    for date in news_csv["time"].unique():
        news_selected = news_csv.loc[news_csv["time"] == date, "news"].iloc[0]
        print(news_selected)
        if isinstance(news_selected, str):
            news_selected = replace_in_text(news_selected)
            try:
                news_json = json.loads(news_selected)
                sections = ["Long-Term Effect on Future Load Consumption",
                            "Short-Term Effect on Today's Load Consumption",
                            "Real-Time Direct Effect on Today's Load Consumption"]
                for section in sections:
                    news_details = process_news_section(news_json, section)
                    for detail in news_details:
                        detail['date'] = date  # Add the date to each news detail
                        result_data.append(detail)
            except json.JSONDecodeError:
                print(f"In {date}, the response is not in JSON format.")
            except Exception as e:
                print(f"An error occurred for {date}: {e}")
        else:
            print(f"In {date}, expected a string but got {type(news_selected).__name__}.")

    # Creating DataFrame from result_data
    parsed_data = pd.DataFrame(result_data)
    parsed_data.to_csv(save_file, index=False, encoding='utf-8')
