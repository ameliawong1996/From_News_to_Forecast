# (NeurIPS 2024) From News to Forecast: Iterative Event Reasoning in LLM-Based Time Series Forecasting

> [[Paper]](https://arxiv.org/abs/2401.13627) <br>
> [Xinlei Wang](https://scholar.google.com/citations?user=BfaMv18AAAAJ&hl=en), Maike Feng, [Jing Qiu](https://scholar.google.com/citations?user=QUclXRoAAAAJ&hl=en), [Jinjin Gu](https://www.jasongt.com/), [Junhua Zhao](https://www.zhaojunhua.org/) <br>
> School of Electrical and Computer Engineering, The University of Sydney; <br>
> School of Science and Engineering, The Chinese University of Hong Kong, Shenzhen; <br>
> Shenzhen Institute of Artificial Intelligence and Robotics for Society <be>

<!-- This repository is for the paper entitled: From News to Forecast: Iterative Event Reasoning in LLM-Based Time Series Forecasting -->

<!-- Code coming soon. -->

This repository contains the code and dataset for our paper: **"From News to Forecast: Iterative Event Reasoning in LLM-Based Time Series Forecasting"**, presented at NeurIPS 2024.

## Abstract

This paper introduces a novel approach to enhance time series forecasting using Large Language Models (LLMs) and Generative Agents. Our method integrates real-world social events, extracted from news, with traditional time series data. This allows our model to respond to unexpected incidents and societal shifts, improving the accuracy of predictions.

The main components of our system are:
- **LLM-based agents**: Automatically filter and analyze relevant news for time series forecasting.
- **Reasoning logic updates**: Refine the selection of news and improve prediction accuracy iteratively.

## Features

- Integration of unstructured news data into numerical time series forecasting.
- Iterative event reasoning through LLMs to continuously refine predictions.
- Application across multiple domains, including energy, exchange, bitcoin, and traffic forecasting.

## Dataset
### Overview

Our dataset is a crucial component of the research and spans several sectors where time series forecasting can be enhanced by integrating real-world events and news data. The dataset includes both structured numerical data and unstructured textual information, offering a unique blend of insights for more accurate and adaptive forecasting. The model integrates structured time series data with unstructured news data to improve forecast accuracy across various domains, including electricity demand, Bitcoin prices, exchange rates, and traffic volume.

The dataset covers the following domains:
1. **Electricity Demand (Australia)**:
   - Half-hourly electricity load data at the state level from the Australian Energy Market Operator (AEMO), covering the period between 2018 and 2022.
2. **Exchange Rates**:
   - Daily exchange rate data, particularly focusing on the Australian dollar between 2018 and 2022.
3. **Bitcoin Prices**:
   - Daily Bitcoin price data covering 2019 to 2021.
4. **Traffic Volume (California)**:
   - Hourly traffic volume data from California roads between 2015 and 2016.

### News Data

The News data is collected from a variety of sources, including:
- **GDELT Project**: A global database that monitors news media worldwide in real-time.
- **Yahoo Finance**: For financial news related to the exchange rate and Bitcoin price domains.
- **News AU**: News for Australian national or international Events.

We also enhance the dataset with supplementary information such as weather data (from OpenWeatherMap), calendar dates, and economic indicators to further enrich the context for forecasting.

### Data Structure

The dataset is structured as follows:
- `data/`: 
  - **Electricity Data**: Contains half-hourly electricity demand data for various Australian states.
  - **Exchange Rate Data**: Daily exchange rate information, particularly AUD-related.
  - **Bitcoin Data**: Daily Bitcoin prices.
  - **Traffic Data**: Hourly traffic volume for California roads.
  - **News Data**: Raw and filtered news articles with metadata for each domain.

For more details about the dataset, refer to the `dataset/` directory.

## Program Details
### Running the Code in Jupyter Notebook

All the steps for data preprocessing, model training, and forecasting are contained within the provided Jupyter Notebook. Follow these steps to run the notebook:

1. **Open the Notebook**: Start by launching the Jupyter Notebook interface and opening the `main.ipynb` file, which contains the complete workflow for time series forecasting with LLMs.

2. **Data Preprocessing**: 
   - loading and preprocessing the time series and news data.
     
3. **News Data Integration**:
   - The LLM agent creates news selection logic based on the time series task, guiding the filtering of relevant news, and aligning it with the time series data
     
4. **Model Training**:
   - fine-tuning a Large Language Model (LLM) using the preprocessed time series and selected news data.

5. **Forecasting**:
   - forecasting future values using both the historical time series and the news data.
  
6. **Evaluation**
   - validating the model’s prediction with validation sets. The evaluation agent checks for missing news that may have influenced the prediction, helping to refine the filtering logic in subsequent iterations.


### Metrics

The notebook uses the following metrics to compare the prediction performances of different forecasting models:

- **MSE (Mean Squared Error)**
- **RMSE (Root Mean Square Error)**
- **MAE (Mean Absolute Error)**
- **MAPE (Mean Absolute Percentage Error)**

### Usage

To use the notebook:
1. Clone the repository and navigate to the project directory:
    ```bash
    git clone https://github.com/your-repo-name.git
    cd your-repo-name
    ```

2. Install the necessary dependencies (make sure `requirements.txt` is provided with the needed packages):
    ```bash
    pip install -r requirements.txt
    ```

3. Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

4. Open and run `main.ipynb`.


### Citation

If you use this code in your research, please cite our paper:
```
@inproceedings{wang2024newsforecast,
   title={From News to Forecast: Iterative Event Reasoning in LLM-Based Time Series Forecasting},
   author={Wang, Xinlei and Feng, Maike and Qiu, Jing and Gu, Jinjin and Zhao, Junhua},
   booktitle={Neural Information Processing Systems},
   year={2024}
}
```

