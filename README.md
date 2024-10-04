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

Our dataset is a crucial component of the research and spans several sectors where time series forecasting can be enhanced by integrating real-world events and news data. The dataset includes both structured numerical data and unstructured textual information, offering a unique blend of insights for more accurate and adaptive forecasting.

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
