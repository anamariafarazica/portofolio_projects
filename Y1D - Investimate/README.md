## Table of Contents
- [Introduction](#i-introduction)
- [Database Creation](#ii-database-creation)
- [Data Cleaning](#iii-data-cleaning)
- [Feature Engineering](#iv-predictive-feature-engineering-and-target-variable)
- [Wavelet Denoising](#v-advanced-predictive-feature-engineering-with-wavelet-denoising)
- [Risk Categorization](#vi-clustering-preprocessing-and-risk-categorization)
- [Merging Features](#vii-merging-wavelet-derived-predictive-features)
- [Database Design](#viii-database-design-considerations-for-future-development)
- [Schema Design](#ix-database-schema-design)
- [Summary](#x-summary-and-conclusions)


# I. Introduction
This project supports the development of an educational and informative web platform for Move Tickers, aimed at helping users explore financial markets through data-driven insights.

We implemented two core machine learning pipelines:

1. **Risk Clustering Model**: Utilizes macroeconomic indicators and technical metrics to segment companies into risk levels (Low, Medium, High). This model allows users to identify patterns in market sensitivity and volatility over time, supporting comparative analysis across companies.

2. **Next-Day Price Direction Model**: Predicts short-term stock movement using a classification approach. It integrates denoised technical indicators and macroeconomic signals to support scenario-based learning and strategy backtesting.

These models are designed not for real-time trading, but to promote financial literacy, pattern recognition, and critical thinking in investment contexts. The system forms the analytical backbone of a web-based tool that helps users simulate strategies and explore market dynamics with real historical data.


# II. Database Creation 

## 2.1. Initial Table Creation
The process begins with the creation of a core table named *fact_company_daily_metrics*. This table is populated with daily stock market data for a select group of companies—**Apple, Microsoft, Palantir, KLA Corporation, Qualcomm, and Crowdstrike Holdings Inc** — sourced from a larger NASDAQ dataset that contains daily inputs for every company in the NASDAQ 100 List. The table includes fields such as:
 - Date
 - Company identifier
 - Opening and closing prices
 - Daily highs and lows
 - Trading volume

## 2.2. Integration of Macroeconomic Indicators
To provide broader economic context, several macroeconomic indicators are added to the table. These include:
 - Unemployment rate
 - GDP growth
 - Inflation (CPI)
 - Stock market volatility (VIX index)
 - Federal funds interest rate

These indicators are sourced from a separate table containing cleaned macroeconomic news data and are matched to the stock data by date.

## 2.3. Enhancement with Technical Indicators
The table is further enriched with commonly used technical indicators in stock market analysis:

- Moving Averages
  - Simple Moving Averages (SMA): 10-day and 50-day moving averages of closing prices for each company
  - SMA Difference: Difference between 10-day and 50-day SMAs to identify short-term vs. long-term trends

- Volatility Indicators
  - Average True Range (ATR): Measures market volatility by averaging the true range over a specified period

- Momentum Indicators
  - MACD (Moving Average Convergence Divergence): Approximated by subtracting a 26-day moving average from a 12-day moving average of closing prices
  - RSI (Relative Strength Index): Calculated over a 14-day window to assess the speed and change of price movements

- Return Calculations
  - 1-Day and 3-Day Returns: Percentage returns over one and three days for each company, providing insight into short-term price movements

# III. Data Cleaning 

A thorough cleaning process is applied to prepare the dataset for analysis and modeling:
- Handling Missing Values:
  - Macroeconomic indicators (*unemployment rate, CPI inflation, federal funds rate*): Missing daily values are filled using spline interpolation to preserve smooth trends.
  - VIX volatility index: 107 missing entries are forward-filled, reflecting financial market practices.
  - GDP growth: Excluded due to excessive missingness and lack of reliable imputation.
  - Final row with residual NaN values: Dropped as it was non-critical.

- Removing Duplicates:
  - Duplicate entries for the same date in the macroeconomic dataset are removed, retaining only one entry per day.

# IV. Predictive Feature Engineering and Target Variable

To enable supervised learning for next-day stock movement prediction:
- Target Variable Creation:
  - For each stock, a *Tomorrow* column is computed by shifting the close_value one day backward within each *company_prefix* group.
  - The *Target* is set to 1 if the next day’s close is higher than the current day’s close, and 0 otherwise.

- Feature Selection:
  - Price-related metrics: *open_value, low_value, high_value, close_value, volume*
  - Technical indicators: *rsi_14, macd, atr*
  - Macroeconomic indicators: *unemployment_rate, inflation_cpi, stock_market_volatility_vix_index, interest_rate_fed_funds*
  - Target columns: *Tomorrow, Target*

- Data Splitting:
  - The dataset is split into individual DataFrames for each company (**AAPL, MSFT, PLTR, KLAC, QCOM, CRWD**), for further training and ease in evaluating each individual company.
  - *date_value* is converted to a datetime.date object and set as the index for efficient time-series operations.

# V. Advanced Predictive Feature Engineering with Wavelet Denoising

An advanced signal processing step is introduced to enhance the predictive quality of the price data:

- Wavelet Denoising:
  - Discrete Wavelet Transform (DWT) with Daubechies 4 (db4) wavelet, decomposition level 3
  - Universal soft threshold applied to wavelet coefficients to suppress noise
  - Signal reconstructed using inverse wavelet transform

- Derived Features from Denoised Series:
  - *Denoised_Close*: Smoothed version of the stock’s price trajectory
  - *Returns*: Percentage change of the denoised close price
  - *RSI*: 14-day RSI computed from the denoised signal
  - *Tomorrow*: Next day’s denoised close value
  - *Target*: Binary classification label indicating whether the price increases the next day

# VI. Clustering Preprocessing and Risk Categorization

A dedicated clustering pipeline is developed for time-sensitive risk profiling:

- Feature Engineering for Clustering:
  - *daily_return*: Percentage change in closing price
  - *volatility_30*: 30-day rolling standard deviation of returns
  - *max_drawdown*: 60-day rolling peak-to-trough loss percentage
  - *avg_volume*: 30-day average trading volume
  - *momentum*: 5-day price momentum

- Handling Missing Values:
  - Two-step imputation: Company-wise mean filling, then backward and forward fill for edge NaNs
  - Final safety check to ensure no missing values

- Clustering and Risk Mapping:
  - StandardScaler used for standardization
  - KMeans clustering segments historical rows into three risk-based groups
  - Clusters mapped to qualitative risk levels (Low, Medium, High) using *volatility_30* and *max_drawdown*
  - Each row assigned a *risk_level*, resulting in a labeled dataset (*df_risk_prep*)
  - Dataset merged back into the main table for enriched behavioral context

# VII. Merging Wavelet-Derived Predictive Features

After wavelet-denoised features are computed for each company:
  - All individual company DataFrames are reset to bring date_value back as a column
  - DataFrames concatenated into a single DataFrame (*all_data*), sorted by date_value
  - Company tickers normalized to uppercase, date_value cast to date-only format
  - Final subset of features selected: Denoised_Close, Returns, RSI, Tomorrow, Target

The main dataset is purged of previous versions of Tomorrow and Target, then enriched via an inner join using *date_value* and *company_prefix*. This ensures only aligned, high-integrity records are retained.

### Result:
The main table now includes denoised technical features alongside macroeconomic, volatility, and classic indicators, enabling more robust, noise-aware model training and analysis

# VIII. Database Design Considerations for Future Development

The database architecture is designed with future growth and enhancement in mind, prioritizing key aspects such as security, privacy, reliability, and scalability.

In terms of *security*, future plans include implementing robust encryption protocols for data both at rest and in transit, ensuring sensitive information remains protected. Role-based access control mechanisms will be introduced to restrict data access strictly to authorized users. Additionally, audit logging is anticipated to provide comprehensive tracking of data access and modifications, supporting compliance and forensic needs.

*Privacy* considerations will focus on minimizing data collection to only what is necessary while developing consent management features that give users greater control over their personal data. Techniques for anonymizing personally identifiable information will be explored to protect user identities, especially in analytics and reporting contexts, aligning the system with regulations such as GDPR and CCPA.

*Reliability* will be enhanced by establishing automated backup processes and incorporating redundancy to prevent data loss and minimize downtime. Future implementations will also include real-time monitoring and alerting systems to proactively identify and address potential database performance issues or failures.

To support *scalability*, the database schema will maintain a modular and normalized structure to facilitate seamless evolution and expansion. Architectural plans involve enabling horizontal scaling through methods like sharding and read replicas, as well as deploying on scalable cloud-managed database services to efficiently handle growing data volumes and user loads.

# IX. Database Schema Design

**Move Tickers** utilizes a dual-schema approach to organize its data effectively, supporting both *market-centric analytics* and *user-centric interactions*. This separation ensures clarity, performance, and scalability for current functionalities and future enhancements.

## **1. Market-Centric Schema**

**Design Type**: *Star Schema*

![Market-Centric Schema](https://github.com/BredaUniversityADSAI/2024-25-y1d-teamwork-group-6/blob/9927a6f8c1ce5bd5e845cc017bba9dafae512eae/Deliverables/ILO%206.3/Database%20Schema%20-%20Stock%20Market%20Orientated%20-%20Image.png)

<u>Purpose</u> 
    This schema forms the analytical backbone of Move Tickers’ financial data processing. It captures historical stock market data, technical indicators, macroeconomic metrics, and risk-level classifications. The core objective is to enable deep exploration of market behavior and machine learning model outputs, such as risk clustering and next-day price direction predictions.

<u>Key Features</u> 

- Fact Table: 
    - fact_stock_daily_metrics — stores daily stock price movements, technical indicators, and derived features aligned with companies and dates.

- Dimension Tables
    - dim_company — holds company metadata (ticker, name, sector, industry, country).
    - dim_date — date attributes supporting time-based analyses
    - dim_risk_level — categorical risk labels for companies derived from clustering models.

<u>Rationale</u> 
    The star schema’s denormalized design minimizes query complexity and maximizes analytical performance when slicing data across companies, dates, or risk levels. This simplicity aligns with the platform’s goal of providing fast, interactive exploration of historical financial patterns and ML insights.

<u>Relevance</u> 
    This schema supports core functionalities such as comparative risk analysis, pattern recognition in stock behavior, and scenario-based learning, which are central to improving financial literacy among users.

## **2. User-Centric Schema**

**Design Type:** *Star Schema*

![User-Centric Schema](https://github.com/BredaUniversityADSAI/2024-25-y1d-teamwork-group-6/blob/9927a6f8c1ce5bd5e845cc017bba9dafae512eae/Deliverables/ILO%206.3/Database%20Schema%20-%20User%20Orientated%20-%20Image.png)

<u>Purpose</u> 
    This schema is dedicated to managing user data, their interactions, and personalized analytics. It tracks user profiles, their prediction requests, trading strategies, and portfolio simulations, enabling a tailored learning experience and strategy evaluation.

<u>Key Features</u> 

- Fact Tables:
    - fact_user_predictions — records user-generated stock predictions linked to ML models.
    - fact_user_strategy_performance — monitors performance metrics of user-selected or created strategies over time.
    - fact_user_simulations — logs portfolio simulation snapshots for practice and scenario testing.

- Dimension Tables:
    - dim_user — user profile and metadata.
    - dim_date — consistent date dimension shared with market schema for temporal analysis.
    - dim_model — details on machine learning models generating predictions.
    - dim_strategy — repository of trading strategies.

<u>Rationale</u> 
    Separating the user schema from the market data avoids cluttering core financial data with user-specific attributes. The star schema facilitates fast querying and aggregation of user behavior and outcomes, which is vital for delivering personalized insights and tracking progress.

<u>Relevance</u>  
    This schema directly supports Move Tickers’ educational mission by enabling interactive features like personalized predictions, strategy backtesting, and simulation-based learning — all crucial for user engagement and skill development.

# X. Summary and Conclusions

The Move Tickers platform provides an educational web environment for exploring financial markets using advanced, data-driven insights. Two core machine learning pipelines—risk clustering and next-day price direction—enable users to analyze market sensitivity, volatility, and short-term stock movements, all with the goal of promoting financial literacy and critical thinking rather than real-time trading.

A robust data engineering process underlies the platform, integrating daily stock data with macroeconomic and technical indicators, applying advanced cleaning, and enhancing predictive features through wavelet denoising. This results in a high-quality, noise-aware dataset for model training and analysis.

The database architecture is designed for future security, privacy, reliability, and scalability, employing a dual star-schema approach to separate market analytics from user interactions. This structure supports fast, interactive exploration, personalized learning, and future growth.

In summary, Move Tickers combines rigorous data preparation, machine learning, and thoughtful database design to empower users with tools for financial analysis and simulation, supporting its mission to improve financial literacy through interactive, scenario-based learning.