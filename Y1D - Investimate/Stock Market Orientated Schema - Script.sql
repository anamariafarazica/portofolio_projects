CREATE TABLE dim_date (
    date_key DATE PRIMARY KEY,
    day INT,
    month INT,
    quarter INT,
    year INT,
    day_of_week VARCHAR(10),
    is_weekend BOOLEAN
);

CREATE TABLE dim_company (
    company_key SERIAL PRIMARY KEY,
    company_prefix VARCHAR(50) UNIQUE NOT NULL
    -- add company_name, sector, etc. if you have
);

CREATE TABLE dim_risk_level (
    risk_level_key SERIAL PRIMARY KEY,
    risk_level VARCHAR(20) UNIQUE NOT NULL,
    description TEXT
);

INSERT INTO dim_date (date_key, day, month, quarter, year, day_of_week, is_weekend)
SELECT DISTINCT 
    date_value AS date_key,
    EXTRACT(DAY FROM date_value) AS day,
    EXTRACT(MONTH FROM date_value) AS month,
    EXTRACT(QUARTER FROM date_value) AS quarter,
    EXTRACT(YEAR FROM date_value) AS year,
    TO_CHAR(date_value, 'Day') AS day_of_week,
    CASE WHEN EXTRACT(DOW FROM date_value) IN (0,6) THEN TRUE ELSE FALSE END AS is_weekend
FROM final_data;

INSERT INTO dim_company (company_prefix)
SELECT DISTINCT company_prefix
FROM final_data;

INSERT INTO dim_risk_level (risk_level)
SELECT DISTINCT risk_level
FROM final_data;


CREATE TABLE fact_stock_daily_metrics (
    -- Surrogate key for fact table (optional but recommended for uniqueness)
    fact_id SERIAL PRIMARY KEY,

    -- Foreign keys (to be linked after dimension tables are created)
    date_key DATE NOT NULL,
    company_key INT NOT NULL,
    risk_level_key INT NOT NULL,

    -- Stock market data
    open_value DECIMAL(18,6) NOT NULL,
    close_value DECIMAL(18,6) NOT NULL,
    low_value DECIMAL(18,6) NOT NULL,
    high_value DECIMAL(18,6) NOT NULL,
    volume BIGINT NOT NULL,

    -- Returns and technical indicators
    return_1d DECIMAL(10,6) NOT NULL,
    return_3d DECIMAL(10,6) NOT NULL,
    sma_10 DECIMAL(18,6) NOT NULL,
    sma_50 DECIMAL(18,6) NOT NULL,
    sma_diff DECIMAL(18,6) NOT NULL,
    rsi_14 DECIMAL(18,6) NOT NULL,
    atr DECIMAL(18,6) NOT NULL,
    macd DECIMAL(18,6) NOT NULL,

    -- Macroeconomic indicators (daily, tied to date)
    unemployment_rate DECIMAL(10,6) NOT NULL,
    inflation_cpi DECIMAL(10,6) NOT NULL,
    stock_market_volatility_vix_index DECIMAL(10,6) NOT NULL,
    interest_rate_fed_funds DECIMAL(10,6) NOT NULL,

    -- Additional metrics
    daily_return DECIMAL(10,6) NOT NULL,
    volatility_30 DECIMAL(10,6) NOT NULL,
    max_drawdown DECIMAL(10,6) NOT NULL,
    avg_volume BIGINT NOT NULL,
    momentum DECIMAL(18,6) NOT NULL,

    -- ML outputs
    cluster INT NOT NULL,
    
    denoised_close DECIMAL(18,6) NOT NULL,
    returns DECIMAL(10,6) NOT NULL,
    rsi DECIMAL(18,6) NOT NULL,
    tomorrow_prediction INT NOT NULL,  -- 0/1 for classification
    target INT NOT NULL,               -- 0/1 actual label

    -- Indexes can be added after loading data for performance

    CONSTRAINT fk_date FOREIGN KEY (date_key) REFERENCES dim_date(date_key),
    CONSTRAINT fk_company FOREIGN KEY (company_key) REFERENCES dim_company(company_key),
    CONSTRAINT fk_risk_level FOREIGN KEY (risk_level_key) REFERENCES dim_risk_level(risk_level_key)
);


INSERT INTO fact_stock_daily_metrics (
    date_key,
    company_key,
    risk_level_key,
    open_value,
    close_value,
    low_value,
    high_value,
    volume,
    return_1d,
    return_3d,
    sma_10,
    sma_50,
    sma_diff,
    rsi_14,
    atr,
    macd,
    unemployment_rate,
    inflation_cpi,
    stock_market_volatility_vix_index,
    interest_rate_fed_funds,
    daily_return,
    volatility_30,
    max_drawdown,
    avg_volume,
    momentum,
    cluster,
    denoised_close,
    returns,
    rsi,
    tomorrow_prediction,
    target
)
SELECT
    fd.date_value,
    dc.company_key,
    dr.risk_level_key,
    fd.open_value,
    fd.close_value,
    fd.low_value,
    fd.high_value,
    fd.volume,
    fd.return_1d,
    fd.return_3d,
    fd.sma_10,
    fd.sma_50,
    fd.sma_diff,
    fd.rsi_14,
    fd.atr,
    fd.macd,
    fd.unemployment_rate,
    fd.inflation_cpi,
    fd.stock_market_volatility_vix_index,
    fd.interest_rate_fed_funds,
    fd.daily_return,
    fd.volatility_30,
    fd.max_drawdown,
    fd.avg_volume,
    fd.momentum,
    fd.cluster,
    fd."Denoised_Close",
	fd."Returns",
	fd."RSI",
	fd."Tomorrow",
	fd."Target"

FROM
    final_data fd
JOIN
    dim_company dc ON fd.company_prefix = dc.company_prefix
JOIN
    dim_risk_level dr ON fd.risk_level = dr.risk_level;

UPDATE dim_risk_level
SET description = CASE risk_level
    WHEN 'Low Risk' THEN 'Large, stable companies with strong earnings and relatively low price volatility compared to peers.'
    WHEN 'Medium Risk' THEN 'Mid-to-large cap firms with moderate volatility, solid fundamentals, but exposed to sector and market cyclicality.'
    WHEN 'High Risk' THEN 'Highly volatile, growth-oriented companies with unpredictable earnings and sensitivity to market sentiment.'
    ELSE description
END
WHERE risk_level IN ('Low Risk', 'Medium Risk', 'High Risk');

ALTER TABLE dim_company
ADD COLUMN company_name VARCHAR(255);

UPDATE dim_company dc
SET company_name = crd.company_name
FROM company_risk_descriptions crd
WHERE dc.company_prefix = crd.ticker;


ALTER TABLE dim_date
DROP COLUMN is_weekend;

ALTER TABLE dim_company
ADD COLUMN sector VARCHAR(100),
ADD COLUMN industry VARCHAR(100),
ADD COLUMN country VARCHAR(100);

UPDATE dim_company
SET sector = 'Technology',
    industry = 'Consumer Electronics',
    country = 'USA'
WHERE company_prefix = 'AAPL';

UPDATE dim_company
SET sector = 'Technology',
    industry = 'Software & Programming',
    country = 'USA'
WHERE company_prefix = 'PLTR';

UPDATE dim_company
SET sector = 'Technology',
    industry = 'Semiconductors',
    country = 'USA'
WHERE company_prefix = 'KLAC';

UPDATE dim_company
SET sector = 'Technology',
    industry = 'Semiconductors',
    country = 'USA'
WHERE company_prefix = 'QCOM';

UPDATE dim_company
SET sector = 'Technology',
    industry = 'Cybersecurity',
    country = 'USA'
WHERE company_prefix = 'CRWD';

UPDATE dim_company
SET sector = 'Technology',
    industry = 'Software & Programming',
    country = 'USA'
WHERE company_prefix = 'MSFT';

CREATE TABLE dim_user (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


