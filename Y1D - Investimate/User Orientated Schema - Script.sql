-- Dimension: Users
CREATE TABLE dim_user (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dimension: Dates
CREATE TABLE dim_date (
    date_key SERIAL PRIMARY KEY,
    date_value DATE NOT NULL UNIQUE,
    year INT,
    month INT,
    day INT,
    weekday INT,
    quarter INT
);

-- Dimension: Models
CREATE TABLE dim_model (
    model_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dimension: Strategies
CREATE TABLE dim_strategy (
    strategy_id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fact: User Predictions
CREATE TABLE fact_user_predictions (
    prediction_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES dim_user(user_id),
    company_key INT NOT NULL,  -- assumes dim_company exists in market schema
    date_key INT NOT NULL REFERENCES dim_date(date_key),
    model_id INT NOT NULL REFERENCES dim_model(model_id),
    predicted_direction VARCHAR(20),
    prediction_confidence FLOAT,
    actual_outcome VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fact: User Strategy Performance
CREATE TABLE fact_user_strategy_performance (
    performance_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES dim_user(user_id),
    strategy_id INT NOT NULL REFERENCES dim_strategy(strategy_id),
    date_key INT NOT NULL REFERENCES dim_date(date_key),
    returns FLOAT,
    max_drawdown FLOAT,
    volatility FLOAT
);

-- Fact: User Simulations
CREATE TABLE fact_user_simulations (
    simulation_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES dim_user(user_id),
    date_key INT NOT NULL REFERENCES dim_date(date_key),
    company_key INT NOT NULL,  -- assumes dim_company exists in market schema
    shares_held INT,
    cash_balance FLOAT,
    portfolio_value FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
