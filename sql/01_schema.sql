-- Customer Churn & Revenue Intelligence Platform
-- Core normalized schema

USE churn_platform;

SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS churn_events;
DROP TABLE IF EXISTS support_tickets;
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS subscriptions;
DROP TABLE IF EXISTS ml_predictions;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id         VARCHAR(20) PRIMARY KEY,
    gender               VARCHAR(10),
    senior_citizen       TINYINT,
    partner              TINYINT,
    dependents            TINYINT,
    signup_date          DATE NOT NULL,
    city                  VARCHAR(60),
    state                 VARCHAR(60),
    country               VARCHAR(60) DEFAULT 'Germany'
) ENGINE=InnoDB;

CREATE TABLE subscriptions (
    subscription_id      INT AUTO_INCREMENT PRIMARY KEY,
    customer_id           VARCHAR(20) NOT NULL,
    contract_type         VARCHAR(20),      -- Month-to-month, One year, Two year
    payment_method        VARCHAR(40),
    paperless_billing      TINYINT,
    internet_service       VARCHAR(20),      -- DSL, Fiber optic, No
    phone_service          TINYINT,
    multiple_lines         VARCHAR(20),
    online_security        VARCHAR(20),
    online_backup           VARCHAR(20),
    device_protection       VARCHAR(20),
    tech_support             VARCHAR(20),
    streaming_tv              VARCHAR(20),
    streaming_movies          VARCHAR(20),
    monthly_charges           DECIMAL(10,2),
    tenure_months              INT,
    start_date                  DATE,
    end_date                    DATE NULL,
    is_active                    TINYINT DEFAULT 1,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
) ENGINE=InnoDB;

CREATE TABLE transactions (
    transaction_id        BIGINT AUTO_INCREMENT PRIMARY KEY,
    customer_id            VARCHAR(20) NOT NULL,
    subscription_id         INT NOT NULL,
    transaction_date         DATE NOT NULL,
    amount                     DECIMAL(10,2) NOT NULL,
    transaction_type           VARCHAR(20),   -- charge, refund, credit
    status                       VARCHAR(20),   -- paid, failed, pending
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (subscription_id) REFERENCES subscriptions(subscription_id)
) ENGINE=InnoDB;

CREATE TABLE support_tickets (
    ticket_id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    customer_id             VARCHAR(20) NOT NULL,
    opened_date               DATE NOT NULL,
    closed_date                DATE NULL,
    category                    VARCHAR(40),  -- billing, technical, cancellation, general
    priority                     VARCHAR(10),  -- low, medium, high
    satisfaction_score            TINYINT NULL, -- 1-5
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
) ENGINE=InnoDB;

CREATE TABLE churn_events (
    churn_id                BIGINT AUTO_INCREMENT PRIMARY KEY,
    customer_id              VARCHAR(20) NOT NULL,
    churn_date                 DATE NOT NULL,
    churn_flag                  TINYINT NOT NULL, -- 1 = churned
    churn_reason                 VARCHAR(60) NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
) ENGINE=InnoDB;

-- ML predictions written back from the Python ML layer
CREATE TABLE ml_predictions (
    customer_id              VARCHAR(20) PRIMARY KEY,
    churn_probability          DECIMAL(6,5),
    risk_tier                    VARCHAR(10),   -- Low, Medium, High
    predicted_clv                 DECIMAL(12,2),
    segment_cluster                 INT,
    model_version                    VARCHAR(20),
    scored_at                          DATETIME,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
) ENGINE=InnoDB;

SET FOREIGN_KEY_CHECKS = 1;

CREATE INDEX idx_sub_customer ON subscriptions(customer_id);
CREATE INDEX idx_txn_customer ON transactions(customer_id);
CREATE INDEX idx_txn_date ON transactions(transaction_date);
CREATE INDEX idx_tickets_customer ON support_tickets(customer_id);
CREATE INDEX idx_churn_customer ON churn_events(customer_id);
