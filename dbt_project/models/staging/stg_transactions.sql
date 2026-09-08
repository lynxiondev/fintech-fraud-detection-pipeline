select
    transaction_id,
    timestamp,
    customer_id,
    amount,
    currency,
    merchant_category,
    payment_method,
    country,
    device_type,
    is_international,
    is_fraud
from {{ ref('transactions') }}
