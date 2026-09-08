with transactions as (

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
    from {{ ref('stg_transactions') }}

),

features as (

    select
        *,
        
        row_number() over (
            partition by customer_id
            order by timestamp, transaction_id
        ) - 1 as customer_transaction_count,

        avg(amount) over (
            partition by customer_id
            order by timestamp, transaction_id
            rows between unbounded preceding and 1 preceding
        ) as customer_avg_amount

    from transactions

)

select *
from features
