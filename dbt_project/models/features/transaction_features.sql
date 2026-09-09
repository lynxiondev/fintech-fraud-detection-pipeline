{{
  config(
    materialized='table'
  )
}}

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

base_features as (

    select
        *,

        -- Historical transaction count (excluye la actual)
        row_number() over (
            partition by customer_id
            order by timestamp, transaction_id
        ) - 1 as customer_transaction_count,

        -- Historical average amount (excluye la actual)
        avg(amount) over (
            partition by customer_id
            order by timestamp, transaction_id
            rows between unbounded preceding and 1 preceding
        ) as customer_avg_amount,

        -- Arrays de historia previa (para unique + is_new)
        array_agg(country) over (
            partition by customer_id
            order by timestamp, transaction_id
            rows between unbounded preceding and 1 preceding
        ) as prev_countries,

        array_agg(device_type) over (
            partition by customer_id
            order by timestamp, transaction_id
            rows between unbounded preceding and 1 preceding
        ) as prev_devices,

        -- Velocity features (excluyen la transacción actual)
        count(*) over (
            partition by customer_id
            order by unix_seconds(cast(timestamp as timestamp))
            range between 3599 preceding and 1 preceding
        ) as transactions_last_1h,

        count(*) over (
            partition by customer_id
            order by unix_seconds(cast(timestamp as timestamp))
            range between 86399 preceding and 1 preceding
        ) as transactions_last_24h,

        sum(amount) over (
            partition by customer_id
            order by unix_seconds(cast(timestamp as timestamp))
            range between 3599 preceding and 1 preceding
        ) as amount_last_1h,

        sum(amount) over (
            partition by customer_id
            order by unix_seconds(cast(timestamp as timestamp))
            range between 86399 preceding and 1 preceding
        ) as amount_last_24h

    from transactions

),

final as (

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
        is_fraud,

        customer_transaction_count,
        customer_avg_amount,

        -- Ratio amount vs historical average
        safe_divide(amount, customer_avg_amount) as amount_vs_customer_avg,

        -- Unique countries / devices vistos hasta antes de esta transacción
        array_length(array(select distinct c from unnest(prev_countries) as c)) as customer_unique_countries,
        array_length(array(select distinct d from unnest(prev_devices) as d)) as customer_unique_devices,

        -- Flags de nuevo país / dispositivo
        case
            when prev_countries is null then 1
            when country in unnest(prev_countries) then 0
            else 1
        end as is_new_country,

        case
            when prev_devices is null then 1
            when device_type in unnest(prev_devices) then 0
            else 1
        end as is_new_device,

        -- Velocity (rellenamos nulls con 0)
        coalesce(transactions_last_1h, 0) as transactions_last_1h,
        coalesce(transactions_last_24h, 0) as transactions_last_24h,
        coalesce(amount_last_1h, 0) as amount_last_1h,
        coalesce(amount_last_24h, 0) as amount_last_24h,

        -- Interaction feature
        is_international * case when payment_method = 'digital_wallet' then 1 else 0 end as international_digital_wallet

    from base_features

)

select * from final