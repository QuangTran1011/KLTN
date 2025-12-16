{{ config(materialized='table') }}
with
raw as (
select
  -- Prevent duplicated rows due to possibly unexpected ingestion error
  distinct *
from
  {{ source('amz_review_rating', 'traindatareview') }}
)
select distinct
    timestamp,
    parent_asin,
    -- item agg
    COUNT(*) OVER (
        PARTITION BY parent_asin 
        ORDER BY UNIX_SECONDS(timestamp)
        RANGE BETWEEN 31536000 PRECEDING AND 1 PRECEDING  -- 365 days in seconds
    ) AS parent_asin_rating_cnt_365d,
    AVG(rating) OVER (
        PARTITION BY parent_asin 
        ORDER BY UNIX_SECONDS(timestamp)
        RANGE BETWEEN 31536000 PRECEDING AND 1 PRECEDING
    ) AS parent_asin_rating_avg_prev_rating_365d,
    COUNT(*) OVER (
        PARTITION BY parent_asin 
        ORDER BY UNIX_SECONDS(timestamp)
        RANGE BETWEEN 7776000 PRECEDING AND 1 PRECEDING  -- 90 days in seconds
    ) AS parent_asin_rating_cnt_90d,
    AVG(rating) OVER (
        PARTITION BY parent_asin 
        ORDER BY UNIX_SECONDS(timestamp)
        RANGE BETWEEN 7776000 PRECEDING AND 1 PRECEDING
    ) AS parent_asin_rating_avg_prev_rating_90d,
    COUNT(*) OVER (
        PARTITION BY parent_asin 
        ORDER BY UNIX_SECONDS(timestamp)
        RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING  -- 30 days in seconds
    ) AS parent_asin_rating_cnt_30d,
    AVG(rating) OVER (
        PARTITION BY parent_asin 
        ORDER BY UNIX_SECONDS(timestamp)
        RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING
    ) AS parent_asin_rating_avg_prev_rating_30d,
    COUNT(*) OVER (
        PARTITION BY parent_asin 
        ORDER BY UNIX_SECONDS(timestamp)
        RANGE BETWEEN 604800 PRECEDING AND 1 PRECEDING  -- 7 days in seconds
    ) AS parent_asin_rating_cnt_7d,
    AVG(rating) OVER (
        PARTITION BY parent_asin 
        ORDER BY UNIX_SECONDS(timestamp)
        RANGE BETWEEN 604800 PRECEDING AND 1 PRECEDING
    ) AS parent_asin_rating_avg_prev_rating_7d
FROM
    raw