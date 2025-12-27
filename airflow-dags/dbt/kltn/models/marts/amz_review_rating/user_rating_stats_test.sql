{{ config(materialized='table') }}
with
raw as (
  select
    -- Prevent duplicated rows due to possibly unexpected ingestion error
    distinct *
  from
    {{ source('amz_review_rating', 'traindatareviewtest') }}
)
, raw_agg as (
-- Dedup the aggregated data by all columns
select distinct
    user_id,
    timestamp,
    COUNT(*) OVER (
        PARTITION BY user_id 
        ORDER BY UNIX_SECONDS(timestamp)
        RANGE BETWEEN 7776000 PRECEDING AND CURRENT ROW  -- 90 days in seconds
    ) AS user_rating_cnt_90d,
    AVG(rating) OVER (
        PARTITION BY user_id 
        ORDER BY UNIX_SECONDS(timestamp)
        RANGE BETWEEN 7776000 PRECEDING AND 1 PRECEDING  -- 90 days in seconds
    ) AS user_rating_avg_prev_rating_90d,
    ARRAY_TO_STRING(
        ARRAY_AGG(parent_asin) OVER (
            PARTITION BY user_id
            ORDER BY timestamp
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
        ), 
        ','
    ) AS user_rating_list_10_recent_asin,
    ARRAY_TO_STRING(
        ARRAY_AGG(CAST(UNIX_SECONDS(timestamp) AS STRING)) OVER (
            PARTITION BY user_id
            ORDER BY timestamp
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
        ), 
        ','
    ) AS user_rating_list_10_recent_asin_timestamp
FROM
    raw
)
-- There are cases where the there are more than 10 preceding rows for a timestamp duplicated column, for example:
-- user A, item X, timestamp 12
-- user A, item Y, timestamp 12
-- But before the above two rows there are many other rows
-- In this case array_agg operation above would result in two aggregated rows with different value where one might contain less collated items
-- So when there is duplicated user_id and timestamp we select the ones with more collated items
, agg_dedup as (
select
     *,
     ROW_NUMBER() OVER (
         PARTITION BY user_id, timestamp 
         ORDER BY ARRAY_LENGTH(SPLIT(user_rating_list_10_recent_asin, ',')) DESC
     ) as dedup_rn
from
    raw_agg
)
, agg_final as (
select
    *
from
    agg_dedup
where 1=1
    and dedup_rn = 1
)
select * from agg_final