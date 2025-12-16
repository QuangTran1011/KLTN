from openai import OpenAI
import numpy as np
import pandas as pd
import time

client = OpenAI(api_key='')

text_about_book = """When Megan Murphy discovered a floppy-eared rabbit gnawing on the hem o her skirt, she meant to give its careless owner a piece of her mind, but Dr. Patrick Hunter was too attractive to stay mad at for long. Soon the two are making Thanksgiving dinner for their families."""

instructions = """You are an expert book classifier. 
Generate 5 concise and meaningful tags representing the book's topic, genre, and literary type. 
Include format when applicable. 
Prefer generalizable tags; avoid very specific historical references unless essential. 
Only output tags separated by commas."""

input_text = f'Description: "{text_about_book}"\n\nTags:'

# response = client.responses.create(
#     model="gpt-4o-mini",
#     instructions=instructions,
#     input=input_text,
#     max_output_tokens=100,  # Đúng rồi!
#     temperature=0.1
# )

# tags = [t.strip() for t in response.output[0].content[0].text.split(",") if t.strip()]
# tags = [t.lower() for t in tags]  

# print(type(metadata_raw_df["features"][0]))

meta_df = pd.read_parquet('features_merged.parquet')

# Giả sử meta_df là DataFrame gốc
sample_df = meta_df.sample(n=3000, random_state=42)  # random_state để reproducible

batch_size = 300
predicted_tags = []

for start in range(0, len(sample_df), batch_size):
    end = start + batch_size
    batch = sample_df.iloc[start:end]
    print(f"Processing batch {start} to {end-1}")

    for idx, row in batch.iterrows():
        input_text = f'Description: "{row["feature_text"]}"\n\nTags:'
        
        try_count = 0
        while try_count < 3:  # retry tối đa 3 lần
            try:
                response = client.responses.create(
                    model="gpt-4o-mini",
                    instructions=instructions,
                    input=input_text,
                    max_output_tokens=100,
                    temperature=0.2
                )
                text_out = response.output[0].content[0].text
                tags = [t.strip().lower() for t in text_out.split(",") if t.strip()]
                break  # thành công → thoát loop
            except Exception as e:
                try_count += 1
                print(f"Request failed for index {idx}, try {try_count}: {e}")
                time.sleep(5)  # chờ 5 giây trước khi thử lại
                tags = []  # nếu fail sau 3 lần, bỏ qua
        
        predicted_tags.append(tags)
        
    time.sleep(60)

sample_df['predicted_tags'] = predicted_tags
sample_df.to_parquet("llm_tag_data.parquet", index=False)