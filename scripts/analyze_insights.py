import pandas as pd
from collections import Counter

try:
    df = pd.read_csv('OnlineNewsPopularity/Data/OnlineNewsPopularity.csv')
    # Clean columns
    df.columns = df.columns.str.strip()
    
    # 1. Text Analysis
    print("--- Top Words in Viral Articles ---")
    top_10_shares = df['shares'].quantile(0.90)
    viral = df[df['shares'] >= top_10_shares]
    
    stopwords = {'the', 'a', 'an', 'in', 'on', 'of', 'for', 'to', 'and', 'is', 'with', 'at', 'by', 'from', 'it', 'that', 'as', 'be', 'video', 'new', 'photos', 'one', 'how', 'what', 'when'}
    
    all_words = []
    for url in viral['url']:
        if not isinstance(url, str): continue
        parts = url.split('/')
        # Filter empty and domain
        parts = [p for p in parts if len(p) > 2 and not p.isdigit() and 'mashable' not in p and 'http' not in p]
        if parts:
            slug = parts[-1]
            words = slug.replace('-', ' ').split()
            all_words.extend([w.lower() for w in words if w.lower() not in stopwords and w.isalpha()])
            
    print(Counter(all_words).most_common(10))
    
    # 2. Channel Analysis
    print("\n--- Median Shares by Channel ---")
    channels = ['data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus', 
                'data_channel_is_socmed', 'data_channel_is_tech', 'data_channel_is_world']
    
    for col in channels:
        name = col.replace('data_channel_is_', '')
        median = df[df[col] == 1]['shares'].median()
        print(f"{name}: {median}")

except Exception as e:
    print(f"Error: {e}")
