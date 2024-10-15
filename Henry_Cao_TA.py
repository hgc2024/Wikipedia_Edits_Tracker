import mwclient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import seaborn as sns
import csv
import os
from datetime import datetime
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import spacy

# Connect to Wikipedia
site = mwclient.Site('en.wikipedia.org')
page = site.pages['Brexit']

# Retrieve the edit history
edit_history = []
for revision in page.revisions(prop='ids|user|timestamp|comment|content', limit=None):
    try:
        revid = revision['revid']
        user = revision.get('user', 'Anonymous')
        timestamp = datetime(*revision['timestamp'][:6])
        comment = revision.get('comment', '')
        content = revision.get('*', '')
        edit_history.append({
            'revid': revid,
            'user': user,
            'timestamp': timestamp,
            'comment': comment,
            'content': content
        })
    except Exception as e:
        print(f"Unexpected error encountered: {e} in revision: {revision}")
    if len(edit_history) % 100 == 0:
        print(f"Retrieved {len(edit_history)} revisions...")

print(f"Total revisions retrieved: {len(edit_history)}")
df = pd.DataFrame(edit_history)
output_dir = 'Desktop/KCL/Career/Application_Materials_UK/UCL/Research_Assistant_Computational_Social_Science/Technical_Assignment/Final_Test_2/'
df.to_csv(output_dir + 'brexit_edit_history.csv', index=False, quoting=csv.QUOTE_ALL)

# Load the data
output_dir = '/home/henry-cao/Desktop/KCL/Career/Application_Materials_UK/UCL/Research_Assistant_Computational_Social_Science/Technical_Assignment/Final_Test_2/'
input_dir = '/run/media/henry-cao/Elements/UCL_RA_Wikipedia_Project/UCL_RA_Wikipedia_v5/'
df = pd.read_csv(input_dir + 'brexit_edit_history.csv', quoting=csv.QUOTE_ALL)

# Pre-processing
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

df['timestamp'] = pd.to_datetime(df['timestamp'])

custom_stop_words = set([
    'user', 'edit', 'article', 'revision', 'date', 'source', 'talktalk', 'specialcontributions', 'section',
    'see', 'removed', 'reference', 'top', 'per', 'edits', 'talk', 'page', 'add', 'ce', 'ref', 'also', 'citation',
    'tag', 'update', 'talkmozarttalk', 'wpijustdontlikeit'
])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in custom_stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if len(token) > 1]
    return tokens

df['processed_comments'] = df['comment'].fillna('').apply(preprocess_text)

# Named Entity Recognition
nlp = spacy.load("en_core_web_lg")

def get_entities(text):
    if not isinstance(text, str):
        return []
    try:
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents]
        return entities
    except Exception as e:
        print(f"Error processing text: {text}")
        print(f"Error: {e}")
        return []

df['entities'] = df['comment'].apply(get_entities)

df['year'] = df['timestamp'].dt.year
entity_evolution = df.groupby('year')['entities'].apply(lambda x: Counter([item for sublist in x for item in sublist]))

years = sorted(df['year'].unique())
entity_evolution_dict = {}
for year in years:
    if year in entity_evolution:
        entity_evolution_dict[year] = entity_evolution[year]
    else:
        entity_evolution_dict[year] = Counter()

entity_evolution_df = pd.DataFrame.from_dict(entity_evolution_dict, orient='columns')
entity_evolution_df['Total'] = entity_evolution_df.sum(axis=1)

drop_entities = [
    '1', 'first', '2', 'two', 'one', 'State', '3', 'three', 
    'second', 'Second', 'Third', 'UCB|use', 'Juncker', 'Contributions/2A02',
    'Second'
]

entity_evolution_df = entity_evolution_df.drop(drop_entities, errors='ignore')

# Heatmap
heatmap_df = entity_evolution_df.sort_values('Total', ascending=False)
top_n = 40
heatmap_df = heatmap_df.head(top_n)
heatmap_df = heatmap_df.drop('Total', axis=1)

plt.figure(figsize=(20, 12))

min_nonzero = heatmap_df[heatmap_df > 0].min().min()
heatmap_df_adj = heatmap_df.replace(0, min_nonzero / 10)

norm = SymLogNorm(linthresh=min_nonzero, linscale=1, vmin=heatmap_df_adj.min().min(), vmax=heatmap_df_adj.max().max())

ax = sns.heatmap(heatmap_df_adj, annot=False, cmap='YlOrRd', 
            norm=norm,
            cbar_kws={'label': 'Mention Count'})

cbar = ax.collections[0].colorbar
tick_locations = [1, 5, 10, 25, 50, 100, 200]
cbar.set_ticks(tick_locations)
cbar.set_ticklabels([f'{x:g}' for x in tick_locations])
cbar.set_label('Mention Count', rotation=270, labelpad=20)

plt.title('Top 30 Entity Mentions Heatmap')
plt.xlabel('Year')
plt.ylabel('Entity')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(output_dir + 'entity_heatmap_custom_scale.png', dpi=300, bbox_inches='tight')
plt.close()

# Edit counts over time
df['date'] = df['timestamp'].dt.date
edits_per_day = df.groupby('date').size()

plt.figure(figsize=(10, 6))
edits_per_day.plot(kind='line')
plt.title('Number of Edits per Day')
plt.xlabel('Date')
plt.ylabel('Number of Edits')
plt.grid(True)
plt.savefig(output_dir + 'edits_per_day.png')
plt.close()