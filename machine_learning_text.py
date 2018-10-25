# Importing modules
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
%matplotlib inline
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

# Read datasets/papers.csv into papers

papers = pd.read_csv('datasets/papers.csv', error_bad_lines=False)

# Print out the first rows of papers
print(papers.head())

# Remove the columns
papers = papers.drop(['id','event_type','pdf_name'], axis=1)

# Print out the first rows of papers
print(papers.head())

# Group the papers by year
groups = papers.groupby('year')

# Determine the size of each group
counts = groups.size()

# Visualise the counts as a bar plot
counts.plot(kind='bar')

# Print the text of the first rows
print(papers['paper_text'].head())

# Remove punctuation
papers['text_processed'] = papers['paper_text'].map(lambda x: re.sub('[,\.!?"()@:;~/]', ' ', x))

# Remove words with string size less than 3
papers['text_processed'] = papers['text_processed'].map(lambda x: re.sub(r'\b\w{,3}\b', '', x))

#remove \n's from processed text
#papers['text_processed'] = papers['text_processed'].map(lambda x: re.sub(r'\n', '', x))

#remove tabs, newlines, whitespace-like from text
papers['text_processed']=papers['text_processed'].map(lambda x: re.sub('\s+', ' ', x).strip())

# Convert the titles to lowercase
papers['text_processed'] = papers['text_processed'].map(lambda x:x.lower())

# Print the processed titles of the first rows
print(papers['text_processed'].head())


# Separate via decade
papers.loc[papers['year']<=2017,'decade']='2007-2017'
papers.loc[papers['year']<2007,'decade']='1997-2006'
papers.loc[papers['year']<1997,'decade']='1987-1996'

# Join the different processed texts together, sampling by decade
papers1 = pd.pivot_table(papers, values='text_processed', index='decade',
                         aggfunc=' '.join)
papers2=papers1.text_processed.astype(str)

# Generate a word cloud for each decade

for idx in papers1.index:
    wc=WordCloud(width=1000,
                 height=600,
                 background_color='black',
                 stopwords=STOPWORDS)
    wc.generate(papers1['text_processed'].loc[idx])
    plt.imshow(wc, interpolation='bilinear')
    plt.xlabel(idx)
    plt.tight_layout(pad=0)
    plt.show()

# Helper function
def plot_10_most_common_words(count_data, count_vectorizer, decade):
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]

    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))

    plt.bar(x_pos, counts,align='center')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words in %s' %(decade))
    plt.ylabel('counts')
    plt.title('10 most common words by decade')
    plt.show()

# Create a loop to iterate over each decade
for idx in papers1.index:
    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer(stop_words='english')

    # Fit and transform the processed titles
    count_data = count_vectorizer.fit_transform([papers1['text_processed'].loc[idx]])

    # Visualise the 10 most common words by decade
    plot_10_most_common_words(count_data, count_vectorizer, idx)


# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

# Tweak the two parameters below (use int values below 15)
number_topics = 1
number_words = 10

# Create a loop to iterate over each decade
for idx in papers1.index:
    count_vectorizer = CountVectorizer(stop_words='english')
    count_data = count_vectorizer.fit_transform([papers1['text_processed'].loc[idx]])

    # Create and fit the LDA model
    lda = LDA(n_components=number_topics)
    lda.fit(count_data)

    # Print the words found by the LDA model
    print("Top words used between %s via LDA:" %(idx))
    print_topics(lda, count_vectorizer, number_words)

