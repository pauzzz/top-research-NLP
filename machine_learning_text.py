# Importing modules
import pandas as pd

# Read datasets/papers.csv into papers

papers = pd.read_csv('datasets/papers.csv', error_bad_lines=False)

# Print out the first rows of papers
print(papers.head())

# Remove the columns
papers=papers.drop(['id','event_type','pdf_name'], axis=1)

# Print out the first rows of papers
print(papers.head())

# Group the papers by year
groups = papers.groupby('year')

# Determine the size of each group
counts = groups.size()

# Visualise the counts as a bar plot
import matplotlib.pyplot
%matplotlib inline
counts.plot(kind='bar')

# Load the regular expression library
import re

# Print the titles of the first rows
print(papers['paper_text'].head())

# Remove punctuation
papers['text_processed'] = papers['paper_text'].map(lambda x: re.sub('[,\.!?]', '', x))

#Remove words with string size less than 3
papers['text_processed'] = papers['text_processed'].map(lambda x: re.sub(r'\b\w{,3}\b', '', x))

#remove \n's from processed text
papers['text_processed'] = papers['text_processed'].map(lambda x: re.sub(r'\n', '', x))

# Convert the titles to lowercase
papers['text_processed'] = papers['text_processed'].map(lambda x:x.lower())

# Print the processed titles of the first rows
papers['text_processed'].head()


# Import the wordcloud library
import wordcloud

# Join the different processed texts together, sampling by year
texts_by_year={}
for year in papers['year']:
    texts_by_year[year]=papers.groupby('year')['text_processed'].

# Create a WordCloud object
wordcloud = wordcloud.WordCloud()

# Generate a word cloud
wordcloud.generate(long_string)

# Visualize the word cloud
wordcloud.to_image()
