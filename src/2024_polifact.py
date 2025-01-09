from bs4 import BeautifulSoup
import pandas as pd
import requests
import urllib.request
import time

speakers = []
dates = []
statements = []
sources = []
labels = []

def scrape_website(page_number):
    page_num = str(page_number)
    url = 'https://www.politifact.com/factchecks/list/?page={}&'.format(page_num)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    statement_footer =  soup.find_all('footer',attrs={'class':'m-statement__footer'}) 
    statement_quote = soup.find_all('div', attrs={'class':'m-statement__quote'}) 
    statement_meta = soup.find_all('div', attrs={'class':'m-statement__meta'})
    target = soup.find_all('div', attrs={'class':'m-statement__meter'})
    
    #extract the author and date 
    for i in statement_footer:
        footer = i.text.strip()
        text = footer.split()
        speaker = " ".join(text[1:text.index('•')])
        speakers.append(speaker)
        date = " ".join(text[text.index('•') + 1:])
        dates.append(date)
        
    #extract the statement_quote
    for i in statement_quote:
        quote = i.find_all('a')
        statements.append(quote[0].text.strip())

    #extract the source
    for i in statement_meta:
        meta = i.find_all('a')
        sources.append(meta[0].text.strip())
        
    #extract the truthfulness
    for i in target:
        fact = i.find('div', attrs={'class':'c-image'}).find('img').get('alt')
        labels.append(fact)
        
#scrape all 2024 data
for i in range(1, 54):
    scrape_website(i)
    
#store data in dataframe
data = pd.DataFrame(columns = ['speaker',  'statement', 'source', 'date', 'label'])
data['speaker'] = speakers
data['statement'] = statements
data['source'] = sources
data['date'] = dates
data['label'] = labels

data.to_csv('2024_polifact.tsv', sep = '\t', index = True, header = False)