# ------------------------------------ 
# 
#
# ------------------------------------

import urllib2
import re
from bs4 import BeautifulSoup
import HTMLParser
import html2text

# remove the html tags from the content
def filter_tags(text):
    text = re.sub(r'</?\w+[^>]*>','',text)
    text = re.sub(r'<!--[^>]*-->','',text)
    return text

# find the values in the content
def find_value(text, key):
    value=[]
    keys = key[0].split()
    words = text.split()
    for k in range(len(words)):
        if words[k].lower()=='dielectric' and words[k+1].lower()=='constant':
            words_sub=words[k-10:k+10]
            for word in words_sub:
                if re.match('[0-9\,\.]+',word):
                    word = re.sub(r'[,;]','',word)
                    try:
                        one_value = float(word)
                        value.append(one_value)
                    except:
                        one_value = 0
    return value


#------------------------------------------ Main Program ---------------------------------------------------


# use bing to search the content
root_url = 'http://www.bing.com/search?q=barium+titanate+dielectric+constant'
root_response = urllib2.urlopen(root_url)
root_content = root_response.read()


# locate the url from the html page, with BeautifulSoup library
root_bs = BeautifulSoup(root_content, "html.parser")
root_a = root_bs.findAll('a', recursive=True)
# remove microsoft's links
alist = []
t1 = 'translator'
t2 = 'go.microsoft'
t=[t1, t2]
for i in root_a:
    if i.has_attr('href'):
        h = i.attrs['href']
        if h.find('http')==0 and h.find(t1)<0 and h.find(t2)<0:
            alist.append(h)
            # print h


#del alist[0]
c=0
for url in alist:
    response = urllib2.urlopen(url)
    content = response.read()
    bs = BeautifulSoup(content, "html.parser")
    text = bs.get_text()
    text = filter_tags(text)
    print find_value(text,['dielectric constant']), url
    break




