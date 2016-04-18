# ------------------------------------  
# xie dian sha ma ??????
# ------------------------------------

import urllib2
import urllib
import re
from bs4 import BeautifulSoup
import HTMLParser
import os

# remove the html tags from the content
def filter_tags(text):
    text = re.sub(r'</?\w+[^>]*>', '', text)
    text = re.sub(r'<\![^>]*>', '', text)
    text = re.sub(r'[^\x00-\x7f]', '', text)
    return text

# find the values in the content
def find_value(text, key):
    value=[]
    keys = key[0].split()
    words = text.split()
    for k in range(len(words)):
        if words[k].lower()=='dielectric' and words[k+1].lower()=='constant':
            words_sub=words[k:k+10]
            #print words_sub
            for word in words_sub:
                if re.match('[0-9,.~]+',word):
                    word = re.sub(r'[,;~]','',word)
                    try:
                        one_value = float(word)
                        value.append(one_value)
                    except:
                        one_value = 0
    return value

# get text of one html page
def get_text(url):
    if url[len(url)-4:len(url)]=='.pdf':
        name = re.sub(r'\/', '', url)
        name = re.sub(r'http:', '', name)
        name = re.sub(r'https:', '', name)
        name = 'pdf/' + name
        if not os.path.isfile(name):
            urllib.urlretrieve(url, name) 
        text = pdf2txt.pdf2txt(name)
    else:
        response = urllib2.urlopen(url)
        content = response.read()
        bs = BeautifulSoup(content, "html.parser")
        text = bs.get_text()
    text = filter_tags(text)
    return text

# get url list from a root url
def get_url_list(root_url):
    root_response = urllib2.urlopen(root_url)
    root_content = root_response.read()
    # locate the url from the html page, with BeautifulSoup library
    root_bs = BeautifulSoup(root_content, "html.parser")
    root_a = root_bs.findAll('a', recursive=True)
    url_list = []
    
    # remove microsoft's links
    t1 = 'translator'
    t2 = 'go.microsoft'
    t=[t1, t2]
    for i in root_a:
        if i.has_attr('href'):
            h = i.attrs['href']
            if h.find('http')==0 and h.find(t1)<0 and h.find(t2)<0:
                url_list.append(h)
                print h
    return url_list

#------------------------------------------ Main Program ---------------------------------------------------

if __name__ == '__main__':
    # use bing to search the content
    root_url = 'http://www.bing.com/search?q=barium+titanate+dielectric+constant'

    url_list = get_url_list(root_url)

    # test code
    #url = 'http://onlinelibrary.wiley.com/doi/10.1111/j.1551-2916.2008.02693.x/abstract'
    #del alist[0]
    #text = get_text(url)
    #print text
    #print find_value(text,['dielectric constant']), url
    # -------------------------

    for url in url_list:
        print url[len(url)-4:len(url)]
        try:
            text = get_text(url)
            print find_value(text,['dielectric constant']), ' from ', url
        except:
            print 'something wrong!', url
        




