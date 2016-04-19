# -*- coding: utf-8 -*-
# -----------------------------------------
# 这是一个用来在网页上找钛酸钡介电常数的爬虫
# -----------------------------------------

import urllib2
import urllib
import re
import HTMLParser
import os
from bs4 import BeautifulSoup
import pdf2txt


# remove the html tags from the content
def filter_tags(text):
    text = re.sub(r'</?\w+[^>]*>', '', text)
    text = re.sub(r'<\![^>]*>', '', text)
    text = re.sub(r'[^\x00-\x7f]', '', text)
    text = re.sub(r'[\(\)\[\]\{\}]', ' ', text)
    return text


# find the values in the content
def find_value(text, key):
    words = text.lower().split()
    value = []
    for onekey in key:
        keys = onekey.split()
        for i_word in range(len(words)):
            finded = True
            for i_key in range(len(keys)):
                if words[i_word + i_key] != keys[i_key]:
                    finded = False
                    break
            if finded:
                words_sub = words[i_word:i_word +10]
                # print words_sub
                for word in words_sub:
                    if re.match('[0-9,.~]+', word):
                        word = re.sub(r'[,;~]', '', word)
                        try:
                            one_value = float(word)
                            value.append(one_value)
                            break
                        except:
                            one_value = 0
    return value


# get text of one html page
def get_text(url):
    if url[len(url) - 4:len(url)] == '.pdf':
        # remove unavailable charactors
        name = re.sub(r'\/', '', url)
        name = re.sub(r'http:', '', name)
        name = re.sub(r'https:', '', name)
        name = 'pdf/' + name
        # download the pdf file if not exists
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


# get url lists from a root url
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
    t = [t1, t2]
    for i in root_a:
        if i.has_attr('href'):
            h = i.attrs['href']
            if h.find('http') == 0 and h.find(t1) < 0 and h.find(t2) < 0:
                url_list.append(h)
                # print h
    return url_list

# Main Program -----------------

if __name__ == '__main__':
    # use bing to search the content
    for first in range(1, 101, 10):
        root_url = 'http://www.bing.com/search?q=barium+titanate+dielectric+constant&first=' + str(first)
        url_list = get_url_list(root_url)

        for url in url_list:
            # print url[len(url)-4:len(url)]
            try:
                text = get_text(url)
                print find_value(text, ['dielectric constant', 'permitivity']), ' from ', url
            except:
                print 'something wrong! from', url
    print 'End.'
