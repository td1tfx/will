import urllib2
import re
from bs4 import BeautifulSoup

url = 'http://www.bing.com/search?q=barium+titanate+permittivity'
response = urllib2.urlopen(url)
content = response.read()

bs = BeautifulSoup(content, "html.parser")
a = bs.findAll('a')

#print a

alist = []
t='translator'
for i in a:
    h = i.attrs['href']
    if h.find('http')==0 and h.find(t)<0 :
        alist.append(i.attrs['href'])
        print alist

#alist = [i.attrs['href'] for i in a if i.attrs['href'][0] != 'j']

print alist
    
#print content



