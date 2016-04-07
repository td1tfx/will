import pdf2txt
import urllib
import re

# find the values in the content
def find_value(text, key):
    value=[]
    keys = key[0].split()
    words = text.split()
    for k in range(len(words)):
        if words[k].lower()=='dielectric' and words[k+1].lower()=='constant':
            words_sub=words[k:k+10]
            print words_sub
            for word in words_sub:
                if re.match('[0-9,.~]+',word):
                    word = re.sub(r'[,;~]','',word)
                    try:
                        one_value = float(word)
                        value.append(one_value)
                    except:
                        one_value = 0
    return value

#url='http://www.iiss.sanu.ac.rs/download/vol40_3/vol40_3_02.pdf'
#urllib.urlretrieve(url, 'www.iiss.sanu.ac.rsdownloadvol40_3vol40_3_02.pdf') 
text = pdf2txt.pdf2txt('www.iiss.sanu.ac.rsdownloadvol40_3vol40_3_02.pdf')
print text
print find_value(text,['dielectric constant'])

