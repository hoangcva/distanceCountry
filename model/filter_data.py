# -*- coding: utf-8 -*-
import os
import shutil

pathfileraw = '../vn_news'
pathcountry = './country'
pathfiledata = '../vn_news_country'
if not (os.path.exists(pathfiledata) or os.path.isdir(pathfiledata)):
    os.makedirs(pathfiledata)

with open(pathcountry, "r") as f:
    listcountry = f.readlines()
listcountry2 = []
for country in listcountry:
    listcountry2.append(country.strip())

listfile = os.listdir(pathfileraw)

for filename in listfile:
    with open(pathfileraw + '/' + filename, 'r') as f:
        text = f.read().split()

    for namecountry in listcountry2:
        if namecountry in text:
            shutil.copy2(pathfileraw + '/' + filename, pathfiledata + '/' + namecountry + filename)
            break
