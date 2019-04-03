#%%

from bs4 import BeautifulSoup
import urllib.request as req

url = "https://ko.wikisource.org/wiki/%EC%A0%80%EC%9E%90:%EC%9C%A4%EB%8F%99%EC%A3%BC"

res = req.urlopen(url)
soup = BeautifulSoup(res, 'html.parser')

# #mw-content-text 바로 아래에 있는 
# # ul 태그 바로 아래에 있는 
# # li 태그 아래에 있는 
# # a 태그를 모두 선택합니다.
a_list = soup.select("#mw-content-text > ul > li a")
for a in a_list:
    name = a.string
    print('-', name)
#mw-content-text > div > ul:nth-child(6) > li > b > a

#mw-content-text > div > table:nth-child(1) > tbody > tr > td:nth-child(2)

#mw-content-text > div > ul:nth-child(6) > li > b > a

#%%

from bs4 import BeautifulSoup
fp = open("books.html", encoding="utf-8")
soup = BeautifulSoup(fp, 'html.parser')

# CSS 선택자로 검색하는 방법
sel = lambda q : print(soup.select_one(q).string)
sel('#nu')
sel('li#nu')
sel("ul > li#nu") 
sel("#bible #nu") 
sel("#bible > #nu") 
sel("ul#bible > li#nu") 
sel("li[id='nu']") 
sel("li:nth-of-type(4)")

# 그 밖의 방법
print(soup.select('li')[3].string)
print(soup.find_all('li')[3].string)

#%%

from bs4 import BeautifulSoup
fp = open('fruits-vegetables.html', encoding='utf-8')
soup = BeautifulSoup(fp, 'html.parser')

# css 선택자로 추출
print(soup.select_one('li:nth-of-type(4)').string)

# find 메서드로 추출
cond = {'data-lo':'us', 'class':'black'}
print(soup.find('li', cond).string)

# find 메서드를 연속적으로 사용하기
print(soup.find(id='ve-list').find('li', cond).string)


#%%

from urllib.parse import urljoin

url = "http://localhost:8080/tour/gallery/view/3?page=1"
res = req.urlopen(url)
soup = BeautifulSoup(res, 'html.parser')

img_list = soup.select('#gallery > div > div:nth-of-type(1) > img')
for img in img_list:
    src = img.attrs['src']
    print(urljoin(url, src))

#%%

import re   # 정규 표현식
html = '''
<ul>
<li><a href="hoge.html">hoge</li>
<li><a href="https://example.com/fuga">fuga*</li>
<li><a href="https://example.com/foo">foo*</li>
<li><a href="http://example.com/aaa">aaa</li>
</ul>'''

soup = BeautifulSoup(html, 'html.parser')

# 정규표현식으로 href에서 https인 것을 추출
li = soup.find_all(href=re.compile(r"https://"))
for e in li:
    print(e.attrs['href'])


#%%

# 상대경로를 절대경로로 변환시
from urllib.parse import urljoin

base = "http://example.com/html/a.html"

print( urljoin (base, "b.html") )
print( urljoin (base, "sub/c.html") ) 
print( urljoin (base, "../index.html") )
print( urljoin (base, "../ img /hoge.png") )
print( urljoin (base, "../ css /") )

print(urljoin(base, '/hoge.html'))
print(urljoin(base, 'http://123.com/wike'))
print(urljoin(base, '//localhost:8080/tour/gallery/view/3?page=1'))