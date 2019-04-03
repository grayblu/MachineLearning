#%%

from bs4 import BeautifulSoup

# 분석하고 싶은 HTML
html = """
<html>
<body>
<h1 id="title">스크레이핑이란?</h1>
<p id="body">웹 페이지를 붂석하는 것</p>
<p>원하는 부붂을 추출하는 것</p>
</body>
</html>
"""

# html 분석하기
soup = BeautifulSoup(html, 'html.parser')


# # 원하는 부분 추출
# h1 = soup.html.body.h1
# p1 = soup.html.body.p
# p2 = p1.next_sibling.next_sibling

# # 요소의 글자 출력하기
# print(h1.string)
# print(p1.string)
# print(p2.string)

# find() 메서드로 원하는 부분 추출
title = soup.find(id="title")
body = soup.find(id="body")

print(title.string)
print(body.string)

#%%

html = """
<html><body>
<ul>
<li><a href="http://www.naver.com">naver</a></li>
<li><a href="http://www.daum.net">daum</a></li>
</ul>
</body></html>
"""

soup = BeautifulSoup(html, 'html.parser')

# find_all() 메서드로 추출
links = soup.find_all("a")

# 링크 목록 출력하기
for a in links:
    href = a.attrs['href']
    text = a.string
    print(text, '>', href)

#%%

from bs4 import BeautifulSoup
import urllib.request as req
url = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp"

# urlopen()으로 데이터 가져오기
res = req.urlopen(url)

# BeautifulSoup으로 분석
soup = BeautifulSoup(res, 'html.parser')

# 원하는 데이터 추출하기
title = soup.find('title').string
wf = soup.find('wf').string
print(title)
print(wf)

#%%

html = """
<html><body>
<div id="meigen">
    <h1>위키북스 도서</h1>
    <ul class="items">
        <li>유니티 게임 이펙트 입문</li>
        <li>스위프트로 시작하는 아이폰 앱 개발 교과서</li>
        <li>모던 웹사이트 디자인의 정석</li>
    </ul>
</div>
</body></html>
"""

soup = BeautifulSoup(html, 'html.parser')

# css 쿼리로 추출
# 타이틀 부분 추출
h1 = soup.select_one('dv#meigen > h1').string
print('h1 = ', h1)

# 목록 부분 추출
li_list = soup.select('div#meigen > ul.items > li')
for li in li_list:
    print('li = ', li.string)
