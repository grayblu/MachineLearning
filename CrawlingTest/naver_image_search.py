import requests
import urllib.parse as parse
import json
import os

import urllib.request

# 네이버 검색 클라이언트 ID, Secret ID 헤더 설정
headers = {
    'X-Naver-Client-ID' : '1V0dEJc8rb3JhF6EBHqz',
    'X-Naver-Client-Secret' : 'uGGhTMxmos'
}

# 검색 url
url = 'https://openapi.naver.com/v1/search/image'

# 검색 파라미터
params = {
    'query' : '버닝썬',
    'start' : 1,
    'display' : 10
}

res = requests.get(url, headers=headers, params=params)
print(res.status_code)  #응답 코드

list = json.loads(res.text) # json 문자열 파싱
print()
for ix, item in enumerate(list['items']):   # enum은 인덱스 추가
    title = item['title']
    link = item['link']
    info = parse.urlparse(link)
    fileName = os.path.split(info.path)[1]
    print(ix, title, fileName)

    # link를 이용하여 파일 다운로드 진행
    # 다운로드
    path = 'c:/temp/download/{}'.format(fileName)
    urllib.request.urlretrieve(link, path)
