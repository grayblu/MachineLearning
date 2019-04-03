# 로그인을 위한 모듈 추출하기
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# 아이디와 비번 지정하기
USER = "test"
PASS = "000000"

# 세션 시작하기
session = requests.session()
# 로그인하기
login_info = {
    'userId' : USER,
    'password': PASS
}

url_login = 'http://localhost:8080/tour/login'
res = session.post(url_login, data=login_info)
res.raise_for_status()  # 오류가 발생하면 예외가 발생
print(res.text)

res = session.get('http://localhost:8080/tour/member/view')
res.raise_for_status()
print(res.text)

