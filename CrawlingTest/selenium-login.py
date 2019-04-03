from selenium import webdriver

USER = "test1"
PASS = "000000"

# PhantomJS 드라이버 추출하기
browser = webdriver.Chrome('C:/chromedriver_win32/chromedriver')
browser.implicitly_wait(3)

# 로그인 페이지에 접근하기
url_login = "https://nid.naver.com/nidlogin.login?mode=form&url=https%3A%2F%2Fwww.naver.com"
browser.get(url_login)
print('로그인 페이지에 접근합니다.')

# 텍스트 박스에 아이디와 비밀번호 입력하기
e = browser.find_element_by_id('id')
e.clear()
e.send_keys(USER)
e = browser.find_element_by_id('pw')
e.clear()
e.send_keys(PASS)

# 입력 양식 전송해서 로그인하기
form = browser.find_element_by_css_selector('input.btn_global[type=submit]')
form.submit()
print('로그인 버튼을 클릭합니다.')

# 페이지의 데이터 가져오기
browser.get('https://mail.naver.com/')

# 목록 출력하기
email = browser.find_element_by_css_selector('#list_for_view > ol > li.\34 1623_li.notRead.unmark._c1\28 mrCore\7c clickTitle\7c 41623\29._d2\28 mcDragndrop\7c html5DragStart\29 > div > div.subject > a._d2\28 mcDragndrop\7c html5DragStart\29 > span > strong')
print(email)