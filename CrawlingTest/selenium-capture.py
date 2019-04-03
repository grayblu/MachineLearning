from selenium import webdriver

url = "https://www.naver.com/"

# PhantomJS 드라이버 추출하기
# browser = webdriver.PhantomJS('C:\phantomjs-2.1.1-windows\phantomjs-2.1.1-windows\bin\phantomjs.exe')

options = webdriver.ChromeOptions()
options.add_argument('headless')
browser = webdriver.Chrome('C:/chromedriver_win32/chromedriver', chrome_options=options)
# 3초 대기하기
browser.implicitly_wait(3)

# url 읽어 들이기
browser.get(url)

browser.save_screenshot('Website.png')

# 브라우저 종료
browser.quit()
