#%%

# 라이브러리 읽어 들이기
import urllib.request

# URL과 저장 경로 지정
url = "https://postfiles.pstatic.net/MjAxOTAzMjZfMTIw/MDAxNTUzNTkyODg3MTc5.YHoXvnZ1HgNHtH_HJ0tmI6MLNmGk-3kA6KS6dlf1nPMg.BD61S5oZ-3PjD5D_YdV0IzXfD4S5GNFhny-MTsoauXcg.JPEG.naverlaw/GettyImages-57441053.jpg?type=w773"
savename = 'test.png'

# 다운로드
urllib.request.urlretrieve(url, savename)
print('저장되었습니다')

#%%

url = "https://postfiles.pstatic.net/MjAxOTAzMjZfMTIw/MDAxNTUzNTkyODg3MTc5.YHoXvnZ1HgNHtH_HJ0tmI6MLNmGk-3kA6KS6dlf1nPMg.BD61S5oZ-3PjD5D_YdV0IzXfD4S5GNFhny-MTsoauXcg.JPEG.naverlaw/GettyImages-57441053.jpg?type=w773"
savename = 'test.png'

# 다운로드
mem = urllib.request.urlopen(url).read()

# 파일로 저장하기
with open(savename, mode='wb') as f:
    f.write(mem)
    print('저장되엇음')

#%%

# IP확인 API로 접근해서 결과 출력하기

url = 'http://api.aoikujira.com/ip/ini'
res = urllib.request.urlopen(url)
data = res.read()

# 바이너리를 문자열로 변환하기
text = data.decode('utf-8')
print(text)

#%%

import sys  # url을 인자로 받아 처리
import urllib.parse as parse

# 명령줄 매개변수 추출
if len(sys.argv) <= 1:
    print("USAGE: download-forecast-argv <Region Number>")
    sys.exit()
regionNumber = sys.argv[1]

# 매개변수를 URL 인코딩
API = 'http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp'
values = {
    'stdId' : regionNumber
}
params = parse.urlencode(values)
url = API + '?' + params
print('url=', url)

# 다운로드
data = urllib.request.urlopen(url).read()
text = data.decode("utf-8")
print(text)