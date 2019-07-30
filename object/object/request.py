import requests

url = 'http://127.0.0.1:8001/'
files = {'file':open('D:\study\dl\Graduation_Project\2.png','rb')}
r = requests.post(url,files=files)
r.text