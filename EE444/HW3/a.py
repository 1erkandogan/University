x = """
GET /sites/eee.metu.edu.tr/files/images/eehistory.jpg HTTP/1.1
Host: eee.metu.edu.tr
Connection: keep-alive
DNT: 1
Upgrade-Insecure-Requests: 1
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7
Accept-Encoding: gzip, deflate
Accept-Language: en-US,en;q=0.9
"""
y = """
HTTP/1.1 200 OK
Date: Sat, 20 May 2023 16:59:44 GMT
Server: Apache
X-XSS-Protection: 1; mode=block
X-Frame-Options: SAMEORIGIN
Content-Security-Policy: frame-ancestors 'self';
X-Content-Type-Options: nosniff
Last-Modified: Fri, 29 Jun 2018 07:29:30 GMT
ETag: "28e79-56fc2ceff5904"
Accept-Ranges: bytes
Content-Length: 167545
Cache-Control: max-age=1209600
Expires: Sat, 03 Jun 2023 16:59:44 GMT
Connection: close
Content-Type: image/jpeg
"""
len_x = 0
len_y = 0
for i in x.split("\n"):
    len_x += len(i)
    print(f"row {i} : {len(i)}")
for j in y.split("\n"):
    len_y += len(j)
    print(f"row {j} : {len(j)}")
print(f"Total length of x is {len_x}")
print(f"Total length of y is {len_y}")