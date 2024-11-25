import hashlib

# 字符串转md5
def str2md5(s):
    m = hashlib.md5()
    m.update(s.encode('utf-8'))
    return m.hexdigest()
