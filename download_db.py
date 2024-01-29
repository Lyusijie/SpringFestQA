import subprocess

def download_folder(url, local_path):
    command = f'wget -r -np -nH --cut-dirs=1 -R index.html -P {local_path} {url}'
    subprocess.run(command, shell=True)

# 指定服务器上的文件夹URL和本地保存路径
folder_url = 'https://b-aide-20240108-f71d317-018318.intern-ai.org.cn/files/data/demo/data_base/vector_db/chroma2/chroma.sqlite3?_xsrf=2%7C70629ef6%7C9bacff043c1cdfed84eb8c83d2147fb0%7C1704724886'
local_path = '/Users/mac/Documents/GitHub/12306QA/data_base/vector_db/'

# 调用下载函数
download_folder(folder_url, local_path)