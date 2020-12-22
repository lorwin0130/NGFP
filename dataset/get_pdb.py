import urllib.request
import requests

lst = []
with open('target.txt','r',encoding='utf-8') as f:
    lst = f.readlines()
num = 0
start = 0
for line in lst:
    if num < start:
        num += 1
        continue
    line = line.strip().split('\t')
    pdb_id, url = line[1], line[2]
    print(pdb_id)
    r = requests.get(url)
    with open('pdb/{}.pdb'.format(pdb_id),'wb') as f:
        f.write(r.content)
    # f = urllib.request.urlretrieve(url,'pdb/{}.pdb'.format(pdb_id))
    num += 1
    print('all: {}/102\n'.format(num))