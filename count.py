import os

names = []
path=''
count = 0
for name in names:
    count + = len(os.listdir(os.path.join(path,name)))

print("Test set size = ",count)