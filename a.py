a = [11,2,3,45,7]
b = ['avs','esd','gdfdf','sdsd','hgf']

c = {}
for i in range(len(a)):
    c[b[i]] = a[i]

print(c)
d = 'y'
for i,j in c.items():
    
    if j == 45:
        d = i

c.pop(d)
        
print(c)    
# c = dict(map(b:b for i in a))
# print(c)