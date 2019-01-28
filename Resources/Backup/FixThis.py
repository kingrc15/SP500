from sys import argv

f = open(argv[1], "rb")
temp = ""

i = 0
length = 0
for line in f:
	length += 1
f.seek(0,0)
print(length)
for line in f:
	j=0
	if i == (length - 1):
		commaCount = 0
		for ch in line:
			if ch == ',':
				commaCount += 1
			if commaCount == 7 and ch == ',':
				temp += "\n"
			else:
				temp += ch
			j += 1
	else:
		temp += line
	i += 1

print(temp[:-1])
f.close()
f = open(argv[1], "wb")
f.write(temp)
f.close()
