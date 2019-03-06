import subprocess

class date:
	day = 0
	month = 'NA'
	year = 0

	def __init__(self, month, day, year):
		self.day = day
		if len(month) < 3:
			self.month = self.ReverseMonthIndex(month)
		else:
			self.month = month
		self.year = year

	def monthIndex(self):
		months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
		if months.index(self.month) < 10:
			return '0' + str(months.index(self.month))
		else:
			return str(months.index(self.month));

	def ReverseMonthIndex(self, month):
		print month
		self.month = int(month)
		months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
		if months[self.month] < 10:
			return '0' + str(months[self.month])
		else:
			return str(months[self.month]);

	def __lt__(self, other):
		if self.year < other.year:
			return True;
		elif self.year > other.year:
			return False;
		else:
			if self.monthIndex() < other.monthIndex():
				return True;
			elif self.monthIndex() > other.monthIndex():
				return False;
			else:
				if self.day < other.day:
					return True;
				else:
					return False;

	def __gt__(self, other):
		return not self.__lt__(other);

	def GetDate(self):
		return self.month, self.day, self.year

def SortFiles(f):				#bubble sort
	with open("Resources/" + f + ".csv",'r') as fi:
		temp = fi.read().split('\n')
		i = 2
		sort = False
		while i < len(temp) and not sort:
			j = 0
			sort = True
			while j < len(temp)-i:
				#print temp
				date1 = date(temp[j][:3],temp[j][4:6],temp[j][7:11])
				date2 = date(temp[j+1][:3],temp[j+1][4:6],temp[j+1][7:11])
				if date1 < date2:
					sort = False
					swap = temp[j]
					temp[j] = temp[j+1]
					temp[j+1] = swap
				j += 1
			i += 1
	with open("Resources/" + f + ".csv",'w') as fi:
		for lines in temp:
			fi.write(lines + '\n')
	return temp;

def Replace(s1, s2):
	i = 1
	j = 1
	count = 0
	for c in s1:
		if c == ",":
			count += 1
		if count == 6:
			break
		i += 1
	
	count = 0
	for c in s2:
		if c == ",":
			count += 1
		if count == 6:
			break
		j += 1

	if s1[i:] > s2[j:]:
		return s1;
	else:	
		return s2;

def Update(l):
	temp = []
	
	i = 0
	while i < len(l) - 1:
		if l[i][:11] == l[i+1][:11]:
			temp.append(Replace(l[i],l[i+1]))
			i += 2
		else:
			temp.append(l[i])
			i += 1
	if i >= len(l):
		temp.append(l[len(l)-1])

	return temp;
		

def DeleteRepeating(f):
	fileName = "Resources/" + f + ".csv"
	with open(fileName, "r") as data:
		temp = []
		for s in data:
			repeat = False
			for l in temp:
				if l == s:
					repeat = True
					break
			if not repeat:
				temp.append(s)
		temp = Update(temp)

	f = open(fileName, "w")
	for l in temp:
		f.write(l)
	f.close

def CleanString(s):
	com = False
	temp = ""

	s = s.strip()
	if s[-1:] == ",":
		s = s[:len(s)-1]
	if s[-1:] == 0:
		s = s[:len(s)-1]
	for c in s:
		if com and c == " ":
			com = False
		else:
			temp += c
		if c == ",":
			com = True

	return temp;
	
