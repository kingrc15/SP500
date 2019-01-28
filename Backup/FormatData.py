import CleanData
import numpy as np


def FormatData(r, minYear = 2000):
	test = []
	testResult = []
	i = 0				#company count
	with open("Resources/Companies.csv",'r') as companies:
		arrayLength = findLen(companies, 2000)
		inP = np.zeros((arrayLength,3033))
		out = np.zeros((arrayLength,505))
		companies.seek(0)
		for company in companies:
			name = company[:len(company)-1]
			with open("Resources/" + name + ".csv",'r') as companyData:
				j = 0
				for line in companyData:
					line = line[:len(line)-1]
					if len(line) > 1 and int(line[7:11]) >= minYear:
						newLine =FormatDate(line).split(',')
						for k in xrange(len(newLine)):
							newLine[k] = newLine[k].strip()
						#inP[j][:3] = newLine[:3]
						#inP[j][i*6+3:i*6+9] = newLine[3:]
						#out[j][i] = newLine[7]
					j += 1
			i += 1
	inP = inP[:][:-6]		#Remove last
	out = out[:][1:]		#Remove first
	

	testinP = inP[:int(np.floor(len(inP)*(1-r)))]				
	testOut = out[:int(np.floor(len(out)*(1-r)))]					
	inP = inP[int(np.floor(len(inP)*(1-r))):]			
	out = out[int(np.floor(len(out)*(1-r))):]
	np.savetxt("Resources/Input.csv",inP, fmt="%8.4f", delimiter=",")
	np.savetxt("Resources/Output.csv",out, fmt="%8.4f", delimiter=",")
	np.savetxt("Resources/TestInput.csv",testinP, fmt="%8.4f", delimiter=",")
	np.savetxt("Resources/TestOutput.csv",testOut, fmt="%8.4f", delimiter=",")
	return (zip(inP, out), zip(testinP, testOut))

#Finds the file with the max length
def findLen(companies, minYear = 2000):
	Max = 0
	for company in companies:
		name = company[:len(company)-1]
		count = 0
		with open("Resources/" + name + ".csv") as companyData:
			for line in companyData:
				if len(line) > 2 and int(line[7:11]) >= minYear:
					#print line[:-1]
					count += 1
		if count > Max:
			Max = count
	return Max;

def FormatDate(line):
	d = CleanData.date(line[:3],line[4:6],line[7:11])
	return d.monthIndex() + ", " + str(d.day) + ", " + str(d.year) + ", " + line[12:];
