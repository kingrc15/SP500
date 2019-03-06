import CleanData
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

class FormatedData(object):
	def __init__(self, r, minYear = 2000):
		self.maxStockValue = 0
		self.minStockValue = 0
		self.maxVolume = 0
		self.minVolume = 0
		self.minDate = CleanData.date('Jan', 1, minYear)
		self.maxDate = CleanData.date('Jan', 1, minYear)
		self.minYear = minYear
		i = 0
		with open("Resources/Companies.csv",'r') as companies:
			self.arrayLength = self.FindLen(companies,2000)
			self.inP = np.zeros((self.arrayLength,3033))
			self.out = np.zeros((self.arrayLength,505))
			companies.seek(0)
			for company in companies:
				name = company[:len(company)-1]
				with open("Resources/" + name + ".csv",'r') as companyData:
					j = 0
					for line in companyData:
						line = line[:len(line)-1]
						if len(line) > 1 and int(line[7:11]) >= minYear:
							newLine = self.FormatDate(line).split(',')
							for k in xrange(len(newLine)):
								newLine[k] = newLine[k].strip()
							newLine = self.Norm(newLine)
							self.inP[j][:3] = newLine[:3]
							self.inP[j][i*6+3:i*6+9] = newLine[3:]
							self.out[j][i] = newLine[7]
						j += 1
				i += 1

		self.inP = np.flip(self.inP[:][:-6])		#Remove last
		self.out = np.flip(self.out[:][1:])		    #Remove first

		o = np.zeros([len(self.out)])
		for p in xrange(len(self.out)):
			o[p] = np.average(self.out[p])

		col_num = np.arange(len(self.out))
		plt.plot(col_num, o, 'ro')
		plt.show()

		self.testinP = np.array(self.inP[:int(np.floor(len(self.inP)*(1-r)))])			
		self.testOut = np.array(self.out[:int(np.floor(len(self.out)*(1-r)))])			
		self.inP = np.array(self.inP[int(np.floor(len(self.inP)*(1-r))):])		
		self.out = np.array(self.out[int(np.floor(len(self.out)*(1-r))):])

		np.save('TestInputs',self.testinP)
		np.save('TestOutputs',self.testOut)
		np.save('Inputs',self.inP)
		np.save('Outputs',self.out)

	def FormatDate(self, line):
		d = CleanData.date(line[:3],line[4:6],line[7:11])
		return d.monthIndex() + "," + str(d.day) + "," + str(d.year) + "," + line[12:];

	def FindMax(self, line):
		d = CleanData.date(line[0], line[1], line[2])
		if d > self.maxDate:
			self.maxDate = d
		elif d < self.minDate:
			self.minDate = d
		if float(line[8]) > self.maxVolume:
			self.maxVolume = float(line[8])
		for price in line[3:8]:
			price = float(price)
			if price > self.maxStockValue:
				self.maxStockValue = price
			elif price < self.minStockValue:
				self.minStockValue = price

	def Norm(self, row):
		row[0] = float(row[0]) / 12.0
		row[1] = float(row[1]) / 31.0
		row[2] = (float(row[2]) - float(self.minDate.GetDate()[2])) / (float(self.maxDate.GetDate()[2]) - float(self.minDate.GetDate()[2]))
		for i in xrange(5):
			row[3 + i] = (float(row[3 + i]) - self.minStockValue) / (self.maxStockValue - self.minStockValue)# / (1.00001 ** float(row[3 + i]))
		row[8] = (float(row[8]) - self.minVolume) / (self.maxVolume - self.minVolume)
		return row

	def FindLen(self, companies, minYear):
		length = 0
		for company in companies:
			company = company[:len(company)-1]
			count = 0
			with open("Resources/" + company + ".csv",'r') as data:
				for line in data:
					if len(line) > 1 and int(line[7:11]) >= self.minYear:
						line = self.FormatDate(line).split(',')
						for k in xrange(len(line)):
							line[k] = line[k].strip()
						self.FindMax(line)
						count += 1
			if count > length:
				length = count

		return length