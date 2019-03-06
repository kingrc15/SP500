import time
import CleanData

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

def get_source(company):
	browser = webdriver.Chrome()

	browser.get("https://finance.yahoo.com/quote/" + company + "/history?period1=-630957600&period2=1546930800&interval=1d&filter=history&frequency=1d")
	time.sleep(1)

	elem = browser.find_element_by_tag_name("body")

	no_of_pagedowns = 450		#1400

	while no_of_pagedowns:
		elem.send_keys(Keys.PAGE_DOWN)
	   	time.sleep(0.05)
	   	no_of_pagedowns-=1

	bs = BeautifulSoup(browser.page_source, 'html.parser')
	
	browser.close()
	browser.quit()
	return bs

def Extract(s):
	s = s[25:len(s)-7]
	return s;

def MonthTest(s):
	Months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
	isMonth = False
	for month in Months:
		if s == month:
			isMonth = True

	return isMonth;

def WriteTempData(data):
	temp = []
	i = 1
	tempStr = ''
	read = False
	for t in data:
		month = MonthTest(t.text[:3])
		if month:
			read = True
		if t.text[:1] == "*":
			read = False
		if read and not t.text[:5] == 'Divid' and not t.text[:5] == 'Stock':
			t = t.text.replace(',','')
			if month:
				tempStr = "\n "
				i = 1
			tempStr += t + ", "
			if i % 7 == 0: 
				if len(temp) > 0:
					tempStr = CleanData.CleanString(tempStr)
					temp.append(tempStr)	
				elif len(temp) == 0:
					tempStr = CleanData.CleanString(tempStr)
					temp.append(tempStr)				
				tempStr = ""
			i += 1
	return temp;

def MergeData(data1, data2):
	NewData = []

	for l1 in data1:
		Repeat = False
		for l2 in data2:
			if l1 == l2:
				Repeat = True
		if not Repeat:
			NewData.append(l1)
	return NewData;

def WriteToFile(data, company):
	with open("Resources/" + company + ".csv", "w+") as f:
		for l in data:
			f.write(str(l) + "\n")
	return;

def Scrape(company):
	marketSite = get_source(company)
	try:
		with open("Resources/" + company + ".csv", "r+") as f:
			url = 'https://finance.yahoo.com/quote/' + company + '/history?period1=521272800&period2=1547103600&interval=1d&filter=history&frequency=1d'	
			fileTemp = []
			for x in f:
				x = CleanData.CleanString(x)
				fileTemp.append(x)
			NewData = MergeData(WriteTempData(marketSite.find_all('span')),fileTemp)
	except:
		WriteToFile(WriteTempData(marketSite.find_all('span')), company)
	return;

def PullData():
	with open("Resources/Companies.csv","r") as companies:
		i = 1
		for company in companies:
			name = company[:len(company)-1]
			name = "ABT"
			Scrape(name)
			CleanData.DeleteRepeating(name)
			CleanData.SortFiles(name)
			print(str(i) + "/505: " + name)
			i += 1
	return;






















