# SP500
Scrapes market data for the S&P 500 companies, formats the data and runs it through a neural net. 

Main
From this function, you can use the PullData, PullData2, Net and FormatData programs. You must have the Numpy library to run most programs.

PullData and PullData2
These programs scrape data from Yahoo Finance. They pull information about companies from a list called Companies.csv that you must store in /Resources. I am using the 505 S&P500 companies but you can change the list to whatever you want. Note that Companies.csv uses company stock symbols. You must have the bs4 library for both of these programs.

PullData2
This is used for your initial scrape. It will create .csv files in your Resources folder. You must have chromium and selenium to run PullData2 since you have to scroll down to load more data. Adjust no_of_pagedowns to scrape more data. At the time I'm writing this, I've noticed that 1400 page downs will get you all stock information for every company. This program will take a long time to run depending on the number of companies and scroll down you choose.

PullData
This is used for adding recent stock information to your files. You must have requests for this program

CleanData
This is a list of function and classes that help with formating and cleaning data.

FormatData
This is a class that collects information, formats and normalizes your data. It saves the data in a .npy files for use later.

Net
This is a neural net class. In order to use it, you must initialize the net. Then you can use either SGD or AdaDelta to train it.
