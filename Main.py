import PullData
import PullData2
import Net2
import FormatData
#import matplotlib.pyplot as plt
#import pandas as pd

#PullData2.PullData()
data = FormatData.FormatedData(.75, 2000)
net = Net2.Net([3033,1000,505],['sig','sig','sig'])
net.GD(zip(data.inP,data.out),100.0,1000000,30,zip(data.testinP, data.testOut))
