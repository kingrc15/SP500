import PullData
import PullData2
import Net2
import FormatData
#import matplotlib.pyplot as plt
#import pandas as pd

#PullData2.PullData()
data = FormatData.FormatData(.75)
net = Net2.Net([3033,100,100,505],['sig','sig','sig','sig'])
net.GD(data[0],100.0,1000000,30,data[1])
