import PullData
import PullData2
import Net
import FormatData

#PullData2.PullData()
data = FormatData.FormatedData(.75, 2000)
net = Net.Net([3033,500,200,505],['TanH','TanH','TanH','sig'], cost_func = 'MSE')
net.GD(zip(data.inP,data.out),500.0,1000000,30,zip(data.testinP, data.testOut))


#3033,500,505  ~  2000    LR = 10  MSE  TanH TanH sig
#3033,400,505  ~  2750 	  LR = 5   MSE  TanH TanH sig