import csv
import os
import datetime
from sklearn.ensemble import RandomForestClassifier

session = []
order = []
orderPred = []
data = []
sessionTest = []

def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)	# skipping column names
		for row in csvFileReader:
			session.append(row[0])
			order.append(int(row[5]))
			data.append([modify(row[1], 'ad'), modify(row[6], 'grp'), getFunnel(row[7])])
	csvfile.close()
	return data

def get_test_data(filename):
	testData = []
	print('directory for test file - ' + os.getcwd())
	with open(filename, 'r') as csvfile1:
		csvFileReader = csv.reader(csvfile1)
		next(csvFileReader)	# skipping column names
		for row in csvFileReader:
			sessionTest.append(row[0])
			testData.append([modify(row[1], 'ad'), modify(row[4], 'grp'), getFunnel(row[5])])
		print('size of test data set is ' + str(len(testData)))
	csvfile1.close()	
	return testData

def getFunnel(s):
	if(s == 'lower'):
		return -1
	elif(s == 'middle'):
		return 0
	elif(s == 'upper'):
		return 1

def modify(data, str):
	if(data == 'NULL'):
		return 0.0
	else:
		return float(data.replace(str, ''))


def calculateAccuracy(order, orderPred):
	correct = 0
	total = len(order)
	for i in range(0,total):
		if(order[i] == orderPred[i]):
			correct += 1
	print(correct)
	prec = (correct/total) * 100
	print("Accuracy = " + str(prec) + "%")

def predict(data, order, testData):
	svr_lin = RandomForestClassifier(n_estimators=100, max_depth=5000, min_samples_leaf=500, random_state=0)
	print("stated fitting data at "+str(datetime.datetime.now()))
	svr_lin.fit(data, order) # fitting the data points in the models
	print("data fitting done.. starting prediction at " + str(datetime.datetime.now()))

	orderPrediction = svr_lin.predict(testData)
	print('prediction on testdata done at ' + str(datetime.datetime.now()))
	newOrder = []
	for i in orderPrediction:
		newOrder.append(int(round(i)))
	return newOrder

def outputFile(sessionData, order):
	rows = []
	rows.append(['id', 'order_placed'])
	for i in range(0,len(sessionData)):
		rows.append([sessionData[i], order[i]])
	with open('testOutput64.csv', 'w', newline='') as writeFile:
		writer = csv.writer(writeFile)
		writer.writerows(rows)
	writeFile.close()

print('started getting data set at '+ str(datetime.datetime.now()))
get_data('FullDataSet/training_data.csv')
testData = get_test_data('FullDataSet/test_data.csv')
print('got the data set at ' + str(datetime.datetime.now()))
print('size of data set is ',len(data))
print('size of test data set is ',len(testData))
#test on dataset
orderPred = predict(data, order, data)
print("worked - size of output on data set is ", len(orderPred))
calculateAccuracy(order, orderPred)
orderPred = predict(data, order, testData)
print("worked - size of output on test data is ", len(orderPred))
outputFile(sessionTest, orderPred)
print('Printed output data')