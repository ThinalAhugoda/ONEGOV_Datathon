import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy

employees = [1,4] # keys
values = [['SEC-001','2021-01-01',834.5694834000001],['SEC-002','2022-01-01',1525.3689896166666]] # values

# initialization
sections = []
date = []
sectionEncoder = LabelEncoder()
dateEncoder = LabelEncoder()

sections = [i[0] for i in values]
date = [i[1] for i in values]
sections = sectionEncoder.fit_transform(sections)
date = dateEncoder.fit_transform(date)
for idx, elmnt in enumerate(values):
    elmnt[0] = sections[idx] # encode section id
    elmnt[1] = date[idx] # encode dates

# reshape to 2d
employees = numpy.array(employees)
values = numpy.array(values)
xTrain, xTest, yTrain, yTest = train_test_split(values, employees, test_size=0.2, random_state=1) 

model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    booster='gblinear', 
    n_estimators=100,              
    learning_rate=0.1                             
)
model.fit(xTrain, yTrain)
employeePredict= model.predict(xTest) 
print(employeePredict[0])


