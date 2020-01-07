from flask import Flask
from flask import request
import json
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# EB looks for an 'application' callable by default.
application = Flask(__name__)

def train(history):

    x= history[1]
    y=history[0]
    #transform history data into cloud coverage list
    cdf = pd.DataFrame([x[x.index(i)]['cloudCover'] for  i in x], columns = ['cloudCover'])

    #transform from list of objects into a panda dataframe
    variables = list(y[0].keys())
    df = pd.DataFrame([[y[y.index(i)][j] for j in variables] for i in y], columns = variables)

    #sets the independant variables of x and the dependant variable y

    X = pd.DataFrame([i.replace(":", "")  for i in df['timestamp'] ], columns=['timestamp'])
    X['cloudCover'] = cdf
    Y = df['solar']

    #splits train and test groups
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #the random forest predictor
    regressor = RandomForestRegressor(n_estimators=20, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    #

    #test algorithm
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    return [regressor, sc]


#returns the solar generation data
def forecast():
    if request.json:
        history = json.loads(request.json)
        rtree, sc = train(history)
        date, lat, lon = history[2]

        #getting forecast data from api
        key = '6262fa908f8eba2bef1deb6cc7a5869d'
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36'}
        url = 'https://api.darksky.net/forecast/{}/{},{},{}?exclude=currently,flags'.format(key, lat, lon,date )
        response = requests.get(url, headers=headers)
        #https://api.darksky.net/forecast/6262fa908f8eba2bef1deb6cc7a5869d/-27.59,-48.548,1575870400?exclude=currently,flags
        result = response.json()['hourly']['data']
        fdf = pd.DataFrame([[result.index(i)*100, i['cloudCover']] for i in result], columns=['timestamp','cloudCover'])
        #scale the variables
        test_forecast = sc.fit_transform(fdf)
        #make prediction
        y_pred = rtree.predict(test_forecast)

        predict = pd.DataFrame([fillZero(str(i//100))+':'+'00' for i in fdf['timestamp']] , columns=['timestamp'])
        predict['solar'] = y_pred
        return json.dumps(predict.values.tolist())

    return 'no post'

def fillZero(n):
    if len(n) == 1:
        return '0'+n
    return n


#12-12-2019 -> previsao
#[0.0, 0.0, 0.0, 0.0, 0.0, 0.024999999999999998, 0.2871, 0.3685999999999999, 1.6073, 1.5404000000000002, 1.9159, 2.0225, 2.0101, 1.9393999999999996, 2.2298, 2.8047000000000004, 0.7227999999999999, 0.7299, 0.08329999999999999, 5.123346928686195e-05, 5.123346928686195e-05, 5.123346928686195e-05, 5.123346928686195e-05, 5.123346928686195e-05]
#13-12-2019 -> previsao
#[["0:00", 0.0], ["1:00", 0.0], ["2:00", 0.0], ["3:00", 0.0], ["4:00", 0.0], ["5:00", 0.0004], ["6:00", 0.0338], ["7:00", 0.1799], ["8:00", 0.3264999999999999], ["9:00", 1.7084999999999997], ["10:00", 2.4797000000000002], ["11:00", 3.0562], ["12:00", 3.5067999999999997], ["13:00", 3.1754000000000002], ["14:00", 3.1891999999999996], ["15:00", 3.2726999999999995], ["16:00", 1.4771], ["17:00", 0.7779999999999999], ["18:00", 0.9533999999999999], ["19:00", 0.14540000000000003], ["20:00", 0.00012363598620270287], ["21:00", 2.3635986202702845e-05], ["22:00", 2.3635986202702845e-05], ["23:00", 2.3635986202702845e-05]]

#Application main routes
application.add_url_rule('/', 'index', forecast)
application.add_url_rule('/request', 'request', forecast, methods=['POST', 'GET'])

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()
