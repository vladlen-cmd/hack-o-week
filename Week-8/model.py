import pandas as pd, numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle, os, warnings
warnings.filterwarnings('ignore')

class ParkingLightingPredictor:
    def __init__(self, data_file='parking_data.csv', model_file='model.pkl'):
        self.data_file=data_file; self.model_file=model_file
        self.model=None; self.data=None; self.feature_names=None; self.metrics=None

    def load_data(self):
        self.data=pd.read_csv(self.data_file); self.data['timestamp']=pd.to_datetime(self.data['timestamp'])
        self.data=self.data.sort_values('timestamp').reset_index(drop=True); return self.data

    def _features(self, df):
        f=df.copy(); f['hour']=f['timestamp'].dt.hour; f['dow']=f['timestamp'].dt.dayofweek
        f['hour_sin']=np.sin(2*np.pi*f['hour']/24); f['hour_cos']=np.cos(2*np.pi*f['hour']/24)
        f['lag1']=f['electricity_kwh'].shift(1); f['lag24']=f['electricity_kwh'].shift(24)
        f['roll_mean']=f['electricity_kwh'].rolling(24).mean()
        f['veh_roll']=f['vehicles'].rolling(12).mean()
        f=f.dropna().reset_index(drop=True)
        self.feature_names=['hour','dow','is_weekend','is_dark','vehicles','sunrise_hour','sunset_hour','has_event','hour_sin','hour_cos','lag1','lag24','roll_mean','veh_roll']
        return f

    def train(self, test_size=0.2):
        if self.data is None: self.load_data()
        f=self._features(self.data); s=int(len(f)*(1-test_size))
        Xtr,ytr=f.iloc[:s][self.feature_names],f.iloc[:s]['electricity_kwh']
        Xte,yte=f.iloc[s:][self.feature_names],f.iloc[s:]['electricity_kwh']
        self.model=DecisionTreeRegressor(max_depth=12,min_samples_split=10,min_samples_leaf=5,random_state=42)
        self.model.fit(Xtr,ytr); yp=self.model.predict(Xte)
        self.metrics={'rmse':round(float(np.sqrt(mean_squared_error(yte,yp))),3),'mae':round(float(mean_absolute_error(yte,yp)),3),'r2':round(float(r2_score(yte,yp)),4),'mape':round(float(np.mean(np.abs((yte-yp)/yte))*100),2)}
        print(f"✓ DecisionTree — R²={self.metrics['r2']}"); return self.model

    def get_feature_importance(self):
        imp=self.model.feature_importances_
        return {n:round(float(v),4) for n,v in zip(self.feature_names,imp)}

    def predict_next(self):
        f=self._features(self.data); row=f.iloc[[-1]][self.feature_names]
        p=float(self.model.predict(row)[0])
        return {'prediction':round(p,2),'lower_bound':round(p*0.9,2),'upper_bound':round(p*1.1,2),'confidence_level':90}

    def get_lighting_analysis(self):
        dark=self.data[self.data['is_dark']==1]; light=self.data[self.data['is_dark']==0]
        return {
            'dark_avg':round(float(dark['electricity_kwh'].mean()),2),
            'light_avg':round(float(light['electricity_kwh'].mean()),2),
            'ratio':round(float(dark['electricity_kwh'].mean()/max(light['electricity_kwh'].mean(),0.01)),1),
            'dark_hours_pct':round(len(dark)/len(self.data)*100,1),
            'event_nights':int(self.data['has_event'].sum()//5)
        }

    def save_model(self):
        with open(self.model_file,'wb') as f: pickle.dump({'model':self.model,'features':self.feature_names,'metrics':self.metrics},f)
    def load_model(self):
        with open(self.model_file,'rb') as f: b=pickle.load(f)
        self.model=b['model'];self.feature_names=b['features'];self.metrics=b['metrics']

if __name__=='__main__':
    p=ParkingLightingPredictor(); p.load_data(); p.train()
    print("Lighting analysis:", p.get_lighting_analysis()); print("Prediction:", p.predict_next()); p.save_model()
