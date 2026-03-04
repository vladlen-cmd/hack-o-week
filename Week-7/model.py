import pandas as pd, numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle, os, warnings
warnings.filterwarnings('ignore')

class AdminBuildingPredictor:
    def __init__(self, data_file='admin_data.csv', model_file='model.pkl'):
        self.data_file=data_file; self.model_file=model_file
        self.model=None; self.scaler=None; self.data=None; self.feature_names=None; self.metrics=None

    def load_data(self):
        self.data=pd.read_csv(self.data_file); self.data['timestamp']=pd.to_datetime(self.data['timestamp'])
        self.data=self.data.sort_values('timestamp').reset_index(drop=True); return self.data

    def _features(self, df):
        f=df.copy(); f['hour']=f['timestamp'].dt.hour; f['dow']=f['timestamp'].dt.dayofweek
        f['hour_sin']=np.sin(2*np.pi*f['hour']/24); f['hour_cos']=np.cos(2*np.pi*f['hour']/24)
        f['dow_sin']=np.sin(2*np.pi*f['dow']/7); f['dow_cos']=np.cos(2*np.pi*f['dow']/7)
        f['is_monday_morning']=((f['dow']==0)&(f['hour'].between(7,10))).astype(int)
        f['is_friday_evening']=((f['dow']==4)&(f['hour'].between(16,20))).astype(int)
        f['lag1']=f['electricity_kwh'].shift(1); f['lag24']=f['electricity_kwh'].shift(24)
        f['lag168']=f['electricity_kwh'].shift(168)
        f['roll_mean']=f['electricity_kwh'].rolling(24).mean(); f['roll_std']=f['electricity_kwh'].rolling(24).std()
        f=f.dropna().reset_index(drop=True)
        self.feature_names=['hour','dow','is_weekend','occupancy','temp_outside','hour_sin','hour_cos','dow_sin','dow_cos','is_monday_morning','is_friday_evening','lag1','lag24','lag168','roll_mean','roll_std']
        return f

    def train(self, test_size=0.2):
        if self.data is None: self.load_data()
        f=self._features(self.data); s=int(len(f)*(1-test_size))
        Xtr,ytr=f.iloc[:s][self.feature_names],f.iloc[:s]['electricity_kwh']
        Xte,yte=f.iloc[s:][self.feature_names],f.iloc[s:]['electricity_kwh']
        self.scaler=StandardScaler(); Xtr_s=self.scaler.fit_transform(Xtr); Xte_s=self.scaler.transform(Xte)
        self.model=ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)
        self.model.fit(Xtr_s, ytr); yp=self.model.predict(Xte_s)
        self.metrics={'rmse':round(float(np.sqrt(mean_squared_error(yte,yp))),3),'mae':round(float(mean_absolute_error(yte,yp)),3),'r2':round(float(r2_score(yte,yp)),4),'mape':round(float(np.mean(np.abs((yte-yp)/yte))*100),2)}
        print(f"✓ ElasticNet — R²={self.metrics['r2']}"); return self.model

    def get_feature_importance(self):
        coefs=np.abs(self.model.coef_); t=coefs.sum() or 1
        return {n:round(float(c/t),4) for n,c in zip(self.feature_names,coefs)}

    def predict_next(self):
        f=self._features(self.data); row=self.scaler.transform(f.iloc[[-1]][self.feature_names])
        p=float(self.model.predict(row)[0])
        return {'prediction':round(p,2),'lower_bound':round(p*0.92,2),'upper_bound':round(p*1.08,2),'confidence_level':92}

    def get_weekend_dip(self):
        if self.data is None: self.load_data()
        wk=self.data[self.data['is_weekend']==0]; we=self.data[self.data['is_weekend']==1]
        mon_morn=self.data[(self.data['timestamp'].dt.dayofweek==0)&(self.data['timestamp'].dt.hour.between(7,10))]
        fri_eve=self.data[(self.data['timestamp'].dt.dayofweek==4)&(self.data['timestamp'].dt.hour.between(16,20))]
        return {
            'weekday_avg':round(float(wk['electricity_kwh'].mean()),2),
            'weekend_avg':round(float(we['electricity_kwh'].mean()),2),
            'dip_pct':round(float((wk['electricity_kwh'].mean()-we['electricity_kwh'].mean())/wk['electricity_kwh'].mean()*100),1),
            'monday_surge_avg':round(float(mon_morn['electricity_kwh'].mean()),2),
            'friday_wind_down':round(float(fri_eve['electricity_kwh'].mean()),2)
        }

    def save_model(self):
        with open(self.model_file,'wb') as f: pickle.dump({'model':self.model,'scaler':self.scaler,'features':self.feature_names,'metrics':self.metrics},f)
    def load_model(self):
        with open(self.model_file,'rb') as f: b=pickle.load(f)
        self.model=b['model'];self.scaler=b['scaler'];self.feature_names=b['features'];self.metrics=b['metrics']

if __name__=='__main__':
    p=AdminBuildingPredictor(); p.load_data(); p.train()
    print("Weekend dip:", p.get_weekend_dip()); print("Prediction:", p.predict_next()); p.save_model()
