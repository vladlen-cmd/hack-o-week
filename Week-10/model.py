import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle, os, warnings
warnings.filterwarnings('ignore')

class SustainabilityTracker:
    def __init__(self, data_file='campus_data.csv', model_file='model.pkl'):
        self.data_file=data_file; self.model_file=model_file
        self.model=None; self.data=None; self.feature_names=None; self.metrics=None

    def load_data(self):
        self.data=pd.read_csv(self.data_file); self.data['timestamp']=pd.to_datetime(self.data['timestamp'])
        self.data=self.data.sort_values('timestamp').reset_index(drop=True); return self.data

    def _features(self, df):
        f=df.copy(); f['hour']=f['timestamp'].dt.hour; f['dow']=f['timestamp'].dt.dayofweek; f['month']=f['timestamp'].dt.month
        f['hour_sin']=np.sin(2*np.pi*f['hour']/24); f['hour_cos']=np.cos(2*np.pi*f['hour']/24)
        f['lag1']=f['total_electricity_kwh'].shift(1); f['lag24']=f['total_electricity_kwh'].shift(24)
        f['roll_mean']=f['total_electricity_kwh'].rolling(24).mean()
        f=f.dropna().reset_index(drop=True)
        self.feature_names=['hour','dow','month','is_weekend','temperature','solar_generation_kwh','hour_sin','hour_cos','lag1','lag24','roll_mean']
        return f

    def train(self):
        if self.data is None: self.load_data()
        f=self._features(self.data); s=int(len(f)*0.8)
        X,y= f.iloc[:s][self.feature_names], f.iloc[:s]['total_electricity_kwh']
        Xt,yt= f.iloc[s:][self.feature_names], f.iloc[s:]['total_electricity_kwh']
        self.model=RandomForestRegressor(n_estimators=150,max_depth=12,random_state=42,n_jobs=-1)
        self.model.fit(X,y); yp=self.model.predict(Xt)
        self.metrics={'rmse':round(float(np.sqrt(mean_squared_error(yt,yp))),3),'r2':round(float(r2_score(yt,yp)),4)}
        print(f"✓ Trained R²={self.metrics['r2']}"); return self.model

    def get_feature_importance(self):
        return {n:round(float(v),4) for n,v in zip(self.feature_names,self.model.feature_importances_)}

    def predict_next(self):
        f=self._features(self.data); p=float(self.model.predict(f.iloc[[-1]][self.feature_names])[0])
        tree_preds=[float(t.predict(f.iloc[[-1]][self.feature_names])[0]) for t in self.model.estimators_[:50]]
        std=float(np.std(tree_preds))
        return {'prediction':round(p,2),'lower_bound':round(p-1.96*std,2),'upper_bound':round(p+1.96*std,2),'confidence_level':95}

    def get_sustainability_summary(self):
        d=self.data; last30=d[d['timestamp']>=d['timestamp'].max()-pd.Timedelta(days=30)]
        return {
            'total_co2_kg':round(float(d['co2_emissions_kg'].sum()),0),
            'total_co2_saved_kg':round(float(d['co2_saved_kg'].sum()),0),
            'total_solar_kwh':round(float(d['solar_generation_kwh'].sum()),0),
            'total_water_l':round(float(d['water_consumption_l'].sum()),0),
            'avg_score':round(float(d['sustainability_score'].mean()),1),
            'score_trend':round(float(last30['sustainability_score'].mean()-d['sustainability_score'].mean()),1),
            'avg_daily_elec':round(float(d.groupby(d['timestamp'].dt.date)['total_electricity_kwh'].sum().mean()),1),
            'solar_pct':round(float(d['solar_generation_kwh'].sum()/max(d['total_electricity_kwh'].sum(),1)*100),1)
        }

    def save_model(self):
        with open(self.model_file,'wb') as f: pickle.dump({'model':self.model,'features':self.feature_names,'metrics':self.metrics},f)
    def load_model(self):
        with open(self.model_file,'rb') as f: b=pickle.load(f)
        self.model=b['model'];self.feature_names=b['features'];self.metrics=b['metrics']

if __name__=='__main__':
    t=SustainabilityTracker(); t.load_data(); t.train()
    print("Summary:", t.get_sustainability_summary()); t.save_model()
