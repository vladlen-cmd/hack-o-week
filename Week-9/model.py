import pandas as pd, numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle, os, warnings
warnings.filterwarnings('ignore')

class LaundryPeakPredictor:
    def __init__(self, data_file='laundry_data.csv', model_file='model.pkl'):
        self.data_file=data_file; self.model_file=model_file
        self.model=None; self.scaler=None; self.data=None; self.feature_names=None; self.metrics=None

    def load_data(self):
        self.data=pd.read_csv(self.data_file); self.data['timestamp']=pd.to_datetime(self.data['timestamp'])
        self.data=self.data.sort_values('timestamp').reset_index(drop=True); return self.data

    def _features(self, df):
        f=df.copy(); f['hour']=f['timestamp'].dt.hour; f['dow']=f['timestamp'].dt.dayofweek
        f['is_sunday']=((f['dow']==6)).astype(int)
        f['hour_sin']=np.sin(2*np.pi*f['hour']/24); f['hour_cos']=np.cos(2*np.pi*f['hour']/24)
        f['lag1']=f['electricity_kwh'].shift(1); f['lag24']=f['electricity_kwh'].shift(24)
        f['roll_mean']=f['electricity_kwh'].rolling(24).mean(); f['roll_std']=f['electricity_kwh'].rolling(24).std()
        f['mach_roll']=f['machines_active'].rolling(6).mean()
        f=f.dropna().reset_index(drop=True)
        self.feature_names=['hour','dow','is_weekend','is_sunday','machines_active','water_temp','hour_sin','hour_cos','lag1','lag24','roll_mean','roll_std','mach_roll']
        return f

    def train(self, test_size=0.2):
        if self.data is None: self.load_data()
        f=self._features(self.data); s=int(len(f)*(1-test_size))
        Xtr,ytr=f.iloc[:s][self.feature_names],f.iloc[:s]['electricity_kwh']
        Xte,yte=f.iloc[s:][self.feature_names],f.iloc[s:]['electricity_kwh']
        self.scaler=StandardScaler(); Xtr_s=self.scaler.fit_transform(Xtr); Xte_s=self.scaler.transform(Xte)
        self.model=KNeighborsRegressor(n_neighbors=7, weights='distance', metric='minkowski')
        self.model.fit(Xtr_s,ytr); yp=self.model.predict(Xte_s)
        self.metrics={'rmse':round(float(np.sqrt(mean_squared_error(yte,yp))),3),'mae':round(float(mean_absolute_error(yte,yp)),3),'r2':round(float(r2_score(yte,yp)),4),'mape':round(float(np.mean(np.abs((yte-yp)/yte))*100),2)}
        print(f"✓ KNN — R²={self.metrics['r2']}"); return self.model

    def get_feature_importance(self):
        # KNN doesn't have native importance — use permutation
        f=self._features(self.data); X=self.scaler.transform(f[self.feature_names]); y=f['electricity_kwh'].values
        base=float(np.mean(np.abs(y-self.model.predict(X))))
        imp={}
        for i,n in enumerate(self.feature_names):
            Xp=X.copy(); Xp[:,i]=np.random.permutation(Xp[:,i])
            imp[n]=round(max(0,float(np.mean(np.abs(y-self.model.predict(Xp))))-base),4)
        t=sum(imp.values()) or 1
        return {k:round(v/t,4) for k,v in imp.items()}

    def predict_next(self):
        f=self._features(self.data); row=self.scaler.transform(f.iloc[[-1]][self.feature_names])
        dists,idxs=self.model.kneighbors(row)
        p=float(self.model.predict(row)[0])
        return {'prediction':round(p,2),'lower_bound':round(p*0.88,2),'upper_bound':round(p*1.12,2),'confidence_level':88}

    def get_peak_analysis(self):
        if self.data is None: self.load_data()
        hourly=self.data.groupby(self.data['timestamp'].dt.hour)['electricity_kwh'].mean()
        peak_hour=int(hourly.idxmax()); peak_val=round(float(hourly.max()),2)
        sun=self.data[self.data['timestamp'].dt.dayofweek==6]
        return {'peak_hour':peak_hour,'peak_avg':peak_val,
                'sunday_avg':round(float(sun['electricity_kwh'].mean()),2),
                'overall_avg':round(float(self.data['electricity_kwh'].mean()),2),
                'max_machines':int(self.data['machines_active'].max()),
                'avg_machines_peak':round(float(self.data[(self.data['timestamp'].dt.hour>=17)&(self.data['timestamp'].dt.hour<=21)]['machines_active'].mean()),1)}

    def save_model(self):
        with open(self.model_file,'wb') as f: pickle.dump({'model':self.model,'scaler':self.scaler,'features':self.feature_names,'metrics':self.metrics},f)
    def load_model(self):
        with open(self.model_file,'rb') as f: b=pickle.load(f)
        self.model=b['model'];self.scaler=b['scaler'];self.feature_names=b['features'];self.metrics=b['metrics']

if __name__=='__main__':
    p=LaundryPeakPredictor(); p.load_data(); p.train()
    print("Peak analysis:", p.get_peak_analysis()); print("Prediction:", p.predict_next()); p.save_model()
