import pandas as pd, numpy as np, pickle, warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def train():
    df = pd.read_csv('laundry_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.weekday

    # Categorize load: low, medium, high based on electricity_kwh
    q33 = df['electricity_kwh'].quantile(0.33)
    q66 = df['electricity_kwh'].quantile(0.66)
    df['category'] = pd.cut(df['electricity_kwh'], bins=[-1, q33, q66, 999], labels=['low', 'medium', 'high'])
    df['cat_code'] = df['category'].cat.codes

    feats = ['hour', 'day_of_week', 'is_weekend', 'machines_active', 'water_temp']
    X, y = df[feats].values, df['cat_code'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    nb = GaussianNB(); nb.fit(X_tr, y_tr)
    nb_preds = nb.predict(X_te)
    nb_acc = round(accuracy_score(y_te, nb_preds), 4)
    print(f"NaiveBayes Accuracy: {nb_acc}")

    # Prophet or fallback forecast
    try:
        from prophet import Prophet
        daily = df.groupby(df['timestamp'].dt.date).agg({'electricity_kwh': 'sum'}).reset_index()
        daily.columns = ['ds', 'y']; daily['ds'] = pd.to_datetime(daily['ds'])
        m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False,
                    changepoint_prior_scale=0.05)
        m.fit(daily)
        future = m.make_future_dataframe(periods=14)
        forecast = m.predict(future)
        forecast_out = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(14)
        forecast_out['ds'] = forecast_out['ds'].astype(str)
        forecast_data = forecast_out.to_dict(orient='records')
        engine = 'Prophet'
        print(f"Prophet forecast: next 14 days generated")
    except ImportError:
        daily = df.groupby(df['timestamp'].dt.date)['electricity_kwh'].sum().values
        avg = float(np.mean(daily[-7:]))
        forecast_data = [{'ds': str(pd.Timestamp.now().date() + pd.Timedelta(days=i)),
                          'yhat': round(avg + np.random.normal(0, avg*0.05), 2),
                          'yhat_lower': round(avg * 0.85, 2), 'yhat_upper': round(avg * 1.15, 2)} for i in range(1, 15)]
        engine = 'MovingAverage-fallback'
        print(f"Using MA fallback (Prophet not installed)")

    bundle = {'nb_model': nb, 'nb_accuracy': nb_acc, 'forecast': forecast_data,
              'features': feats, 'engine': engine,
              'category_thresholds': {'q33': round(float(q33), 2), 'q66': round(float(q66), 2)}}

    with open('model.pkl', 'wb') as f: pickle.dump(bundle, f)
    print(f"Engine: {engine} | NB Acc: {nb_acc}")
    return bundle

if __name__ == '__main__':
    train()
