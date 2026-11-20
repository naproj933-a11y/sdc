from __future__ import annotations
import os, re, random, warnings
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

BASE = Path(__file__).resolve().parent
# Support the repository layout where trained model files are inside ./SDC.
MODEL_DIR = BASE / 'SDC' if (BASE / 'SDC').is_dir() else BASE
DATA = BASE / 'data' if (BASE / 'data').is_dir() else (BASE / 'SDC' / 'data')
CLASS_NAMES = {0: 'Good / Satisfactory / Moderate', 1: 'Poor', 2: 'Very Poor', 3: 'Severe'}
M1 = joblib.load(MODEL_DIR / 'M1_PM25_Regression.joblib')
M2 = joblib.load(MODEL_DIR / 'M2_Delhi_Classification.joblib')
M4 = joblib.load(MODEL_DIR / 'M4_Station_Clustering.joblib')
M1_FEATURES = list(M1['features'])
M2_FEATURES = list(M2.feature_names_in_)
M4_FEATURES = list(M4.feature_names_in_)
app = FastAPI(title='SDC Air Quality Model Testing API', version='1.0.0', description='Individual, combined and random historical-day testing for M1, M2 and M4.')

def _daily_m2():
    fire = pd.read_csv(DATA/'Layer1_Fire_Augmented.csv', parse_dates=['acq_date'])
    weather = pd.read_csv(DATA/'Layer2_Weather_Regional_4Zones.csv', parse_dates=['time'])
    air = pd.read_csv(DATA/'Layer3_AirQuality_Regional_4Zones.csv', parse_dates=['Timestamp'])
    def slug(s): return re.sub(r'[^0-9a-zA-Z]+', '_', s).strip('_').lower()
    def daily_mean(df, date_col, cols, prefix):
        out = pd.DataFrame(index=sorted(df[date_col].dropna().unique()))
        for c in cols:
            if re.search(r'direction|_wd_|^WD', c):
                rad = np.deg2rad(df[c])
                out[prefix+slug(c)+'_sin'] = np.sin(rad).groupby(df[date_col]).mean()
                out[prefix+slug(c)+'_cos'] = np.cos(rad).groupby(df[date_col]).mean()
            else: out[prefix+slug(c)] = df.groupby(date_col)[c].mean()
        return out
    air['date'] = air.Timestamp.dt.normalize(); pmcol='PM2.5 (ug/m3)'
    skip=r'^_id$|lat|lon|hour|day_of|month|year|weekend|ratio|proxy|^TOT|pm2|aqi'
    acols=[c for c in air.select_dtypes('number').columns if not re.search(skip,c,re.I) or c==pmcol]; acols.remove(pmcol)
    pm=air.groupby('date')[pmcol]
    ad=daily_mean(air,'date',acols,'aq_')
    ad['pm25']=pm.mean(); ad['pm25_max']=pm.max(); ad['pm25_min']=pm.min(); ad['pm25_std']=pm.std()
    ad['pm25_last']=air.sort_values('Timestamp').groupby('date')[pmcol].last()
    ad['pm25_share_gt250']=pm.apply(lambda s:(s>250).mean())
    zm=air.groupby(['date','Zone'])[pmcol].mean().unstack(); ad['pm25_zone_spread']=zm.max(axis=1)-zm.min(axis=1)
    ad['pm25_count']=pm.count()
    weather['date']=weather.time.dt.normalize(); wcols=[c for c in weather.select_dtypes('number').columns if not re.search(r'^lat|^lon',c)]
    wd=daily_mean(weather,'date',wcols,'wx_'); wg=weather.groupby('date')
    wd['wx_temp_max']=wg['temperature_2m (°C)'].max(); wd['wx_temp_min']=wg['temperature_2m (°C)'].min()
    wd['wx_blh_min']=wg['boundary_layer_height_m'].min(); wd['wx_gust_max']=wg['wind_gusts_10m (km/h)'].max()
    wd['wx_rain_total']=wg['precipitation (mm)'].sum()
    rr=weather.precipitation_intensity.map({'none':0,'light':1,'moderate':2,'heavy':3}); wd['wx_rain_intensity']=rr.groupby(weather.date).max()
    fire['date']=fire.acq_date.dt.normalize(); fg=fire.groupby('date')
    fd=pd.DataFrame({'fire_n':fg.size(),'fire_frp_sum':fg.frp.sum(),'fire_frp_max':fg.frp.max(),'fire_high_conf':fg.confidence.apply(lambda s:(s=='h').sum())})
    days=pd.date_range(ad.index.min(),ad.index.max()); fd=fd.reindex(days).fillna(0)
    for n in (3,7,14):
        fd[f'fire_n_last{n}d']=fd.fire_n.rolling(n,min_periods=1).sum(); fd[f'fire_frp_last{n}d']=fd.fire_frp_sum.rolling(n,min_periods=1).sum()
    D=ad.reindex(days).join(wd).join(fd); D.index.name='date'; D['pm25']=D.pm25.where(D.pm25_count>=12); D=D.drop(columns='pm25_count')
    for lag in (1,2,3,7): D[f'pm25_lag{lag}']=D.pm25.shift(lag)
    for win in (3,7,14): D[f'pm25_avg{win}d']=D.pm25.rolling(win,min_periods=1).mean()
    D['pm25_change']=D.pm25-D.pm25_lag1
    tomorrow=D.index+pd.Timedelta(days=1); D['month_sin']=np.sin(2*np.pi*tomorrow.month/12); D['month_cos']=np.cos(2*np.pi*tomorrow.month/12)
    D['tomorrow_is_weekend']=(tomorrow.dayofweek>=5).astype(int)
    return D


def _m1_frame():
    aq=pd.read_csv(DATA/'Layer3_AirQuality_Regional_4Zones.csv',parse_dates=['Timestamp'])
    wx=pd.read_csv(DATA/'Layer2_Weather_Regional_4Zones.csv',parse_dates=['time'])
    fr=pd.read_csv(DATA/'Layer1_Fire_Augmented.csv',parse_dates=['acq_date'])
    aq['day']=aq.Timestamp.dt.normalize(); g=aq.groupby(['day','Zone'])
    zpm=g['PM2.5 (ug/m3)'].mean().where(g['PM2.5 (ug/m3)'].count()>=4).unstack()
    da=pd.DataFrame({'pm25':zpm.mean(axis=1),'zones_reporting':zpm.count(axis=1)})
    da['pm10']=aq.groupby('day')['PM10 (ug/m3)'].mean(); da['no2']=aq.groupby('day')['NO2 (ug/m3)'].mean(); da.loc[da.zones_reporting<3,'pm25']=np.nan
    wx['day']=wx.time.dt.normalize(); zw=wx.groupby(['day','zone']).agg(temp_mean=('temperature_2m (°C)','mean'),humidity_mean=('relative_humidity_2m (%)','mean'),wind_speed_mean=('wind_speed_10m (km/h)','mean'),blh_mean=('boundary_layer_height_m','mean'),blh_min=('boundary_layer_height_m','min'),rain_total=('precipitation (mm)','sum'),pressure_mean=('pressure_hPa','mean'),upwind_punjab_ws_mean=('upwind_punjab_ws_kmh','mean'))
    dw=zw.groupby('day').mean(); dfire=fr.groupby('acq_date').agg(fire_count=('frp','size'),fire_frp=('frp','sum'))
    days=pd.date_range('2020-11-01','2025-12-31'); d=da.reindex(days).join(dw).join(dfire); d[['fire_count','fire_frp']]=d[['fire_count','fire_frp']].fillna(0)
    wxcols=list(dw.columns)
    def fill_short(s):
        na=s.isna(); size=na.groupby(na.ne(na.shift()).cumsum()).transform('sum')
        return s.where(~(na&(size<=2)),s.interpolate(limit_area='inside'))
    d[wxcols]=d[wxcols].apply(fill_short)
    d['pm25_lag0']=d.pm25; d['pm25_lag1']=d.pm25.shift(1); d['pm25_roll3']=d.pm25.rolling(3,min_periods=2).mean(); d['pm25_roll7']=d.pm25.rolling(7,min_periods=4).mean()
    d['pm25_diff1']=d.pm25_lag0-d.pm25_lag1; d['pm10_lag0']=d.pm10; d['pm25_pm10_ratio']=d.pm25_lag0/d.pm10_lag0; d['no2_lag0']=d.no2
    d['ventilation_index']=d.wind_speed_mean*d.blh_mean; d['fc_wind_speed_mean']=d.wind_speed_mean.shift(-1); d['fc_blh_mean']=d.blh_mean.shift(-1); d['fc_rain_total']=d.rain_total.shift(-1)
    d['fc_ventilation_index']=d.fc_wind_speed_mean*d.fc_blh_mean; d['fire_count_3d']=d.fire_count.rolling(3,min_periods=1).sum(); d['fire_frp_3d']=d.fire_frp.rolling(3,min_periods=1).sum()
    d['month']=d.index.month; d['is_weekend']=(d.index.dayofweek>=5).astype(int)
    return d


def _m4_features():
    air=pd.read_csv(DATA/'Layer3_AirQuality_Regional_4Zones.csv',parse_dates=['Timestamp'])
    air=air[['Timestamp','Station ID','Station Name','Zone','PM2.5 (ug/m3)','PM10 (ug/m3)']].copy()
    air['pm25']=pd.to_numeric(air['PM2.5 (ug/m3)'],errors='coerce'); air['pm10']=pd.to_numeric(air['PM10 (ug/m3)'],errors='coerce')
    for c in ['pm25','pm10']: air.loc[(air[c]<0)|(air[c]>2500),c]=np.nan
    air=air.dropna(subset=['Timestamp','Station ID','Station Name','Zone']); air['date']=air.Timestamp.dt.floor('D')
    daily=air.groupby(['Station ID','Station Name','Zone','date'],as_index=False).agg(pm25=('pm25','mean'),pm10=('pm10','mean'),pm25_count=('pm25','count'),pm10_count=('pm10','count'))
    daily.loc[daily.pm25_count<4,'pm25']=np.nan; daily.loc[daily.pm10_count<4,'pm10']=np.nan; daily.loc[daily.pm10<=0,'pm10']=np.nan
    daily['is_weekend']=daily.date.dt.dayofweek>=5
    total=daily.groupby('date').pm25.transform('sum'); count=daily.groupby('date').pm25.transform('count'); other_count=count-daily.pm25.notna().astype(int)
    daily['other_station_mean']=(total-daily.pm25.fillna(0))/other_count.replace(0,np.nan); daily.loc[other_count<2,'other_station_mean']=np.nan
    daily['pm25_anomaly']=daily.pm25-daily.other_station_mean
    def episode(g):
        g=g.sort_values('date'); lengths=[]; run=0; last=None
        for dt,v in zip(g.date,g.pm25):
            consecutive=last is not None and (dt-last).days==1
            if pd.notna(v) and v>=121: run=run+1 if consecutive and run>0 else 1
            else:
                if run: lengths.append(run)
                run=0
            last=dt
        if run: lengths.append(run)
        return float(np.mean(lengths)) if lengths else 0.0
    rows=[]
    for sid,g in daily.groupby('Station ID'):
        valid=g.dropna(subset=['pm25']); weekdays=valid.loc[~valid.is_weekend,'pm25']; weekends=valid.loc[valid.is_weekend,'pm25']; paired=g.dropna(subset=['pm25','pm10'])
        rows.append({'Station ID':sid,'Station Name':g['Station Name'].iloc[0],M4_FEATURES[0]:valid.pm25_anomaly.mean(),M4_FEATURES[1]:100*valid.pm25.ge(121).mean(),M4_FEATURES[2]:weekdays.mean()-weekends.mean(),M4_FEATURES[3]:episode(g)})
    return pd.DataFrame(rows).sort_values('Station ID').reset_index(drop=True)



def _plain_value(value):
    """Convert pandas/numpy values into JSON/Pydantic-friendly Python values."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _feature_input_model(model_name: str, feature_names: list[str], defaults: dict):
    """Create explicit Swagger fields with defaults from the latest saved CSV row."""
    fields = {}
    for feature in feature_names:
        value = _plain_value(defaults.get(feature))
        fields[feature] = (
            float | None,
            Field(default=value, description='Latest available saved CSV value; editable')
        )
    return create_model(model_name, **fields)


def _load_latest_csv_defaults():
    """Build the most recent usable M1/M2 rows and current M4 station profiles."""
    d1 = _m1_frame()
    valid1 = d1.dropna(subset=['pm25_lag0'])
    if valid1.empty:
        m1_defaults = {}
    else:
        m1_defaults = {
            feature: _plain_value(valid1.iloc[-1].get(feature))
            for feature in M1_FEATURES
        }

    d2 = _daily_m2()
    valid2 = d2.dropna(subset=['pm25'])
    if valid2.empty:
        m2_defaults = {}
    else:
        m2_defaults = {
            feature: _plain_value(valid2.iloc[-1].get(feature))
            for feature in M2_FEATURES
        }

    station_frame = _m4_features()
    m4_defaults = []
    for _, row in station_frame.iterrows():
        item = {
            'station_name': str(row.get('Station Name', '')),
        }
        for feature in M4_FEATURES:
            item[feature] = _plain_value(row.get(feature))
        m4_defaults.append(item)

    return m1_defaults, m2_defaults, m4_defaults


# These defaults are loaded from the latest available saved CSV data when the API starts.
# They are displayed in Swagger and can be edited before submitting each endpoint.
try:
    _M1_DEFAULTS, _M2_DEFAULTS, _M4_DEFAULT_STATIONS = _load_latest_csv_defaults()
except Exception as exc:
    warnings.warn(f'Could not load latest CSV defaults for Swagger: {exc}')
    _M1_DEFAULTS, _M2_DEFAULTS, _M4_DEFAULT_STATIONS = {}, {}, []

M1FeatureInput = _feature_input_model('M1FeatureInput', M1_FEATURES, _M1_DEFAULTS)
M2FeatureInput = _feature_input_model('M2FeatureInput', M2_FEATURES, _M2_DEFAULTS)

_m4_station_fields = {
    'station_name': (str | None, Field(default=None, description='Station name')),
}
for _feature in M4_FEATURES:
    _m4_station_fields[_feature] = (
        float | None,
        Field(
            default=_plain_value(_M4_DEFAULT_STATIONS[0].get(_feature)) if _M4_DEFAULT_STATIONS else None,
            description='Latest saved station-profile value; editable'
        )
    )
M4StationInput = create_model('M4StationInput', **_m4_station_fields)


class M1Input(BaseModel):
    features: M1FeatureInput = Field(default=M1FeatureInput())


class M2Input(BaseModel):
    features: M2FeatureInput = Field(default=M2FeatureInput())


class M4Input(BaseModel):
    stations: list[M4StationInput] = Field(
        default=_M4_DEFAULT_STATIONS,
        description='Latest saved station-profile feature values from the CSV; editable'
    )


def _predict_m1(values):
    row=pd.DataFrame([values]).reindex(columns=M1_FEATURES)
    if pd.isna(row['pm25_lag0'].iloc[0]): raise ValueError('pm25_lag0 (today PM2.5) is required')
    missing=[f for f in M1_FEATURES if pd.isna(row[f].iloc[0])]
    for name,a,b in [('ventilation_index','wind_speed_mean','blh_mean'),('fc_ventilation_index','fc_wind_speed_mean','fc_blh_mean')]:
        if name in missing and row[[a,b]].notna().all(axis=None): row[name]=row[a].iloc[0]*row[b].iloc[0]; missing.remove(name)
    if 'pm25_diff1' in missing and row[['pm25_lag0','pm25_lag1']].notna().all(axis=None): row['pm25_diff1']=row.pm25_lag0.iloc[0]-row.pm25_lag1.iloc[0]; missing.remove('pm25_diff1')
    change=float(M1['model'].predict(row[M1_FEATURES])[0]); pred=float(np.expm1(np.log1p(float(row.pm25_lag0.iloc[0]))+change))
    return {'predicted_pm25':round(pred,1),'interval_low':float(round(pred*np.exp(M1['q10']),1)),'interval_high':float(round(pred*np.exp(M1['q90']),1)),'inputs_missing_filled_with_model_median':missing}


def _predict_m2(values):
    row=pd.DataFrame([values]).reindex(columns=M2_FEATURES)
    if row.isna().all(axis=1).iloc[0]: raise ValueError('At least one valid M2 feature is required')
    missing=[f for f in M2_FEATURES if pd.isna(row[f].iloc[0])]
    pred=int(M2.predict(row)[0]); result={'class_id':pred,'class_name':CLASS_NAMES.get(pred,str(pred)), 'inputs_missing_filled_with_model_median':missing}
    if hasattr(M2,'predict_proba'):
        probs=M2.predict_proba(row)[0]; result['class_probabilities']={CLASS_NAMES.get(int(c),str(c)):round(float(p),4) for c,p in zip(M2.classes_,probs)}
    return result

@app.get('/')
def root(): return {'project':'SDC','api':'SDC Air Quality Model Testing API','docs':'/docs','modes':['individual','combined','random-day','live-date-prediction']}
@app.get('/health')
def health(): return {'status':'ok','models_loaded':['M1_PM25_Regression','M2_Delhi_Classification','M4_Station_Clustering'],'feature_counts':{'M1':len(M1_FEATURES),'M2':len(M2_FEATURES),'M4':len(M4_FEATURES)}}
@app.get('/schemas')
def schemas(): return {'M1_features':M1_FEATURES,'M2_features':M2_FEATURES,'M4_features':M4_FEATURES}

@app.post('/test/m1')
def test_m1(body:M1Input):
    try:
        values = body.features.model_dump()
        return {'model':'M1_PM25_Regression','input':values,'output':_predict_m1(values)}
    except Exception as e: raise HTTPException(400,str(e))
@app.post('/test/m2')
def test_m2(body:M2Input):
    try:
        values = body.features.model_dump()
        return {'model':'M2_Delhi_Classification','input':values,'output':_predict_m2(values)}
    except Exception as e: raise HTTPException(400,str(e))
@app.post('/test/m4')
def test_m4(body:M4Input):
    try:
        station_rows = []
        for station in body.stations:
            item = station.model_dump()
            item['Station Name'] = item.pop('station_name', None)
            station_rows.append(item)
        frame = pd.DataFrame(station_rows)
        missing = [c for c in M4_FEATURES if c not in frame]
        if missing:
            raise ValueError('Missing M4 feature columns: ' + ', '.join(missing))
        if frame[M4_FEATURES].isna().any().any():
            raise ValueError('M4 feature values cannot be missing')
        labels = M4.predict(frame[M4_FEATURES])
        out = []
        for i, label in enumerate(labels):
            out.append({
                'station_name': frame.iloc[i].get('Station Name') or f'Station {i+1}',
                'cluster': int(label)
            })
        return {
            'model': 'M4_Station_Clustering',
            'input': station_rows,
            'output': {'station_assignments': out}
        }
    except Exception as e: raise HTTPException(400,str(e))

@app.post('/test/combined')
def test_combined():
    # Current dataset's latest complete engineered row. M4 is a station-profile result, not a next-day forecast.
    try:
        d1=_m1_frame(); valid1=d1.dropna(subset=[c for c in M1_FEATURES if c not in ['pm10_lag0','no2_lag0','pm25_pm10_ratio']]).copy()
        if valid1.empty: raise ValueError('No usable M1 rows')
        date1=valid1.index[-1]; m1=_predict_m1(valid1.loc[date1].to_dict())
        d2=_daily_m2(); valid2=d2.dropna(subset=['pm25']); date2=valid2.index[-1]; m2=_predict_m2(valid2.loc[date2].to_dict())
        st=_m4_features(); labels=M4.predict(st[M4_FEATURES]); stations=[{'station_id':str(r['Station ID']),'station_name':r['Station Name'],'cluster':int(labels[i])} for i,(_,r) in enumerate(st.iterrows())]
        return {'mode':'combined','M1':{'input_date':str(date1.date()),'prediction_for':str((date1+pd.Timedelta(days=1)).date()),'input':valid1.loc[date1].reindex(M1_FEATURES).where(pd.notna(valid1.loc[date1].reindex(M1_FEATURES)),None).to_dict(),'output':m1},'M2':{'input_date':str(date2.date()),'prediction_for':str((date2+pd.Timedelta(days=1)).date()),'output':m2},'M4':{'profile_basis':'full historical station dataset','output':{'station_assignments':stations}},'note':'M1 and M2 forecast next-day outcomes. M4 clusters historical station profiles; it is not a daily forecast.'}
    except Exception as e: raise HTTPException(500,str(e))

@app.post('/test/random-day')
def test_random_day(seed:int|None=None):
    # Randomly choose an eligible historical origin date. Compare next-day prediction with recorded target.
    try:
        rng=random.Random(seed)
        d1=_m1_frame(); d2=_daily_m2()
        common=d1.index.intersection(d2.index)
        candidates=[]
        for dt in common[:-1]:
            nxt=dt+pd.Timedelta(days=1)
            if nxt not in d1.index or nxt not in d2.index: continue
            r1=d1.loc[dt]; actual1=d1.loc[nxt,'pm25']
            if pd.isna(actual1) or pd.isna(r1.get('pm25_lag0')): continue
            # M1's feature design includes next-day weather forecast columns (shifted from the recorded dataset).
            if any(pd.isna(r1.get(c)) for c in ['fc_wind_speed_mean','fc_blh_mean','fc_rain_total']): continue
            r2=d2.loc[dt]; actual2=d2.loc[nxt,'pm25']
            if pd.isna(actual2): continue
            candidates.append((dt,nxt,r1,r2,float(actual1),float(actual2)))
        if not candidates: raise ValueError('No eligible historical date found for random-day testing')
        dt,nxt,r1,r2,a1,a2=rng.choice(candidates)
        p1=_predict_m1(r1.to_dict()); p2=_predict_m2(r2.to_dict())
        actual_class=int(np.select([a2<=90,(a2>90)&(a2<121),(a2>=121)&(a2<=250),a2>250],[0,1,2,3],default=-1))
        m4=_m4_features(); labels=M4.predict(m4[M4_FEATURES])
        return {'mode':'random-day-backtest','random_seed':seed,'input_date':str(dt.date()),'prediction_date':str(nxt.date()),'M1':{'prediction':p1,'actual_pm25':round(a1,1),'absolute_error':round(abs(p1['predicted_pm25']-a1),1)},'M2':{'prediction':p2,'actual_class_id':actual_class,'actual_class_name':CLASS_NAMES[actual_class],'correct':p2['class_id']==actual_class},'M4':{'station_profile_clusters':[{'station_name':r['Station Name'],'cluster':int(labels[i])} for i,(_,r) in enumerate(m4.iterrows())]},'method_note':'Historical backtest: the randomly selected origin day is used as input; the following day is held out for comparison. M1 next-day weather columns use recorded next-day weather as a retrospective forecast proxy, matching the notebook feature construction; this is not a live weather forecast.'}
    except Exception as e: raise HTTPException(500,str(e))

# ---------------- LIVE DATE-BASED PREDICTION ----------------
# Uses Open-Meteo forecast/CAMS endpoints for Delhi's four training zones.
# These are gridded model estimates, not CPCB station observations.
from datetime import datetime, date as Date, timedelta as Timedelta
from zoneinfo import ZoneInfo
import requests

DELHI_ZONES = {
    'Central': {'latitude': 28.6289, 'longitude': 77.2409},
    'East':    {'latitude': 28.6469, 'longitude': 77.3160},
    'North':   {'latitude': 28.7972, 'longitude': 77.1331},
    'South':   {'latitude': 28.5528, 'longitude': 77.2183},
}
WEATHER_HOURLY = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'apparent_temperature', 'precipitation', 'wind_speed_10m',
    'wind_direction_10m', 'wind_gusts_10m', 'cloud_cover',
    'pressure_msl', 'boundary_layer_height', 'wind_speed_100m',
    'wind_direction_100m', 'shortwave_radiation',
    'vapour_pressure_deficit', 'visibility'
]
AIR_HOURLY = ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide',
              'sulphur_dioxide', 'ozone']

class LivePredictionInput(BaseModel):
    date: Date = Field(..., description='Delhi local date to predict, YYYY-MM-DD')


def _get_hourly(url: str, params: dict) -> dict:
    try:
        response = requests.get(url, params=params, timeout=25)
        response.raise_for_status()
        payload = response.json()
        if 'hourly' not in payload:
            raise ValueError('Weather provider returned no hourly data')
        return payload['hourly']
    except requests.RequestException as exc:
        raise ValueError(f'Could not retrieve live data from Open-Meteo: {exc}') from exc


def _day_values(hourly: dict, day: Date) -> dict:
    times = hourly.get('time', [])
    indexes = [i for i, stamp in enumerate(times) if stamp[:10] == day.isoformat()]
    out = {}
    for key, values in hourly.items():
        if key == 'time' or not isinstance(values, list):
            continue
        picked = [values[i] for i in indexes if i < len(values) and values[i] is not None]
        out[key] = float(np.mean(picked)) if picked else None
        if key == 'precipitation':
            out['precipitation_sum'] = float(np.sum(picked)) if picked else None
        if key == 'wind_gusts_10m':
            out['wind_gusts_10m_max'] = float(np.max(picked)) if picked else None
        if key == 'boundary_layer_height':
            out['boundary_layer_height_min'] = float(np.min(picked)) if picked else None
        if key == 'pm2_5':
            out['pm2_5_max'] = float(np.max(picked)) if picked else None
            out['pm2_5_min'] = float(np.min(picked)) if picked else None
            out['pm2_5_std'] = float(np.std(picked, ddof=1)) if len(picked) > 1 else 0.0
            out['pm2_5_last'] = float(picked[-1]) if picked else None
            out['pm2_5_share_gt250'] = float(np.mean(np.array(picked) > 250)) if picked else None
    return out


def _fetch_live_delhi_days(start_day: Date, end_day: Date) -> dict:
    # Include a seven-day context window so lag/rolling features can be built.
    past_start = start_day - Timedelta(days=14)
    all_days = {}
    for zone, coords in DELHI_ZONES.items():
        common = {
            **coords, 'timezone': 'Asia/Kolkata',
            'past_days': 20, 'forecast_days': 5,
        }
        weather_params = {**common, 'hourly': ','.join(WEATHER_HOURLY)}
        air_params = {**common, 'hourly': ','.join(AIR_HOURLY)}
        weather = _get_hourly('https://api.open-meteo.com/v1/forecast', weather_params)
        air = _get_hourly('https://air-quality-api.open-meteo.com/v1/air-quality', air_params)
        for offset in range((end_day - past_start).days + 1):
            day = past_start + Timedelta(days=offset)
            all_days.setdefault(day, {})[zone] = {
                'weather': _day_values(weather, day),
                'air': _day_values(air, day),
            }
    return all_days


def _mean_key(rows: list[dict], key: str):
    vals = [r.get(key) for r in rows if r.get(key) is not None and np.isfinite(r.get(key))]
    return float(np.mean(vals)) if vals else None


def _aggregate_live_day(zone_data: dict) -> dict:
    weather_rows = [v['weather'] for v in zone_data.values()]
    air_rows = [v['air'] for v in zone_data.values()]
    result = {}
    for key in set().union(*(r.keys() for r in weather_rows + air_rows)):
        rows = air_rows if key in AIR_HOURLY or key.startswith('pm2_5_') else weather_rows
        result[key] = _mean_key(rows, key)
    # Preserve a zone-to-zone PM2.5 spread, used by the classifier.
    zone_pm = [v['air'].get('pm2_5') for v in zone_data.values()]
    zone_pm = [v for v in zone_pm if v is not None]
    result['pm25_zone_spread'] = float(max(zone_pm) - min(zone_pm)) if len(zone_pm) >= 2 else None
    result['zones_reporting'] = len(zone_pm)
    return result


def _build_live_feature_rows(days: dict, target_day: Date) -> tuple[dict, dict, dict]:
    # Both trained models use day t inputs to predict day t+1.
    origin = target_day - Timedelta(days=1)
    daily = {d: _aggregate_live_day(z) for d, z in days.items()}
    if origin not in daily:
        raise ValueError(f'Live provider did not return required input date {origin}')
    cur = daily[origin]
    prev = daily.get(origin - Timedelta(days=1), {})

    # M1: exactly the training feature names, with target-day weather in fc_*.
    m1 = {
        'pm25_lag0': cur.get('pm2_5'),
        'pm25_lag1': prev.get('pm2_5'),
        'pm25_roll3': _mean_key([daily.get(origin - Timedelta(days=i), {}) for i in range(3)], 'pm2_5'),
        'pm25_roll7': _mean_key([daily.get(origin - Timedelta(days=i), {}) for i in range(7)], 'pm2_5'),
        'pm25_diff1': (cur.get('pm2_5') - prev.get('pm2_5')) if cur.get('pm2_5') is not None and prev.get('pm2_5') is not None else None,
        'pm10_lag0': cur.get('pm10'),
        'pm25_pm10_ratio': (cur.get('pm2_5') / cur.get('pm10')) if cur.get('pm2_5') is not None and cur.get('pm10') not in (None, 0) else None,
        'no2_lag0': cur.get('nitrogen_dioxide'),
        'temp_mean': cur.get('temperature_2m'),
        'humidity_mean': cur.get('relative_humidity_2m'),
        'wind_speed_mean': cur.get('wind_speed_10m'),
        'blh_mean': cur.get('boundary_layer_height'),
        'blh_min': cur.get('boundary_layer_height_min'),
        'rain_total': cur.get('precipitation_sum'),
        'pressure_mean': cur.get('pressure_msl'),
        'ventilation_index': (cur.get('wind_speed_10m') * cur.get('boundary_layer_height')) if cur.get('wind_speed_10m') is not None and cur.get('boundary_layer_height') is not None else None,
        'fc_wind_speed_mean': daily.get(target_day, {}).get('wind_speed_10m'),
        'fc_blh_mean': daily.get(target_day, {}).get('boundary_layer_height'),
        'fc_rain_total': daily.get(target_day, {}).get('precipitation_sum'),
        'fc_ventilation_index': (daily.get(target_day, {}).get('wind_speed_10m') * daily.get(target_day, {}).get('boundary_layer_height')) if daily.get(target_day, {}).get('wind_speed_10m') is not None and daily.get(target_day, {}).get('boundary_layer_height') is not None else None,
        'month': origin.month,
        'is_weekend': int(origin.weekday() >= 5),
        # Not supplied by the public forecast endpoint; model imputer handles these and response reports them.
        'upwind_punjab_ws_mean': None, 'fire_count_3d': None, 'fire_frp_3d': None,
    }

    # M2: same origin-day inputs as training; target date is the next-day label.
    m2 = {}
    aq_map = {
        'aq_pm10_ug_m3': 'pm10', 'aq_no2_ug_m3': 'nitrogen_dioxide',
        'aq_nh3_ug_m3': 'ammonia', 'aq_so2_ug_m3': 'sulphur_dioxide',
        'aq_ozone_ug_m3': 'ozone', 'aq_co_mg_m3': 'carbon_monoxide',
    }
    for feature, source in aq_map.items():
        value = cur.get(source)
        m2[feature] = (value / 1000.0 if feature == 'aq_co_mg_m3' and value is not None else value)
    # Meteorological columns in the original AQ table are represented by
    # same-day Open-Meteo weather proxies (unit conversions are explicit).
    m2.update({
        'aq_at_degc': cur.get('temperature_2m'),
        'aq_rh': cur.get('relative_humidity_2m'),
        'aq_ws_m_s': (cur.get('wind_speed_10m') / 3.6) if cur.get('wind_speed_10m') is not None else None,
        'aq_wd_deg_sin': float(np.sin(np.deg2rad(cur['wind_direction_10m']))) if cur.get('wind_direction_10m') is not None else None,
        'aq_wd_deg_cos': float(np.cos(np.deg2rad(cur['wind_direction_10m']))) if cur.get('wind_direction_10m') is not None else None,
        'aq_rf_mm': cur.get('precipitation_sum'),
        'aq_sr_w_mt2': cur.get('shortwave_radiation'),
        'aq_bp_mmhg': (cur.get('pressure_msl') * 0.750061683) if cur.get('pressure_msl') is not None else None,
        'aq_pressure_hpa': cur.get('pressure_msl'),
        'aq_boundary_layer_height_m': cur.get('boundary_layer_height'),
        'aq_dew_point_c': cur.get('dew_point_2m'),
        'aq_vapour_pressure_deficit_hpa': cur.get('vapour_pressure_deficit'),
        'aq_fog_visibility_km': (cur.get('visibility') / 1000.0) if cur.get('visibility') is not None else None,
        'aq_cloud_cover_pct': cur.get('cloud_cover'),
    })
    m2.update({
        'pm25': cur.get('pm2_5'), 'pm25_max': cur.get('pm2_5_max'),
        'pm25_min': cur.get('pm2_5_min'), 'pm25_std': cur.get('pm2_5_std'),
        'pm25_last': cur.get('pm2_5_last'), 'pm25_share_gt250': cur.get('pm2_5_share_gt250'),
        'pm25_zone_spread': cur.get('pm25_zone_spread'),
        'wx_temperature_2m_c': cur.get('temperature_2m'),
        'wx_relative_humidity_2m': cur.get('relative_humidity_2m'),
        'wx_dew_point_2m_c': cur.get('dew_point_2m'),
        'wx_apparent_temperature_c': cur.get('apparent_temperature'),
        'wx_precipitation_mm': cur.get('precipitation'),
        'wx_wind_speed_10m_km_h': cur.get('wind_speed_10m'),
        'wx_wind_gusts_10m_km_h': cur.get('wind_gusts_10m'),
        'wx_cloud_cover': cur.get('cloud_cover'), 'wx_pressure_hpa': cur.get('pressure_msl'),
        'wx_boundary_layer_height_m': cur.get('boundary_layer_height'),
        'wx_wind_speed_100m_kmh': cur.get('wind_speed_100m'),
        'wx_shortwave_radiation_wm2': cur.get('shortwave_radiation'),
        'wx_vapour_pressure_deficit_hpa': cur.get('vapour_pressure_deficit'),
        'wx_fog_visibility_km': (cur.get('visibility') / 1000.0) if cur.get('visibility') is not None else None,
        'wx_temp_max': None, 'wx_temp_min': None, 'wx_blh_min': cur.get('boundary_layer_height_min'),
        'wx_gust_max': cur.get('wind_gusts_10m_max'), 'wx_rain_total': cur.get('precipitation_sum'),
        'wx_rain_intensity': (0 if not cur.get('precipitation_sum') else 1),
        'pm25_lag1': prev.get('pm2_5'),
        'pm25_lag2': daily.get(origin - Timedelta(days=2), {}).get('pm2_5'),
        'pm25_lag3': daily.get(origin - Timedelta(days=3), {}).get('pm2_5'),
        'pm25_lag7': daily.get(origin - Timedelta(days=7), {}).get('pm2_5'),
        'pm25_avg3d': _mean_key([daily.get(origin - Timedelta(days=i), {}) for i in range(3)], 'pm2_5'),
        'pm25_avg7d': _mean_key([daily.get(origin - Timedelta(days=i), {}) for i in range(7)], 'pm2_5'),
        'pm25_avg14d': _mean_key([daily.get(origin - Timedelta(days=i), {}) for i in range(14)], 'pm2_5'),
        'pm25_change': (cur.get('pm2_5') - prev.get('pm2_5')) if cur.get('pm2_5') is not None and prev.get('pm2_5') is not None else None,
        'month_sin': float(np.sin(2 * np.pi * target_day.month / 12)),
        'month_cos': float(np.cos(2 * np.pi * target_day.month / 12)),
        'tomorrow_is_weekend': int(target_day.weekday() >= 5),
    })
    # Directional features use circular encoding.
    for src, prefix in [('wind_direction_10m', 'wx_wind_direction_10m'), ('wind_direction_100m', 'wx_wind_direction_100m_deg')]:
        angle = cur.get(src)
        if angle is not None:
            m2[prefix + ('_sin' if '10m' in src else '_sin')] = float(np.sin(np.deg2rad(angle)))
            m2[prefix + ('_cos' if '10m' in src else '_cos')] = float(np.cos(np.deg2rad(angle)))
    # Additional AQ source features that have no Open-Meteo equivalent remain None and are imputed.
    for feature in M2_FEATURES:
        m2.setdefault(feature, None)
    for feature in M1_FEATURES:
        m1.setdefault(feature, None)
    return m1, m2, daily


@app.post('/predict/live')
def predict_live(body: LivePredictionInput):
    """Predict Delhi's selected date using live/forecast inputs from Open-Meteo."""
    try:
        today = datetime.now(ZoneInfo('Asia/Kolkata')).date()
        if body.date < today - Timedelta(days=1) or body.date > today + Timedelta(days=4):
            raise ValueError(f'Date must be between {(today-Timedelta(days=1)).isoformat()} and {(today+Timedelta(days=4)).isoformat()} (Delhi local forecast window).')
        target = body.date
        days = _fetch_live_delhi_days(target - Timedelta(days=1), target)
        m1_values, m2_values, daily = _build_live_feature_rows(days, target)
        m1_result = _predict_m1(m1_values)
        m2_result = _predict_m2(m2_values)
        # M4 is a static station-profile model, not a date-specific forecast.
        station_frame = _m4_features()
        station_labels = M4.predict(station_frame[M4_FEATURES])
        station_clusters = [
            {'station_name': row['Station Name'], 'cluster': int(station_labels[i])}
            for i, (_, row) in enumerate(station_frame.iterrows())
        ]
        target_weather = daily.get(target, {})
        return {
            'mode': 'live-date-prediction', 'location': 'Delhi, India',
            'requested_date': target.isoformat(), 'input_date': (target-Timedelta(days=1)).isoformat(),
            'M1': {'prediction_for': target.isoformat(), 'output': m1_result},
            'M2': {'prediction_for': target.isoformat(), 'output': m2_result},
            'weather_for_requested_date': {
                'temperature_2m_c': target_weather.get('temperature_2m'),
                'relative_humidity_percent': target_weather.get('relative_humidity_2m'),
                'dew_point_c': target_weather.get('dew_point_2m'),
                'apparent_temperature_c': target_weather.get('apparent_temperature'),
                'wind_speed_10m_kmh': target_weather.get('wind_speed_10m'),
                'wind_direction_10m_degrees': target_weather.get('wind_direction_10m'),
                'wind_gusts_10m_kmh': target_weather.get('wind_gusts_10m_max'),
                'precipitation_total_mm': target_weather.get('precipitation_sum'),
                'cloud_cover_percent': target_weather.get('cloud_cover'),
                'pressure_hpa': target_weather.get('pressure_msl'),
                'boundary_layer_height_m': target_weather.get('boundary_layer_height'),
                'wind_speed_100m_kmh': target_weather.get('wind_speed_100m'),
                'shortwave_radiation_w_m2': target_weather.get('shortwave_radiation'),
                'visibility_km': (target_weather.get('visibility') / 1000.0) if target_weather.get('visibility') is not None else None,
            },
            'air_quality_model_estimates_for_requested_date': {
                'pm25_ug_m3': target_weather.get('pm2_5'),
                'pm10_ug_m3': target_weather.get('pm10'),
                'carbon_monoxide_ug_m3': target_weather.get('carbon_monoxide'),
                'nitrogen_dioxide_ug_m3': target_weather.get('nitrogen_dioxide'),
                'sulphur_dioxide_ug_m3': target_weather.get('sulphur_dioxide'),
                'ozone_ug_m3': target_weather.get('ozone'),
            },
            'zone_inputs_for_requested_date': {
                zone: {'weather': values['weather'], 'air_quality': values['air']}
                for zone, values in days.get(target, {}).items()
            },
            'M4': {'profile_basis': 'historical station profiles; not a daily prediction', 'station_profile_clusters': station_clusters},
            'data_source': 'Open-Meteo weather forecast and CAMS air-quality model estimates for four Delhi zone coordinates; not CPCB station observations.',
            'limitations': [
                'The trained models were fitted on historical CPCB/weather/fire datasets. Open-Meteo CAMS inputs have a different source and spatial resolution, so live predictions are experimental and require validation.',
                'Unavailable pollutants, Punjab upwind-wind features and fire features are left missing and handled by the trained model imputer; see the missing-feature lists.',
                'M1 predicts next-day PM2.5 from the previous day inputs plus requested-day forecast weather. M2 predicts the requested-day category from previous-day features.',
                'M4 assigns historical station-profile clusters and does not forecast daily air quality.'
            ]
        }
    except Exception as e:
        raise HTTPException(400, str(e))
