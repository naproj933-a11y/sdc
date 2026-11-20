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
