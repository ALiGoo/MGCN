import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import (
    mean_squared_error,
    mean_squared_log_error,
    mean_absolute_error,
    r2_score,
)

dataset_period = [datetime.strptime('2013-07-01', '%Y-%m-%d'), datetime.strptime('2017-10-01', '%Y-%m-%d')]
test_period = [datetime.strptime('2017-10-01', '%Y-%m-%d') - timedelta(days=80), datetime.strptime('2017-10-01', '%Y-%m-%d')]
valid_period = [test_period[0] - timedelta(days=40), test_period[0]]
train_period = [dataset_period[0], valid_period[0]]
predict_time = "H"

log_path = "logs/mgcn"
warnings.filterwarnings("ignore")

def caculate_loss(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r2_score(y_true, y_pred)

    print("rmse:%f  rmsle:%f  mae:%f  corr:%f  r2:%f" % (rmse, rmsle, mae, corr, r2))

features = pd.read_hdf('process_data/features_201307_201709.h5', key='features')
station_info = pd.read_hdf("process_data/features_201307_201709.h5", key="info")
alive_df = pd.read_hdf("process_data/features_201307_201709.h5", key="alive")
alive_df = alive_df.query('@test_period[0] <= time < @test_period[1]')
predict_df = features[['time','stationid','y_in', 'y_out']].query('@test_period[0] <= time < @test_period[1] & @alive_df.is_alive == 1')
predicts = np.load(f"{log_path}/predicts.npy")
predict_df['y_in_pred'] = predicts[:,0]
predict_df['y_out_pred'] = predicts[:,1]

predict_df_in = (
    predict_df
    .rename(columns={"y_in_pred": "y_pred", "y_in": "y"})
)
predict_df_out = (
    predict_df
    .rename(columns={"y_out_pred": "y_pred", "y_out": "y"})
)

stationid_sort = predict_df.groupby('stationid')['y_in'].sum().sort_values(ascending=False).index

# Total Loss

print("IN")
caculate_loss(predict_df_in.y.values, predict_df_in.y_pred.values)
print("\n")
print("OUT")
caculate_loss(predict_df_out.y.values, predict_df_out.y_pred.values)

# Top 5 Loss

top_in = predict_df_in[predict_df_in.stationid.isin(stationid_sort[:5])]
top_out = predict_df_out[predict_df_out.stationid.isin(stationid_sort[:5])]
print("IN")
caculate_loss(top_in.y.values, top_in.y_pred.values)
print("\n")

print("OUT")
caculate_loss(top_out.y.values, top_out.y_pred.values)

# Top 10 Loss

top_in = predict_df_in[predict_df_in.stationid.isin(stationid_sort[:10])]
top_out = predict_df_out[predict_df_out.stationid.isin(stationid_sort[:10])]
print("IN")
caculate_loss(top_in.y.values, top_in.y_pred.values)
print("\n")

print("OUT")
caculate_loss(top_out.y.values, top_out.y_pred.values)

# 尖峰 Loss

"""
先把y值為0數據拿掉，再以95百分為高峰門檻
"""
print("IN threashold:%i" % (predict_df_in.query("y > 0").y.quantile(0.95)))
print("OUT threashold:%i" % (predict_df_out.query("y > 0").y.quantile(0.95)))

top_in = predict_df_in.query("y >= 20")
top_out = predict_df_out.query("y >= 19")
print("IN")
caculate_loss(top_in.y.values, top_in.y_pred.values)
print("\n")

print("OUT")
caculate_loss(top_out.y.values, top_out.y_pred.values)