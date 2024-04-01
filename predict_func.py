import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoRegResults

def predict_labels_z_score(data: pd.DataFrame, threshold=1.6) -> np.ndarray:
    
    model = AutoRegResults.load("model_params.pkl")
    
    #Получаем историю поставок, чтобы посчитать агрегированное количество бенчмарков
    historic_data: pd.DataFrame = pd.read_csv("data.csv", parse_dates=["date_of", "ctl_loading_date"])
    historic_data = historic_data[historic_data["date_of"].isin(data["date_of"].unique())]
    
    #Складываем входные данные и данные из прошлого
    all_data: pd.DataFrame = pd.concat([historic_data, data]).drop_duplicates()
    values: np.ndarray = pd.DataFrame([0] * len(data["ctl_loading"].unique()), columns=["val"])
    values.index = data["ctl_loading"].unique()
    for loading in data["ctl_loading"].unique():
        loading_data = data[data["ctl_loading"] == loading]
        date_of = loading_data["date_of"].unique()[0]
        ctl_date = loading_data["ctl_loading_date"].unique()[0]
        values.loc[loading, "val"] = len(all_data[(all_data["date_of"] == date_of) & (all_data["ctl_loading_date"] <= ctl_date)].drop_duplicates(subset=["benchmark_id"]))
        
    #Для каждой поставки предсказываем её значение поставок в date_of 
    ctl_dates: np.ndarray = pd.DataFrame(data.groupby(["ctl_loading"])["date_of"].unique().str[0].astype("datetime64[ns]")).reset_index()
    
    preds: pd.DataFrame = pd.DataFrame(model.predict(ctl_dates["date_of"].min(), ctl_dates["date_of"].max()))
    preds.reset_index(inplace=True)
    preds.rename(columns={0: "quotes_number", "index": "date_of"}, inplace=True)
    
    ctl_dates.rename(columns={0: "date_of"}, inplace=True)
    ctl_dates = pd.merge(ctl_dates, preds, on=["date_of"])
    ctl_dates.set_index("ctl_loading", inplace=True)
    ctl_dates["quotes_number"] -= values["val"]
    
    musigma: pd.DataFrame = pd.read_csv("musigma.csv")
    mu = musigma.loc[0, "mu"]
    sigma = musigma.loc[0, "sigma"]
    
    ctl_dates["quotes_number"] -= mu
    ctl_dates["quotes_number"] /= sigma
    ctl_dates["quotes_number"] = ctl_dates["quotes_number"].apply(abs)
    pred = (ctl_dates["quotes_number"] < threshold).replace({False: "Anomaly", True: "Normal"})
    return pred
    