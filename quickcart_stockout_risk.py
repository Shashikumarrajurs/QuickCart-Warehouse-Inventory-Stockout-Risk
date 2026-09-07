from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

BASE=Path(".")
stores=pd.read_csv(BASE/"dim_stores.csv")
skus=pd.read_csv(BASE/"dim_skus.csv")
suppliers=pd.read_csv(BASE/"dim_suppliers.csv",na_values=["N/A","missing","--","NA","null"],keep_default_na=True)
events=pd.read_csv(BASE/"dim_events.csv",parse_dates=["date"])
fact=pd.read_csv(BASE/"fact_inventory_daily.csv",parse_dates=["date"])

df=fact.merge(stores,on="store_id",validate="many_to_one")
df=df.merge(skus,on=["sku_id","supplier_id"],validate="many_to_one")
df=df.merge(suppliers,on="supplier_id",validate="many_to_one")
df=df.merge(events,on="date",validate="many_to_one")

df["supplier_reliability_clean"]=df["reliability_score"].fillna(suppliers["reliability_score"].median())
df["reorder_gap"]=df["reorder_point"]-df["closing_stock"]
df["days_of_cover_ratio"]=df["days_of_cover"]/df["lead_time_days_expected"].replace(0,np.nan)
df=df.sort_values(["store_id","sku_id","date"])
df["is_recent_reorder"]=(df.groupby(["store_id","sku_id"])["reorder_placed"].transform(lambda s:s.eq("Y").shift(1).fillna(False).rolling(3,min_periods=1).max()).astype(int))
df["day_of_month"]=df["date"].dt.day
festival_start=pd.Timestamp("2026-10-22")
df["days_since_festival_start"]=np.where(df["date"]>=festival_start,(df["date"]-festival_start).dt.days,-1)
df["city_display_clean"]=df["city_display"].str.title()
df["demand_multiplier_effective"]=np.where(df["festive_relevant"].eq("Y"),df["demand_multiplier_festive"],df["demand_multiplier_other"])

features=["opening_stock","units_demanded","units_sold","closing_stock","reorder_point","lead_time_days_expected","sales_velocity_7d","days_of_cover","reorder_gap","days_of_cover_ratio","supplier_reliability_clean","base_lead_time_days","lead_time_variance_days","is_recent_reorder","day_of_month","days_since_festival_start","demand_multiplier_effective","sqft","opened_year","baseline_daily_orders","unit_price_inr","shelf_life_days","is_perishable","festive_relevant","popularity_tier","category","store_size","city","event_type"]
train=df[df.date<="2026-10-23"]; test=df[df.date>="2026-10-24"]
cat=[c for c in features if df[c].dtype=="object"]; num=[c for c in features if c not in cat]

def make_pre():
    return ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
                              ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)])

model=Pipeline([("pre",make_pre()),("model",RandomForestClassifier(n_estimators=500,min_samples_leaf=2,class_weight="balanced_subsample",random_state=42,n_jobs=-1))])
model.fit(train[features],train["stockout_risk"])
probs=model.predict_proba(test[features])
classes=list(model.named_steps["model"].classes_)
ii=classes.index("Imminent"); non=[i for i,c in enumerate(classes) if c!="Imminent"]
pred=np.array(["Imminent" if r[ii]>=0.35 else classes[non[np.argmax(r[non])]] for r in probs])
print(classification_report(test["stockout_risk"],pred,digits=4))
print(confusion_matrix(test["stockout_risk"],pred,labels=["Safe","At-Risk","Imminent"]))
