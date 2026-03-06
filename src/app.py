import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Supply Chain ML Dashboard", layout="wide")

@st.cache_resource
def load_model():
    with open("supervised_model_final_boost.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("Supply Chain Prediction Platform")

st.sidebar.header("Input Order Data")

Days_for_shipment_scheduled = st.sidebar.slider("Days for shipment scheduled",0,30,5)
Benefit_per_order = st.sidebar.number_input("Benefit per order",0.0,1000.0,100.0)
Order_Item_Discount = st.sidebar.number_input("Item Discount",0.0,500.0,10.0)
Order_Item_Discount_Rate = st.sidebar.slider("Discount rate",0.0,1.0,0.1)
Order_Item_Profit_Ratio = st.sidebar.slider("Profit ratio",-1.0,1.0,0.2)
Order_Item_Quantity = st.sidebar.slider("Quantity",1,100,3)
Price_Per_Unit = st.sidebar.number_input("Price per unit",0.0,10000.0,50.0)

Type_num = st.sidebar.slider("Order Type",0,5,1)
Category_Name_num = st.sidebar.slider("Category",0,50,10)
Customer_City_num = st.sidebar.slider("Customer City",0,500,20)
Customer_Country_num = st.sidebar.slider("Customer Country",0,50,3)
Customer_Segment_num = st.sidebar.slider("Customer Segment",0,5,2)
Customer_State_num = st.sidebar.slider("Customer State",0,100,10)
Customer_Zipcode_num = st.sidebar.slider("Customer Zip",0,99999,28000)
Department_Name_num = st.sidebar.slider("Department",0,20,5)

Order_City_num = st.sidebar.slider("Order City",0,500,50)
Order_Country_num = st.sidebar.slider("Order Country",0,50,5)
Order_State_num = st.sidebar.slider("Order State",0,100,15)
Order_Status_num = st.sidebar.slider("Order Status",0,10,3)
Shipping_Mode_num = st.sidebar.slider("Shipping Mode",0,10,2)

shipping_day_num = st.sidebar.slider("Shipping day",1,31,10)
shipping_month_num = st.sidebar.slider("Shipping month",1,12,6)

Logistics_Corridor_ID = st.sidebar.slider("Logistics Corridor",0,20,3)

input_dict = {
"Days_for_shipment_scheduled":Days_for_shipment_scheduled,
"Benefit_per_order":Benefit_per_order,
"Order_Item_Discount":Order_Item_Discount,
"Order_Item_Discount_Rate":Order_Item_Discount_Rate,
"Order_Item_Profit_Ratio":Order_Item_Profit_Ratio,
"Order_Item_Quantity":Order_Item_Quantity,
"Type_num":Type_num,
"Category_Name_num":Category_Name_num,
"Customer_City_num":Customer_City_num,
"Customer_Country_num":Customer_Country_num,
"Customer_Segment_num":Customer_Segment_num,
"Customer_State_num":Customer_State_num,
"Department_Name_num":Department_Name_num,
"Order_City_num":Order_City_num,
"Order_Country_num":Order_Country_num,
"Order_State_num":Order_State_num,
"Order_Status_num":Order_Status_num,
"Shipping_Mode_num":Shipping_Mode_num,
"Customer_Zipcode_num":Customer_Zipcode_num,
"shipping_day_num":shipping_day_num,
"shipping_month_num":shipping_month_num,
"Price_Per_Unit":Price_Per_Unit,
"Logistics_Corridor_ID":Logistics_Corridor_ID
}

input_df = pd.DataFrame([input_dict])

st.subheader("Input Data")
st.dataframe(input_df)

if st.button("Run Prediction"):

    prediction = model.predict(input_df)[0]

    if hasattr(model,"predict_proba"):
        prob = model.predict_proba(input_df)[0]
    else:
        prob = None

    col1,col2 = st.columns(2)

    with col1:
        st.metric("Prediction",prediction)

    with col2:
        if prob is not None:
            st.metric("Confidence",round(max(prob)*100,2))

st.divider()

st.header("Feature Analysis")

if hasattr(model,"feature_importances_"):

    features = input_df.columns
    importance = model.feature_importances_

    fi_df = pd.DataFrame({
        "feature":features,
        "importance":importance
    }).sort_values("importance",ascending=False)

    fig = px.bar(fi_df,x="importance",y="feature",orientation="h",title="Feature Importance")

    st.plotly_chart(fig,use_container_width=True)

st.divider()

st.header("Supply Chain Geography")

map_data = pd.DataFrame({
"lat":np.random.uniform(35,60,200),
"lon":np.random.uniform(-10,30,200),
"volume":np.random.randint(10,500,200)
})

layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_data,
    get_position="[lon, lat]",
    get_radius="volume*100",
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=40,
    longitude=5,
    zoom=3
)

r = pdk.Deck(layers=[layer],initial_view_state=view_state,tooltip={"text":"Orders: {volume}"})

st.pydeck_chart(r)

st.divider()

st.header("Operational Analytics")

sim_data = pd.DataFrame({
"month":range(1,13),
"orders":np.random.randint(200,1000,12),
"delays":np.random.randint(10,200,12)
})

fig1 = px.line(sim_data,x="month",y="orders",title="Orders Over Time")
fig2 = px.line(sim_data,x="month",y="delays",title="Delivery Delays")

col1,col2 = st.columns(2)

with col1:
    st.plotly_chart(fig1,use_container_width=True)

with col2:
    st.plotly_chart(fig2,use_container_width=True)

st.divider()

st.header("Business KPIs")

k1,k2,k3,k4 = st.columns(4)

k1.metric("Total Orders",np.random.randint(5000,20000))
k2.metric("Avg Shipping Days",round(np.random.uniform(2,7),2))
k3.metric("Profit Ratio",round(np.random.uniform(0.1,0.4),2))
k4.metric("Late Deliveries",np.random.randint(100,500))

st.divider()

st.header("Scenario Simulator")

qty = st.slider("Simulate Quantity",1,50,5)
price = st.slider("Simulate Price",1,500,50)

tmp = input_df.copy()
tmp["Order_Item_Quantity"] = qty
tmp["Price_Per_Unit"] = price

pred_sim = model.predict(tmp)[0]

st.write("Simulated prediction:",pred_sim)

st.divider()

st.header("Raw Prediction API")

st.code(input_df.to_json(orient="records"))

st.caption("End-to-end Supply Chain Machine Learning dashboard")
