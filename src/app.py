import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import pydeck as pdk
import plotly.graph_objects as go

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Supply Chain ML Dashboard",
    layout="wide",
    page_icon="📦"
)

# -------------------------
# STYLE
# -------------------------
st.markdown("""
<style>
.main { background-color: #CCA465; }
h1 { text-align:center; color:#CCA465; }
.block-container { padding-top:2rem; }
[data-testid="stSidebar"] { background-color:#CCA465; }
[data-testid="stSidebar"] label { color:white; }
.table-style { background-color:white; border-radius:10px; padding:15px; box-shadow:0px 2px 8px rgba(0,0,0,0.08); }
</style>
""", unsafe_allow_html=True)

# -------------------------
# LOAD MODELS
# -------------------------
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model("supervised_model_final_boost.pkl")
model2 = load_model("unsupervised_kmeans_final.pkl")

# -------------------------
# LOAD MAPPINGS
# -------------------------
with open("category_mappings.json") as f:
    mappings = json.load(f)

with open("country_coords.json") as f:
    country_coords = json.load(f)

# -------------------------
# UTILITY FUNCTIONS
# -------------------------
def sidebar_select(label, options, default_idx=0):
    """Generic sidebar selectbox returning selected value and index"""
    selected = st.sidebar.selectbox(label, options, index=default_idx)
    return selected, options.index(selected)

def sidebar_slider(label, min_val, max_val, default_val):
    """Generic sidebar slider"""
    return st.sidebar.slider(label, min_val, max_val, default_val)

def sidebar_number(label, default_val, min_val=None, max_val=None):
    """Generic sidebar number input"""
    if min_val is not None and max_val is not None:
        return st.sidebar.number_input(label, min_val, max_val, default_val)
    else:
        return st.sidebar.number_input(label, value=default_val)

# -------------------------
# TITLE
# -------------------------
st.title("📦 DataCo Supply Chain Risk Predictor")
st.markdown("<center>Predict delivery delays and identify logistic risk clusters in real-time.</center>", unsafe_allow_html=True)

# -------------------------
# SIDEBAR INPUTS
# -------------------------
st.sidebar.header("📥 Input Order Data")

# Step 1: Order Logistics
st.sidebar.subheader("Step 1: Order Logistics")
Customer_City, Customer_City_num = sidebar_select("Destination City", mappings["Customer_City"])
shipping_days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
Shipping_Day, shipping_day_num = sidebar_select("Shipping Day", shipping_days)
Shipping_Mode, Shipping_Mode_num = sidebar_select(
    "Shipping Mode", ["Standard Class","Second Class","First Class","Same Day"]
)

# Step 2: Product & Payment
st.sidebar.subheader("Step 2: Product & Payment")
Category_Name, Category_Name_num = sidebar_select("Category", mappings["Category_Name"])
Type, Type_num = sidebar_select("Payment Type", ["DEBIT","TRANSFER","CASH","PAYMENT","CREDIT"])

# Step 3: Constraints
st.sidebar.subheader("Step 3: Constraints")
Days_for_shipment_scheduled = sidebar_slider("Scheduled Days", 1, 10, 3)
Price_Per_Unit = sidebar_number("Unit Price ($)", 150.0)
Benefit_per_order = sidebar_number("Expected Benefit ($)", 50.0)

# Step 4: Shipping info
st.sidebar.subheader("Step 4: Shipping info")
Department_Name, Department_Name_num = sidebar_select("Department", mappings["Department_Name"])
Order_City, Order_City_num = sidebar_select("Order City", mappings["Order_City"])
Order_Country, Order_Country_num = sidebar_select("Order Country", mappings["Order_Country"])
Order_State, Order_State_num = sidebar_select("Order State", mappings["Order_State"])
Order_Status, Order_Status_num = sidebar_select("Order Status", mappings["Order_Status"])
Logistics_Corridor_ID = sidebar_slider("Logistics Corridor", 0, 20, 3)

predict_button = st.sidebar.button("🚀 Run Prediction")

# -------------------------
# MODEL INPUT DATAFRAME
# -------------------------
input_dict = {
    "Days_for_shipment_scheduled": Days_for_shipment_scheduled,
    "Benefit_per_order": Benefit_per_order,
    "Price_Per_Unit": Price_Per_Unit,
    "Shipping_Mode_num": Shipping_Mode_num,
    "shipping_day_num": shipping_day_num,
    "Type_num": Type_num,
    "Category_Name_num": Category_Name_num,
    "Customer_City_num": Customer_City_num,
    "Department_Name_num": Department_Name_num,
    "Order_City_num": Order_City_num,
    "Order_Country_num": Order_Country_num,
    "Order_State_num": Order_State_num,
    "Order_Status_num": Order_Status_num,
    "Logistics_Corridor_ID": Logistics_Corridor_ID
}

input_df = pd.DataFrame([input_dict])

# -------------------------
# DISPLAY INPUT TABLE
# -------------------------
display_dict = {k.replace("_"," ").title(): v for k,v in input_dict.items()}
display_table = pd.DataFrame(list(display_dict.items()), columns=["Variable","Value"])
st.subheader("📊 Input Data Overview")
st.dataframe(display_table, use_container_width=True, hide_index=True)

# -------------------------
# MAP
# -------------------------
st.subheader("🌍 Shipping Route")
if Customer_City in country_coords and Order_Country in country_coords:
    origin = country_coords[Customer_City]
    destination = country_coords[Order_Country]

    map_data = pd.DataFrame({"lat":[origin[1],destination[1]],"lon":[origin[0],destination[0]]})
    arc = pd.DataFrame({"start_lon":[origin[0]],"start_lat":[origin[1]],"end_lon":[destination[0]],"end_lat":[destination[1]]})

    layer_points = pdk.Layer(
        "ScatterplotLayer", data=map_data, get_position='[lon, lat]', get_radius=250000, get_fill_color='[0,102,255]'
    )

    layer_arc = pdk.Layer(
        "ArcLayer", data=arc, get_source_position='[start_lon,start_lat]', get_target_position='[end_lon,end_lat]',
        get_source_color=[0,150,255], get_target_color=[255,100,0], get_width=6
    )

    view_state = pdk.ViewState(
        latitude=(origin[1]+destination[1])/2, longitude=(origin[0]+destination[0])/2, zoom=1
    )

    st.pydeck_chart(
        pdk.Deck(layers=[layer_arc, layer_points], initial_view_state=view_state,
                 map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json")
    )

# -------------------------
# PREDICTION
# -------------------------
if predict_button:
    prediction = model.predict(input_df)[0]
    label_map = {0:"On Time",1:"Late"}
    pred_label = label_map.get(prediction, prediction)
    st.subheader("📦 Prediction Result")
    if pred_label=="Late":
        st.error(pred_label)
    else:
        st.success(pred_label)

    if hasattr(model,"predict_proba"):
        proba = model.predict_proba(input_df)[0]
        fig = go.Figure(data=[go.Bar(x=["On Time","Late"], y=proba)])
        fig.update_layout(title="Prediction Probabilities", xaxis_title="Result", yaxis_title="Probability")
        st.plotly_chart(fig, use_container_width=True)