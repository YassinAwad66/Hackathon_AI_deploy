import streamlit as st
import pandas as pd
import altair as alt
import geopandas as gpd
from shapely.geometry import Polygon
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import json
import base64
import io
from PIL import Image
from Fortyguard import FortyGuardClient


# -------------------------
# FortyGuard client
# -------------------------

client = FortyGuardClient(
    api_key=st.secrets["API_KEY"]
)


def _scalar(value):
    """Unwrap a single-element list/tuple (e.g. [26.1]) into its scalar value
    so it can be passed to st.metric, which only accepts numbers/strings."""
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        if len(value) == 1:
            return value[0]
        return ", ".join(str(v) for v in value)
    return value


def _decode_image(b64_string):
    """Decode a base64 image string (optionally with a data: prefix) into a PIL Image."""
    if not b64_string:
        return None
    if b64_string.startswith("data:"):
        b64_string = b64_string.split(",", 1)[1]
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64_string)))
    except Exception:
        return None


# -------------------------
# USA Coordinates
# -------------------------
countries = gpd.read_file("ne_10m_admin_0_countries.shp")
USA = countries[countries["ADMIN"] == "United States of America"]

# -------------------------
# Page Configration
# -------------------------
st.set_page_config(
    page_title=" Forecasting heat(FortyGaurd)",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Custom CSS   # access to use HTML & CSS
# -------------------------
st.markdown("""
<style>
    /* 1. Standardized Sidebar Width */
    [data-testid="stSidebar"] {
        min-width: 350px !important;
        max-width: 350px !important;
    }

    /* Hide Sidebar Collapse Button */
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    /* 2. Standardized Sidebar Font Sizing */
    [data-testid="stSidebar"] * {
        font-size: 1.2rem !important;
    }

    [data-testid="stSidebarNav"] span {
        font-size: 1.35rem !important;
        font-weight: 600 !important;
    }
    /*Main background*/
    .stApp {
    background: url('https://images.unsplash.com/photo-1513002749550-c59d786b8e6c?q=80&w=1920&auto=format&fit=crop') no-repeat center center fixed !important;
    background-size: cover !important;
}

    /* Remove default top padding */
    .block-container{
    padding-top: 2rem;
    padding-top: 3rem;
    max-width: 1200px;
    }

    /* Header */
    .header {
    text-align: center;
    padding: 40px 0 35px 0;
    }

    .header h1{
    font-size: 3.2rem;
    font-weight: 800;
    margin-bottom: 5px;
    letter-spacing: -1px;
    }

    .header p{
    font-size: 1.5rem;
    color: #121212;
    margin-top: 0;
    }

    /* Second Title */
    .second-title {
    font-size: 3rem;
    font-weight: 900;
    margin: 25px 0 15px 0;
    color: #121212;
    }

    /* Cards */
    .card{
    background: rgba(30, 41, 59, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 18px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }

    /* Prediction button */
    .stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    font-size: 2rem;
    font-weight: 900;
    border: none;
    background: linear-gradient(
    90deg,
    #f97316

    );
    color: white;
    transition: 0.2s;
    }
    .stButton > button:hover{
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(249,115,2,0.25);
    }

    /*Input labels*/
    label{
    font-weight: 900!important;
    }

    /*Forecast table*/
    .forecast-card{
    background: rgba(30,41,59,0.75);
    border-radius: 18px;
    padding: 15px;
    border: 1px solid rgba(148,163,184,0.15);
    }

    /*forecast rows*/
    .forecast-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 10px;
    border-bottom: 1px solid rgba(148,163,184,0.1);
    transition: 0.2s
    }
    .forecast-row:last-child {
    border-bottom: none;
    }
    .forecast-row:hover{
    background: rgba(148,163,184,0.1);
    border-radius: 10px;
    }
    .forecast-time {
    color: #cbd5e1;
    font-size: 2rem;
    font-weight: 700;
    }
    .forecast-temp {
    color: #fb923c;
    font-size: 2rem;
    font-weight: 700;
    }

    /*Status badge*/
    .status{
    display: inline-block;
    padding: 6px 12px;
    background: rgba(34,197,184,0.12);
    color: #4ade80;
    font-size: 1rem;
    font-weight: 600;
    border-radius: 20px;
    }



    </style>
    """, unsafe_allow_html=True)


# -------------------------
# Header
# -------------------------
st.markdown("""
<div class="header">
    <div style="font-size: 3rem;">🌡️</div>
    <h1>HeatCast</h1>
    <p>
        AI-powered temperature forecasting for your location
    </p>

<div class="status">● Forecast system ready</div>
</div>
""", unsafe_allow_html=True)


if "forecast_data" not in st.session_state:
    st.session_state.forecast_data = None

if "forecast_latitude" not in st.session_state:
    st.session_state.forecast_latitude = None

if "forecast_longitude" not in st.session_state:
    st.session_state.forecast_longitude = None

if "heatmap_data" not in st.session_state:
    st.session_state.heatmap_data = None

if "heat_polygon" not in st.session_state:
    st.session_state.heatmap_polygon = None

if "heatmap_hours" not in st.session_state:
    st.session_state.heatmap_hours = None

if "heatmap_granularity" not in st.session_state:
    st.session_state.heatmap_granularity = None

if "heatmap_filter_type" not in st.session_state:
    st.session_state.heatmap_filter_type = None

if "heatmap_start_date" not in st.session_state:
    st.session_state.heatmap_start_date = None

if "heatmap_start_time" not in st.session_state:
    st.session_state.heatmap_start_time = None

# --- Point-lookup state (environmental / satellite / street view) ---
if "selected_point" not in st.session_state:
    st.session_state.selected_point = None  # {"lat","lon","tile_id","avg_temp"}

if "env_params_result" not in st.session_state:
    st.session_state.env_params_result = None

if "satellite_result" not in st.session_state:
    st.session_state.satellite_result = None

if "streetview_result" not in st.session_state:
    st.session_state.streetview_result = None

# -------------------------
# Select Polygon
# -------------------------
st.markdown(
    '<div class="second-title"> Select Area</div',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="card">
        <h3>Draw your area</h3>
        <p style="color:"#cbd5e1;">
            Draw a polygon on the map to select the area you want to analyze.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
## create map
draw_map = folium.Map(
    location=[40.7128, -74.0060],
    zoom_start=11,
    tiles="OpenStreetMap"
)
## Drawing
Draw(
    export=True,
    draw_options={
        "polyline": False,
        "rectangle": False,
        "circle": False,
        "marker": False,
        "circlemarker": False,
        "polygon": True
    },
    edit_options={
        "edit": True,
        "remove": True
    }
).add_to(draw_map)
## Display map
draw_map_data = st_folium(
    draw_map,
    width=1200,
    height=600,
    key="polygon_map",
    returned_objects=["last_active_drawing"]
)
if draw_map_data and draw_map_data["last_active_drawing"] is not None:
    selected_polygon = draw_map_data["last_active_drawing"]
    if selected_polygon["geometry"]["type"] == "Polygon":
        st.session_state.heatmap_polygon = (
            selected_polygon["geometry"]["coordinates"]
        )


# -------------------------
# Get Saved Polygon
# -------------------------
predict = False

if st.session_state.heatmap_polygon is not None:

    coordinates = st.session_state.heatmap_polygon

    st.markdown(
        """
        <div style="
            background: rgba(15, 23, 42, 0.9);
            color: #f8fafc;
            padding: 12px 18px;
            border-radius: 10px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            font-weight: 700;
            text-align: center;
        ">
            ✓ Polygon Selected Successfully
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("heatmap_form"):
        st.markdown(
            '<div class="second-title">Forecast time</div>'
            , unsafe_allow_html=True
        )
        st.markdown(
            '<p style="color:#0f172a; font-size:1.2rem; font-weight:800;">'
            'How many hours from now you want predict temperature?'
            '</p>',
            unsafe_allow_html=True
        )
        hours = st.slider(
            "Forecast hours",
            min_value=1,
            max_value=12,
            value=1,
            label_visibility="collapsed"
        )
        ## Granularity
        st.markdown('<p style="color:#0f172a; font-size:1.2rem; font-weight:800;">'
                     '' 'Enter Heatmap Granularity (meters)' '</p>'
                     , unsafe_allow_html=True)
        granularity = st.selectbox(
            "Granularity",
            options=[60, 80, 100],
            index=2,
            label_visibility="collapsed"
        )
        ## filter type
        if hours == 1:
            filter_type = 1
        else:
            filter_type = 2

        predict = st.form_submit_button(
            "🌡️ Generate Heatmap",
            use_container_width=True
        )

# -------------------------
# Heatmap
# -------------------------
if predict:
    ## prepare polygon for API
    polygon_aoi = {
        "type": "Polygon",
        "coordinates": coordinates
    }
    ## Current Date & Time
    now = datetime.now()

    next_hour = now + timedelta(hours=1)

    start_date = next_hour.strftime("%Y-%m-%d")
    start_time = next_hour.strftime("%H:00")

    end_datetime = next_hour + timedelta(hours=hours)
    if hours == 1:
        filter_type = 1
        heatmap_response = client.create_heatmap(
            polygon_aoi=polygon_aoi,
            start_date=start_date,
            start_time=start_time,
            filter_type=filter_type,
            granularity=granularity,
            analytic_type="tcm",
            verbose=False
        )
    else:
        if end_datetime.date() == next_hour.date():
            filter_type = 2
            end_time = end_datetime.strftime("%H:00")

            heatmap_response = client.create_heatmap(
                polygon_aoi=polygon_aoi,
                start_date=start_date,
                start_time=start_time,
                end_time=end_time,
                filter_type=filter_type,
                granularity=granularity,
                analytic_type="tcm",
                verbose=False
            )
        else:
            filter_type = 4
            end_date = end_datetime.strftime("%Y-%m-%d")
            heatmap_response = client.create_heatmap(
                polygon_aoi=polygon_aoi,
                start_date=start_date,
                end_date=end_date,
                filter_type=filter_type,
                granularity=granularity,
                analytic_type="tcm",
                verbose=False
            )

    st.session_state.heatmap_data = heatmap_response
    st.session_state.heatmap_polygon = coordinates
    st.session_state.heatmap_hours = hours
    st.session_state.heatmap_granularity = granularity
    st.session_state.heatmap_filter_type = filter_type
    # Reused by the point-lookup APIs below (environmental / satellite need a
    # start_date + start_time; we anchor them on the same forecast start hour).
    st.session_state.heatmap_start_date = start_date
    st.session_state.heatmap_start_time = start_time

    # A new heatmap invalidates any previously selected tile / cached lookups.
    st.session_state.selected_point = None
    st.session_state.env_params_result = None
    st.session_state.satellite_result = None
    st.session_state.streetview_result = None

if (
    st.session_state.heatmap_data is not None
    and st.session_state.heatmap_polygon is not None
):
    result = st.session_state.heatmap_data.get("result", {})
    heatmap_geojson = result.get("map_data")
    if heatmap_geojson and heatmap_geojson.get("features"):
        st.markdown(
            '<div class="second-title">Temperature Heatmap</div>',
            unsafe_allow_html=True
        )
        st.info(
            "📍 Click a tile on the map below to see environmental, "
            "satellite, and street-view details for that point."
        )

        first_point = st.session_state.heatmap_polygon[0][0]
        heatmap_map = folium.Map(
            location=[first_point[1], first_point[0]],
            zoom_start=12,
            tiles="OpenStreetMap"
        )
        temperatures = []
        for feature in heatmap_geojson["features"]:
            temp = feature["properties"].get("average_temperature")
            if temp is not None:
                temperatures.append(temp)
        if temperatures:
            min_temp = min(temperatures)
            max_temp = max(temperatures)

            def style_heatmap(feature):
                temp = feature["properties"].get(
                    "average_temperature",
                    min_temp
                )
                if max_temp == min_temp:
                    fraction = 0
                else:
                    fraction = (
                        temp - min_temp
                    ) / (max_temp - min_temp)

                red = int(255 * fraction)
                blue = int(255 * (1 - fraction))
                return {
                    "fillColor": f"#{red:02x}00{blue:02x}",
                    "color": "#00000000",
                    "weight": 0,
                    "fillOpacity": 0.7
                }
            folium.GeoJson(
                heatmap_geojson,
                style_function=style_heatmap,
                tooltip=folium.GeoJsonTooltip(
                    fields=["tile_id",
                            "average_temperature",
                            "min_temperature",
                            "max_temperature"],
                    aliases=[
                        "Tile",
                        "Average Temperature",
                        "Minimum Temperature",
                        "Maximum Temperature"],
                    localize=True)).add_to(heatmap_map)
            click_data = st_folium(
                heatmap_map,
                width=1200,
                height=600,
                key="temperature_heatmap",
                returned_objects=["last_object_clicked"]
            )

            # ---- Resolve a click into the nearest tile (lat/lon + avg temp) ----
            if click_data and click_data.get("last_object_clicked"):
                click_lat = click_data["last_object_clicked"]["lat"]
                click_lon = click_data["last_object_clicked"]["lng"]

                best_tile = None
                best_dist = None
                for feature in heatmap_geojson["features"]:
                    geom = feature.get("geometry", {})
                    if geom.get("type") != "Polygon":
                        continue
                    ring = geom.get("coordinates", [[]])[0]
                    if not ring:
                        continue
                    c_lon = sum(pt[0] for pt in ring) / len(ring)
                    c_lat = sum(pt[1] for pt in ring) / len(ring)
                    dist = (c_lat - click_lat) ** 2 + (c_lon - click_lon) ** 2
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_tile = {
                            "lat": c_lat,
                            "lon": c_lon,
                            "tile_id": feature["properties"].get("tile_id"),
                            "avg_temp": feature["properties"].get("average_temperature"),
                        }

                if best_tile is not None and best_tile != st.session_state.selected_point:
                    st.session_state.selected_point = best_tile
                    # New tile -> clear stale results from the previous one.
                    st.session_state.env_params_result = None
                    st.session_state.satellite_result = None
                    st.session_state.streetview_result = None
        else:
            st.error("No heatmap data was returned from FortyGuard.")

# -------------------------
# Location Insights (environmental parameters / satellite / street view)
# -------------------------
if st.session_state.selected_point is not None:
    sel = st.session_state.selected_point

    st.markdown('<div class="second-title">📍 Location Insights</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="card">
            Selected tile <b>{sel.get('tile_id')}</b> — lat {sel['lat']:.5f}, lon {sel['lon']:.5f}
            (avg. temperature: {sel.get('avg_temp')}°C)
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_env, tab_sat, tab_street = st.tabs(
        ["🌡️ Environmental Parameters", "🛰️ Satellite Segmentation", "🚶 Street View Segmentation"]
    )

    # ---- Environmental Parameters ----
    with tab_env:
        if st.button("Fetch environmental parameters", key="fetch_env"):
            with st.spinner("Fetching environmental parameters..."):
                try:
                    env_response = client.environmental_parameters(
                        latitude=sel["lat"],
                        longitude=sel["lon"],
                        temperature=sel["avg_temp"] if sel.get("avg_temp") is not None else 20.0,
                        start_date=st.session_state.heatmap_start_date,
                        start_time=st.session_state.heatmap_start_time,
                        filter_type=1,
                    )
                    st.session_state.env_params_result = env_response.get("result", {})
                except Exception as e:
                    st.error(f"Environmental parameters request failed: {e}")

        if st.session_state.env_params_result:
            env_result = st.session_state.env_params_result
            location = (env_result.get("locations") or [{}])[0]
            params = location.get("parameters", {})

            col1, col2, col3 = st.columns(3)
            col1.metric("Heat index (°C)", _scalar(params.get("heat_index_celsius")))
            col2.metric("Wet-bulb temp (°C)", _scalar(params.get("wet_bulb_temperature_celsius")))
            col3.metric("Relative humidity (%)", _scalar(params.get("relative_humidity_percent")))

            col4, col5, col6 = st.columns(3)
            col4.metric("Apparent temp (°C)", _scalar(params.get("apparent_temperature_celsius")))
            col5.metric("Cloud cover (octas)", _scalar(params.get("cloud_cover_octas")))
            col6.metric("Precipitation (mm)", _scalar(params.get("precipitation_mm")))

            solar = location.get("solar_irradiance", {}).get("clear_sky", {})
            if solar:
                st.markdown("**Clear-sky solar irradiance (W/m²)**")
                sol1, sol2, sol3 = st.columns(3)
                sol1.metric("GHI", _scalar(solar.get("ghi")))
                sol2.metric("DNI", _scalar(solar.get("dni")))
                sol3.metric("DHI", _scalar(solar.get("dhi")))

            aqi_keys = [
                k for k in params
                if "air_quality" in k or k in ("aqi_us_co", "co2_ppm", "methane_ppb")
            ]
            if any(params.get(k) is not None for k in aqi_keys):
                st.markdown("**Air quality**")
                st.json({k: params.get(k) for k in aqi_keys})

    # ---- Satellite Segmentation ----
    with tab_sat:
        granularity_sat = st.session_state.heatmap_granularity or 80
        if st.button("Fetch satellite segmentation", key="fetch_sat"):
            with st.spinner("Fetching satellite segmentation..."):
                try:
                    sat_response = client.satellite_segmentation(
                        latitude=sel["lat"],
                        longitude=sel["lon"],
                        start_date=st.session_state.heatmap_start_date,
                        start_time=st.session_state.heatmap_start_time,
                        filter_type=1,
                        granularity=granularity_sat,
                    )
                    st.session_state.satellite_result = sat_response.get("result", {})
                except Exception as e:
                    st.error(f"Satellite segmentation request failed: {e}")

        if st.session_state.satellite_result:
            sat_result = st.session_state.satellite_result
            st.caption(f"Imagery year: {sat_result.get('image_year', 'n/a')}")

            originals = sat_result.get("orignal_image") or sat_result.get("original_image") or []
            if isinstance(originals, str):
                originals = [originals]
            seg_b64 = sat_result.get("segmentation", {}).get("image_content")

            img_col1, img_col2 = st.columns(2)
            orig_img = _decode_image(originals[0]) if originals else None
            mask_img = _decode_image(seg_b64)
            if orig_img is not None:
                img_col1.image(orig_img, caption="Original satellite tile", use_container_width=True)
            if mask_img is not None:
                img_col2.image(mask_img, caption="Segmentation mask", use_container_width=True)

            segments = sat_result.get("segmentation", {}).get("segments", {})
            if segments:
                st.markdown("**Land-cover breakdown**")
                seg_df = pd.DataFrame(
                    sorted(segments.items(), key=lambda kv: kv[1], reverse=True),
                    columns=["class", "coverage_pct"],
                )
                st.dataframe(seg_df, hide_index=True, use_container_width=True)

    # ---- Street View Segmentation ----
    with tab_street:
        c1, c2, c3 = st.columns(3)
        vertical_angle = c1.slider("Vertical angle (°)", -30.0, 30.0, 10.0, 1.0)
        horizontal_angle = c2.slider("Horizontal angle (°)", 0.0, 270.0, 90.0, 15.0)
        back_view = c3.checkbox("Include back view", value=False)

        if st.button("Fetch street view segmentation", key="fetch_street"):
            with st.spinner("Fetching street view segmentation..."):
                try:
                    street_response = client.street_view_segmentation(
                        latitude=sel["lat"],
                        longitude=sel["lon"],
                        vertical_angle=vertical_angle,
                        horizontal_angle=horizontal_angle,
                        back_view=back_view,
                    )
                    st.session_state.streetview_result = street_response.get("result", {})
                except Exception as e:
                    st.error(f"Street view segmentation request failed: {e}")

        if st.session_state.streetview_result:
            street_result = st.session_state.streetview_result
            front = street_result.get("front", {})

            img_col1, img_col2 = st.columns(2)
            street_img = _decode_image(front.get("original_image"))
            street_mask = _decode_image(front.get("segmented_image"))
            if street_img is not None:
                img_col1.image(street_img, caption="Street view", use_container_width=True)
            if street_mask is not None:
                img_col2.image(street_mask, caption="Segmentation", use_container_width=True)

            segments = front.get("segments", {})
            if segments:
                st.markdown("**Class coverage**")
                seg_df = pd.DataFrame(
                    sorted(segments.items(), key=lambda kv: kv[1], reverse=True),
                    columns=["class", "coverage_pct"],
                )
                st.dataframe(seg_df, hide_index=True, use_container_width=True)

            back = street_result.get("back")
            if back_view and back:
                st.markdown("---")
                st.markdown("**Back view**")
                img_col3, img_col4 = st.columns(2)
                back_img = _decode_image(back.get("original_image"))
                back_mask = _decode_image(back.get("segmented_image"))
                if back_img is not None:
                    img_col3.image(back_img, caption="Street view (back)", use_container_width=True)
                if back_mask is not None:
                    img_col4.image(back_mask, caption="Segmentation (back)", use_container_width=True)