import streamlit as st
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import folium
from streamlit_folium import st_folium
from datetime import datetime
from Fortyguard import FortyGuardClient

# Adjust this import if your ClimateAI class lives in a different module.
from climate_ai import ClimateAI

st.set_page_config(
    page_title="Heat Guardian (Chatbot)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -------------------------
# FortyGuard client
# -------------------------

client = FortyGuardClient(
    api_key=st.secrets["API_KEY"]
)


@st.cache_resource
def _get_climate_ai():
    """One ClimateAI instance (and its ThreadPoolExecutor/genai client) shared
    across reruns instead of rebuilt on every message."""
    return ClimateAI()


def _scalar(value):
    """Unwrap a single-element list/tuple (e.g. [25.2]) into its scalar value —
    filter_type=1 responses are documented as scalars but can still come back
    as single-element lists for some parameters."""
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        if len(value) == 1:
            return value[0]
        return ", ".join(str(v) for v in value)
    return value


def _get_anchor_temperature(lat, lon, start_date, start_time):
    """environmental_parameters() requires an ambient-temperature anchor.
    The FortyGuard docs recommend sourcing it from create_heatmap, so we run
    a tiny heatmap (~200m box) around the point and average its tile(s)."""
    delta = 0.002  # roughly a 200m box
    polygon_aoi = {
        "type": "Polygon",
        "coordinates": [[
            [lon - delta, lat - delta],
            [lon + delta, lat - delta],
            [lon + delta, lat + delta],
            [lon - delta, lat + delta],
            [lon - delta, lat - delta],
        ]],
    }
    response = client.create_heatmap(
        polygon_aoi=polygon_aoi,
        start_date=start_date,
        start_time=start_time,
        filter_type=1,
        granularity=100,
        analytic_type="tcm",
        verbose=False,
    )
    features = response.get("result", {}).get("map_data", {}).get("features", [])
    temps = [
        f["properties"].get("average_temperature")
        for f in features
        if f["properties"].get("average_temperature") is not None
    ]
    if not temps:
        return None
    return sum(temps) / len(temps)


def _format_environmental_context(env_result, start_date, start_time):
    """Build both a display-friendly summary and a plain-text system-context
    string (for whenever the real chatbot is wired in) from an
    environmental_parameters() result."""
    location = (env_result.get("locations") or [{}])[0]
    params = location.get("parameters", {})
    solar = location.get("solar_irradiance", {}).get("clear_sky", {})

    summary = {
        "Anchor temperature (°C)": _scalar(location.get("temperature")),
        "Heat index (°C)": _scalar(params.get("heat_index_celsius")),
        "Apparent temp (°C)": _scalar(params.get("apparent_temperature_celsius")),
        "Wet-bulb temp (°C)": _scalar(params.get("wet_bulb_temperature_celsius")),
        "Relative humidity (%)": _scalar(params.get("relative_humidity_percent")),
        "Cloud cover (octas)": _scalar(params.get("cloud_cover_octas")),
        "Precipitation (mm)": _scalar(params.get("precipitation_mm")),
        "Air quality index": _scalar(params.get("air_quality:idx")),
    }

    lines = [f"Environmental conditions at {start_date} {start_time} (local forecast hour):"]
    for label, value in summary.items():
        if value is not None:
            lines.append(f"- {label}: {value}")
    if solar:
        lines.append(
            f"- Clear-sky solar irradiance (W/m²): GHI {_scalar(solar.get('ghi'))}, "
            f"DNI {_scalar(solar.get('dni'))}, DHI {_scalar(solar.get('dhi'))}"
        )

    return summary, "\n".join(lines)


def _build_environmental_data_dict():
    """Flatten the stored environmental_parameters() result (plus lat/lon/time)
    into the flat dict shape ClimateAI.analyze() expects as environmental_data."""
    env_result = st.session_state.get("environmental_context")
    if not env_result:
        return None

    location = (env_result.get("locations") or [{}])[0]
    params = location.get("parameters", {})

    data = {
        "latitude": st.session_state.get("latitude"),
        "longitude": st.session_state.get("longitude"),
        "temperature_c": _scalar(location.get("temperature")),
        "heat_index_c": _scalar(params.get("heat_index_celsius")),
        "apparent_temperature_c": _scalar(params.get("apparent_temperature_celsius")),
        "wet_bulb_temperature_c": _scalar(params.get("wet_bulb_temperature_celsius")),
        "humidity_percent": _scalar(params.get("relative_humidity_percent")),
        "cloud_cover_octas": _scalar(params.get("cloud_cover_octas")),
        "precipitation_mm": _scalar(params.get("precipitation_mm")),
        "aqi": _scalar(params.get("air_quality:idx")),
        "timestamp": st.session_state.get("environmental_timestamp"),
    }
    return {k: v for k, v in data.items() if v is not None}


def _render_assistant_extras(result_data):
    """Render the structured fields of a ClimateAnalysisResult (stored as a
    dict on the message) below the answer text inside a chat bubble."""
    badge_bits = []
    if result_data.get("domain"):
        badge_bits.append(f"**Domain:** {result_data['domain']}")
    if result_data.get("topic"):
        badge_bits.append(f"**Topic:** {result_data['topic']}")
    if result_data.get("confidence"):
        badge_bits.append(f"**Confidence:** {result_data['confidence']}")
    if result_data.get("risk_level"):
        badge_bits.append(f"**Risk:** {result_data['risk_level']}")
    if badge_bits:
        st.caption(" · ".join(badge_bits))

    if result_data.get("is_forecast"):
        st.info("This is a forecast/projection, not a guaranteed reading.")

    if result_data.get("data_missing"):
        st.warning("The model flagged that some data needed to fully answer this was missing.")

    if result_data.get("explanation"):
        st.markdown(result_data["explanation"])

    key_metrics = result_data.get("key_metrics") or []
    if key_metrics:
        metric_cols = st.columns(min(4, len(key_metrics)))
        for i, m in enumerate(key_metrics):
            unit = f" {m['unit']}" if m.get("unit") else ""
            metric_cols[i % len(metric_cols)].metric(m["name"], f"{m['value']}{unit}")

    recommended_actions = result_data.get("recommended_actions") or []
    if recommended_actions:
        st.markdown("**Recommended actions:**")
        for action in recommended_actions:
            st.markdown(f"- {action}")

    chart_series = result_data.get("chart_series") or []
    if chart_series:
        chart_df = None
        for series in chart_series:
            s = pd.Series(
                {p["label"]: p["value"] for p in series["points"]},
                name=series["name"],
            )
            chart_df = s.to_frame() if chart_df is None else chart_df.join(s, how="outer")
        if chart_df is not None and not chart_df.empty:
            if result_data.get("chart_label"):
                st.caption(result_data["chart_label"])
            st.line_chart(chart_df)

    if result_data.get("web_search_used"):
        sources = result_data.get("external_sources") or []
        footer = "🔎 Answer informed by a live web search"
        if sources:
            footer += f" — sources: {', '.join(sources)}"
        st.caption(footer)


# -------------------------
# USA
# -------------------------
countries = gpd.read_file("ne_10m_admin_0_countries.shp")
USA = countries[countries["ADMIN"] == "United States of America"]

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
    /* 1. Sidebar Container (Width Adjustment) */
    [data-testid="stSidebar"] {
        min-width: 350px !important;
        max-width: 350px !important;
    }
    
    /* 2. Hide Sidebar Collapse Button */
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    /* 3. Increase Font Size in Sidebar Elements */
    [data-testid="stSidebar"] * {
        font-size: 1.2rem !important;
    }
    
    [data-testid="stSidebarNav"] span {
        font-size: 1.35rem !important;
        font-weight: 600 !important;
    }

    /* Main background */
        .stApp {
    background: url('https://images.unsplash.com/photo-1513002749550-c59d786b8e6c?q=80&w=1920&auto=format&fit=crop') no-repeat center center fixed !important;
    background-size: cover !important;
}

    
    .block-container {
        padding-top: 3rem;
        max-width: 1200px;
    }

    /* Second title (matches the Forecast page) */
    .second-title {
        font-size: 3rem;
        font-weight: 900;
        margin: 25px 0 15px 0;
        color: #121212;
    }

    /* Card (matches the Forecast page) */
    .card {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        color: #e2e8f0;
    }
    
    /* Header */
    .chat-header {
        text-align: center;
        padding-bottom: 20px;
    }
    .chat-header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .chat-header p {
        font-size: 1.1rem;
        color: #121212;
    }
    
    /* Chat container styling applied directly to Streamlit Container */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(30,21,59,0.75) !important;
        border: 1px solid rgba(148,163,184,0.15) !important;
        border-radius: 18px !important;
        padding: 20px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15) !important;
    }
    
    /* Welcome message */
    .welcome {
        text-align: center;
        padding: 60px 20px;
    }
    .welcome-icon {
        font-size: 4rem;
    }
    .welcome h2 {
        color: #e2e8f0;
    }
    .welcome p {
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)  

# -------------------------
# Header
# -------------------------
st.markdown("""
<div class="chat-header">
    <div style="font-size: 3rem;">🤖</div>
    <h1>Heat Guardian Assistant</h1>
    <p>Ask questions about heat, temperature, safety and environmental conditions</p>
</div>    
""", unsafe_allow_html=True)

# -------------------------
# Session state
# -------------------------
if "chat_selected_point" not in st.session_state:
    st.session_state.chat_selected_point = None  # {"lat", "lon"}

if "environmental_context" not in st.session_state:
    st.session_state.environmental_context = None  # raw API result

if "environmental_summary" not in st.session_state:
    st.session_state.environmental_summary = None  # dict for display

if "chat_system_context" not in st.session_state:
    st.session_state.chat_system_context = None  # plain-text, ready for the chatbot

if "environmental_timestamp" not in st.session_state:
    st.session_state.environmental_timestamp = None  # ISO timestamp the env data was fetched for

# -------------------------
# Location Input (pick a point on the map)
# -------------------------
st.markdown(
    '<div class="second-title">Select Location</div>',
    unsafe_allow_html=True
)
st.markdown(
    """
    <div class="card">
        Click a point on the map below to select the location you want to ask about.
    </div>
    """,
    unsafe_allow_html=True
)

point_map = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles="OpenStreetMap")

if st.session_state.chat_selected_point is not None:
    folium.Marker(
        [st.session_state.chat_selected_point["lat"], st.session_state.chat_selected_point["lon"]],
        tooltip="Selected location",
        icon=folium.Icon(color="orange", icon="fire", prefix="fa"),
    ).add_to(point_map)

point_map_data = st_folium(
    point_map,
    width=1200,
    height=500,
    key="chat_location_map",
    returned_objects=["last_clicked"],
)

if point_map_data and point_map_data.get("last_clicked"):
    clicked = point_map_data["last_clicked"]
    new_point = {"lat": clicked["lat"], "lon": clicked["lng"]}
    if new_point != st.session_state.chat_selected_point:
        st.session_state.chat_selected_point = new_point
        # A new point invalidates any previously fetched environmental data.
        st.session_state.environmental_context = None
        st.session_state.environmental_summary = None
        st.session_state.chat_system_context = None
        st.session_state.environmental_timestamp = None
        st.rerun()

if st.session_state.chat_selected_point is not None:
    sel = st.session_state.chat_selected_point
    st.caption(f"Selected point: {sel['lat']:.5f}, {sel['lon']:.5f}")

# -------------------------
# Confirm Location
# -------------------------
confirm_location = st.button(
    "Confirm Location",
    use_container_width=True
)

# -------------------------
# USA Validation + Environmental Lookup
# -------------------------
if confirm_location:
    if st.session_state.chat_selected_point is None:
        st.warning("Please click a point on the map first.")
    else:
        sel = st.session_state.chat_selected_point
        location = Point(sel["lon"], sel["lat"])
        is_inside_usa = USA.geometry.covers(location).any()
        if is_inside_usa:
            st.success("Location is inside the USA")
            st.session_state.latitude = sel["lat"]
            st.session_state.longitude = sel["lon"]

            # Use "now" for the API call, rounded to the current hour.
            now = datetime.now()
            start_date = now.strftime("%Y-%m-%d")
            start_time = now.strftime("%H:00")

            with st.spinner("Fetching current environmental conditions..."):
                try:
                    anchor_temp = _get_anchor_temperature(sel["lat"], sel["lon"], start_date, start_time)
                    if anchor_temp is None:
                        st.session_state.environmental_context = None
                        st.session_state.environmental_summary = None
                        st.session_state.chat_system_context = None
                        st.warning("Could not determine a temperature anchor for this point.")
                    else:
                        env_response = client.environmental_parameters(
                            latitude=sel["lat"],
                            longitude=sel["lon"],
                            temperature=anchor_temp,
                            start_date=start_date,
                            start_time=start_time,
                            filter_type=1,
                        )
                        env_result = env_response.get("result", {})
                        summary, system_context = _format_environmental_context(
                            env_result, start_date, start_time
                        )
                        st.session_state.environmental_context = env_result
                        st.session_state.environmental_summary = summary
                        st.session_state.chat_system_context = system_context
                        st.session_state.environmental_timestamp = f"{start_date}T{start_time}:00"
                except Exception as e:
                    st.session_state.environmental_context = None
                    st.session_state.environmental_summary = None
                    st.session_state.chat_system_context = None
                    st.session_state.environmental_timestamp = None
                    st.error(f"Environmental data request failed: {e}")
        else:
            st.error("Please select a location inside the USA")
            st.session_state.latitude = None
            st.session_state.longitude = None
            st.session_state.environmental_context = None
            st.session_state.environmental_summary = None
            st.session_state.chat_system_context = None
            st.session_state.environmental_timestamp = None

# -------------------------
# Current Conditions (what the chatbot will be given)
# -------------------------
if st.session_state.environmental_summary is not None:
    st.markdown('<div class="second-title">Current Conditions</div>', unsafe_allow_html=True)
    summary = st.session_state.environmental_summary
    cols = st.columns(4)
    items = [(k, v) for k, v in summary.items() if v is not None]
    for i, (label, value) in enumerate(items):
        cols[i % 4].metric(label, value)

    with st.expander("Raw context passed to the chatbot"):
        st.text(st.session_state.chat_system_context)

# -------------------------
# Chat History
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

latitude = st.session_state.get("latitude")
longitude = st.session_state.get("longitude")
forecast = st.session_state.get("forecast_data")

# -------------------------
# Chat Container (Bordered Native Container)
# -------------------------
with st.container(border=True):
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="welcome">
            <div class="welcome-icon">🌡️</div>
            <h2>How Can I Help You?</h2>
            <p>Ask me anything about heat and environmental conditions.</p>
        </div>        
        """, unsafe_allow_html=True)
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("result"):
                    _render_assistant_extras(message["result"])

# -------------------------
# Chat Input
# -------------------------
user_question = st.chat_input("Ask Heat Guardian something...") 

# -------------------------
# Handle Question
# -------------------------
if user_question:
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    env_data = _build_environmental_data_dict()
    if env_data is None:
        assistant_message = {
            "role": "assistant",
            "content": (
                "I don't have environmental data yet — pick a point on the map "
                "above and click **Confirm Location** first, then ask again."
            ),
        }
    else:
        try:
            climate_ai = _get_climate_ai()
            with st.spinner("Analyzing..."):
                result = climate_ai.analyze(user_question, env_data)
            assistant_message = {
                "role": "assistant",
                "content": result.answer,
                "result": result.model_dump(),
            }
        except Exception as e:
            assistant_message = {
                "role": "assistant",
                "content": f"Sorry, something went wrong generating a response. ({e})",
            }

    st.session_state.messages.append(assistant_message)
    st.rerun()