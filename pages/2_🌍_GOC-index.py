import os
import pandas as pd
import pydeck as pdk
import geopandas as gpd
import streamlit as st
import leafmap.colormaps as cm
from leafmap.common import hex_to_rgb
import sys

st.set_page_config(layout="wide")

st.sidebar.title("About")
st.sidebar.info(
    """
    Web App URL: <https://geospatial.streamlitapp.com>
    GitHub repository: <https://github.com/giswqs/streamlit-geospatial>
    """
)

st.sidebar.title("Contact")
st.sidebar.info(
    """
    Maarten Vonk: <https://www.HCSS.nl>
    [GitHub](https://github.com/MaartenCVonk) 
    """
)

link_prefix = "https://storage.googleapis.com/location-grid-gis-layers/"
directory = os.path.join(os.getcwd(), "data/")
data_links = {
    "monthly_current": {
        "national": directory + "MEI_FIN_19012023181514641.csv",
        "subnational": directory + "MEI_FIN_19012023181514641.csv",
}
}

#@st.cache
def get_inventory_data(url):
    df = pd.read_csv(url)
    df['month_date_yyyymm'] = df['TIME'].replace('-', '', regex=True).astype(int)
    df = df[['TIME', 'Country', 'Value', 'month_date_yyyymm']]
    return df


def filter_weekly_inventory(df, week):
    df = df[df["week_end_date"] == week]
    return df


def get_start_end_year(df):
    start_year = int(str(df["month_date_yyyymm"].min())[:4])
    end_year = int(str(df["month_date_yyyymm"].max())[:4])
    return start_year, end_year


def get_periods(df):
    return [str(d) for d in list(set(df["month_date_yyyymm"].tolist()))]


#@st.cache
def get_geom_data(category):
    prefix = ("https://storage.googleapis.com/location-grid-gis-layers/")
    links = {
        "national":  directory+ "world-administrative-boundaries.geojson",
        "subnational": prefix + "fra_admin1.geojson",
    }
    gdf = gpd.read_file(links[category])
    if category == 'national':
        gdf = gdf[['name','geometry']]
        gdf.rename(columns={'name': 'Name'}, inplace=True)
    if category == 'subnational':
        gdf = gdf[['admin1_name','geometry']]
        gdf.rename(columns={'admin1_name': 'Name'}, inplace=True)
    return gdf


def join_attributes(gdf, df, category):

    new_gdf = None
    if category == "national":
        new_gdf = gdf
        new_gdf['TIME'] = 202112
        new_gdf['Value'] = 1
    elif category == "subnational":
        new_gdf = gdf
        new_gdf['TIME'] = 202112
        new_gdf['Value'] = 1
    return new_gdf


def select_non_null(gdf, col_name):
    new_gdf = gdf[~gdf[col_name].isna()]
    return new_gdf


def select_null(gdf, col_name):
    new_gdf = gdf[gdf[col_name].isna()]
    return new_gdf

def get_data_dict(name):
    in_csv = os.path.join(os.getcwd(), "data/goc-data.csv")
    df = pd.read_csv(in_csv)
    label = list(df[df["Name"] == name]["Label"])[0]
    desc = list(df[df["Name"] == name]["Description"])[0]
    return label, desc


def app():

    st.title("Organised Crime Index")
    st.markdown(
        """**Introduction:** Organised crime index.
    """
    )

    row1_col1, row1_col2, row1_col3, row1_col4, row1_col5 = st.columns(
        [1, 1, 0.6, 1.4, 2]
    )
    with row1_col1:
        frequency = st.selectbox("Monthly/weekly data", ["Monthly", "Weekly"])
    with row1_col2:
        types = ["Current month data", "Historical data"]
        if frequency == "Weekly":
            types.remove("Current month data")
        cur_hist = st.selectbox(
            "Current/historical data",
            types,
        )
    with row1_col3:
        if frequency == "Monthly":
            scale = st.selectbox(
                "Scale", ["National", "Subnational"], index=0
            )
        else:
            scale = st.selectbox("Scale", ["National", 'Subnational'], index=0)

    gdf = get_geom_data(scale.lower())

    if frequency == "Monthly":
        if cur_hist == "Current month data":
            inventory_df = get_inventory_data(data_links["monthly_current"][scale.lower()])
            selected_period = get_periods(inventory_df)[0]
        else:
            with row1_col2:
                inventory_df = get_inventory_data(data_links["monthly_historical"][scale.lower()])
                start_year, end_year = get_start_end_year(inventory_df)
                periods = get_periods(inventory_df)
                with st.expander("Select year and month", True):
                    selected_year = st.slider(
                        "Year",
                        start_year,
                        end_year,
                        value=start_year,
                        step=1,
                    )
                    selected_month = st.slider(
                        "Month",
                        min_value=1,
                        max_value=12,
                        value=int(periods[0][-2:]),
                        step=1,
                    )
                selected_period = str(selected_year) + str(selected_month).zfill(2)
                if selected_period not in periods:
                    st.error("Data not available for selected year and month")
                    selected_period = periods[0]
                inventory_df = inventory_df[
                    inventory_df["month_date_yyyymm"] == int(selected_period)
                ]

    #data_cols = get_data_columns(inventory_df, scale.lower(), frequency.lower())
    data_cols = ['Value']

    with row1_col4:
        selected_col = st.selectbox("Attribute", data_cols)
    with row1_col5:
        show_desc = st.checkbox("Show attribute description")
        if show_desc:
            try:
                label, desc = get_data_dict(selected_col.strip())
                markdown = f"""
                **{label}**: {desc}
                """
                st.markdown(markdown)
            except:
                st.warning("No description available for selected attribute")

    row2_col1, row2_col2, row2_col3, row2_col4, row2_col5, row2_col6 = st.columns(
        [0.6, 0.68, 0.7, 0.7, 1.5, 0.8]
    )

    palettes = cm.list_colormaps()
    with row2_col1:
        palette = st.selectbox("Color palette", palettes, index=palettes.index("Blues"))
    with row2_col2:
        n_colors = st.slider("Number of colors", min_value=2, max_value=20, value=8)
    with row2_col3:
        show_nodata = st.checkbox("Show nodata areas", value=True)
    with row2_col4:
        show_3d = st.checkbox("Show 3D view", value=False)
    with row2_col5:
        if show_3d:
            elev_scale = st.slider(
                "Elevation scale", min_value=1, max_value=1000000, value=1, step=10
            )
            with row2_col6:
                st.info("Press Ctrl and move the left mouse button.")
        else:
            elev_scale = 1

    gdf = join_attributes(gdf, inventory_df, scale.lower())
    gdf_null = select_null(gdf, selected_col)
    gdf = select_non_null(gdf, selected_col)
    gdf = gdf.sort_values(by=selected_col, ascending=True)


    colors = cm.get_palette(palette, n_colors)
    colors = [hex_to_rgb(c) for c in colors]

    for i, ind in enumerate(gdf.index):
        index = int(i / (len(gdf) / len(colors)))
        if index >= len(colors):
            index = len(colors) - 1
        gdf.loc[ind, "R"] = colors[index][0]
        gdf.loc[ind, "G"] = colors[index][1]
        gdf.loc[ind, "B"] = colors[index][2]

    initial_view_state = pdk.ViewState(
        latitude=53,
        longitude=10,
        zoom=3,
        max_zoom=16,
        pitch=0,
        bearing=0,
        height=600,
        width=None,
    )

    min_value = gdf[selected_col].min()
    max_value = gdf[selected_col].max()
    color = "color"
    # color_exp = f"[({selected_col}-{min_value})/({max_value}-{min_value})*255, 0, 0]"
    color_exp = f"[R, G, B]"

    geojson = pdk.Layer(
        "GeoJsonLayer",
        gdf,
        pickable=True,
        opacity=0.5,
        stroked=True,
        filled=True,
        extruded=show_3d,
        wireframe=True,
        get_elevation=f"{selected_col}",
        elevation_scale=elev_scale,
        # get_fill_color="color",
        get_fill_color=color_exp,
        get_line_color=[0, 0, 0],
        get_line_width=2,
        line_width_min_pixels=1,
    )

    geojson_null = pdk.Layer(
        "GeoJsonLayer",
        gdf_null,
        pickable=True,
        opacity=0.2,
        stroked=True,
        filled=True,
        extruded=False,
        wireframe=True,
        # get_elevation="properties.ALAND/100000",
        # get_fill_color="color",
        get_fill_color=[200, 200, 200],
        get_line_color=[0, 0, 0],
        get_line_width=2,
        line_width_min_pixels=1,
    )

    # tooltip = {"text": "Name: {NAME}"}

    # tooltip_value = f"<b>Value:</b> {median_listing_price}""
    tooltip = {
        "html": "<b>Name:</b> {Name}<br><b>Value:</b> {"
        + selected_col
        + "}<br><b>Date:</b> "
        + selected_period
        + "",
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }

    layers = [geojson]
    if show_nodata:
        layers.append(geojson_null)

    r = pdk.Deck(
        layers=layers,
        initial_view_state=initial_view_state,
        map_style="light",
        tooltip=tooltip,
    )

    row3_col1, row3_col2 = st.columns([6, 1])

    with row3_col1:
        st.pydeck_chart(r)
    with row3_col2:
        st.write(
            cm.create_colormap(
                palette,
                label=selected_col.replace("_", " ").title(),
                width=0.2,
                height=3,
                orientation="vertical",
                vmin=min_value,
                vmax=max_value,
                font_size=10,
            )
        )
    # row4_col1, row4_col2, row4_col3 = st.columns([1, 2, 3])
    # with row4_col1:
    #     show_data = st.checkbox("Show raw data")
    # with row4_col2:
    #     show_cols = st.multiselect("Select columns", data_cols)
    # with row4_col3:
    #     show_colormaps = st.checkbox("Preview all color palettes")
    #     if show_colormaps:
    #         st.write(cm.plot_colormaps(return_fig=True))
    # if show_data:
    #     if scale == "National":
    #         st.dataframe(gdf[["NAME", "GEOID"] + show_cols])
    #     elif scale == "State":
    #         st.dataframe(gdf[["NAME", "STUSPS"] + show_cols])
    #     elif scale == "County":
    #         st.dataframe(gdf[["NAME", "STATEFP", "COUNTYFP"] + show_cols])
    #     elif scale == "Metro":
    #         st.dataframe(gdf[["NAME", "CBSAFP"] + show_cols])
    #     elif scale == "Zip":
    #         st.dataframe(gdf[["GEOID10"] + show_cols])


app()
