import datetime
import os
import pathlib
import requests
import zipfile
import pandas as pd
import pydeck as pdk
import geopandas as gpd
import streamlit as st
import leafmap.colormaps as cm
from leafmap.common import hex_to_rgb

print('hi')
print(os.getcwd())
directory = os.path.join(os.getcwd(), "data/us_states.geojson")
directory1 = os.path.join(os.getcwd(), "data/world-administrative-boundaries.geojson")
directory2 = os.path.join(os.getcwd(), "data/MEI_FIN_19012023181514641.csv")
print(directory)
gdf = gpd.read_file(directory1)
data = pd.read_csv(directory2)
print(gdf.columns)
print(gdf[['name']])
data['TIME'] = data['TIME'].replace('-', '', regex=True).astype(int)
print(data.iloc[0:4,0:8])