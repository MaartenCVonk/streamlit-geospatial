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
import sys

print('hi')
print(os.getcwd())
directory = os.path.join(os.getcwd(), "data/us_counties.geojson")
directory1 = os.path.join(os.getcwd(), "data/world-administrative-boundaries.geojson")
directory2 = os.path.join(os.getcwd(), "data/MEI_FIN_19012023181514641.csv")
print(directory)
gdf = gpd.read_file(directory1)
print(sys.getsizeof(gdf))
print(gdf.columns)
#gdf = gpd.read_file(directory)
#print(sys.getsizeof(gdf))

print(gdf.columns)
data = pd.read_csv(directory2)
print(gdf.columns)
gdf.rename(columns={'name':'Country'}, inplace=True)
data['TIME'] = data['TIME'].replace('-', '', regex=True).astype(int)
data = data.iloc[0:100,:]
print(data.columns)

new_gdf = gdf.merge(data, on="Country", how="inner")
sys.getsizeof(new_gdf)
print(sys.getsizeof(new_gdf))