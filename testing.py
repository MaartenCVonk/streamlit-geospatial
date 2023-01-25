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
directory2 = os.path.join(os.getcwd(), "data/data_iafq_firearms_trafficking.xlsx")
print(directory)
gdf = gpd.read_file(directory1)
print(gdf.columns)

data = pd.read_excel(directory2, skiprows=2)
data = data[['Iso3_code','Indicator','Year','VALUE']]
data.rename(columns={'Iso3_code':'iso3'}, inplace=True)
data = data.merge(gdf, on='iso3', how='left')
print(data.head(5))
print(data.columns)
print(data['iso3'])
print(data)
print(data.Indicator.unique())
sys.exit()
gdf.rename(columns={'name':'Country'}, inplace=True)
data['TIME'] = data['TIME'].replace('-', '', regex=True).astype(int)
data = data.iloc[0:100,:]
print(data.columns)

new_gdf = gdf.merge(data, on="Country", how="inner")
sys.getsizeof(new_gdf)
print(sys.getsizeof(new_gdf))