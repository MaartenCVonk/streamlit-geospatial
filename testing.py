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
data = data[['Iso3_code','Indicator','Year','VALUE','Dimension', 'Category']]
data = data[(data['Dimension'] == 'Total') & (data['Category'] == 'Total')]
data.drop(columns = ['Dimension', 'Category'], inplace=True)
data.rename(columns={'Iso3_code':'iso3'}, inplace=True)
data = data.groupby(['iso3','Year','Indicator']).mean().reset_index() #TODO: information is being lost by aggregating here, reformulate
data.set_index(['iso3','Year','Indicator'], inplace=True)
data = data.unstack(['Indicator']).reset_index().droplevel(level=0, axis=1).reset_index(drop=True) #TODO rewrite, preferably without explicitnly naming columns
data.columns = ['iso3','Year','Ammunition seized', 'Arms seized',
       'Individuals arrested/suspected for illicit trafficking in weapons',
       'Individuals convicted for illicit trafficking in weapons',
       'Individuals prosecuted for illicit trafficking in weapons',
       'Individuals targeted by criminal justice system due to illicit trafficking in weapons',
       'Instances/cases of seizures', 'Parts and components seized']
sys.exit()
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