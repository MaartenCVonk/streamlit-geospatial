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
import googlemaps
from datetime import datetime

print('hi')
print(os.getcwd())
directory = os.path.join(os.getcwd(), "data/us_counties.geojson")
directory1 = os.path.join(os.getcwd(), "data/world-administrative-boundaries.geojson")
directory2 = os.path.join(os.getcwd(), "data/data_iafq_firearms_trafficking.xlsx")

api_key = 'AIzaSyBi0F3a_ib1WTH3V-Npo08rwUf7FthgJ6k'
gmaps = googlemaps.Client(key=api_key)
directions_result = gmaps.directions("Sydney Town Hall",
                                     "Parramatta, NSW",
                                     mode="transit",
                                     departure_time= datetime.now())
print(directions_result)
