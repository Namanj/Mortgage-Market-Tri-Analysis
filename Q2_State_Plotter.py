import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.prepared import prep
import seaborn as sns

from lxml import etree
from descartes import PolygonPatch
import fiona
from itertools import chain
from pygeocoder import Geocoder
import time

import shapefile as shp

from Q2_Branch_Recommender import GetNewLocationRecommendations

random.seed(0)


class initiate(object):
	"""docstring for initiate"""
	def __init__(self, state, num):
		self._state = state
		self._num = num
		self._cities = None
		self._df_cities_lat_long = None
		self._state_shapefile = None
		self._df_map = None
		self._state_points = None
		self.state_polygon = None

	def get_data(self):
		"""
		Read in raw data from excel files. 
		Get city coordinates via Geocoder for every city in self._state
		"""
		df = pd.read_excel('1.3 Portfolio data for last three years full.xlsx', sheetname = 'Branch', skiprows = 3, \
										names = ['S.No','City', 'State', 'Branch Opening Date'])

		df.drop('S.No', axis = 1, inplace = True)
		df_by_state = df.groupby(['State', 'City', 'Branch Opening Date']).count()

		self._cities = df_by_state.ix[self._state].reset_index()['City']
		self._df_cities_lat_long = pd.DataFrame(columns = ['City', 'Lat', 'Long'])

		for i in range(len(self._cities)):
			(latitude, longitude) = Geocoder.geocode(self._cities[i] + ' ' + self._state).coordinates
			self._df_cities_lat_long.loc[i] = [self._cities[i], latitude, longitude]
			time.sleep(.1) 		# delay of .1 second to not exceed query limit with geocode

		pass		

	def read_shapefile(self):
		"""
		Read in raw shapefile via Basemap
		"""
		self._num = ['0{}'.format(self._num) if self._num < 10 else self._num][0]
		path = "gov data/maps/eci/PC_Data/States/S{num}/S{num}_PC.shp".format(num = self._num)
		shp = fiona.open(path)
		bds = shp.bounds
		shp.close()

		extra = 0.01
		ll = (bds[0], bds[1])
		ur = (bds[2], bds[3])
		self._coords = list(chain(ll, ur))
		w, h = self._coords[2] - self._coords[0], self._coords[3] - self._coords[1]
		
		self._state_shapefile = Basemap(
			lon_0 = self._df_cities_lat_long['Long'].mean(), 
			lat_0 = self._df_cities_lat_long['Lat'].mean(), 
			projection = 'tmerc', ellps = 'WGS84',
			llcrnrlon = self._coords[0] - extra * w,					# llcrnrlon = Lower Left Corner Longitude
			llcrnrlat = self._coords[1] - extra + 0.01 * h,				# llcrnrlat = Lower Left Corner Latitude
			urcrnrlon = self._coords[2] + extra * w,					# urcrnrlon = Upper Right Corner Longitude
			urcrnrlat = self._coords[3] + extra + 0.01 * h,				# urcrnrlat = Upper Right Corner Latitude
			lat_ts = 0, resolution = 'i', suppress_ticks = True)

		_ = self._state_shapefile.readshapefile(
			'gov data/maps/eci/PC_Data/States/S{num}/S{num}_PC'.format(num = self._num),
			self._state, drawbounds = False,
			color = 'none', zorder = 2)

		pass

	def map_df(self):
		"""
		Setup self._df_map
		"""
		a = 'state_info'.format(state = self._state)
		self._df_map = pd.DataFrame(
							{
							'poly': [ Polygon(xy) for xy in self._state_shapefile.UP],
							'PC_NAME': [district['PC_NAME'] for district in self._state_shapefile.UP_info]
							})

		# Create Point objects in map coordinates from dataframe long and lat values
		map_points = pd.Series([ Point( self._state_shapefile(mapped_x, mapped_y)) for mapped_x, mapped_y in \
						zip( self._df_cities_lat_long['Long'], self._df_cities_lat_long['Lat'])])
		self._state_points = MultiPoint( list(map_points.values))
		self.state_polygon = prep( MultiPolygon( list(self._df_map['poly'].values)))

		# calculate points that fall within the state boundary
		self._state_points = filter(self.state_polygon.contains, self._state_points)

		# draw state districts from polygons
		self._df_map['districts'] = self._df_map['poly'].map( lambda item: PolygonPatch(
			item,
			fc='#555555',
			ec='#787878', lw=.25, alpha=.9,
			zorder=4))

		pass

	def plot_formatted_shapefile_with_locations(self, list_of_new_points):
		"""
		Plot both existing and recommended office locations on formatted State shapefile.

		Parameters
		----------
		list_of_new_points : list
			List of Longtitude & latitude for location recommendations.
		"""
		fig = plt.figure()
		ax = fig.add_subplot(111, axisbg = 'w', frame_on = False)

		# we don't need to pass points to m() because we calculated using map_points and shapefile polygons
		dev = self._state_shapefile.scatter(
			[geom.x for geom in self._state_points],
			[geom.y for geom in self._state_points],
			20, marker = 'o', lw = .25, facecolor = '#33ccff', edgecolor = 'w',
			alpha = 0.9, antialiased = True, label = 'Blue Shubham office locations', zorder = 3)
		
		
		map_points = pd.Series([Point(self._state_shapefile(mapped_x, mapped_y)) for mapped_x, mapped_y in \
					zip(list_of_new_points['Long'], list_of_new_points['Lat'])])
		_new_state_points = MultiPoint(list(map_points.values))
		dev = self._state_shapefile.scatter(
			[geom.x for geom in _new_state_points],
			[geom.y for geom in _new_state_points],
			20, marker = 'o', lw = .25, facecolor = '#32ff5f', edgecolor = 'w',
			alpha = 0.9, antialiased = True, zorder = 3)
			
		# plot office locations by adding the PatchCollection to the axes instance
		ax.add_collection(PatchCollection(self._df_map['districts'].values, match_original = True))
		
		# copyright and source data info
		smallprint = ax.text(
			1.03, 0.1,
			'\nContains Shubham Housing Finance Proprietary Data',
			ha = 'right', va = 'bottom',
			size = 8, color = '#555555',
			transform = ax.transAxes)

		# Draw a map scale
		self._state_shapefile.drawmapscale(
			self._coords[0] + 0.08, self._coords[1] + 0.015,
			self._coords[0], self._coords[1], 10.,
			barstyle = 'fancy', labelstyle = 'simple',
			fillcolor1 = 'w', fillcolor2 = '#555555',
			fontcolor = '#555555', zorder = 5)

		plt.title("Shubham Housing Finance Office Locations, {State}".format(State = self._state))
		plt.tight_layout()

		# this will set the image width to 722px at 100dpi
		fig.set_size_inches(10, 10)

		plt.savefig('data/{State}_visualized.png'.format(State = self._state), dpi = 100, alpha = True)
		plt.show()
			
		pass	


if __name__ == '__main__':
	state = 'UP'
	num = 24
	
	start = initiate(state, num)
	start.get_data()
	start.read_shapefile()
	start.map_df()

	new_locations = GetNewLocationRecommendations(state, start._df_cities_lat_long)
	new_locations.initiate()

	start.plot_formatted_shapefile_with_locations(new_locations.list_of_new_points)

