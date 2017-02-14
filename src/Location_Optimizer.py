from __future__ import division
import pandas as pd
import numpy as np
import random
import cPickle as pickle
import fiona

from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter
from shapely.geometry import Point, shape, MultiPoint
from geopy.distance import vincenty
from pygeocoder import Geocoder
import time

from scipy.optimize import basinhopping

random.seed(0)


class RandomDisplacement(object):
	'''
	__call__ method is called when we take a random step
	'''
	def __init__(self, stepsize = 1):
		self.stepsize = stepsize

	def __call__(self, x):
		xnew = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
		return xnew


class AcceptTestCondition(object):
	'''
	__call__ method is called to check whether a step should be accepted
	'''
	def __init__(self, xmin, xmax, stepsize = 1):
		self.stepsize = stepsize
		self.xmin = xmin
		self.xmax = xmax

		num = 24
		num = ['0{}'.format(num) if num < 10 else num][0]
		path = "gov data/maps/eci/PC_Data/States/S{num}/S{num}_PC.shp".format(num = num)
		self._fc = fiona.open(path)

	def __call__(self, f_new , x_new , f_old , x_old):
		point = Point(x_new[0], x_new[1])
		for feature in self._fc:
			if shape(feature['geometry']).contains(point):
				return True


class CallbackCondition(object):
	'''
	__call__ method is called everytime a global minima is accepted
	'''
	def __init__(self):
		self.list_of_new_points = []
		self.num_of_points = 0

	def __call__(self, x_new, f, accept):
		if len(self.list_of_new_points) > 0:
			distances = []
			distances = [map(lambda item: vincenty(x_new, item).miles > 35, self.list_of_new_points)]
			if not all(distances):
				return False
		if accept:
			self.list_of_new_points.append(x_new)
			self.num_of_points += 1
		
		if self.num_of_points == 15:
			return True


class GradientValue(object):
	'''
	__call__ method is called when we have to calculate the value of the 
	cost function at a step
	'''
	def __init__(self, state, df_cities_lat_long):
		self._area_profitability_df = None
		self._area_GDP_growth_df = None
		self._df = df_cities_lat_long
		self._state = state
		self._cols = None
		self._other_cols = None

		self._transformed_area_profitability = None
		self._transformed_area_GDP_growth = None
		self._other_area_GDP_growth_df = None

		self.get_data()
		self.transform_data()

	def __call__(self, x_new):
		distances = []
		for i in range(len(self._cols)):
			new_point = (x_new[0], x_new[1])
			area_office = self._df[['Long', 'Lat']].ix[i].values
			distances.append(vincenty(new_point, area_office).miles)

		k = 3

		closest_office_indices = sorted([dist for dist in enumerate(distances)], key = lambda x : x[1])[ : k]

		gradient_value = self.calculate_gradient_value(closest_office_indices, distances, x_new)

		return gradient_value

	def get_data(self):
		"""
		Get and clean data
		"""
		self._df = self._df.set_index('City').drop(['Modinagar', 'Sahibabad']).reset_index()		
		self._cols = self._df['City']

		self._area_profitability_df = pd.read_excel('data/Shubham MIS/Shubham Branch Summary - November 16.xlsx', \
						sheetname = 'Branch Profitability', parse_cols =  np.append(0, np.arange(43,68)))
		self._area_profitability_df = self._area_profitability_df.T[self._cols]

		self._area_GDP_growth_df = pd.read_csv('gov data/Districtwise_GDP_and_growth_rate_based_at_current_price_2004-05_Uttar_Pradesh.csv', header = 0)
		self._area_GDP_growth_df.rename(index = str, columns={'Kanpur Nagar': "Kanpur"}, inplace = True)
		self._area_GDP_growth_df = self._area_GDP_growth_df.ix[8:14, 2:].reset_index(drop = True)
		self._other_cols = map(lambda item: item if item not in self._cols else None, self._area_GDP_growth_df.columns)
		self._other_cols = [item for item in self._other_cols if item is not None]
		lst = map(lambda col: [col, 1.0 * self._area_GDP_growth_df[col].isnull().sum()], self._other_cols)
		self._other_cols = pd.DataFrame(lst)[pd.DataFrame(lst)[1] < 1][0].values

		pass

	def transform_data(self):
		"""
		Transform Profitability and GDP data
		"""
		scale = StandardScaler(with_mean = False)
		m_scale = MinMaxScaler(feature_range=(-5, 5))

		def tranform_area_profitability(area):
			s = scale.fit_transform(self._area_profitability_df[area])
			m = m_scale.fit_transform(s)

			return m.mean()

		def tranform_area_GDP_data(area):
			values = self._area_GDP_growth_df[area]
			s = scale.fit_transform(values)
			values = m_scale.fit_transform(s)

			# values = map(lambda item: float(item.replace(',', '')), values)
			# sign = Counter(np.sign(np.diff(values[:-1]))).most_common()[1][0]
			area_GDP = np.diff(values)[-1] * np.diff(values[:-1]).mean() #* sign

			return area_GDP

		self._transformed_area_profitability = map(lambda area: -1.0 * tranform_area_profitability(area), self._cols)
		self._transformed_area_GDP_growth = map(lambda area: tranform_area_GDP_data(area), self._cols)
		self._transformed_other_area_GDP_growth = map(lambda area: tranform_area_GDP_data(area), self._other_cols)
		
		_other_area_coordinates = []
		for i in range(len(self._other_cols)):
			_other_area_coordinates.append(Geocoder.geocode(self._other_cols[i] + ' ' + self._state).coordinates)
			time.sleep(.1) 		# delay of .1 second to not exceed query limit with geocode

		self._other_cols = _other_area_coordinates

		pass

	def calculate_gradient_value(self, closest_office_indices, distances, x_new):
		"""
		Plot both existing and recommended office locations on formatted State shapefile

		Parameters
		----------
		closest_office_indices : list
			List of Longtitude & latitude for location recommendations.
		distances : list
			List of distances between x_new and its k nearest existing offices
		x_new : list
			Longtitude & latitude of the new location

		Returns
		----------
		gradient_value : float
			The value of the cost function at Point x_new
		"""
		gradient_value = 0

		points = []
		for combo in closest_office_indices:
			gradient_value = ( self._transformed_area_profitability[combo[0]] * 10  +  0.5 * self._transformed_area_GDP_growth[combo[0]] ) * distances[combo[0]]**2
		
		for i in range(len(self._other_cols)):
			gradient_value += vincenty(x_new, self._other_cols[i]).miles * self._transformed_other_area_GDP_growth[i] * -1.0 

		points.append(x_new)

		return gradient_value


class GetNewLocationRecommendations(object):
	def __init__(self, state, df):
		self._state = state
		self._df = df
		self.list_of_new_points = None
		pass

	def find_new_locations(self):
		# the starting point, starting from roughly the middle of the polygon
		num = 24
		num = ['0{}'.format(num) if num < 10 else num][0]
		path = "gov data/maps/eci/PC_Data/States/S{num}/S{num}_PC.shp".format(num = num)
		fc = fiona.open(path)
		coords = fc.bounds

		start_pos = [np.mean([coords[0], coords[2]]), np.mean([coords[1], coords[3]])]

		# the bounds
		xmin = [coords[0], coords[1]]
		xmax = [coords[2], coords[3]]

		# rewrite the bounds in the way required by L-BFGS-B
		bounds = [(low, high) for low, high in zip(xmin, xmax)]
		minimizer_kwargs = dict(method = "L-BFGS-B", bounds = bounds)

		# define the new step taking routine, acceptance criteria and 
		# event to trigger after acceptance of point and pass them to basinhopping
		get_gradient_value = GradientValue(self._state, self._df)
		take_step = RandomDisplacement()
		accept_test = AcceptTestCondition(xmin, xmax)
		callback = CallbackCondition()

		result = basinhopping(get_gradient_value, start_pos, niter = 100, minimizer_kwargs = minimizer_kwargs, interval = 10, T = 3.5,\
									take_step = take_step, accept_test = accept_test, callback = callback)
		
		self.list_of_new_points = pd.DataFrame(callback.list_of_new_points, columns = ['Long', 'Lat'])

