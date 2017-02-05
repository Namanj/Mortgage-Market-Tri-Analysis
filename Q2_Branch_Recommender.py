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

from scipy.optimize import basinhopping

random.seed(0)


class RandomDisplacement(object):
	def __init__(self, stepsize = 1):
		self.stepsize = stepsize

	def __call__(self, x):
		xnew = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
		return xnew


class PolygonCheck(object):
	def __init__(self, xmin, xmax, stepsize = 1):
		self.stepsize = stepsize
		self.xmin = xmin
		self.xmax = xmax

		num = 24
		num = ['0{}'.format(num) if num < 10 else num][0]
		path = "gov data/maps/eci/PC_Data/States/S{num}/S{num}_PC.shp".format(num = num)
		self._fc = fiona.open(path)

	def __call__(self, f_new , x_new , f_old , x_old ):
		point = Point(x_new[0], x_new[1])
		for feature in self._fc:
			if shape(feature['geometry']).contains(point):
				return True
	

class CallbackCondition(object):
	def __init__(self):
		self.list_of_new_points = []
		self.num_of_points = 0

	def __call__(self, x_new, f, accept):
		if accept:
			self.list_of_new_points.append(x_new)
			self.num_of_points += 1
		
		if self.num_of_points == 10:
			return True

	def function():
		pass
	

class GradientValue(object):
	def __init__(self, state, df_cities_lat_long):
		self._area_profitability_df = None
		self._area_GDP_growth_df = None
		self._df = df_cities_lat_long
		self._state = state
		self._cols = None

		self._transformed_area_profitability = None
		self._transformed_area_GDP_growth = None

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
		self._df = self._df.set_index('City').drop(['Modinagar', 'Sahibabad']).reset_index()
		# self._df = self._df.set_index('City').drop(['HimmatNagar', 'Wapi']).reset_index()
		# self._df = self._df.set_index('City').drop(['Chandrapur', 'Baramati' ,'Chandan Nagar' ,'Kalyan' ,'Pimpri chinwad' ,'Shrirampur']).reset_index()
		
		self._cols = self._df['City']

		self._area_profitability_df = pd.read_excel('data/Shubham MIS/Shubham Branch Summary - November 16.xlsx', \
						sheetname = 'Branch Profitability', parse_cols =  np.append(0, np.arange(43,68)))
		self._area_profitability_df = self._area_profitability_df.T[self._cols]

		self._area_GDP_growth_df = pd.read_csv('gov data/Districtwise_GDP_and_growth_rate_based_at_current_price_2004-05_Uttar_Pradesh.csv', header = 0)
		self._area_GDP_growth_df.rename(index = str, columns={'Kanpur Nagar': "Kanpur"}, inplace = True)
		self._area_GDP_growth_df = self._area_GDP_growth_df.ix[8:15, :].reset_index(drop = True)
		self._area_GDP_growth_df = self._area_GDP_growth_df[self._cols]

		pass

	def transform_data(self):
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
			sign = Counter(np.sign(np.diff(values[:-1]))).most_common()[1][0]
			area_GDP = np.diff(values)[-1] * np.diff(values[:-1]).mean() * sign

			return area_GDP

		# import pdb
		# pdb.set_trace()
		self._transformed_area_profitability = map(lambda area: -1.0 * tranform_area_profitability(area), self._cols)
		self._transformed_area_GDP_growth = map(lambda area: tranform_area_GDP_data(area), self._cols)

		pass

	def calculate_gradient_value(self, closest_office_indices, distances, x_new):
		"""
		Plot both existing and recommended office locations on formatted State shapefile.

		Parameters
		----------
		closest_office_indices : list
			List of Longtitude & latitude for location recommendations.
		distances : list
			List of Longtitude & latitude for location recommendations.
		x_new : list
			List of Longtitude & latitude for location recommendations.
		"""
		gradient_value = 0

		# print type(self._transformed_area_profitability)
		# print self._transformed_area_profitability
		# print type(self._transformed_area_GDP_growth)
		# print self._transformed_area_GDP_growth
		# print len(self._transformed_area_GDP_growth)
		points = []
		for combo in closest_office_indices:
			gradient_value = ( self._transformed_area_profitability[combo[0]] * 10  +  self._transformed_area_GDP_growth[combo[0]] ) * distances[combo[0]]**2
		points.append(x_new)

		# print self._transformed_area_profitability
		# print self._transformed_area_GDP_growth
		# import sys
		# sys.exit()
		# dist = []
		# if len(points) > 0:
		# 	for point in points:
		# 		dist.append(vincenty(point, x_new).miles)
		# 	gradient_value -= (1 / np.sum(dist))
			
		# print self._transformed_area_profitability
		# print self._transformed_area_GDP_growth

		return gradient_value


class GetNewLocationRecommendations(object):
	def __init__(self, state, df):
		self._state = state
		self._df = df
		self.list_of_new_points = None
		pass

	def initiate(self):

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

		# define the new step taking routine, acceptance criteria and event to trigger after acceptance of point and pass them to basinhopping
		get_gradient_value = GradientValue(self._state, self._df)
		take_step = RandomDisplacement()
		accept_test = PolygonCheck(xmin, xmax)
		callback = CallbackCondition()

		result = basinhopping(get_gradient_value, start_pos, niter = 100, minimizer_kwargs = minimizer_kwargs, interval = 20, T = 2.5,\
									take_step = take_step, accept_test = accept_test, callback = callback)
		
		self.list_of_new_points = pd.DataFrame(callback.list_of_new_points, columns = ['Long', 'Lat'])

