import pandas as pd
import numpy as np
import random
import cPickle as pickle

# from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point

from scipy.optimize import basinhopping

random.seed(0)

class RandomDisplacementPolygonCheck(object):
    def __init__(self, state_polygon, state_shapefile, stepsize = 10):
        self.stepsize = stepsize
        self._state_polygon = state_polygon
        self._state_shapefile = state_shapefile

    def __call__(self, x):
        while True:
            xnew = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
            
            # Checking whether proposed point is within the Polygons boundry
            if filter(self._state_polygon.contains, [Point(self._state_shapefile(xnew[0], xnew[1]))]):
                break          
        return xnew
    

class CallbackCondition(object):
    def __init__(self):
        self.list_of_new_points = []
        self.num_of_points = 0

    def __call__(self, x_new, f, accept):
        if accept:
            self.list_of_new_points.append((x_new, f))
    
class BranchRecommender(object):
	def __init__(self, df, state_polygon, _state_shapefile):
		self._df = df
		self._state_polygon = state_polygon
		self._state_shapefile = _state_shapefile
		pass

	def initiate(self):
		# an example function with multiple minima
		def f(x): return np.sin(np.linalg.norm(x) * np.pi)

		# the starting point, starting from roughly the middle of the polygon
		x0 = [self._df['Long'].mean(), self._df['Lat'].mean()]

		# the bounds
		xmin = [self._df['Long'].min(), self._df['Lat'].min()]
		xmax = [self._df['Long'].max(), self._df['Lat'].max()]

		# rewrite the bounds in the way required by L-BFGS-B
		bounds = [(low, high) for low, high in zip(xmin, xmax)]
		minimizer_kwargs = dict(method = "L-BFGS-B", bounds = bounds)

		# define the new step taking routine and pass it to basinhopping
		take_step = RandomDisplacementPolygonCheck(self._state_polygon, self._state_shapefile)
		callback = CallbackCondition()

		list_of_new_points = []
		result = basinhopping(f, x0, niter = 10, minimizer_kwargs = minimizer_kwargs, interval = 100, take_step = take_step, callback = callback)
		list_of_new_points.append(result)
		
		print callback.list_of_new_points
		# print list_of_new_points


if __name__ == '__main__':
	br = BranchRecommender()
	br.initiate()

