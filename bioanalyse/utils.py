from typing import Any, Callable, Hashable, Iterable, List, Mapping, Optional, Tuple, Union
from abc import ABC, abstractmethod
Number = Union[int, float]

from igraph import Graph
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.sparse.csr import csr_matrix
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
from umap.umap_ import fuzzy_simplicial_set
from warnings import warn

import _class_utils

# BEGIN ABSTRACT CLASS SampleBase
class SampleBase(ABC):
	"""The base class from which all specific sample implementations should derive
	Any subclasses must implement the property 'data' to ensure compatibility with filters
	"""
	# BEGIN ABSTRACT PROPERTIES
	@property
	@abstractmethod
	def data(self) -> pd.DataFrame:
		"""Implement this to ensure compatibility with filters
		It should return all data that you want to then filter
		"""
		pass
	# END ABSTRACT PROPERTIES

	# BEGIN DUNDER FUNCTIONS
	# BEGIN FUNCTION __contains__
	def __contains__(self, key) -> bool:
		"""Returns True if hasattr(self, key) and not callable(getattr(self, key))
		Makes mapping and objects interchangeable

		(Partially) Reimplement only for faster response
		"""
		return hasattr(self, key) and not callable(getattr(self, key))
	# END FUNCTION __contains__

	# BEGIN FUNCTION __getitem__
	def __getitem__(self, key) -> Any:
		"""Recursive for every item in key

		if self is list and key in list: return list[key] for Any(key); otherwise try it as np.ndarray
		elif self is Series: Series[key] > Series[i] for i in key
		elif self is DataFrame:
			if key is float: DataFrame.loc > DataFrame[key]
			elif key is int: DataFrame.loc[row] > DataFrame[col] > DataFrame.iloc[key]
			elif key is Series: DataFrame.loc[row] > DataFrame[col] > DataFrame[key]
			elif key is ndarray:
				if ndim == 1: DataFrame.loc[rows] > DataFrame[cols] > DataFrame[key]
				else: DataFrame[key]
			elif key is tuple:
				if len == 2: DataFrame[multirow] > DataFrame[multicol] > DataFrame.loc, DataFrame.iloc, (DataFrame.iloc/loc[key0])[key1]
				else: DataFrame[multirow] > DataFrame[multicol] > DataFrame.loc[key[:2]][key[2:], DataFrame.iloc[key[:2]][key[2:], (DataFrame.iloc/loc[key0])[key1][key[2:]]
			else: DataFrame[row] > DataFrame[col] > DataFrame.iloc
		else:
			if key is tuple: self[key] > self[key[0]][key[1:]] > [self[i][key[1:]] for i in key[0]]
			else: self[key] > self.key

		Raises a KeyError, with the original error at the top of the traceback with a separate KeyError for each recursion

		(Partially) Reimplement only for faster response
		Recommended to only handle specific cases and otherwise `return super().__getitem__(key)`
		"""
		return _class_utils._getitem(self, key)
	# END FUNCTION __getitem__
	# END DUNDER FUNCTIONS
# END ABSTRACT CLASS SampleBase


# BEGIN KWARGS LISTS
# BEGIN TRANSFORMATIONS
PCA_KWARGS = ['n_components', 'whiten', 'svd_solver', 'iterated_power', 'random_state']
UMAP_KWARGS = ['n_neighbors','metric', 'metric_kwds', 'output_metric', 'output_metric_kwds', 'n_epochs', 'learning_rate', 'init',
				'min_dist', 'spread', 'low_memory', 'set_op_mix_ratio', 'local_connectivity', 'repulsion_strength',
				'negative_sample_rate', 'transform_queue_size', 'a', 'b', 'random_state', 'angular_rp_forest', 'target_n_neighbors',
				'target_metric', 'target_metric_kwds', 'target_weight', 'transform_seed', 'force_approximation_algorithm',
				'verbose', 'unique']
TSNE_KWARGS = ['n_components', 'perplexity', 'early_exaggeration', 'learning_rate', 'n_iter', 'n_iter_without_progress', 'min_grad_norm',
				'metric', 'init', 'verbose', 'random_state', 'method', 'angle', 'n_jobs']
CLUSTER_KWARGS = ['leiden', 'resolution'] + ['initial_membership', 'n_iterations', 'seed', 'node_sizes'] + PCA_KWARGS + UMAP_KWARGS + \
					['pca_random_state', 'umap_random_state']
# END TRANSFORMATIONS

# BEGIN FIGURE/AXES
FIGURE_KWARGS = ['num', 'figsize', 'dpi', 'facecolor', 'edgecolor', 'FigureClass', 'clear', 'linewidth', 'subplotpars',
					'tight_layout', 'constrained_layout']
IMAGEGRID_KWARGS = ['nrows_ncols', 'ngrids', 'direction', 'axes_pad', 'add_all', 'share_all', 'aspect', 'label_mode', 'cbar_mode',
					'cbar_location', 'cbar_pad', 'cbar_size', 'cbar_set_cax', 'axes_class']
ADD_SUBPLOT_KWARGS = ['projection',  'polar', 'sharex', 'sharey', 'label', 'adjustable', 'agg_filter', 'alpha', 'anchor', 'animated', 'aspect',
						'autoscale_on', 'autoscalex_on', 'autoscaley_on', 'axes_locator', 'axisbelow', 'clip_box', 'clip_on', 'clip_path',
						'contains', 'facecolor', 'fc', 'frame_on', 'gid', 'in_layout', 'navigate', 'navigate_mode',
						'path_effects', 'picker', 'position', 'rasterization_zorder', 'rasterized', 'sketch_params', 'snap', 'title', 'transform',
						'url', 'visible', 'xbound', 'xlabel', 'xlim', 'xmargin', 'xscale', 'xticklabels', 'xticks', 'ybound', 'ylabel', 'ylim',
						'ymargin', 'yscale', 'yticklabels', 'yticks', 'zorder']
SUBPLOTS_KWARGS = ['sharex', 'sharey', 'squeeze', 'subplot_kw', 'gridspec_kw'] + FIGURE_KWARGS
SUBPLOTS_ADJUST_KWARGS = ['hspace', 'wspace', 'left', 'bottom', 'right', 'top']
# END FIGURE/AXES

# BEGIN PLOT
COLLECTIONS_KWARGS = ['edgecolors', 'facecolors', 'linestyles', 'capstyle', 'joinstyle', 'antialiaseds', 'offsets', 'transOffset', 'norm',
						'pickradius', 'hatch', 'urls', 'offset_position', 'zorder']
PLOT_KWARGS = ['scalex', 'scaley', 'data', 'agg_filter', 'alpha', 'animated', 'antialiased', 'aa', 'clip_box', 'color', 'c', 'contains', 
				'dash_capstyle', 'dash_joinstyle', 'dashes', 'drawstyle', 'ds', 'fillstyle', 'gid', 'in_layout', 'label', 'linestyle', 'ls',
				'linewidth', 'lw', 'marker', 'markeredgecolor', 'mec', 'markeredgewidth', 'mew', 'markerfacecolor', 'mfc', 'markerfacecoloralt',
				'mfcalt', 'markersize', 'ms', 'markevery', 'path_effects', 'picker', 'pickradius', 'rasterized', 'sketch_params', 'snap',
				'solid_capstyle', 'solid_joinstyle', 'transform', 'url', 'visible', 'xdata', 'ydata', 'zorder']
SCATTER_KWARGS = ['s', 'c', 'color', 'marker', 'cmap', 'vmin', 'vmax', 'alpha', 'linewidths', 'verts', 'plotnonfinite',
					'data'] + COLLECTIONS_KWARGS
LEGEND_KWARGS = ['loc', 'bbox_to_anchor', 'ncol', 'prop', 'fontsize', 'numpoints', 'scatterpoints', 'scatteryoffsets',
					 'markerscale', 'markerfirst', 'frameon', 'fancybox', 'shadow', 'framealpha', 'facecolor', 'edgecolor', 'mode',
					 'bbox_transform', 'title', 'tile_fontsize', 'borderpad', 'labelspacing', 'handlelength', 'handletextpad', 'borderaxespad',
					 'columnspacing', 'handler_map']
COLORBAR_KWARGS = ['use_gridspec', 'orientation', 'fraction', 'pad', 'shrink', 'aspect', 'anchor', 'panchor', 'extend', 'extendfrac',
					'extendrect', 'spacing', 'ticks', 'format', 'drawedges', 'boundaries', 'values']
# END PLOT

# BEGIN LABELING
TICK_KWARGS = ['minor']
TEXT_KWARGS = ['x', 'y', 'text', 'color', 'verticalalignment', 'horizontalalignment', 'multialignment', 'fontproperties', 'rotation',
				'linespacing', 'rotation_mode', 'usetex', 'wrap', 'agg_filter', 'alpha', 'animated', 'backgroundcolor', 'bbox', 'clip_box',
				'clip_on', 'clip_path', 'c', 'contains', 'fontfamily', 'font_properties', 'fontsize', 'size', 'fontstretch', 'stretch',
				'fontstyle', 'style', 'fontvariant', 'variant', 'fontweight', 'weight', 'gid', 'ha', 'in_layout', 'label', 'ma', 'path_effects',
				'picker', 'position', 'rasterized', 'sketch_params', 'snap', 'transform', 'url', 'va', 'visible', 'zorder']
TITLE_KWARGS = ['fontdict', 'loc', 'pad'] + TEXT_KWARGS
# END LABELING

# BEGIN MISC
CSV_KWARGS = ['sep','delimiter','header','names','index_col','usecols','squeeze','prefix','mangle_dupe_cols','dtype','engine','converters','true_values',
			  'false_values','skipinitialspace','skiprows','skipfooter','nrows','na_values','keep_default_na','na_filter','verbose','skip_blank_lines',
			  'parse_dates','infer_datetime_format','keep_date_col','date_parser','dayfirst','cache_dates','iterator','chunksize','compression','thousands',
			  'decimal','lineterminator','quotechar','quoting','doublequote','escapechar','comment','encoding','dialect','error_bad_lines','warn_bad_lines',
			  'delim_whitespace','low_memory','memory_map','float_precision']
# END MISC
# END KWARGS LISTS

# BEGIN FUNCTION cluster
def cluster(data : Union[np.ndarray, pd.DataFrame], *, leiden : bool = True, resolution : Number = 2, verbose : bool = False, **kwargs) -> np.ndarray:
	"""Returns Leiden or Louvain clustering labels of the rows in the given data
	Uses PCA and UMAP to find neighbors

	Parameters
	----------
	data : np.ndarray, pd.DataFrame
		the values to cluster
		rows are individual points
		columns are values
	*
	leiden : bool = True
		whether to default to the Leiden algorithm if installed
		ignored if module `leidenalg` is not installed
	resolution : Number = 2
		the density limit that defines clusters
		all clusters are guaranteed to have density >= resolution
		only applies if using Leiden
	verbose : bool = False
		Whether or not to print what's happening
	**kwargs
		passed variously to sklearn.decomposition.PCA, umap.umap_.fuzzy_simplicial_set, leidenalg.find_partition
		extra kwargs ignored silently

	Returns
	-------
	np.ndarray (data.shape[0],)
		the cluster membership for each row

	Selected kwargs
	---------------
	n_components : int = 50
		the number of components to reduce to in PCA
	n_neighbors : int = sqrt(data.shape[0]).astype(int)
		the size of the local neighborhood
	metric : str = 'euclidean'
		the metric used to calculate distance in the high dimensional space
		many common metrics are predefined: eg. 'euclidean', 'manhattan', 'chebyshev', 'correlation'
	n_iterations : int =  -1
		number of iterations to run the Leiden algorithm
		if -1, runs until no improvement	
	seed : int = None
		seed for Leiden algorithm random number generator
		if None, leidenalg uses a random seed by default

	umap_random_state : always passed to UMAP
	pca_random_state : always passed to PCA
	random_state 
		passed to UMAP through sklearn.utils.check_random_state, but overridden by umap_random_state
		default None
	"""
	if verbose: print('Validating...')
	if isinstance(data, pd.DataFrame): data = data.values
	elif not isinstance(data, np.ndarray): raise TypeError('data must be an np.ndarray or pd.DataFrame')
	if not isinstance(resolution, (int,float)): raise TypeError('resolution must be a positive float')
	elif resolution < 0: raise ValueError('resolution must be a positive float')

	if leiden:
		if verbose: print('Trying leidenalg import')
		try: import leidenalg
		except ImportError:
			leiden = False # don't try it later
			warn('Using Louvain as leidenalg is not installed')

	LEIDEN_KWARGS = ['initial_membership', 'n_iterations', 'seed', 'node_sizes']

	if 'n_components' not in kwargs: kwargs['n_components'] = 50
	if 'n_neighbors' not in kwargs: kwargs['n_neighbors'] = np.sqrt(data.shape[0]).astype(int)
	if 'metric' not in kwargs: kwargs['metric'] = 'euclidean'
	if 'n_iterations' not in kwargs: kwargs['n_iterations'] = -1

	if 'umap_random_state' not in kwargs: 
		if 'random_state' in kwargs: kwargs['umap_random_state'] = check_random_state(kwargs.pop('random_state'))
		else: kwargs['umap_random_state'] = check_random_state(None)
	if 'pca_random_state' in kwargs: kwargs['random_state'] = kwargs['pca_random_state']


	if verbose: print('Training PCA...')
	pc = PCA(**{k:kwargs[k] for k in PCA_KWARGS if k in kwargs}).fit_transform(data)

	if verbose: print('Calculating distances...')
	kwargs['random_state'] = kwargs.pop('umap_random_state') # must be there
	del kwargs['n_components']
	adj = fuzzy_simplicial_set(pc, **{k:kwargs[k] for k in UMAP_KWARGS if k in kwargs})

	sources, targets = adj.nonzero()
	g = Graph(directed=leiden) # undirected for Louvain
	g.add_vertices(adj.shape[0])  # this adds adj.shape[0] vertices
	g.add_edges(list(zip(sources, targets)))

	if verbose: print('Clustering...')
	if leiden: # now guaranteed to work
		part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution,
										weights=adj[sources, targets].A1, **{k:kwargs[k] for k in LEIDEN_KWARGS if k in kwargs})
	else:
		part = g.community_multilevel(weights=adj[sources, targets].A1)

	# print(part.membership)
	return np.array(part.membership)
# END FUNCTION cluster


# BEGIN CLASS Filter
class Filter:
	"""A filter for an object with DataFrame data
	Filters target data rows (pd.DataFrame) on any other entries/attributes of target
	Minimal memory, as doesn't copy the target or its data
	Any change in the applied filters will call self.target._notify(self) if it exists

	Parameters
	----------
	target : object
		the target whose index is to be filtered
		target[data_key] or target.data_key must be a pd.DataFrame
		target has other attributes accessible as entries
	filters : Iterable[tupl] [optional]
		passed individually to `self.addFilter()`
		see said function for format
		provided for convenience
	*
	data_key : Hashable = 'raw_data'
		the name of the key or attribute the data is saved 
		if not str, only try as key
		if str, attributes are checked before keys

	Attributes
	----------
	result
		the data after applying the filters
	idx
		whether each row passes the filters
	filters
		a list of the filters currently applied as (by, type, cutoff)
	target
		the object this filter targets

	Selected Methods
	----------------
	reset
		undo any applied filtering
	*
		available filter options appear as methods
		use dir(self) to list all available methods
	"""
	# BEGIN DUNDER FUNCTIONS
	# BEGIN FUNCTION __init__
	def __init__(self, target : Any, filter_options : Optional[Tuple[str, ...]] = None, *, data_key : Hashable = '_data') -> None:
		"""Initializes the Filter"""
		if isinstance(data_key, str) and hasattr(target, data_key):
			if not isinstance(getattr(target, data_key), pd.DataFrame):
				raise TypeError('target needs the attribute/entry ' + str(data_key) + ' that is a pd.DataFrame')
			self._isiter = False
		else:
			try: iter(target)
			except TypeError: raise ValueError('target needs the attribute/entry ' + str(data_key) + ' that is a pd.DataFrame')
			else:
				if data_key not in target or not isinstance(target[data_key], pd.DataFrame):
					raise TypeError('target needs the attribute/entry ' + str(data_key) + ' that is a pd.DataFrame')
				self._isiter = True

		if not isinstance(filter_options, Iterable): raise TypeError('filter_options must be an Iterable of filters')
		elif not all([isinstance(i, Iterable) and len(i) >= 2 for i in filter_options]):
			raise TypeError('every filter option must be a Iterable of len 2+. See Filter.addFilter() for format')

		self._target = target
		self.data_key = data_key
		self._idx = pd.Series(np.ones(self.data.shape[0]).astype(bool), index=self.data.index)

		self._filter_options = {}
		if filter_options: [self.add_option(f) for f in filter_options]
		self._applied_filters = []
	# END FUNCTION __init__

	# BEGIN FUNCTION __repr__
	def __repr__(self) -> str:
		return 'Filter(target={0!r}, filter_options={1}, data_key={2})'.format(self.target, self.filter_options, self.data_key)
	# END FUNCTION __repr__

	# BEGIN FUNCTION __getitem__
	def __getitem__(self, key) -> Any:
		"""if key in list, returns list[key] for all key
		if key[0] is Iterable and all(key[0] in self), returns all key[0][key[1:]]
		"""
		return _class_utils._getitem(self, key)
	# END FUNCTION __getitem__

	# BEGIN PROPERTIES
	@property
	def data(self) -> pd.DataFrame:
		"""A reference back to the data in the target"""
		return self.target[self.data_key] if self._isiter else getattr(self.target, self.data_key)
	@property
	def result(self) -> pd.DataFrame:
		"""The values after current filtering"""
		return self.data.iloc[self.idx.values]
	@property
	def idx(self) -> pd.Series:
		"""The filtered indices at present"""
		return self._idx
	@property
	def filters(self) -> Tuple[Hashable, str, Tuple]:
		"""A list of the filters currently applied as (by, type, how)
		External filters show the cutoff as '`func.__name__`(X, `*args`, `**kwargs`)
		"""
		return self._applied_filters
	@property
	def filter_options(self) -> Mapping[Hashable, Union[Iterable[Tuple[Hashable, Any]], Tuple[Hashable, Any]]]:
		"""A dict of available filters as 'by' : value
		`value` can be a single tupl or list of tupl
		each tupl is (name if given else by, how)
		"""
		return self._filter_options
	@property
	def target(self) -> Any:
		"""The target of this Filter"""
		return self._target
	# END PROPERTIES

	# BEGIN METHODS
	# BEGIN FUNCTION reset
	def reset(self) -> 'Filter':
		"""Turn off all applied filters
		Return self for stacking"""
		self._idx = pd.Series(np.ones(self.data.shape[0]).astype(bool), index=self.data.index) # make all true
		self._applied_filters = [] # wipe self.filters

		# Notify and return
		if hasattr(self.target, '_notify'): self.target._notify(self)
		return self
	# END FUNCTION reset

	# BEGIN FUNCTION add_option
	def add_option(self, f : Iterable) -> 'Filter':
		"""Adds a new filter method option
		Filters may raise errors upon first use, as only types and existence are checked, not shapes etc.

		Parameters
		----------
		f : Iterable[Hashable, str/callable]
			must be length 2+
			f[0] is passed as 'by' to outside, linear, or secant
			f[1] is a tuple[str,str] or Callable[[self.target[self.key]], Union[pd.Series[bool], np.ndarray[bool]]
				f[1][0] in ['linear', 'bimodal', 'secant']
				f[1][1] in ['above', 'below']
				if f[1][0] == 'bimodal': f[1][2] is 'method' for self.bimodal
			f[2] if given is the name of the filter, as str
			f[3:] ignored silently if given
		Returns self
		"""
		# BEGION VALIDATION
		if not isinstance(f, Iterable) or len(f) < 2: raise TypeError('f must be an Iterable of length 2+')

		# EXTRACT
		by, how, *name = f
		if name: name = name[0] # ignore f[3:] if given
		
		if not by in self.target and not hasattr(self.target, by): raise ValueError('target has no item ' + str(by))
		if not (isinstance(how, Iterable) and how[0] in ['linear', 'secant', 'bimodal'] and how[1] in ['above', 'below']) and not callable(how):
			raise ValueError('All `how` in filters must be a combination of ([\'secant\', \'linear\', \'bimodal\'], [\'above\', \'below\'])' + \
				', or a callable for outside, not ' + str(how))
		elif isinstance(how, Iterable) and how[0] == 'bimodal' and len(how) == 3 and 	not how[2] in ['isodata', 'li', 'local', 'mean', 'minimum', 'niblack', 'otsu', 'sauvola', 'triangle', 'yen']:
			raise ValueError('method must be a valid skimage.filters.threshold_... function')
		if not name and not isinstance(by, str): raise RuntimeError('If no name is given as f[2], f[0] must be a str')
		if not name and hasattr(self, by): raise RuntimeError('Can\'t add a second filter for ' + str(by) + ' without names')
		elif name and hasattr(self, name): raise RuntimeError('Can\'t add a second filter with the same name')
		# END VALIDATION

		# BEGIN SET UP FILTER LAMBDAS
		if callable(how): setattr(self, name if name else by, lambda *args, **kwargs: self.external(by, how, *args, **kwargs))
		elif how[0] == 'linear': setattr(self, name if name else by, lambda cutoff: self.linear(by, cutoff, how[1]))
		elif how[0] == 'secant': setattr(self, name if name else by, lambda cutoff: self.secant(by, cutoff, how[1]))
		elif how[0] == 'bimodal': setattr(self, name if name else by, lambda **kwargs:
							self.bimodal(by, how[1], how[2] if len(how) == 3 else 'minimum', **kwargs))
		# END SET UP FILTER LAMBDAS

		# BEGIN ADD FILTERS TO OPTIONS
		if not by in self._filter_options: self._filter_options[by] = (name if name else by,how)
		elif not isinstance(self._filter_options[by], list): self._filter_options[by] = [self._filter_options[by], (name if name else by,how)]
		else: self._filter_options[by].append((name if name else by, how))
		# END ADD FILTERS TO OPTIONS

		return self
	# END FUNCTION add_option

	# BEGIN FUNCTION remove_option
	def remove_option(self, f : Hashable) -> 'Filter':
		"""Neatly removes an existing filter option
		does nothing if the given filter does not exist
		give filter by its attribute name, check `hasattr(self, f)`
		does not update self.idx
		"""
		if hasattr(self, f): # double check we can do this first
			delattr(self, f) # actual meat

			# BEGIN REMOVE FROM FILTER OPTIONS
			for k,v in self.filter_options.items():
				if isinstance(v, list): # if multiple filters for same `by`
					for n,i in enumerate(v):
						if i[0] == f:
							del self._filter_options[k][n]
							break
					if len(self.filter_options[k]) == 1: self._filter_options[k] = self._filter_options[k][0]
					elif len(self.filter_options[k]) == 0: del self._filter_options[k]
					break
				elif v[0] == f: # if only filter for `by`
					del self._filter_options[k]
			# END REMOVE FROM FILTER OPTIONS

		return self
	# END FUNCTION remove_option

	# BEGIN FILTER METHODS
	# BEGIN FUNCTION external
	def external(self, by : Hashable, func : Callable[[Any], Union[pd.Series, np.ndarray]], *args, **kwargs) -> 'Filter':
		"""Implements external filters
		Notifies target back by self.target._notify(self)

		Parameters
		----------
		by : Hashable
			the value to get from this.target
		func : callable
			the external filter function
			takes as first positional argument the data from this.target[by]
			returns a boolean mask of the same len as this.idx
				either a Series with matching index as this.idx
					or an array in the same order
		*args, **kwargs
			passed to func after this.target[by]

		Returns self
		"""
		# CALCULATE AND FILTER
		idx = func(self.target[by], *args, **kwargs)
		self._idx = np.logical_and(self._idx, idx)

		# Get name for self.filters
		cutoff =  func.__name__ + '(X, ' + ', '.join([str(i) for i in args] if isinstance(args, tuple) else [str(args)]) + ','.join([str(k)+'='+str(v) for k,v in kwargs.items()]) + ')'
		self._applied_filters.append((by, 'external', cutoff))
		# END CALCULATE AND FILTER

		# Notify and return
		if hasattr(self.target, '_notify'): self.target._notify(self)
		return self
	# END FUNCTION external

	# BEGIN FUNCTION bimodal
	def bimodal(self, by : Hashable, direction : str = 'above', method : str = 'minimum', **kwargs) -> 'Filter':
		"""Uses a bimodal distribution and keeps only one of the distribution

		Parameters
		----------
		by : Hashable
			the value to get from the target
		direction : str in ['above', 'below']
			which distribution to KEEP
		method : str = 'minimum'
			which skimage.filters.threshold_`...` method to use
		**kwargs
			passed to skimage.filters.threshold_`method`

		Returns self
		"""
		# BEGIN VALIDATION
		if not isinstance(direction, str): raise TypeError('direction must be \'above\' or \'below\'')
		elif not direction in ['above', 'below']: raise ValueError('direction must be \'above\' or \'below\'')
		if not method in ['isodata', 'li', 'local', 'mean', 'minimum', 'niblack', 'otsu', 'sauvola', 'triangle', 'yen']:
			raise ValueError('method must be a valid skimage.filters.threshold_... function')
		else: exec('from skimage.filters import threshold_'+method)

		if isinstance(by, str) and hasattr(self.target, by): data = getattr(self.target, by)
		elif self._isiter: data = self.target[by]
		else: raise RuntimeError('target does not have an attribute/key ' + str(by))
		# END VALIDATION

		# Get threshold
		th = eval('threshold_'+method)(data, **kwargs)

		# BEGIN FILTER
		idx = data >= th if direction == 'above' else data <= th
		self._idx = np.logical_and(self._idx, idx)
		self._applied_filters.append((by, 'bimodal', (direction, method)))
		# END FILTER

		# Notify and return
		if hasattr(self.target, '_notify'): self.target._notify(self)
		return self
	# END FUNCTION bimodal

	# BEGIN FUNCTION linear
	def linear(self, by : Hashable, cutoff : Union[Number, Iterable[Number]] , direction : str = 'above') -> 'Filter':
		"""A linear cutoff

		Parameters
		----------
		by : Hashable
			the value to get from the target
		cutoff : Number, Iterable[Number, Number]
			if Iterable, (low, high) - keep values in between
			if Number, see direction for outcome
		direction : str in ['above', 'below']
			if cutoff is Number, which direction to KEEP
			ignored if cutoff is Iterable

		Returns self
		"""
		# BEGIN VALIDATE cutoff
		if isinstance(cutoff, (int,float)):
			if not isinstance(direction, str): raise TypeError('direction must be \'above\' or \'below\'')
			elif not direction in ['above', 'below']: raise ValueError('direction must be \'above\' or \'below\'')
			else:
				# BEGIN HANDLE int/float(cutoff)
				if direction == 'above': cutoff = (cutoff, np.inf)
				elif direction == 'below': cutoff = (-np.inf, cutoff)
				else: pass # unreachable
				# END HANDLE int/float(cutoff)
		elif not isinstance(cutoff, Iterable) or len(cutoff) != 2 or not all([isinstance(i, (int,float)) for i in cutoff]):
			raise TypeError('cutoff must be int/float or Iterable[int/float, int/float] as (low, high)')
		elif cutoff[1] < cutoff[0]: cutoff = (cutoff[1], cutoff[0]) # reverse if it's backwards
		else: pass # verify cutoff
		# END VALIDATE cutoff

		# BEGIN VALIDATE by AND CALCULATE
		if isinstance(by, str) and hasattr(self.target, by):
			idx = np.logical_and(getattr(self.target, by) >= cutoff[0], getattr(self.target, by) <= cutoff[1])
		elif self._isiter: idx = np.logical_and(self.target[by] >= cutoff[0], self._target[by] <= cutoff[1])
		else: raise RuntimeError('target does not have an attribute/key ' + str(by))
		# BEGIN VALIDATE by AND CALCULATE

		# BEGIN FILTER
		self._idx = np.logical_and(self._idx, idx)
		self._applied_filters.append((by, 'linear', cutoff))
		# END FILTER

		# Notify and return
		if hasattr(self.target, '_notify'): self.target._notify(self)
		return self
	# END FUNCTION linear

	# BEGIN FUNCTION secant
	def secant(self, by : Hashable, cutoff : Number, direction : str = 'above') -> 'Filter':
		"""A secant-distance cutoff
		Based on methods from the Lau lab

		Parameters
		----------
		by : Hashable
			the value to get from this.target
		cutoff : Number in [0,100] = 0
			the percentile of additional cells to keep
			15-25 is recommended, but default is 0
			values less than 1 are treated as a fraction and multiplied by 100
		direction : str in ['above', 'below'] = 'above'
			the direction for the maximum secant distance to move the threshold
			if 'above', takes more points
			if 'below', takes fewer points

		Returns self
		"""
		# BEGIN VALIDATE cutoff
		if isinstance(cutoff, (int,float)):
			if cutoff < 1: cutoff *= 100 # move to right range
			if cutoff < 0 or cutoff > 100: raise ValueError('cutoff must be an float/int in [0,100] or [0,1]')
		else: raise TypeError('cutoff must be an int in [0,100]')
		# END VALIDATE cutoff
		
		# BEGIN VALIDATE by AND GET DATA
		if isinstance(by, str) and hasattr(self.target, by): data = getattr(self.target, by)
		elif self._isiter: data = self.target[by]
		else: raise RuntimeError('target does not have an attribute/key ' + str(by))
		# END VALIDATE by AND GET DATA

		# BEGIN CALCULATE
		sorted_idx = getattr(self.target, by).values.argsort()[::-1]
		cum = np.cumsum(data[sorted_idx].values)
		x = np.arange(0, len(data))
		sec = x * cum[len(data)-1]/len(data)
		dist = np.abs(cum-sec)
		d0 = dist.argmax()
		cut = np.percentile(dist, 100-cutoff)
		idx = (d0 + (dist[d0:] <= cut).argmax()) if direction == 'above' else (dist[:d0] >= cut).argmax()
		# END CALCULATE

		# BEGIN FILTER
		keep = np.zeros(len(data)).astype(bool) # all False
		keep[sorted_idx[:idx]] = True # set those who come before the cutoff to True
		self._idx = np.logical_and(self._idx, keep)
		self._applied_filters.append((by, 'secant', (direction, cutoff)))
		# END FILTER

		# Notify and return
		if hasattr(self.target, '_notify'): self.target._notify(self)
		return self
	# END FUNCTION secant
	# END FILTER METHODS
	# END METHODS

# END CLASS Filter




# BEGIN COLORS
# BEGIN COLOR LISTS
# stolen directly from scanpy

# Colorblindness adjusted vega_10
# See https://github.com/theislab/scanpy/issues/387
VEGA_10 = list(map(mpl.colors.to_hex, mpl.cm.tab10.colors))	
VEGA_10[2] = '#279e68'  # green
VEGA_10[4] = '#aa40fc'  # purple
VEGA_10[8] = '#b5bd61'  # kakhi
VEGA_10 = np.array(VEGA_10)

# see 'category20' on https://github.com/vega/vega/wiki/Scales#scale-range-literals
VEGA_20 = list(map(mpl.colors.to_hex, mpl.cm.tab20.colors))

# reorderd, some removed, some added
VEGA_20 = [
    *VEGA_20[0:14:2], *VEGA_20[16::2],  # dark without grey
    *VEGA_20[1:15:2], *VEGA_20[17::2],  # light without grey
    '#ad494a', '#8c6d31',  # manual additions
]
VEGA_20[2] = VEGA_10[2]
VEGA_20[4] = VEGA_10[4]
VEGA_20[7] = VEGA_10[8]  # kakhi shifted by missing grey
VEGA_20 = np.array(VEGA_20)

# https://graphicdesign.stackexchange.com/questions/3682/where-can-i-find-a-large-palette-set-of-contrasting-colors-for-coloring-many-d
# orig reference http://epub.wu.ac.at/1692/1/document.pdf
ZEILEIS_28 = np.array([
    "#023fa5", "#7d87b9", "#bec1d4", "#d6bcc0", "#bb7784", "#8e063b", "#4a6fe3",
    "#8595e1", "#b5bbe3", "#e6afb9", "#e07b91", "#d33f6a", "#11c638", "#8dd593",
    "#c6dec7", "#ead3c6", "#f0b98d", "#ef9708", "#0fcfc0", "#9cded6", "#d5eae7",
    "#f3e1eb", "#f6c4e1", "#f79cd4",
    '#7f7f7f', "#c7c7c7", "#1CE6FF", "#336600",  # these last ones were added
])

# from http://godsnotwheregodsnot.blogspot.de/2012/09/color-distribution-methodology.html
GODSNOT_102 = np.array([
    # "#000000",  # remove the black, as often, we have black colored annotation
    "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059", # 7
    "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87", # 11 x 8
    "#5A0007", "#809693", "#6A3A4C", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
    "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
    "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
    "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
    "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
    "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
    "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
    "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
    "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
    "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
    "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", # 7
]) # 2*7 + 11*8 = 14 + 88 = 102
# END COLOR LISTS

# BEGIN FUNCTION colors
def colors(c : Union[np.ndarray, pd.Series, pd.DataFrame, Iterable, Mapping]) -> Union[List[str], np.ndarray, pd.Series, dict]:
	"""Return a list of distinguishable hex color values
	Uses the smallest possible palette to maximize differentiability
	Palettes stolen from scanpy, which accounted for colorblindness
	Up to 102 different colors

	Parameters
	----------
	c : np.array, pd.Series, pd.DataFrame, Iterable, Mapping
		each unique entry is given a separate color
		repeats are given the same color
		flattens all input via array.flatten()

		np.array returns an np.array
		pd.Series, pd.DataFrame return pd.Series
		all other Iterables are returned as list
		all Mappings are returned as dict

	Returns
	-------
	list[str], np.ndarray[str], pd.Series[str]
		the hex colors for each entry as given
	"""
	# Validate
	if not isinstance(c, (pd.Series, pd.DataFrame, np.ndarray, Iterable, Mapping)): raise TypeError('c must be a list, np.ndarray, or pd.Series')

	# BEGIN PREPREPARATION
	if isinstance(c, Mapping):
		# BEGIN HANDLE Mapping(c)
		c = np.array([[k,v] for k,v in c.items()])
		k, d = c.T.tolist()
		d = np.array(d)
		c = {i[0]:i[1] for i in c}
		# END HANDLE Mapping(c)
	elif not isinstance(c, (list, np.ndarray, pd.Series, pd.DataFrame)): c = list(c) # for weird c
	# END PREPREPARATION

	# BEGIN PREPARATION
	if isinstance(c, Mapping): pass # d is set above as array
	elif isinstance(c, (pd.Series, pd.DataFrame)): d = c.values # d is array
	elif isinstance(c, list): d = np.array(c) # d is still an array
	else: d = c # c is already an array
	d = d.flatten() # guaranteed to work
	# END PREPARATION

	# BEGIN GET COLORS
	u, g = np.unique(d, return_inverse=True)
	if len(u) > 102: raise NotImplementedError('colors can only handle up to 102 unique entries')
	elif len(u) > 28: ret = GODSNOT_102[g]
	elif len(u) > 20: ret = ZEILEIS_28[g]
	elif len(u) > 10: ret = VEGA_20[g]
	else: ret = VEGA_10[g]
	# END GET COLORS

	# BEGIN RETURN
	if isinstance(c, list): return ret.tolist() # if given list/Iterable, return list
	elif isinstance(c, (pd.DataFrame, pd.Series)): return pd.Series(ret, index=c.index) # if given pandas, return pandas
	elif isinstance(c, dict): return {k:v for k,v in zip(k, ret)} # if given Mapping, return dict
	else: return ret # otherwise was array, return array
	# END RETURN
# END FUNCTION colors

# BEGIN INTERNAL FUNCTIONS
# BEGIN INTERNAL FUNCTION
def _luminance(color):
	"""Calculate the luminance of a trusted color"""
	r,g,b,_ = tuple(mpl.colors.to_rgba_array(color).flatten()) # get as rgb in [0,1]

	Y = 0.2126*r + 0.7152*g + 0.0722*b # calculate CIE XYZ luminance Y/Yn in [0,1]
	# see https://en.wikipedia.org/wiki/SRGB#The_reverse_transformation

	L = 116*(np.cbrt(Y) if Y > (6/29)**3 else Y/(3*(6/29)**2) + 4/29) - 16 # CIELAB lightness L* in [0,100]
	# see https://en.wikipedia.org/wiki/CIELAB_color_space#RGB_and_CMYK_conversions
	return L
# END INTERNAL FUNCTION
# END INTERNAL FUNCTIONS

ANNOT_PREF = {
	# Preferences for utils functions used for annotations
	'light_dark': 75, # the luminosity percent at which you which from 'light_annot' above to 'dark_annot' below
	'light_annot': 'w', # the lighter color that the annot can be
	'dark_annot': 'k', # the darker color that the annot will be
	'light_background' : 'w', # the lighter background color of the annot, used with 'dark_annot'
	'dark_background' : 'k', # the darker background color of the annot, used with 'light_annot'
	'light_bg_alpha' : 0, # the alpha for 'light_background'
	'dark_bg_alpha' : 0.5, # the alpha for 'dark_background'
}

# BEGIN FUNCTION set_annot_pref
def set_annot_pref(pref : Mapping) -> None:
	"""To set ANNOT_PREF, which controls some elements of annotation

	Parameters
	----------
	pref : Mapping
		'light_dark' : 0 <= Number <= 100
			the luminosity above which dark annotations are used and below it for light annotations
		'light_annot' : color
			valid matplotlib color that the annotation text is on a dark background
		'dark_annot' : color
			valid matplotlib color that the annotation text is on a light background
		'light_background' : color
			valid matplotlib color for the lighter backgound used under 'dark_annot'
		'dark_background' : color
			valid matplotlib color for the darker background used under 'light_annot'
		'light_bg_alpha' : Number in [0,1] or [1,100]
			alpha value for 'light_background'
			[1,100] divided by 100 into [0,1]
		'dark_bg_alpha' : Number in [0,1] or [1,100]
			alpha value for 'dark_background'
			[1,100] divided by 100 into [0,1]

	"""
	if not isinstance(pref, Mapping): raise TypeError('pref must be a Mapping')	
	if 'light_dark' in pref:
		if not isinstance(pref['light_dark'], (int,float)): raise TypeError('pref[\'light_dark\'] must be a Number in [0,100]')
		elif 0 <= pref['light_dark'] and 100 >= pref['light_dark']: raise ValueError('pref[\'light_dark\'] must be a Number in [0,100]')
	if 'light_annot' in pref and not mpl.colors.is_color_like(pref['light_annot']): raise ValueError('pref[\'light_annot\'] must be a color')
	if 'dark_annot' in pref and not mpl.colors.is_color_like(pref['dark_annot']): raise ValueError('pref[\'dark_annot\'] must be a color')
	if 'light_background' in pref and not mpl.colors.is_color_like(pref['light_background']):
		raise ValueError('pref[\'light_background\'] must be a color')
	if 'dark_background' in pref and not mpl.colors.is_color_like(pef['dark_background']):
		raise ValueError('pref[\'dark_background\'] must be a color')
	if 'light_bg_alpha' in pref:
		if not isinstance(pref['light_bg_alpha'], (int,float)): raise TypeError('pref[\'light_bg_alpha\'] must be a Number in [0,1] or [0,100]')
		elif 0 < pref['light_bg_alpha']: raise ValueError('pref[\'light_bg_alpha\'] must be a Number in [0,1] or [0,100]')
		# must be a break here
		if pref['light_bg_alpha'] > 1: pref['light_bg_alpha'] /= 100 # if [0,100] scale to [0,1]
		if pref['light_bg_alpha'] > 1: raise ValueError('pref[\'light_bg_alpha\'] must be a Number in [0,1] or [0,100]')
		else: pass # validated
	if 'dark_bg_alpha' in pref:
		if not isinstance(pref['dark_bg_alpha'], (int, float)): raise TypeError('pref[\'dark_bg_alpha\'] must be a Number in [0,1] or [0,100]')
		elif 0 < pref['dark_bg_alpha']: raise ValueError('pref[\'dark_bg_alpha\'] must be a Number in [0,1] or [0,100]')
		# must be a break here
		if pref['dark_bg_alpha'] > 1: pref['dark_bg_alpha'] /= 100
		if pref['dark_bg_alpha'] > 1: raise ValueError('pref[\'dark_bg_alpha\'] must be a Number in [0,1] or [0,100]')
		else: pass # validated

	global ANNOT_PREF
	ANNOT_PREF.update(pref)
# END FUNCTION set_annot_pref

# BEGIN FUNCTION annot_color
def annot_color(color):
	"""Returns the color argument for an annotation on a background of the given color

	Parameters
	----------
	color : str
		valid matplotlib color of the background

	Returns
	-------
	color : str
		the color argument to give to annotate() on data of c=color
		valid matplotlib color
	"""
	if not mpl.colors.is_color_like(color): raise ValueError('must pass a color')
	return ANNOT_PREF['light_annot'] if _luminance(color) <= ANNOT_PREF['light_dark'] else ANNOT_PREF['dark_annot']
# END FUNCTION annot_color

# BEGIN FUNCTION annot_bbox
def annot_bbox(color):
	"""Returns the bbox argument for the annotation given the color

	Parameters
	----------
	color : str
		valid matplotlib color of the background

	Returns
	-------
	bbox : dict
		the bbox argument to give to annotate() if passing color=color
		kwargs for matplotlib.patches.Rectangle
	"""
	if not mpl.colors.is_color_like(color): raise ValueError('must pass a color')

	return {'facecolor':ANNOT_PREF['dark_background'], 'alpha':ANNOT_PREF['dark_bg_alpha']} if _luminance(color) <= ANNOT_PREF['light_dark'] else \
			{'facecolor':ANNOT_PREF['light_background'], 'alpha':ANNOT_PREF['light_bg_alpha']}
# END FUNCTION annot_bbox
# END COLORS