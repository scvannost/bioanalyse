from frozendict import frozendict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
from warnings import warn
from umap import UMAP

# for Sample.plot()
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import ImageGrid

# for typing
from typing import Any, Callable, Hashable, IO, Iterable, Mapping, Tuple, Optional, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
Number = Union[int,float]

# for bioanalyse
import utils

# UNIVERSALS
HOUSEKEEPING_SPECIES = ['homo', 'mus']


# BEGIN CLASS Sample
class Sample(utils.SampleBase):
	"""A single single-cell RNA sequencing sample, after all data processing

	Parameters
	----------
	data : pd.DataFrame
		the log-scaled, normalized, etc expression data
	*
	sparse : bool = True
		whether or not to use a sparse datatype
		handles kwargs 'index_col' internally if passed
	fill_value : float = 0
		if sparse, the value to use as the fill_value
		ignored otherwise
	minimal : bool = False
		whether to only store minimal data; self.layers will always be empty
		leads to a lot of calculating on the fly, increasing computation time
	skip_validation : bool = False
		whether to skip validating anything
	verbose : bool = False
		Whether or not to print what's happening
	pca_ : PCA = None
		a PCA instance fit to data
	umap_ : UMAP = None
		a UMAP instance fit to data
	tnse_ : TSNE = None
		a TSNE instance fit to data

	
	Class Methods
	-------------
	from_csv()
		load an scRNAseq sample from a csv file
	from_pickle()
		load an scRNAseq sample from a pickled pd.DataFrame


	Values can be retrieved from data/expression by:
		column(s): self['d'/'e', key]
		row(s) by name: self['d'/'e', key] if key is not a column
		cell(s): self['d'/'e', key1, key2] by any combination of loc/iloc
	Shape along an axis is either self.shape[axis] or self['shape', axis]
	All self.layers are also available as keys; eg self.layers['pca'] == self['pca']


	Selected Attributes
	-------------------
	data, d : pd.DataFrame
		the data as given
	shape : np.ndarray
		the shape of self.data
	expression, e : pd.DataFrame
		equivalent to self.data
	layers : {}
		calculated values stored for easy recall
		keys are the function names that calculated the values
		entries can also be accessed as simply self[key]
		remove layers using self.wipe_layer(key) to run again with different parameters
	"""
	# BEGIN DUNDER FUNCTIONS
	# BEGIN FUNCTION __init__
	def __init__(self, data : pd.DataFrame, *, sparse : bool = True, fill_value : float = 0, minimal : bool = False, skip_validation : bool = False, verbose : bool = False, pca_ : Optional[PCA] = None, umap_ : Optional[UMAP] = None, tsne_ : Optional['sklearn.manifold.TSNE'] = None, **kwargs) -> None:
		"""Initializes the container"""
		# BEGIN VALIDATION
		if not isinstance(skip_validation, bool) or not skip_validation:
			if verbose: print('Validating...')
			if not isinstance(data, pd.DataFrame): raise TypeError('data must be a pd.DataFrame')
			elif not isinstance(sparse, bool): raise TypeError('sparse must be a bool')
			elif sparse and not isinstance(fill_value, (float, int)): raise TypeError('fill_value must be a float if sparse')
			elif not isinstance(minimal, bool): raise TypeError('minimal must be a bool')
			else: pass

			if not minimal:
				if pca_ is not None and not isinstance(pca_, PCA): raise TypeError('pca_ must be a sklearn.decomposition.PCA if given')
				if umap_ is not None and not isinstance(umap_, UMAP): raise TypeError('umap_ must be a umap.UMAP if given')
				if tnse_ is not None:
					from sklearn.manifold import TSNE
					if not isinstance(tsne_, TSNE): raise TypeError('tsne_ must be a sklearn.manifold.TSNE if given')
		# END VALIDATION

		# BEGIN SET UP
		if verbose: print('Building structure...')
		self._minimal = minimal
		self._data = data if not sparse else data.astype(pd.SparseDtype(float,fill_value=fill_value))
		self._pca_ = pca_ if not minimal else None
		self._umap_ = umap_ if not minimal else None
		self._tsne_ = tsne_ if not minimal else None
		self._layers = {}
		# END SET UP
	# END FUNCTION __init__

	# BEGIN FUNCTION __repr__
	def __repr__(self) -> str:
		return 'Sample(shape={0})'.format(self.shape)
	# END FUNCTION __repr__

	# BEGIN FUNCTION __contains__
	def __contains__(self, key) -> bool:
		ret = key in ['data', 'expression', 'd', 'e', 'layers', 'shape'] # these are special
		if ret: return ret # if True, say so

		else: # since self.layers is a dict and `~Hashable in dict` fails
			try: return key in self.layers # can return from here too
			except: return False
	# END FUNCTION __contains__

	# BEGIN FUNCTION __getitem__
	def __getitem__(self, key) -> Any:
		if isinstance(key, str):
			# for faster routing instead of try/catch in _getitem
			if key in ['data', 'd', 'e', 'expression']: return self.data
			elif key == 'layers': return self.layers
			elif key == 'shape': return self.data.shape

			# to handle returning for self.layers
			elif key in self.layers: return self.layers[key]
			else: pass

		# default otherwise
		return super().__getitem__(key)
	# END FUNCTION __getitem__
	# END DUNDER FUNCTIONS

	# BEGIN PROPERTIES
	@property
	def data(self) -> pd.DataFrame:
		"""The data as given"""
		return self._data
	@property
	def idx(self):
		"""The index of the data as given"""
		return self._data.index
	@property
	def columns(self):
		"""The columns of the data as given"""
		return self._data.columns
	@property
	def shape(self) -> Tuple[int]:
		"""The shape of the given data"""
		return self.data.shape
	@property
	def expression(self) -> pd.DataFrame:
		"""Same as self.data"""
		return self.data
	@property
	def layers(self) -> Mapping[str, Any]:
		"""Any calculated values"""
		return self._layers
	# @property
	# def classifier(self) -> _Classifier:
	# 	"""The classifier associated with this Sample"""
	# 	return self._classifier
	@property
	def pca_(self) -> PCA:
		"""The PCA object associated with this Sample"""
		return self._pca_
	@property
	def umap_(self) -> UMAP:
		"""The UMAP object associated with this Sample"""
		return self._umap_
	@property
	def tsne_(self) -> 'sklearn.manifold.TSNE':
		"""The TSNE object associated with this Sample"""
		return self._tsne_
	# END PROPERTIES

	# BEGIN PROPERTY CONTROL METHODS
	# BEGIN FUNCTION wipe_layer
	def wipe_layer(self, key : str) -> Union['Sample', 'Processor']:
		"""Removes a layer from self.layers
		Returns self"""
		if key in self.layers:
			del self._layers[key]
		return self
	# END FUNCTION wipe_layer
	# BEGIN PROPERTY CONTROL METHODS

	# BEGIN CLASS METHODS
	# BEGIN FUNCTION from_csv
	@classmethod
	def from_csv(cls, file : Union[str, IO], *, sparse : bool = True, minimal : bool = False, skip_validation : bool = False, verbose : bool = False, **kwargs) -> 'Sample':
		"""Load a sample directly from a csv
		
		Parameters
		----------
		file : str, filelike
			the file containing the data
			rows are cells, columns are genes
			column name should be gene symbols
		*
		sparse : bool = True
			whether or not to use a sparse datatype
			handles the following kwargs internally:
				sep : str = ','
				header : int, Iterable[int] = 0 # NotImplementedError
				index_col : int, Iterable[int] = (0,1)
				usecols : int, str, Iterable[int,str] = None
				dtype : type, str = 'float16'
				fill_value : float = 0
		minimal : bool = False
			whether to only store minimal data; self.layers will always be empty
			leads to a lot of calculating on the fly, increasing computation time
		skip_validation : bool = False
			whether to skip validating anything
		verbose : bool = False
			Whether or not to print what's happening
		**kwargs
			passed to pd.read_csv if not sparse
		"""
		if not isinstance(skip_validation, bool) or not skip_validation:
			if verbose: print('Validating...')
			if not isinstance(sparse, bool): raise TypeError('sparse must be a bool')
			elif sparse and not isinstance(fill_value, (float, int)): raise TypeError('fill_value must be a float if sparse')
			elif not isinstance(minimal, bool): raise TypeError('minimal must be a bool')
			elif not (isinstance(file,str) or hasattr(file, 'readline')): raise TypeError('file must be str or file-like')
			else: pass

		if 'index_col' not in kwargs: kwargs['index_col'] = 0
		if verbose: print('Loading...')
		if sparse:
			sep = kwargs['delimiter'] if 'delimiter' in kwargs else kwargs['sep'] if 'sep' in kwargs else ','
			header = kwargs['header'] if 'header' in kwargs else [0]
			usecols = kwargs['usecols'] if 'usecols' in kwargs else None
			
			dtype = kwargs['dtype'] if 'dtype' in kwargs else 'float16'
			fill_value = kwargs['fill_value'] if 'fill_value' in kwargs else 0
			sparse_dtype = pd.SparseDtype(dtype,fill_value=fill_value)

			index_col = kwargs['index_col']
			if isinstance(index_col, Iterable): index_col = list(index_col)
			else: index_col = [index_col]

			df = []
			if not hasattr(file, 'readline'): f = open(file, 'r')
			else: f = file

			temp = np.array(f.readline().rstrip().split(sep))
			dat = [i for i in range(len(temp)) if i not in idx]
			temp = pd.Series(temp[dat], name=tuple(temp[idx].tolist()))

			for c in f:
				d = np.array(c.split(sep))
				df.append(pd.Series(d[dat], name=tuple(d[idx].tolist())).astype(sparse_dtype))
				
			if not hasattr(file, 'readline'): f.close()
			df = pd.concat(df, axis=1, copy=False).T
			df.columns = temp

			if verbose:
				print('Cells:', df.shape[0])
				print('Genes:', df.shape[1])
				print('Building structure...')
			return cls(df, minimal=minimal, skip_validation=skip_validation, verbose=verbose)
		else:
			df = pd.read_csv(file, **{i:kwargs[i] for i in utils.CSV_KWARGS if i in kwargs})
			if verbose:
				print('Cells:', data.shape[0])
				print('Genes:', data.shape[1])
				print('Building structure...')
			return cls(df, minimal=minimal, skip_validation=skip_validation, verbose=verbose)
	# END FUNCTION from_csv

	# BEGIN FUNCTION from_pickle
	@classmethod
	def from_pickle(cls, file : Union[str, IO], *, sparse : bool = True, minimal : bool = False, skip_validation : bool = False, verbose : bool = False) -> 'Sample':
		"""Load a sample directly from a pickled pd.DataFrame

		Parameters
		----------
		file : str, filelike
			the file containing the data
			rows are cells, columns are genes
			column name should be gene symbols
		*
		sparse : bool = True
			whether or not to use a sparse datatype
			handles the following kwargs internally:
				dtype : type, str = 'float16'
				fill_value : float = 0
		minimal : bool = False
			whether to only store minimal data; self.layers will always be empty
			leads to a lot of calculating on the fly, increasing computation time
		skip_validation : bool = False
			whether to skip validating anything
		verbose : bool = False
			Whether or not to print what's happening
		"""
		from pickle import load

		if not isinstance(skip_validation, bool) or not skip_validation:
			if verbose: print('Validating...')
			if not isinstance(sparse, bool): raise TypeError('sparse must be a bool')
			elif sparse and not isinstance(fill_value, (float, int)): raise TypeError('fill_value must be a float if sparse')
			elif not isinstance(minimal, bool): raise TypeError('minimal must be a bool')
			elif not (isinstance(file,str) or hasattr(file, 'readline')): raise TypeError('file must be str or file-like')
			else: pass


		if verbose: print('Loading...')
		if hasattr(file, 'read') and hasattr(file, 'readline'): data = load(file)
		else:
			with open(file, 'rb') as f: data = load(f)

		if sparse:
			dtype = kwargs['dtype'] if 'dtype' in kwargs else 'float16'
			fill_value = kwargs['fill_value'] if 'fill_value' in kwargs else 0
			sparse_dtype = pd.SparseDtype(dtype,fill_value=fill_value)

			data = data.astype(sparse_dtype)

		if verbose:
			print('Cells:', data.shape[0])
			print('Genes:', data.shape[1])
			print('Building structure...')
		
		return cls(data, sparse=sparse, fill_value=fill_value, minimal=minimal, skip_validation=skip_validation, verbose=verbose)
	# END FUNCTION from_pickle
	# END CLASS METHODS


	# BEGIN METHODS
	# BEGIN TRANSFORMATIONS
	# BEGIN FUNCTION pca
	def pca(self, verbose : bool = False, **kwargs) -> pd.DataFrame:
		"""Returns the PCA scores of the expression data
		Compare self.pca_loadings, self.init_pca

		For full kwargs, see documentation sklearn.decomposition.PCA

		Selected kwargs
		---------------
		verbose : bool = False
			Whether or not to print what's happening
			Internal
		n_components : int,float = None
			if >= 1, the number of components to keep
			elif < 1, the fraction of total variance to keep
			if None, all components are kept

		Returns
		-------
		pd.DataFrame
			the scores for each sample on each PC
			columns/PCs have names as 'PC1', 'PC2', ...
			the index is the same as self.data.index
		"""
		# Return if known
		if 'pca' in self.layers:
			if verbose: print('Fetching...')
			return self.layers['pca']

		# Transform
		if self.pca_ is not None:
			if verbose: print('Calculating with existing PCA instance...')
			scores = self.pca_.transform(self['expression'])
		else:
			if verbose: print('Training...')
			pca = PCA(**kwargs).fit(self['expression'])
			scores = pca.transform(self['expression'])
			self._pca_ = pca
		ret = pd.DataFrame(scores, index=self.data.index)

		# Save and return
		if not self._minimal: self._layers['pca'] = ret
		return ret
	# END FUNCTION pca

	# BEGIN FUNCTION umap
	def umap(self, verbose : bool = False, **kwargs) -> pd.DataFrame:
		"""Runs PCA and returns the UMAP embeddings of the PCA scores
		Uses self.pca_.transform if it exists, otherwise runs PCA internally
		See also self.umap()

		For full PCA kwargs, see sklearn.decomposition.PCA
		For full UMAP kwargs, see umap-learn.readthedocs.io

		Special kwargs
		--------------
		verbose : bool = False
			Whether or not to print what's happening
			Internal
		n_components : int,float = 50
			if >= 1, the number of PCA components to keep
			elif < 1, the fraction of total variance to keep
			if None, all components are kept
		n_neighbors : int = np.sqrt(self.shape[0]).astype(int)
			size of local neighborhood in number of neighboring points
			larger values result in more global view, while smaller preserves local structure
			in general, [2,100]



		pca_components : always given to PCA
		n_components : equivalent to but overridden by pca_components
		umap_components : always given to UMAP

		umap_random_state : always given to UMAP
		pca_random_state : always given to PCA
		random_state
			passed to UMAP through sklearn.utils.check_random_state, but overridden by umap_random_state
			default None


		Returns
		-------
		pd.DataFrame
			the location of each sample on each embedded axis
			index is the same as self.data.index
		"""
		# Return if known
		if 'umap' in self.layers:
			if verbose: print('Fetching...')
			return self.layers['umap']

		# BEGIN HANDLE KWARGS FOR PCA
		if 'n_components' not in kwargs: kwargs['n_components'] = kwargs['pca_components'] if 'pca_components' in kwargs else 50
		if 'umap_random_state' not in kwargs:
			kwargs['umap_random_state'] = check_random_state(kwargs.pop('random_state') if 'random_state' in kwargs else None)
		if 'pca_random_state' in kwargs: kwargs['random_state'] = check_random_state(kwargs['pca_random_state'])
		# END HANDLE KWARGS FOR PCA

		# PCA transform
		if verbose: print('PCA...')
		if self.pca_ is not None: scores = self.pca_.transform(self['expression'])
		else: scores = self.pca(**{k:kwargs[k] for k in kwargs if k in utils.PCA_KWARGS})

		# BEGIN HANDLE KWARGS FOR UMAP
		if 'n_neighbors' not in kwargs: kwargs['n_neighbors'] = np.sqrt(self.shape[0]).astype(int)
		if 'umap_components' in kwargs: kwargs['n_components'] = kwargs.pop('umap_components')
		elif 'n_components' in kwargs: del kwargs['n_components']
		else: pass
		if 'umap_random_state' in kwargs: kwargs['random_state'] = kwargs.pop('umap_random_state')
		elif 'random_state' in kwargs: del kwargs['random_state']
		else: pass
		# END HANDLE KWARGS FOR UMAP
		
		# UMAP transform
		if self.umap_ is not None:
			print('Calculating using existing UMAP instance...')
			ret = self.umap_.transform(scores)
		else:
			print('Transforming...')
			umap = UMAP(**{k:kwargs[k] for k in kwargs if k in utils.UMAP_KWARGS}).fit(scores)
			self._umap_ = umap
			ret = umap.transform(scores)

		ret = pd.DataFrame(ret, index=self.data.index)

		# Save and return
		if not self._minimal: self._layers['umap'] = ret
		return ret
	# END FUNCTION umap

	# BEGIN FUNCTION tsne
	def tsne(self, verbose : bool = False, **kwargs) -> pd.DataFrame:
		"""Runs PCA and returns the tSNE embeddings of the PCA scores
		Uses self.pca_.transform if it exists, otherwise runs PCA internally
		See also self.tsne()

		For full PCA kwargs, see sklearn.decomposition.PCA
		For full tSNE kwargs, see sklearn.manifold.TSNE

		Special kwargs
		---------------------------------------
		verbose : bool = False
			Whether or not to print what's happening
			Internal
		n_components : int,float = 50
			if >= 1, the number of PCA components to keep
			elif < 1, the fraction of total variance to keep
			if None, all components are kept
		tsne_components : int = 2
			the number of TSNE components to calculate
		learning_rate : float = 1000
			generally [10.0, 1000.0]
			if too high, approximately equal spacing between neighbors
			if too low, dense cloud with few outliers
		perplexity : float = np.sqrt(self.shape[0]).astype(int)
			related to n_neighbors
			larger datasets require larger perplexities, consider [5, 50]
		n_jobs : int = -1
			the number of parallel jobs to run for neighbors search
			None means 1, -1 means all
			only applicable if MulticoreTSNE is installed

		pca_components : always given to PCA
		n_components : always given to PCA, but overridden by pca_components
		tsne_components : always given to TSNE

		tsne_random_state : always given to TSNE
		random_state : always given to TSNE, but overridden by tsne_random_state
		pca_random_state : always given to PCA

		Returns
		-------
		pd.DataFrame
			the location of each sample on each embedded axis
			index is the same as self.data.index
		"""
		# Return if known
		if 'tsne' in self.layers:
			if verbose: print('Fetching...')
			return self.layers['tsne']

		# Get TSNE and handle n_jobs
		try:
			if verbose: print('Trying MulticoreTSNE import')
			from MulticoreTSNE import MulticoreTSNE as TSNE
			if 'n_jobs' not in kwargs: kwargs['n_jobs'] = -1
		except ImportError:
			warn('Consider downloading MulticoreTSNE, as it runs faster than sklearn.manifold.TSNE even for n_jobs=1')
			if 'n_jobs' in kwargs: del kwargs['n_jobs']
			from sklearn.manifold import TSNE

		# Internal defaults
		if 'learning_rate' not in kwargs: kwargs['learning_rate'] = 1000
		if 'perplexity' not in kwargs: kwargs['perplexity'] = np.sqrt(self.shape[0]).astype(int)

		# BEGIN HANDLE KWARGS FOR PCA
		if 'n_components' not in kwargs: kwargs['n_components'] = kwargs['pca_components'] if 'pca_components' in kwargs else 50
		if 'tsne_random_state' not in kwargs and 'random_state' in kwargs: kwargs['tsne_random_state'] = kwargs.pop('random_state')
		if 'pca_random_state' in kwargs: kwargs['random_state'] = kwargs['pca_random_state']
		# END HANDLE KWARGS FOR PCA
		
		# PCA transform
		if verbose: print('PCA...')
		if self.pca_ is not None: scores = self.pca_.transform(self['expression'])
		else: scores = self.pca(**{k:kwargs[k] for k in kwargs if k in utils.PCA_KWARGS})

		# BEGIN HANDLE KWARGS FOR TSNE
		if 'tsne_components' in kwargs: kwargs['n_components'] = kwargs['tsne_components']
		elif 'n_components' in kwargs: del kwargs['n_components']
		else: pass
		if 'tsne_random_state' in kwargs: kwargs['random_state'] = kwargs['tsne_random_state']
		elif 'random_state' in kwargs: del kwargs['random_state']
		else: pass
		# END HANDLE KWARGS FOR TSNE

		# TSNE transform
		if self.tsne_ is not None:
			if verbose: print('Calculating using existing TSNE instance...')
			ret = self.tsne_.transform(scores)
		elif 'n_jobs' not in kwargs: # not MulticoreTSNE has transform, so save and use
			if verbose: print('Transforming with sklearn...')
			tsne = TSNE(**{k:kwargs[k] for k in kwargs if k in utils.TSNE_KWARGS}).fit(scores)
			self._tsne_ = tsne
			ret = tsne.transform(scores)
		else: # MulticoreTSNE doesn't have transform, so have to run every time
			if verbose: print('Transforming with MulticoreTSNE...')
			ret = TSNE(**{k:kwargs[k] for k in kwargs if k in utils.TSNE_KWARGS}).fit_transform(scores)
		ret = pd.DataFrame(ret, index=self.data.index)

		# Save and return
		if not self._minimal: self._layers['tsne'] = ret
		return ret
	# END FUNCTION tsne
	# END TRANSFORMATIONS

	# BEGIN EXTRA PCA METHODS
	# BEGIN FUNCTION pca_loadings
	def pca_loadings(self, verbose : bool = False, **kwargs) -> pd.DataFrame:
		"""Returns the PCA loadings of the expression data
		Compare self.pca, self.init_pca

		For full kwargs, see documentation sklearn.decomposition.PCA

		Selected kwargs
		---------------
		verbose : bool = False
			Whether or not to print what's happening
			Internal
		n_components : int,float = None
			if >= 1, the number of components to keep
			elif < 1, the fraction of total variance to keep
			if None, all components are kept

		Returns
		-------
		pd.DataFrame
			the loadings for each PC in feature space
			rows/PCs have names as 'PC1', 'PC2', ...
			columns are the same as self.columns
		"""
		# Return if known
		if 'pca_loadings' in self.layers:
			if verbose: print('Fetching...')
			return self.layers['pca_loadings']

		# Calculate
		if self.pca_ is not None:
			if verbose: print('Fetching...')
			loads = self.pca_.components_
		else:
			if verbose: print('Training PCA...')
			if 'pca_components' in kwargs: kwargs['n_components'] = kwargs['pca_components']
			pca = PCA(**{k:kwargs[k] for k in utils.PCA_KWARGS if k in kwargs}).fit(self['expression'])
			loads = pca.components_
			self._pca_ = pca

		# Save and return
		ret = pd.DataFrame(loads)
		if not self._minimal: self._layers['pca_loadings'] = ret
		return ret
	# END FUNCTION pca_loadings

	# BEGIN FUNCTION init_pca
	def init_pca(self, verbose : bool = False, **kwargs) -> PCA:
		"""Creates self.pca_ trained with the expression data
		Call this function again to replace self.pca_, or use `del this.pca` to remove entirely
		Compare self.pca, self.pca_loadings

		For full kwargs, see documentation sklearn.decomposition.PCA

		Selected kwargs
		---------------
		verbose : bool = False
			Whether or not to print what's happening
			Internal
		n_components : int,float = None
			if >= 1, the number of components to keep
			elif < 1, the fraction of total variance to keep
			if None, all components are kept

		Returns self.pca_
		Selected Attributes
		-------------------
		components_
			loadings as [components, features]
		explained_variance_
			the amount of variance explained by each component
		explained_variance_ratio_
			the percent variance explained by each component
		transform(X)
			returns the scores for the given X
		inverse_transform(X)
			returns the values of the given X in the original space
		"""
		# Fit, save, and return
		if verbose: print('Training...')
		if not self._minimal:
			self._pca_ = PCA(**{k:kwargs[k] for k in utils.PCA_KWARGS if k in kwargs}).fit(self['expression'])
			return self.pca_
		else:
			return PCA(**{k:kwargs[k] for k in utils.PCA_KWARGS if k in kwargs}).fit(self['expression'])
	# END FUNCTION init_pca
	# END EXTRA PCA METHODS

	# # BEGIN FUNCTION classify
	# def classify(self, markers : Union[str, IO, pd.DataFrame], *, confidence : float = 0.95, gmm_components : int = 5, k_fit : int = 5, num_features_pca : int = 500, variance_to_keep : float = 0.95, verbose : bool = True) -> 'Processor':
	# 	"""Classify the cells using the given marker table

	# 	Parameters
	# 	----------
	# 	markers : str, file-like, pd.DataFrame
	# 		if pd.DataFrame: the marker table itself
	# 		else: the marker table file(name) in .csv format, passed to `pd.read_csv`
	# 		Each row is a class, each column is a marker feature
	# 		Each entry is one of: '', 'AND', 'NOT', np.nan
	# 		The class displays all 'AND' features and no 'NOT' features
	# 	*
	# 	confidence : float in (0,1] = 0.95
	# 		the confidence level at which to assign cell type labels
	# 		values of <0.5 are not recommended
	# 		values above 1 are treated as percents and divided by 100
	# 	gmm_components : int > 0 = 5
	# 		The maximum number of clusters to look for in the marker features
	# 		Must be positive
	# 	k_fit : int > 0 = 5
	# 		Number of folds to do when determining the number of clusters
	# 		Must be positive
	# 	num_features_pca : int  > 0 = 500
	# 		The number of highest variance features to use in classification
	# 		Must be positive
	# 	variance_to_keep : (int,float) in (0,1] = 0.95
	# 		The fraction of variance to keep during dimensionality-reduction
	# 		Must be >0 and ≤1
	# 	verbose : bool = True
	# 		Whether or not to print what's happening

	# 	Returns self
	# 	"""
	# 	# BEGIN VALIDATION
	# 	if not isinstance(confidence, (int,float)): raise TypeError('confidence must be a float in (0,1]')
	# 	elif confidence > 1: confidence /= 100
	# 	if confidence <= 0 or confidence > 1: raise ValueError('confidence must be a float in (0,1]')
	# 	# rest immediately checked by _Classifier.__init__()
	# 	# END VALIDATION

	# 	# BEGIN PREPARE CLASSIFIER
	# 	self._classifier = _Classifier(self, markers, key='expression', skip_validation=True, verbose=verbose,
	# 								params={'gmm_components':gmm_components, 'k_fit':k_fit,
	# 										'num_features_pca':num_features_pca, 'variance_to_keep':variance_to_keep})
	# 	self.classifier.train(verbose=verbose)
	# 	# END PREPARE CLASSIFIER

	# 	# BEGiN CLASSIFY
	# 	if verbose: print('Classifying...')
	# 	temp = self.classifier.classify(confidence=confidence)
	# 	labels = np.array(['Not classified']*self._data.shape[0])
	# 	labels[self.idx] = temp
	# 	# END CLASSIFY

	# 	# SAVE
	# 	self._data.index = pd.MultiIndex.from_arrays((labels, self._data.index))
	# # END FUNCTION classify
	
	# BEGIN FUNCTION cluster
	def cluster(self, *, leiden : bool = True, resolution : float = 2, verbose : bool = False, **kwargs) -> pd.Series:
		"""Returns Leiden or Louvain clustering
		Uses PCA and UMAP to find neighbors

		Parameters
		----------
		*
		leiden : bool = True
			whether to default to the Leiden algorithm if installed
			ignored if module `leidenalg` is not installed
		resolution : float = 2
			the density limit that defines clusters
			all clusters are guaranteed to have density >= resolution
			only applies if using Leiden
		verbose : bool = False
			Whether or not to print what's happening
		**kwargs
			passed variously to sklearn.decomposition.PCA, umap.umap_.fuzzy_simplical_set, leidenalg.find_partition
			extra kwargs ignored silently

		Returns
		-------
		pd.Series
			index is same as self.expression
			values are cluster membership

		Special kwargs
		--------------
		n_components : int = 50
			the number of components to reduce to
		n_neighbors : int = np.sqrt(self.expression.shape[0]).astype(int)
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
		# Return if known
		if 'cluster' in self.layers:
			if verbose: print('Fetching...')
			return self.layers['cluster']

		# Cluster
		ret = pd.Series(utils.cluster(self.expression, leiden=leiden, resolution=resolution, verbose=verbose, **kwargs), index=self.data.index)

		# Save and return
		if not self._minimal: self._layers['cluster'] = ret
		return ret
	# END FUNCTION cluster

	# BEGIN FUNCTION plot
	def plot(self, paint : Optional[Union[str, Iterable[str]]] = None, *, method : str = 'umap', by_cluster : Optional[Union[str, Callable[[pd.Series], Number]]] = None, skip_validation : bool = False, ax : Optional[Axes] = None, xlabel : Optional[Union[str, Mapping[str, Any]]] = None, ylabel : Optional[Union[str, Mapping[str, Any]]] = None, verbose : bool = True, **kwargs) -> Tuple[Figure, Union[Axes, np.ndarray]]:
		"""Plots the data in many ways

		Parameters
		----------
		paint : str,Iterable[str] = None
			the gene/column to color the points by
			if 'cluster', cells will be clustered and colored by cluster number
				'cluster' will always be the last subplot no matter its position in paint
			if None, no coloring is added, c is used if given

			makes a different subplot for each painting
			if cmap not given, plt.cm.coolwarm is the default
			if norm not given, matplotlib.colors.Normalize is the default

			For Processor: if in ['counts', 'genes', 'mito', 'div', 'reldiv'], colored by that value
		*
		method : str = 'umap'
			which method to call to get values
			valid: 'umap', 'tsne', 'pca'
		by_cluster : Union[str, Callable[[Series], Number]] = None
			if None, color by individual cell
			else: color each cluster by self.data[paint].groupby(self.layers['cluster']).`by_cluster`()
			str in ['all', 'any', 'count', 'first', 'last', 'max', 'mean', 'median', 'min', 'prod', 'size', 'sem', 'std', 'sum', 'var']
		skip_validation : bool = False
			whether to skip validation

		ax : Axes = None
			the axes on which to draw the plot
			if None, create new subplots
			ignored if paint is Iterable
		xlabel : str, dict = None
			the label for the x-axis of the graph
			if None, '`METHOD`0'
			if dict, the kwargs to pass to ax.set_xlabel
				use key 'xlabel' for the label itself
		ylabel : str, dict = None
			the label for the y-axis of the graph
			if None, '`METHOD`1'
			if dict, the kwargs to pass to ax.set_ylabel
				use key 'ylabel' for the label itself
		verbose : bool = True
			whether or not to print what's going on
		**kwargs
			passed variously to self.`method`, plt.subplots, plt.subplots_adjust, ax.scatter
			use 'tsne_method' for the 'method' kwarg of tsne
			extra kwargs ignored silently

		Special kwargs
		--------------
		s : int = 1
			the size of the scatter dots
			Different from default

		tsne_method
			passed to TSNE
		edgecolor, facecolor, frameon
			passed to plt.figure
		title_alpha, title_c, title_color, title_zorder
			passed to ax.set_title
		alpha, c, color, zorder
			passed to ax.scatter
		legend_edgecolor, legend_facecolor, legend_frameon
			passed to ax.legend

		grid_aspect : always passed to mpl_toolkits.axes_grid1.ImageGrid
		cbar_aspect : always passed to ax.colorbar
		aspect
			if paint == 'cluster': equivalent to but overridden by grid_aspect
			else: equivalent to but overridden by cbar_aspect

		pca_components : always given to PCA
		tsne_components : always given to TSNE
		umap_components : always given to UMAP
		n_components
			if 'pca' in method: equivalent to but overridden by pca_components
			else: equivalent to but overriden by tsne_components or umap_components
		
		title_fontsize, title_loc : always passed to ax.set_title
		legend_fontsize, legend_loc : always passed to ax.legend
		fontsize
			if 'cluster' in paint: equivalent to but overridden by legend_fontsize
			else: equivalent to but overridden by title_fontsize
		loc
			if 'cluster' in paint: equivalent to but overridden by legend_loc
			else: equivalent to but overridden by title_loc

		norm : overriden by Normalize() if paint is Iterable[str]

		cbar_pad : always passed to ax.colorbar
		title_pad : always passed to ax.set_title
		pad
			if paint == 'cluster': equivalent to but overridden by title_pad
			else: equivalent to but overridden by cbar_pad

		tsne_random_state : always given to TSNE
		umap_random_state : always given to UMAP
		pca_random_state : always given to PCA
		random_state
			if 'tsne' in method: equivalent to but overridden by tsne_random_state
			if 'umap' in method: equivalent to but overridden by umap_random_state
			else: equivalent to but overridden by pca_random_state
			
		Returns
		-------
		fig, ax
			the Figure and Axes on which the plot is drawn
			if ax is given and paint is not an Iterable, the Figure is None
		"""
		# BEGIN VALIDATION
		if ylabel is None: ylabel = method.upper() + '1' # defaults before checking
		if xlabel is None: xlabel = method.upper() + '0' # defaults before checking

		if not isinstance(skip_validation, bool) or not skip_validation:
			if verbose: print('Validating...')
			if method not in ['pca', 'umap', 'tsne']:
				raise ValueError('method must be \'pca\', \'umap\', or \'tsne\'')
			if by_cluster is not None: # optional
				if not callable(by_cluster) and (isinstance(by_cluster, str) and by_cluster not in ['all', 'any', 'count', 'first', 'last', 'max',
																			'mean', 'median', 'min', 'prod', 'size', 'sem', 'std', 'sum', 'var']):
					raise ValueError('by_cluster must be callable or str in [\'all\', \'any\', \'count\', \'first\', \'last\', \'max\', \'mean\', ' + \
																		'\'median\', \'min\', \'prod\', \'size\', \'sem\', \'std\', \'sum\', \'var\']')

			if paint is not None:
				if not isinstance(paint, str) and not (isinstance(paint, Iterable) and all([isinstance(i, str) for i in paint])):
					raise TypeError('paint must be str or Iterable[str]')
				elif isinstance(paint, str) and not (paint in self.columns or paint == 'cluster'):
					raise ValueError('paint must be \'cluster\' or a column of self.data')
				elif not isinstance(paint, str) and isinstance(paint, Iterable) and not all([i in self.columns or i == 'cluster' or i in self for i in paint]):
					raise ValueError('All entries in paint must be \'cluster\' or a column of self.data')
				else: pass


			if not isinstance(xlabel, str) or (isinstance(xlabel, dict) and 'xlabel' in xlabel): raise TypeError('xlabel must be str or dict of kwargs')
			if not isinstance(ylabel, str) or (isinstance(ylabel, dict) and 'ylabel' in ylabel): raise TypeError('ylabel must be str or dict of kwargs')

		# END VALIDATION

		# BEGIN PREPARE KWARGS
		# build methods kwargs
		kw_method = utils.TSNE_KWARGS if 'tsne' in method else utils.UMAP_KWARGS if 'umap' in method else []
		kw_method += utils.PCA_KWARGS + ['pca_components', 'pca_random_state']
		kw_method += ['tsne_components', 'tsne_random_state'] if 'tsne' in method else []
		kw_method += ['umap_components', 'umap_random_state'] if 'umap' in method else []

		if 'n_components' in kwargs:
			# if pca: n_components equivalent to but overridden by pca_components
			if 'pca_components' not in kwargs: kwargs['pca_components'] = kwargs.pop('n_components')
			# otherwise equivalent to but overriden by tsne/umap_components
			elif 'tsne' in method and 'tsne_components' not in kwargs: kwargs['tsne_components'] = kwargs.pop('n_components')
			elif 'umap' in method and 'umap_components' not in kwargs: kwargs['umap_components'] = kwargs.pop('n_components')
			# otherwise unneeded
			else: del kwargs['n_components']

		if 'random_state' in kwargs:
			# equivalent to but overriden to tsne/umap_random_state
			if 'tsne' in method and 'tsne_random_state' not in kwargs: kwargs['tsne_random_state'] = kwargs.pop('random_state')
			elif 'umap' in method and 'umap_random_state' not in kwargs:kwargs['umap_random_state'] = kwargs.pop('random_state')
			# otherwise equivalent to but overriden by pca_random_state
			elif 'pca' in method and 'pca_random_state' not in kwargs: kwargs['pca_random_state'] = kwargs.pop('random_state')
			# otherwise unneeded
			else: del kwargs['n_components']

		# because can't reuse kwargs, but different variables now
		if 'tsne_method' in kwargs: kwargs['method'] = kwargs.pop('tsne_method')
		# overall default because matplotlib default is huge
		if 's' not in kwargs: kwargs['s'] = 1
		# END PREPARE KWARGS

		# BEGIN TRANSFORM
		if verbose: print('Fetching ' + method + '...') if method in self.layers else print('Transforming data...')
		data = getattr(self, method)(**{k:kwargs[k] for k in kw_method if k in kwargs})
		# END TRANSFORM

		# END PREPARE FOR by_cluster
		if by_cluster is not None:
			if not 'cluster' in self.layers:
				if verbose: print('Fetching ' + clusters + '...') if method in self.layers else print('Clustering...')
				self.cluster(**{k:kwargs[k] for k in utils.CLUSTER_KWARGS if k in kwargs})

			# make sure we're showing the clusters themselves
			if paint is None: paint = 'cluster'
			elif 'cluster' not in paint:
				if isinstance(paint, str): paint = [paint, 'cluster']
				else: paint.append('cluster')
			else: pass
		# END PREPARE FOR by_cluster

		# BEGIN HANDLE paint
		# BEGIN HANDLE Iterable[str](paint)
		if paint is not None and not isinstance(paint, str):
			temp = paint.copy()
			# get only the ones that will work, warn about rest
			paint = [i for i in paint if i in self.columns or \
										(i in ['counts', 'genes', 'mito', 'div', 'reldiv', 'cluster'] and hasattr(self, i))]
			[warn(str(i) + ' not valid value for paint, ignoring') for i in temp if i not in paint]
			if len(paint) == 0: paint = None # for consistency
		# END HANDLE Iterable[str](paint)

		# BEGIN HANDLE str(paint) == 'cluster' vs not
		if paint == 'cluster':
			if 'aspect' in kwargs and 'grid_aspect' not in kwargs: # no colorbar, so aspect defaults to grid_aspect
				kwargs['grid_aspect'] = kwargs.pop('aspect')
			if 'pad' in kwargs and 'title_pad' not in kwargs: # no colorbar, so pad defaults to title_pad
				kwargs['title_pad'] = kwargs.pop('pad')
		else:
			if 'cbar_aspect' in kwargs: kwargs['aspect'] = kwargs.pop('cbar_aspect') # else cbar_aspect overrides
			# ImageGrid looks for cbar_pad instead of pad, so need both
			if 'cbar_pad' in kwargs: kwargs['pad'] = kwargs['cbar_pad'] # cbar_pad overrides pad
			# if pad not overridden by cbar_pad, pad doubles
			elif 'pad' in kwargs: kwargs['cbar_pad'] = kwargs['pad']
			else: pass
		# END HANDLE str(paint) == 'cluster' vs not

		# BEGIN HANDLE 'cluster' (not) in paint
		if paint is not None and 'cluster' in paint: # works for Iterable[str] and str=='cluster'
			if 'legend_loc' in kwargs: kwargs['loc'] = kwargs['legend_loc'] # legend_loc overrides loc
			if 'legend_fontsize' in kwargs: kwargs['fontsize'] = kwargs['legend_fontsize'] # legend_fontsize overrides fontsize
		else:
			if 'loc' in kwargs and 'title_loc' not in kwargs: # no legend, so loc defaults to title_loc
				kwargs['title_loc'] = kwargs.pop('loc')
			if 'fontsize' in kwargs and 'title_fontsize' not in kwargs: # no legend, so fontsize defaults to title_fontsize
				kwargs['title_fontsize'] = kwargs.pop('fontsize')
		# END HANDLE 'cluster' (not) in paint
		# END HANDLE paint


		# BEGIN PLOTTING
		# BEGIN NO COLORING
		if paint is None:
			# BEGIN MAKE FIGURE/AXES
			if ax is None: # if not given, then need to make it
				fig, ax = plt.subplots(**{k:kwargs[k] for k in utils.SUBPLOTS_KWARGS if k in kwargs})
				plt.subplots_adjust(**{k:kwargs[k] for k in utils.SUBPLOTS_ADJUST_KWARGS if k in kwargs})
			else: fig = None
			# END MAKE FIGURE/AXES

			# BEGIN PLOT
			ax.scatter(data[0], data[1], **{k:kwargs[k] for k in utils.SCATTER_KWARGS if k in kwargs})
			# END PLOT

			# BEGIN LABELING
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_xlabel(xlabel) if isinstance(xlabel, str) else ax.set_xlabel(**xlabel)
			ax.set_ylabel(ylabel) if isinstance(ylabel, str) else ax.set_ylabel(**ylabel)
			# END LABELING
		# END NO COLORING

		# BEGIN MANY COLORS
		elif not isinstance(paint, str):
			# BEGIN HANDLE KWARGS
			if 'nrows_ncols' not in kwargs: # if not given, then need to make it
				if 'ncol' in kwargs: kwargs['nrows_ncols'] = (np.ceil(len(paint)/kwargs['ncol']).astype(int), kwargs['ncol']) # ncol dominates
				elif 'nrow' in kwargs: kwargs['nrows_ncols'] = (kwargs['nrow'], np.ceil(len(paint)/kwargs['nrow'].astype(int))) # nrow otherwise
				else: kwargs['nrows_ncols'] = (np.ceil(len(paint)/4).astype(int), 4) # completely on our own
			rows, cols = kwargs['nrows_ncols'] # save for easier access

			# Internal defaults
			if 'figsize' not in kwargs: kwargs['figsize'] = ( 5*cols, 5*np.ceil(len(paint)/cols) )
			if 'cmap' not in kwargs: kwargs['cmap'] = plt.cm.coolwarm
			if 'axes_pad' not in kwargs: 
				kwargs['axes_pad'] = ((kwargs['wspace'] if 'wspace' in kwargs else 0.2) * kwargs['figsize'][0]/cols,
									  (kwargs['hspace'] if 'hspace' in kwargs else 0.2) * kwargs['figsize'][1]/rows)
			if 'cbar_size' not in kwargs: kwargs['cbar_size'] = kwargs['figsize'][0]/cols / 20
			if 'cbar_mode' not in kwargs: kwargs['cbar_mode'] = 'each'
			if 'cbar_pad' not in kwargs: kwargs['cbar_pad'] = 0.05 * kwargs['figsize'][0]/20
			# END HANDLE KWARGS

			# if present, move cluster to the end
			if 'cluster' in paint:
				paint.remove('cluster')
				paint = paint + ['cluster']

			# BEGIN MAKE FIGURE/AXES
			# BEGIN HANDLE grid_aspect VS aspect
			kw_grid = {k:kwargs[k] for k in utils.IMAGEGRID_KWARGS if k in kwargs}
			if 'grid_aspect' in kwargs: kw_grid['aspect'] = kwargs['grid_aspect']
			elif 'aspect' in kw_grid: del kw_grid['aspect']
			else: pass
			# END HANDLE grid_aspect VS aspect

			fig = plt.figure(1, **{k:kwargs[k] for k in kwargs if k in utils.FIGURE_KWARGS})
			ax = ImageGrid(fig, 111,  **kw_grid) # always make a new one here, ignore given ax
			# END MAKE FIGURE/AXES

			# BEGIN PREPARE TITLE KWARGS
			kw_title = {k:kwargs[k] for k in utils.TITLE_KWARGS if k in kwargs}
			if 'title_c' in kwargs: kw_title['c'] = kwargs['title_c'] # title_c overrides c
			elif 'c' in kw_title: del kw_title['c']
			else: pass
			if 'title_color' in kwargs: kw_title['color'] = kwargs.pop('title_color') # title_color overrides color
			elif 'color' in kw_title: del kw_title['color']
			else: pass
			if 'title_zorder' in kwargs: kw_title['zorder'] = kwargs.pop('title_zorder') # title_zorder overrides zorder
			elif 'zorder' in kw_title: del kw_title['zorder']
			else: pass
			if 'title_loc' in kwargs: kw_title['loc'] = kwargs.pop('title_loc') # title_loc overrides loc
			elif 'loc' in kw_title: del kw_title['loc']
			else: pass
			if 'title_pad' in kwargs: kw_title['pad'] = kwargs.pop('title_pad') # title_pad overrides pad
			elif 'pad' in kw_title: del kw_title['pad']
			else: pass
			if 'title_alpha' in kwargs: kw_title['alpha'] = kwargs.pop('title_alpha') # title_alpha overrides alpha
			elif 'alpha' in kw_title: del kw_title['alpha']
			else: pass
			# END PREPARE TITLE KWARGS

			# BEGIN PLOT
			for n,i in enumerate(paint):
				kwargs['norm'] = Normalize() # reset because different scales

				if by_cluster is not None and i != 'cluster': # by_cluster for value
					if verbose: print('Fetching ' + i + ' by cluster...')
					if i in self.columns:
						kwargs['c'] = getattr(self['e', i].groupby(self.layers['cluster']), by_cluster)() # gene by cluster
					elif i in ['counts', 'genes', 'mito', 'div', 'reldiv'] and hasattr(self, i):
						kwargs['c'] = getattr(self[i].groupby(self.layers['cluster']), by_cluster)() # gene by cluster
					else: pass # unreachable
				elif i in self.columns: # gene
					if verbose: print('Fetching ' + i + '...')
					kwargs['c'] = self['e', i]
				elif i in ['counts', 'genes', 'mito', 'div', 'reldiv'] and hasattr(self, i): # calculated value
					if verbose: print('Fetching ' + i + '...')
					kwargs['c'] = getattr(self, i)
				elif i == 'cluster': # cluster
					if verbose: print('Fetching cluster...') if 'cluster' in self.layers else print('Clustering...')
					self.cluster(**{k:kwargs[k] for k in utils.CLUSTER_KWARGS if k in kwargs}, verbose=verbose)
					kwargs['c'] = utils.colors(self.layers['cluster'])
				else: pass # unreachable

				# BEGIN LABELING
				ax[n].set_xticks([])
				ax[n].set_yticks([])
				ax[n].set_title(i + (' (cluster ' + by_cluster + ')' if by_cluster is not None and i != 'cluster' else ''), **kw_title)
				ax[n].set_xlabel(xlabel) if isinstance(xlabel, str) else ax[n].set_xlabel(**xlabel)
				ax[n].set_ylabel(ylabel) if isinstance(ylabel, str) else ax[n].set_ylabel(**ylabel)
				# END LABELING

				# BEGIN HANDLE CLUSTER
				if i == 'cluster':
					c = kwargs.pop('c') # np.ndarray

					h = [ax[n].scatter(data[0][self.layers['cluster'] == i], data[1][self.layers['cluster'] == i], c=c[self.layers['cluster'] == i],
										**{k:kwargs[k] for k in utils.SCATTER_KWARGS if k in kwargs})
						for i in sorted(self['cluster'].unique())]

					# BEGIN PREPARE LEGEND KWARGS
					kw_legend = {k:kwargs[k] for k in utils.LEGEND_KWARGS if k in kwargs}

					# Internal defaults
					if 'ncol' not in kw_legend: kw_legend['ncol'] = np.ceil(len(h)/13).astype(int)
					if 'loc' not in kw_legend: kw_legend['loc'] = 'upper left'
					if 'markerscale' not in kw_legend: kw_legend['markerscale'] = 6/20 * kwargs['figsize'][0] / kwargs['s']
					if 'frameon' not in kw_legend: kw_legend['frameon'] = False
					if 'columnspacing' not in kw_legend: kw_legend['columnspacing'] = .15/20 * kwargs['figsize'][0]
					if 'handletextpad' not in kw_legend: kw_legend['handletextpad'] = 0.01/20 * kwargs['figsize'][0]
					if 'labelspacing' not in kw_legend: kw_legend['labelspacing'] = 0.3/15 * kwargs['figsize'][1]

					# Overridings
					if 'legend_facecolor' in kwargs: kw_legend['facecolor'] = kwargs.pop('legend_facecolor') # legend_facecolor overrides facecolor
					elif 'facecolor' in kw_legend: del kw_legend['facecolor']
					else: pass
					if 'legend_edgecolor' in kwargs: kw_legend['edgecolor'] = kwargs.pop('legend_edgecolor') # legend_edgecolor overrides edgecolor
					elif 'edgecolor' in kw_legend: del kw_legend['edgecolor']
					else: pass
					if 'legend_frameon' in kwargs: kw_legend['frameon'] = kwargs.pop('legend_frameon') # legend_frameon overrides frameon
					elif 'frameon' in kw_legend: del kw_legend['frameon']
					else: pass
					# END HANDLE PREPARE KWARGS

					# BEGIN LEGEND
					ax.cbar_axes[n].set_axis_off()
					ax.cbar_axes[n].legend(h, range(len(h)), **kw_legend)
					# END LEGEND

					# BEGIN CLUSTER ANNOTATION
					if 'umap' in method:
						for i in range(len(h)):
							xy = data[self['cluster'] == i].median()[:2]
							color = c[self['cluster'] == i][0]
							ax[n].annotate(str(i), xy=tuple(xy), color=utils.annot_color(color), bbox=utils.annot_bbox(color))
					# END CLUSTER ANNOTATION
				# END HANDLE CLUSTER

				# BEGIN HANDLE by_cluster
				elif by_cluster is not None:
					c = kwargs.pop('c') # pd.Series

					ax[n].scatter(data[0],data[1], c=c[self.layers['cluster']], **{k:kwargs[k] for k in utils.SCATTER_KWARGS if k in kwargs})
					
					# BEGIN COLORBAR
					sm = ScalarMappable(norm=kwargs['norm'], cmap=kwargs['cmap'])
					sm.set_array(c.values)

					plt.colorbar(sm, cax=ax.cbar_axes[n], **{k:kwargs[k] for k  in utils.COLORBAR_KWARGS if k in kwargs})
					# END COLORBAR
				# END HANDLE by_cluster

				# BEGIN HANDLE OTHER
				else:
					ax[n].scatter(data[0], data[1], **{k:kwargs[k] for k in utils.SCATTER_KWARGS if k in kwargs})
					
					# BEGIN COLORBAR
					sm = ScalarMappable(norm=kwargs['norm'], cmap=kwargs['cmap'])
					sm.set_array(kwargs['c'])

					plt.colorbar(sm, cax=ax.cbar_axes[n], **{k:kwargs[k] for k  in utils.COLORBAR_KWARGS if k in kwargs})
					# END COLORBAR
				# END HANDLE OTHER
			# END PLOT

			# BEGIN PREPARE AXES
			for n in range(len(paint), len(ax)):
				ax[n].set_axis_off()
				ax[n-cols].set_xlabel(xlabel) if isinstance(xlabel, str) else ax[n-cols].set_xlabel(**xlabel) # move xlabel to previous row
				ax.cbar_axes[n].set_axis_off()

			for i in range(len(ax)):
				ax[i].cax = ax.cbar_axes[i]

			ax = [[ax[cols*i+j] for j in range(cols)] for i in range(np.ceil(len(paint)/cols).astype(int))]
			ax = np.squeeze(np.array(ax))
			# END PREPARE AXES
		# END MANY COLORS

		# BEGIN SINGLE COLOR
		else:
			# BEGIN MAKE FIGURE/AXES
			if ax is None: # if not given, then need to make it
				fig, ax = plt.subplots(**{k:kwargs[k] for k in utils.SUBPLOTS_KWARGS if k in kwargs})
				plt.subplots_adjust(**{k:kwargs[k] for k in utils.SUBPLOTS_ADJUST_KWARGS if k in kwargs})
			else: fig = None
			# END MAKE FIGURE/AXES

			# can't have by_cluster here unless paint=='cluster'
			if paint in self.columns:
				kwargs['c'] = self['e', paint]
			elif paint in ['counts', 'genes', 'mito', 'div', 'reldiv'] and hasattr(self, paint):
				kwargs['c'] = getattr(self, paint)
			elif paint == 'cluster':
				if verbose: print('Clustering...')
				kwargs['c'] = utils.colors(self.cluster(**{k:kwargs[k] for k in utils.CLUSTER_KWARGS if k in kwargs}).unique())
			else: pass # unreachable

			# BEGIN PLOT
			# BEGIN HANDLE 'cluster'
			if paint == 'cluster':
				# BEGIN PLOT
				c = kwargs.pop('c')
				h = [ax.scatter(data[0][self['cluster'] == i], data[1][self['cluster'] == i], c=c[i],
									**{k:kwargs[k] for k in utils.SCATTER_KWARGS if k in kwargs})
					for i in sorted(self['cluster'].unique())]
				# END PLOT

				# Internal default
				if not 'figsize' in kwargs: kwargs['figsize'] = (8, 6)

				# BEGIN LEGEND
				# BEGIN PREPARE LEGEND KWARGS
				leg = {k.replace('legend_',''):kwargs[k] for k in utils.LEGEND_KWARGS if k in kwargs}
				# Internal defaults
				if 'ncol' not in leg: leg['ncol'] = np.ceil(len(h)/13).astype(int)
				if 'markerscale' not in leg: leg['markerscale'] = 6/20 * kwargs['figsize'][0] / kwargs['s']
				if 'frameon' not in leg: leg['frameon'] = False
				if 'columnspacing' not in leg: leg['columnspacing'] = .15/20 * kwargs['figsize'][0]
				if 'handletextpad' not in leg: leg['handletextpad'] = 0.01/20 * kwargs['figsize'][0]
				if 'labelspacing' not in leg: leg['labelspacing'] = 0.3/15 * kwargs['figsize'][1]
				# END PREPARE LEGEND KWARGS
				ax.legend(h, range(len(h)), **leg)
				# END LEGEND
				
				# BEGIN CLUSTER ANNOTATION
				if 'umap' in method:
					for i in range(len(h)):
						xy = data[self['cluster'] == i].median()[:2]
						color = c[i]
						ax.annotate(str(i), xy=tuple(xy), color=utils.annot_color(color), bbox=utils.annot_bbox(color))
				# END CLUSTER ANNOTATION
			# END HANDLE 'cluster'

			# BEGIN HANDLE OTHERS
			else:
				# BEGIN PLOT
				# Internal defaults
				if 'norm' not in kwargs: kwargs['norm'] = Normalize()
				if 'cmap' not in kwargs: kwargs['cmap'] = plt.cm.coolwarm
				ax.scatter(data[0], data[1], **{k:kwargs[k] for k in utils.SCATTER_KWARGS if k in kwargs})
				# END PLOT

				# BEGIN COLORBAR
				sm = ScalarMappable(norm=kwargs['norm'], cmap=kwargs['cmap'])
				sm.set_array(kwargs['c'])
				plt.colorbar(sm, ax=ax, **{k:kwargs[k] for k in utils.COLORBAR_KWARGS if k in kwargs})
				# END COLORBAR
			# END HANDLE OTHERS

			# BEGIN LABELING
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_xlabel(xlabel) if isinstance(xlabel, str) else ax.set_xlabel(**xlabel)
			ax.set_ylabel(ylabel) if isinstance(ylabel, str) else ax.set_ylabel(**ylabel)
			# END LABELING
		# END SINGLE COLOR

		return fig, ax
	# END FUNCTION plot
	# END METHODS
# END CLASS Sample






# BEGIN CLASS MultiSample
class MultiSample(Sample):
	"""A single object from multiple single-cell RNA sequencing sample, after all data processing

	Parameters
	----------
	data : pd.DataFrame, Mapping[str, pd.DataFrame], Iterable[pd.DataFrame]
		the log-scaled, normalized, etc expression data
		index is expected to be a MultiIndex
			levels[0] is the sample
			levels[1] is unique by row
	*
	names : Optional[Iterable[str]] = None
		the names to give the data, in the same order as in @data
		only used if data is a list
	sparse : bool = True
		whether or not to use a sparse datatype
			handles the following kwargs internally:
				dtype : type, str = 'float16'
				fill_value : float = 0
		whether to only store minimal data; self.layers will always be empty
		leads to a lot of calculating on the fly, increasing computation time
	skip_validation : bool = False
		whether to skip validating anything
	verbose : bool = False
		Whether or not to print what's happening
	pca_ : PCA = None
		a PCA instance fit to data
	umap_ : UMAP = None
		a UMAP instance fit to data
	tnse_ : TSNE = None
		a TSNE instance fit to data

	
	Class Methods
	-------------
	from_csvs()
		load scRNAseq samples from individual csv files
	from_single_csv()
		load scRNAseq samples from single csv file
	from_many()
		combine multiple Sample instances into a single MultiSample
	"""
	# BEGIN DUNDER FUNCTIONS
	# BEGIN FUNCTION __init__
	def __init__(self, data : Union[pd.DataFrame, Mapping[str, pd.DataFrame], Iterable[pd.DataFrame]], *, names : Optional[Iterable[str]] = None, sparse : bool = True, minimal : bool = False, skip_validation : bool = False, verbose : bool = False, pca_ : Optional[PCA] = None, umap_ : Optional[UMAP] = None, tsne_ : Optional['sklearn.manifold.TSNE'] = None, **kwargs) -> None:
		if not isinstance(skip_validation, bool) or not skip_validation:
			if verbose: print('Initial validation...')
			if not isinstance(data, pd.DataFrame) and not isinstance(data, Mapping) and isinstance(data, Iterable):
				if names is None: raise ValueError('Must provide @names if @data is Iterable')
				elif not len(names) == len(data): raise ValueError('@names and @data are not the same length')
				else: data = {n:d for n,d in zip(names,data)}

			if isinstance(data, Mapping):
				if not all([isinstance(i, str) for i in data.keys()]): raise TypeError('All keys must be str')
				elif not all([isinstance(i, pd.DataFrame) for i in data.values()]): raise TypeError('All values must be pd.DataFrame')
				else:
					dtype = kwargs['dtype'] if 'dtype' in kwargs else 'float16'
					fill_value = kwargs['fill_value'] if 'fill_value' in kwargs else 0

					if only_overlap:
						if verbose: print('Taking only overlap...')
						genes = [set(v.columns) for v in data.values()]
						for i in genes[1:]:
							genes[0] = genes[0].intersection(i)
						data = {k:v[genes[0]] for k,v in data.items()}

					data = pd.concat(data, axis=0, sort=True, copy=False).astype(dtype).fillna(fill_value)
			elif not isinstance(data, pd.DataFrame): raise TypeError('data must be pd.DataFrame, Mapping[str : pd.DataFrame], or Iterable[pd.DataFrame]')
			else: pass
		else:
			if not isinstance(data, pd.DataFrame) and not isinstance(data, Mapping) and isinstance(data, Iterable):
				data = {n:d for n,d in zip(names,data)}

			if isinstance(data, Mapping):
				dtype = kwargs['dtype'] if 'dtype' in kwargs else 'float16'
				fill_value = kwargs['fill_value'] if 'fill_value' in kwargs else 0

				if only_overlap:
					if verbose: print('Taking only overlap...')
					genes = [set(v.columns) for v in data.values()]
					for i in genes[1:]:
						genes[0] = genes[0].intersection(i)
					data = {k:v[genes[0]] for k,v in data.items()}

				data = pd.concat(data, axis=0, sort=True, copy=False).astype(dtype).fillna(fill_value)
			elif not isinstance(data, pd.DataFrame): raise TypeError('data must be pd.DataFrame, Mapping[str : pd.DataFrame], or Iterable[pd.DataFrame]')
			else: pass

		if sparse: data = data.astype(pd.SparseDtype(dtype,fill_value=fill_value))
		data = data[data.columns[(data != 0).any()]]

		if verbose:
			print('Cells:', data.shape[0])
			print('Genes:', data.shape[1])
			print('Building structure...')
		super().__init__(data=data, sparse=sparse, fill_value=fill_value, minimal=minimal, skip_validation=skip_validation, pca_=pca_ if not minimal else None, umap_=umap_ if not minimal else None, tsne_=tsne_ if not minimal else None)
	# END FUNCTION __init__
	# END DUNDER FUNCTIONS

	# BEGIN CLASS METHODS
	# BEGIN CLASS METHOD from_csvs
	@classmethod
	def from_csvs(cls, files : Union[Iterable[Union[str, IO]], Mapping[str, Union[str,IO]]], *, samples : Optional[Iterable[str]] = None, sparse : bool = True, minimal : bool = False, skip_validation : bool = False, only_overlap : bool = False, verbose : bool = False, **kwargs) -> 'MultiSample':
		"""Load multiple samples, each from their own csv file
		If not passed explicitly, `index_col` is assumed to be 0

		Parameters
		----------
		files : Union[Iterable[Union[str, IO]], Mapping[str, Union[str,IO]]]
			if Mapping, name: file
			elif Iterable[str], name = file.split('/')[-1].split('.')[0]
				eg. path/to/file.example.csv -> file
			elif Iterable[IO], name = file.__name__
		*
		samples : Optional[Iterable[str]] = None
			the names to give the samples, in the same order as in @files
			only used if files is a list
		sparse : bool = True
			whether or not to use a sparse datatype
			handles the following kwargs internally:
				sep : str = ','
				header : int, Iterable[int] = 0 # NotImplementedError
				index_col : int, Iterable[int] = 0
				usecols : int, str, Iterable[int,str] = None
				dtype : type, str = 'float16'
				fill_value : float = 0
		minimal : bool = False
			whether to only store minimal data; self.layers will always be empty
			leads to a lot of calculating on the fly, increasing computation time
		skip_validation : bool = False
			whether to skip validating anything
		only_overlap : bool = False
			whether or not to only use the overlapping genes
			if False, fills unmeasured genes with fill_value
		verbose : bool = False
			whether or not to print what's going on
		**kwargs
			passed to pd.read_csv if not sparse
		"""
		if not isinstance(skip_validation, bool) or not skip_validation:
			if verbose: print('Validating...')
			if not isinstance(sparse, bool): raise TypeError('sparse must be a bool')
			elif sparse and not isinstance(fill_value, (float, int)): raise TypeError('fill_value must be a float if sparse')
			elif not isinstance(minimal, bool): raise TypeError('minimal must be a bool')
			else: pass
			
			if isinstance(files, Mapping):
				if not all([isinstance(i, str) for i in files.keys()]): raise TypeError('All keys must be str')
				elif not all([isinstance(i,str) or hasattr(i,'readline') for i in files.values()]): raise TypeError('All values must be str or file-like')
				else: pass
			elif isinstance(files, Iterable):
				if not all([isinstance(i,str) or hasattr(i, 'readline') for i in files]): raise TypeError('All files must be str or file-like')
				files = {f:s for f,s in zip(files,samples)}
			else: raise TypeError('files must be a Mapping[str: str/file-like] or Iterable[str/file-like]')
		elif not isinstance(files, Mapping) and isinstance(files, Iterable):
			files = {f:s for f,s in zip(files,samples)}
		else: pass

		if 'index_col' not in kwargs: kwargs['index_col'] = 0
		if sparse:
			sep = kwargs['delimiter'] if 'delimiter' in kwargs else kwargs['sep'] if 'sep' in kwargs else ','
			header = kwargs['header'] if 'header' in kwargs else [0]
			usecols = kwargs['usecols'] if 'usecols' in kwargs else None

			dtype = kwargs['dtype'] if 'dtype' in kwargs else 'float16'
			fill_value = kwargs['fill_value'] if 'fill_value' in kwargs else 0
			sparse_dtype = pd.SparseDtype(dtype,fill_value=fill_value)

			index_col = kwargs['index_col']
			if isinstance(index_col, Iterable): index_col = list(index_col)
			else: index_col = [index_col]
		
			dfs = {}
			for k,v in files.items():
				if verbose: print('Loading ' + k + '...')
				df = []

				if not hasattr(v, 'readline'): f = open(v, 'r')
				else: f = v

				temp = np.array(f.readline().rstrip().split(sep))
				dat = [i for i in range(len(temp)) if i not in idx]
				temp = pd.Series(temp[dat], name=tuple(temp[idx].tolist()))

				for c in f:
					d = np.array(c.split(sep))
					df.append(pd.Series(d[dat], name=tuple(d[idx].tolist())).astype(sparse_dtype))
				
				if not hasattr(v, 'readline'): f.close()
				df = pd.concat(df, axis=1, copy=False).T
				df.columns = temp
				if verbose: print('Loaded size:', df.shape)

				if len(dfs):
					if only_overlap:
						df = df[dfs[list(dfs.keys())[0]].columns.intersection(df.columns)]
						dfs = {k:v[df.columns] for k,v in dfs.items()}
					else:
						col = set(dfs[list(dfs.keys())[0]].columns) - set(df.columns)
						cols = set(df.columns) - set(dfs[list(dfs.keys())[0]].columns)
						
						if len(col): df = pd.concat([df,pd.DataFrame(np.tile(fill_value, (df.shape[0],len(col))), index=df.index,columns=col,copy=False
													).astype(sparse_dtype)],sort=False, axis=1)
						if len(cols): dfs = {k:pd.concat([v,pd.DataFrame(np.tile(fill_value, (v.shape[0],len(cols))), index=v.index,columns=cols,copy=False
														 ).astype(sparse_dtype)], sort=False, axis=1) for k,v in dfs.items()}

				dfs[k] = df
				if verbose: print('Kept size:', df.shape)

			if verbose:
				print('Cells:', {k:v.shape[0] for k,v in dfs.items()})
				print('Total cells:', sum([v.shape[0] for v in dfs.values()]))
				print('Genes:', dfs[list(dfs.keys())[0]].shape[1])
				print('Building structure...')
			return cls(dfs, minimal=minimal, skip_validation=skip_validation)
		else:
			if verbose: print('Loading...')
			dfs = {k:pd.read_csv(v, **{i:kwargs[i] for i in utils.CSV_KWARGS if i in kwargs}) for k,v in files.items()}

			if verbose:
				print('Cells:', {k:v.shape[0] for k,v in dfs.items()})
				print('Total cells:', sum([v.shape[0] for v in dfs.values()]))
				print('Genes:', dfs[list(dfs.keys())[0]].shape[1])
				print('Building structure...')
			return cls(dfs, minimal=minimal, skip_validation=skip_validation)
	# END CLASS METHOD from_csvs

	# BEGIN CLASS METHOD from_single_csv
	@classmethod
	def from_single_csv(cls, file : Union[str, IO], *, sparse : bool = True, minimal : bool = False, skip_validation : bool = False, verbose : bool = False, **kwargs) -> 'MultiSample':
		"""Load multiple samples from a single csv file
		the first column is assumed to be the sample identifier and the second a unique identifier for each row
		ie `index_col=(0,1)` is passed to pd.read_csv if not otherwise given

		Parameters
		----------
		file : Union[str, IO]
			the file containing the data
			rows are cells, columns are genes
			column name should be gene symbols

			assumes first column is sample, second column is unique identifier
			otherwise, pass `index_col` kwarg for pd.read_csv
		*
		sparse : bool = True
			whether or not to use a sparse datatype
			handles the following kwargs internally:
				sep : str = ','
				header : int, Iterable[int] = 0 # NotImplementedError
				index_col : int, Iterable[int] = (0,1)
				usecols : int, str, Iterable[int,str] = None
				dtype : type, str = 'float16'
		minimal : bool = False
			whether to only store minimal data; self.layers will always be empty
			leads to a lot of calculating on the fly, increasing computation time
		skip_validation : bool = False
			whether to skip validating anything
		verbose : bool = False
			whether or not to print what's going on
		**kwargs
			passed to pd.read_csv if not sparse
		"""
		if not isinstance(skip_validation, bool) or not skip_validation:
			if not isinstance(sparse, bool): raise TypeError('sparse must be a bool')
			elif sparse and not isinstance(fill_value, (float, int)): raise TypeError('fill_value must be a float if sparse')
			elif not isinstance(minimal, bool): raise TypeError('minimal must be a bool')
			elif not (isinstance(file,str) or hasattr(file, 'readline')): raise TypeError('file must be str or file-like')
			else: pass
		
		if verbose: print('Loading...')
		if 'index_col' not in kwargs: kwargs['index_col'] = (0,1)
		if sparse:
			sep = kwargs['delimiter'] if 'delimiter' in kwargs else kwargs['sep'] if 'sep' in kwargs else ','
			header = kwargs['header'] if 'header' in kwargs else [0]
			usecols = kwargs['usecols'] if 'usecols' in kwargs else None

			dtype = kwargs['dtype'] if 'dtype' in kwargs else 'float16'
			fill_value = kwargs['fill_value'] if 'fill_value' in kwargs else 0
			sparse_dtype = pd.SparseDtype(dtype,fill_value=fill_value)

			index_col = kwargs['index_col']
			if isinstance(index_col, Iterable): index_col = list(index_col)
			else: index_col = [index_col]

			df = []
			if not hasattr(file, 'readline'): f = open(file, 'r')
			else: f = file

			temp = np.array(f.readline().rstrip().split(sep))
			dat = [i for i in range(len(temp)) if i not in idx]
			temp = pd.Series(temp[dat], name=tuple(temp[idx].tolist()))

			for c in f:
				d = np.array(c.split(sep))
				df.append(pd.Series(d[dat], name=tuple(d[idx].tolist())).astype(pd.SparseDtype(float, fill_value=fill_value)))
				
			if not hasattr(file, 'readline'): f.close()
			df = pd.concat(df, axis=1, copy=False).T
			df.columns = temp

			if verbose:
				print('Cells:', df.shape[0])
				print('Genes:', df.shape[1])
				print('Building structure...')
			return cls(df, minimal=minimal, skip_validation=skip_validation)
		else:
			df = pd.read_csv(file, **{i:kwargs[i] for i in utils.CSV_KWARGS if i in kwargs})
			if verbose:
				print('Cells:', df.shape[0])
				print('Genes:', df.shape[1])
				print('Building structure...')
			return cls(df, minimal=minimal, skip_validation=skip_validation)
	# END CLASS METHOD from_single_csv

	# BEGIN CLASS METHOD from_many
	@classmethod
	def from_many(cls, samples : Union[Mapping[str, Sample], Iterable[Sample]], *, names : Optional[Iterable[str]] = None, sparse : bool = True, minimal : bool = False, skip_validation : bool = False, verbose : bool = False, **kwargs) -> 'MultiSample':
		"""Create a MultiSample from multiple Samples

		Parameters
		----------
		samples : Mapping[str, Sample], Iterable[Sample]
			the Sapmles to condense into a single MultiSample
			uses .data
		*
		names : Optional[Iterable[str]] = None
			the names to give the samples, in the same order as in @samples
			only used if samples is a list
		sparse : bool = True
			whether or not to use a sparse datatype
			handles the following kwargs internally:
				dtype : type, str = 'float16'
				fill_value : float = 0
		minimal : bool = False
			whether to only store minimal data; self.layers will always be empty
			leads to a lot of calculating on the fly, increasing computation time
		skip_validation : bool = False
			whether to skip validating anything
		verbose : bool = False
			whether or not to print what's going on
		**kwargs
			if sparse: dtype, fill_Value
			all other ignored silently
		"""
		if not isinstance(skip_validation, bool) or not skip_validation:
			if not isinstance(sparse, bool): raise TypeError('sparse must be a bool')
			elif sparse and not isinstance(fill_value, (float, int)): raise TypeError('fill_value must be a float if sparse')
			elif not isinstance(minimal, bool): raise TypeError('minimal must be a bool')
			else: pass

			if isinstance(samples, Iterable): samples = {n:s for n,s in zip(names,samp)}

			if isinstance(samples, Mapping) and not all([isinstance(v, Sample) for v in samples.values()]): raise ValueError('samples must be Mapping[str:Sample], or Iterable[Sample] with names given')
			else: raise TypeError('samples must be Mapping[str:Sample], or Iterable[Sample] with names given')
		elif isinstance(samples, Iterable): samples = {n:s for n,s in zip(names,samples)}
		else: pass

		if not sparse:
			dfs = {k:v.data for k,v in samples.items()}
			if verbose:
				print('Cells:', {k:v.shape[0] for k,v in dfs.items()})
				print('Total cells:', sum([v.shape[0] for v in dfs.values()]))
				print('Genes:', dfs[list(dfs.keys())[0]].shape[1])
				print('Building structure...')
			return cls(dfs, minimal=minimal, skip_validation=skip_validation)
		else:
			dtype = kwargs['dtype'] if 'dtype' in kwargs else 'float16'
			fill_value = kwargs['fill_value'] if 'fill_value' in kwargs else 0
			sparse_dtype = pd.SparseDtype(dtype,fill_value=fill_value)

			dfs = {k:v.data.astype(pd.SparseDtype(float,fill_value=fill_value)) for k,v in samples.items()}
			if verbose:
				print('Cells:', {k:v.shape[0] for k,v in dfs.items()})
				print('Total cells:', sum([v.shape[0] for v in dfs.values()]))
				print('Genes:', dfs[list(dfs.keys())[0]].shape[1])
				print('Building structure...')
			return cls(dfs, minimal=minimal, skip_validation=skip_validation)
	# END CLASS METHOD from_many
	# END CLASS METHODS
# END CLASS MultiSample






# BEGIN CLASS Processor
class Processor(Sample):
	"""A container for processing single-cell RNA sequencing data

	Parameters
	----------
	data : pd.DataFrame
		the unnormalized count data with dtype = int
		the columns are genes and the rows are cells
	*
	sparse : bool = True
		whether or not to use a sparse datatype
		handles kwargs 'index_col' internally if passed
	fill_value : float = 0
		if sparse, the value to use as the fill_value
		ignored otherwise
	minimal : bool = False
		whether to only store minimal data; self.layers will always be empty
		leads to a lot of calculating on the fly, increasing computation time
	skip_validation : bool = False
		whether to skip validating anything
	counts, genes, mito : pd.Series [optional]
		can give any combination of precomputed values
		each must be have the same index as data
			counts is the sum of counts per cell
			genes is the number of unique genes per cell
			mito is the mitochondrial fraction per cell
	log : callable = np.log2
		the logarithm function to use
		given a pd.DataFrame, must return a pd.DataFrame with indices and columns unchanged
	mito_marker : str = 'mt-'
		the prefix used for mitochondrial gene names
		cAsE sEnSiTiVe
	normalized : str = None
		the normalization already applied to the data
		valid entries are 'cell', 'hk'/'housekeeping', 'hkcell', 'gf-icf'
		not case sensitive
	verbose : bool = False
		whether or not to print what's going on

	Class Methods
	-------------
	from_csv()
		load an scRNAseq count matrix from a csv file
	from_pickle()
		load an scRNAseq count matrix from a pickled pd.DataFrame



	Values can be retrieved from data/expression by:
		column(s): self['d'/'e', key]
		row(s) by name: self['d'/'e', key] if key is not a column
		cell(s): self['d'/'e', key1, key2] by any combination of loc/iloc
	Shape along an axis is either self.shape[axis] or self['shape', axis]
	All attributes are also available as keys; eg self.mito == self['mito']
	All self.layers are also available as keys; eg self.layers['pca'] == self['pca']

		

	Selected Attributes
	-------------------
	data, d : pd.DataFrame
		the filtered count data
	shape : np.ndarray
		the shape of the filtered data
	expression, e : pd.DataFrame
		the log-transformed expression values of the filtered data
		calculated at time of request
	counts, genes, mito : pd.Series
		the filtered total counts, number unique genes, and mitochondrial fraction
	filter
		the filter for this sample
		has the following filters by default:
			counts(cutoff) -> minimum if single value, range if 2-tuple
			genes(cutoff) -> minimum if single value, range if 2-tuple
			mito(cutoff) -> maximum if single value, range if 2-tuple
			sec_counts(cutoff) -> percentile beyond minimum to keep for a secant-based filter on counts
			reldiv() -> takes upper modal peak by reldiv
		on filter change:
			clears self.layers, self.pca_, self.umap_, self.tsne_
			reruns gf_icf
	normalized : {}
		whether or not the data has been normalized
		'cell':True means cell_normalize has been successfully called
		'hk':list means hk_normalize has been successfully called with the given list
		'gf-icf':True means gf_icf has been successfully called

		if 'gf-icf' is True, then 'cell' and 'hk' must both be False
		if 'cell' or 'hk' is True, then 'gf-icf' must be False
	layers : {}
		calculated values stored for easy recall
		keys are the function names that calculated the values
		entries can also be accessed as simply self[key]
		cleared on change of self.filter
		remove layers using self.wipe_layer(key) to run again with different parameters
	raw, raw_shape, raw_counts, raw_genes, raw_mito
		similar to data, shape, etc but without filtering
	"""
	# BEGIN DUNDER FUNCTIONS
	# BEGIN FUNCTION __init__
	def __init__(self, data : pd.DataFrame, *, sparse : bool = True, fill_value : float = 0, minimal : bool = False, skip_validation : bool = False, counts : Optional[pd.Series] = None, genes : Optional[pd.Series] = None, mito : Optional[pd.Series] = None, log : Callable[[pd.Series], pd.Series] = np.log2, mito_marker : str = 'mt-', normalized : Optional[str] = None, verbose : bool = False) -> None:
		"""Initializes the container and calculates QC metrics"""
		# BEGIN VALIDATION
		if not isinstance(skip_validation, bool) or not skip_validation:
			if verbose: print('Validating...')
			if not isinstance(data, pd.DataFrame): raise TypeError('data must be a pd.DataFrame')
			if not isinstance(sparse, bool): raise TypeError('sparse must be a bool')
			elif sparse and not isinstance(fill_value, (float, int)): raise TypeError('fill_value must be a float if sparse')
			elif not isinstance(minimal, bool): raise TypeError('minimal must be a bool')
			else: pass

			if not minimal:
				if counts:
					if not isinstance(counts, pd.Series): raise TypeError('counts must be a pd.Series if given')
					if not counts.shape[0] == data.shape[0] or not all(counts.index == data.index):
						raise ValueError('counts must have the same index as data')
				if genes:
					if not isinstance(genes, pd.Series): raise TypeError('genes must be a pd.Series if given')
					if not genes.shape[0] == data.shape[0] or not all(genes.index == data.index):
						raise ValueError('genes must have the same index as data')
				if mito:
					if not isinstance(mito, pd.Series): raise TypeError('mito must be a pd.Series if given')
					if not mito.shape[0] == data.shape[0] or not all(mito.index == data.index):
						raise ValueError('mito must have the same index as data')
				if not callable(log): raise TypeError('log must be callable')
				if not isinstance(mito_marker, str): raise TypeError('mito_marker must be a str')
				if normalized and isinstance(normalized, str) and normalized.lower() not in ['cell', 'hk', 'housekeeping', 'hkcell', 'gf-icf']:
						raise ValueError('normalized must be one of [\'cell\', \'hk\'/\'housekeeping\', \'hkcell\' \'gf-icf\']')
		# END VALIDATION

		# BEGIN SET UP
		# BEGIN SET UP NUMBERS
		if verbose: print('Building structure...')
		self._minimal = minimal
		self._raw_data = data if not sparse else data.astype(pd.SparseDtype(float,fill_value=fill_value))
		self._data = self._raw_data.copy() # to keep separate
		self.mito_marker = mito_marker

		if normalized: # pre-normalized data
			self._normalized = {k:False for k in ['cell', 'hk', 'gf-icf']}
			if normalized.lower() in self._normalized: self._normalized[normalized.lower()] = True
			elif normalized.lower() == 'housekeeping': self._normalized['hk'] = True
			else: # normalized = 'hkcell'
				self._normalized['hk'] = True
				self._normalized['cell'] = True
			self._normalized = frozendict(self._normalized)

			self._raw_counts = pd.Series(index=data.index) if not minimal else None
			self._raw_genes = pd.Series(index=data.index) if not minimal else None
			self._raw_div = pd.Series(index=data.index) if not minimal else None
			self._raw_reldiv = pd.Series(index=data.index) if not minimal else None
			self._raw_mito = pd.Series(index=data.index) if not minimal else None
			self._filter = utils.Filter(self, [('raw_data', self._filter_clusters, 'cluster')])

		else: # raw count data
			if verbose and not minimal: print('Calculating summary statistics...')
			self._raw_counts = counts if counts else data.sum(axis=1) if not minimal else None
			self._raw_genes = genes if genes else (data != 0).sum(axis=1) if not minimal else None
			self._raw_div = np.log1p(self._raw_genes) if not minimal else None
			self._raw_reldiv = pd.Series(stats.zscore(self._raw_div), index=data.index) if not minimal else None
			if not data.columns.str.isdigit().any(): # no fillna method on Index[bool]
				self._raw_mito = mito if mito else data.loc[:, data.columns.str.startswith(mito_marker)].sum(axis=1) / self._raw_counts if not minimal else None
			else: # throws NaNs, fillna on Index[object]
				self._raw_mito = mito if mito else data.loc[:, data.columns.str.startswith(mito_marker).fillna(False)].sum(axis=1) / self._raw_counts if not minimal else None
			self._normalized = {k:False for k in ['cell', 'hk', 'gf-icf']}
			self._filter = utils.Filter(self, [('raw_counts', ('linear', 'above'), 'counts'), ('raw_genes', ('linear', 'above'), 'genes'),
												('raw_mito', ('linear', 'below'), 'mito'), ('raw_counts', ('secant', 'above'), 'sec_counts'),
												('raw_reldiv', ('bimodal', 'above'), 'reldiv'),
												('raw_data', self._filter_clusters, 'cluster')])
		# END SET UP NUMBERS

		self.log = log
		self._pca_, self._umap_, self._tsne_ = None, None, None
		self._layers = {}
		# END SET UP
	# END FUNCTION __init__

	# BEGIN FUNCTION __repr__
	def __repr__(self) -> str:
		return 'Processor(raw_shape{}, shape={}, applied_filters={}, normalized={})'.format(
				self.raw_shape, self.shape, len(self.filter.filters), self._normalized)
	# END FUNCTION __repr__

	# BEGIN FUNCTION __contains__
	def __contains__(self, key) -> bool:
		if isinstance(key, Iterable) and not isinstance(key, str): return all([i in self for i in key]) # handled by _getitem

		# Simple in
		ret = key in ['data', 'd', 'counts', 'genes', 'mito', 'div', 'reldiv', 'shape', 'expression', 'e', 'layers', 'idx', 'filtered'
						'raw_data', 'raw_counts', 'raw_genes', 'raw_mito', 'raw_div', 'raw_reldiv', 'raw_shape', 'normalized']
		if ret: return ret # return it if we already know

		else: # since self.layers is a dict and `~Hashable in dict` fails
			try: return key in self.layers # can return from here too
			except: return False
	# END FUNCTION __contains__

	# BEGIN FUNCTION __getitem__
	def __getitem__(self, key) -> Any:
		# BEGIN HANDLE str
		if isinstance(key, str):
			if key in ['data', 'd']: return self.data
			elif key in ['expression', 'e']: return self.expression
			elif hasattr(self, key) and not callable(getattr(self, key)): return getattr(self, key)
			elif key in self.layers: return self.layers[key]
			else: pass
		elif isinstance(key, Iterable):
			if len(key) > 1 and len(key) <= 3 and not self.normalized['gf-icf'] and key[0] in ['e','expression']:
				# need to take the log1p before returning
				ret = self.__getitem__(tuple(['d'] + list(key))) # get it as if it was a data request
				return self.log(ret +1) # log1p
			elif isinstance(key[0], Iterable) and not isinstance(key[0], str):
				ret = super().__getitem__(key)
				if all([isinstance(i, pd.Series) for i in ret]): return pd.concat(ret, keys=key[0], axis=1, sort=False)
				else: return pd.Series(ret, index=key[0])
			else: pass
		else: pass

		return super().__getitem__(key) # can fall out of above
	# END FUNCTION __getitem__
	# END DUNDER FUNCTIONS

	# BEGIN PROPERTIES
	# BEGIN SUBOBJECTS
	@property
	def filter(self) -> utils.Filter:
		"""The filter for this Processor sample"""
		return self._filter
	@property
	def normalized(self) -> Mapping[str, bool]:
		"""Whether or not this sample has been normalized by cell or housekeeping ('hk')
		If GF-ICF has been applied, the value is True for no normalization, or the normalization entry."""
		return self._normalized
	@property
	def layers(self) -> Mapping[str, pd.DataFrame]:
		"""Any calculated values, such as tsne, clusters, etc.
		Reset upon filter change or (un)normalization
		Keys are the same as the functions the data come from
		You can remove layers using self.wipe_layer(key) to run again with different parameters
		"""
		return self._layers
	# END SUBOBJECTS

	# BEGIN FILTERED PROPERTIES
	@property
	def data(self) -> pd.DataFrame:
		"""The filtered count data, possibly normalized
		Access the unfiltered data as self.raw_data"""
		return self.filter.result
	@property
	def d(self) -> pd.DataFrame:
		"""The filtered count data, possibly normalized
		Access the unfiltered data as self.raw_data"""
		return self.data
	@property
	def expression(self) -> pd.DataFrame:
		"""The log-transformed filtered data, possibly normalized
		Same as self.data if GF-ICF normalized, i.e. not log-transformed"""
		return self.log(self.data +1) if not self.normalized['gf-icf'] else self.data
	@property
	def e(self) -> pd.DataFrame:
		"""The log-transformed filtered data, possibly normalized"""
		return self.expression

	@property
	def shape(self) -> np.ndarray:
		"""The shape of the filtered data
		Access the unfiltered shape as self.raw_shape"""
		return (sum(self.filter.idx), self.raw_data.shape[1])
	@property
	def idx(self):
		"""The index of the filtered data"""
		return self.filter.idx
	@property
	def columns(self):
		"""The columns of the data
		Same as self.raw_columns"""
		return self._raw_data.columns
	@property
	def index(self):
		"""The index of the filtered data"""
		return self.idx.index
	
	@property
	def counts(self) -> pd.Series:
		"""The total counts for each cell"""
		if not self._minimal: return self.raw_counts.loc[self.idx]
		else: return self.data.sum(axis=1)
	@property
	def genes(self) -> pd.Series:
		"""The number of detected in each cell"""
		if not self._minimal: return self.raw_genes.loc[self.idx]
		else: return (self.data != 0).sum(axis=1)
	@property
	def mito(self) -> pd.Series:
		"""The fraction of mitochondrial counts in each cell"""
		if not self._minimal: return self.raw_mito.loc[self.idx]
		elif not self.columns.str.isdigit().any(): self.data.loc[:, self.columns.str.startswith(mito_marker)].sum(axis=1) / self.counts
		else: return self.data.loc[:, self.columns.str.startswith(self.mito_marker).fillna(False)].sum(axis=1) / self.counts
	@property
	def div(self) -> pd.Series:
		"""The diversity of each cell, equal to np.log1p(self.genes)"""
		if not self._minimal: return self.raw_div[self.idx]
		else: return np.log1p(self.genes)
	@property
	def reldiv(self) -> pd.Series:
		"""The relative diversity of each cell, equal to sp.stats.stats.zscore(self.div)"""
		if not self._minimal: return self.raw_reldiv[self.idx]
		else: return pd.Series(stats.zscore(self.div), index=self.idx)
	# END FILTERED PROPERTIES
	
	# BEGIN RAW PROPERTIES
	@property
	def raw_data(self) -> pd.DataFrame:
		"""The data as passed to the constructor"""
		return self._raw_data
	@property
	def raw_shape(self) -> np.ndarray:
		"""The shape of the full raw data
		Access the filtered shape as self.shape"""
		return self.raw_data.shape
	@property
	def raw_index(self):
		"""The index of the raw data"""
		return self._raw_data.index
	@property
	def raw_columns(self):
		"""The columns of the data
		Same as self.columns"""
		return self._raw_data.columns

	@property
	def raw_counts(self) -> pd.Series:
		"""The total counts for each cell, unfiltered"""
		if not self._minimal: return self._raw_counts
		else: return self.raw_data.sum(axis=1)
	@property
	def raw_genes(self) -> pd.Series:
		"""The number of detected in each cell, unfiltered"""
		if not self._minimal: return self._raw_genes
		else: return (self.raw_data != 0).sum(axis=1)
	@property
	def raw_mito(self) -> pd.Series:
		"""The fraction of mitochondrial counts in each cell, unfiltered"""
		if not self._minimal: return self._raw_mito
		elif not self.raw_data.columns.str.isdigit().any(): self.raw_data.loc[:, self.raw_data.columns.str.startswith(mito_marker)].sum(axis=1) / self.raw_counts
		else: return self.raw_data.loc[:, self.raw_data.columns.str.startswith(self.mito_marker).fillna(False)].sum(axis=1) / self.raw_counts
	@property
	def raw_div(self) -> pd.Series:
		"""The diversity of each cell, equal to np.log1p(self.raw_genes)"""
		if not self._minimal: return self._raw_div
		else: return np.log1p(self.raw_genes)
	@property
	def raw_reldiv(self) -> pd.Series:
		"""The relative diversity of each cell, equal to sp.stats.stats.zscore(self.raw_div"""
		if not self._minimal: return self._raw_reldiv
		else: return pd.Series(stats.zscore(self.raw_div), index=self.raw_index)
	# END RAW PROPERTIES
	# END PROPERTIES
	

	# BEGIN CLASS METHODS
	# BEGIN FUNCTION from_csv
	@classmethod
	def from_csv(cls, file : Union[str, IO], *, log : Callable[[pd.DataFrame], pd.DataFrame] = np.log2, mito_marker : str = 'mt-', normalized : Optional[str] = None, sparse : bool = True, minimal : bool = False, skip_validation : bool = False, verbose : bool = False, **kwargs) -> 'Processor':
		"""Load a sample directly from a csv
		
		Parameters
		----------
		file : str, filelike
			the file containing the data
			rows are cells, columns are genes
			column name should be gene symbols
		*
		log : callable = np.log2
			the logarithm function to use
			given a pd.DataFrame, must return a pd.DataFrame with indices and columns unchanged
		mito_marker : str = 'mt-'
			the prefix used for mitochondrial gene names
			cAsE sEnSiTiVe
		normalized : str = None
			the normalization already applied to the data
			valid entries are 'cell', 'hk'/'housekeeping', 'hkcell', 'gf-icf'
			not case sensitive
		sparse : bool = True
			whether or not to use a sparse datatype
			handles the following kwargs internally:
				sep : str = ','
				header : int, Iterable[int] = 0 # NotImplementedError
				index_col : int, Iterable[int] = (0,1)
				usecols : int, str, Iterable[int,str] = None
				dtype : type, str = 'float16'
				fill_value : float = 0
		minimal : bool = False
			whether to only store minimal data; self.layers will always be empty
			leads to a lot of calculating on the fly, increasing computation time
		skip_validation : bool = False
			whether to skip validating anything
		verbose : bool = False
			whether or not to print what's going on
		**kwargs
			passed to pd.read_csv if not sparse
			pass `index_col=0` if the first column is index
		"""
		if not isinstance(skip_validation, bool) or not skip_validation:
			if verbose: print('Validating...')
			if not isinstance(sparse, bool): raise TypeError('sparse must be a bool')
			elif sparse and not isinstance(fill_value, (float, int)): raise TypeError('fill_value must be a float if sparse')
			elif not isinstance(minimal, bool): raise TypeError('minimal must be a bool')
			elif not (isinstance(file,str) or hasattr(file, 'readline')): raise TypeError('file must be str or file-like')
			else: pass

		if verbose: print('Loading...')
		if 'index_col' not in kwargs: kwargs['index_col'] = 0
		if sparse:
			sep = kwargs['delimiter'] if 'delimiter' in kwargs else kwargs['sep'] if 'sep' in kwargs else ','
			header = kwargs['header'] if 'header' in kwargs else [0]
			usecols = kwargs['usecols'] if 'usecols' in kwargs else None
			
			dtype = kwargs['dtype'] if 'dtype' in kwargs else 'float16'
			fill_value = kwargs['fill_value'] if 'fill_value' in kwargs else 0
			sparse_dtype = pd.SparseDtype(dtype,fill_value=fill_value)

			index_col = kwargs['index_col']
			if isinstance(index_col, Iterable): index_col = list(index_col)
			else: index_col = [index_col]

			df = []
			if not hasattr(file, 'readline'): f = open(file, 'r')
			else: f = file

			temp = np.array(f.readline().rstrip().split(sep))
			dat = [i for i in range(len(temp)) if i not in idx]
			temp = pd.Series(temp[dat], name=tuple(temp[idx].tolist()))

			for c in f:
				d = np.array(c.split(sep))
				df.append(pd.Series(d[dat], name=tuple(d[idx].tolist())).astype(sparse_dtype))
				
			if not hasattr(file, 'readline'): f.close()
			df = pd.concat(df, axis=1, copy=False).T
			df.columns = temp

			if verbose:
				print('Cells:', df.shape[0])
				print('Genes:', df.shape[1])
				print('Building structure...')
			return cls(df, log=log, mito_marker=mito_marker, normalized=normalized, minimal=minimal, skip_validation=skip_validation, verbose=verbose)
		else:
			df = pd.read_csv(file, **{i:kwargs[i] for i in utils.CSV_KWARGS if i in kwargs})
			if verbose:
				print('Cells:', df.shape[0])
				print('Genes:', df.shape[1])
				print('Building structure...')
			return cls(df, log=log, mito_marker=mito_marker, normalized=normalized, minimal=minimal, skip_validation=skip_validation, verbose=verbose)
	# END FUNCTION from_csv

	# BEGIN FUNCTION from_pickle
	@classmethod
	def from_pickle(cls, file : Union[str, IO], *, log : Callable[[pd.DataFrame], pd.DataFrame] = np.log2, mito_marker : str = 'mt-', normalized : Optional[str] = None, sparse : bool = True, minimal : bool = False, skip_validation : bool = False, verbose : bool = False) -> 'Processor':
		"""Load a sample directly from a pickled pd.DataFrame

		Parameters
		----------
		file : str, filelike
			the file containing the data
			rows are cells, columns are genes
			column name should be gene symbols
		*
		log : callable = np.log2
			the logarithm function to use
			given a pd.DataFrame, must return a pd.DataFrame with indices and columns unchanged
		mito_marker : str = 'mt-'
			the prefix used for mitochondrial gene names
			cAsE sEnSiTiVe
		normalized : str = None
			the normalization already applied to the data
			valid entries are 'cell', 'hk'/'housekeeping', 'hkcell', 'gf-icf'
			not case sensitive
		sparse : bool = True
			whether or not to use a sparse datatype
			handles the following kwargs internally:
				dtype : type, str = 'float16'
				fill_value : float = 0
		minimal : bool = False
			whether to only store minimal data; self.layers will always be empty
			leads to a lot of calculating on the fly, increasing computation time
		skip_validation : bool = False
			whether to skip validating anything
		verbose : bool = False
			whether or not to print what's going on
		"""
		from pickle import load

		if not isinstance(skip_validation, bool) or not skip_validation:
			if verbose: print('Validating...')
			if not isinstance(sparse, bool): raise TypeError('sparse must be a bool')
			elif sparse and not isinstance(fill_value, (float, int)): raise TypeError('fill_value must be a float if sparse')
			elif not isinstance(minimal, bool): raise TypeError('minimal must be a bool')
			elif not (isinstance(file,str) or hasattr(file, 'readline')): raise TypeError('file must be str or file-like')
			else: pass

		if hasattr(file, 'read') and hasattr(file, 'readline'): data = load(file)
		else:
			with open(file, 'rb') as f: data = load(f)

		if sparse:
			dtype = kwargs['dtype'] if 'dtype' in kwargs else 'float16'
			fill_value = kwargs['fill_value'] if 'fill_value' in kwargs else 0
			sparse_dtype = pd.SparseDtype(dtype,fill_value=fill_value)

			data = data.astype(sparse_dtype)

		if verbose:
			print('Cells:', data.shape[0])
			print('Genes:', data.shape[1])
			print('Building structure...')
		return cls(data, log=log, mito_marker=mito_marker, normalized=normalized, minimal=minimal, skip_validation=skip_validation)
	# END FUNCTION from_pickle
	# END CLASS METHODS

	# BEGIN INTERNAL METHODS
	# BEGIN FUNCTION _notify
	def _notify(self, parent = None) -> None:
		"""An internal function to rerun gf_icf if needed and clear self.layers, self.pca_, self.umap_, self.tsne_
		Called by self.filter when a filtering method is run or upon self.filter.reset()"""	
		# BEGIN RERUN gf_icf
		if self.normalized['gf-icf']:
			self.undo_normalize()
			self.gf_icf()
		# END RERUN gf_icf

		# BEGIN CLEARING
		self._layers = {}
		self._pca_ = None
		self._umap_ = None
		self._tsne_ = None
		# END CLEARING
	# END FUNCTION _notify

	# BEGIN FUNCTION pick_clusters
	def _filter_clusters(self, X : pd.DataFrame, clusters : Iterable[int]) -> pd.Series:
		"""An internal additional filter method that provides an external function for filtering based on clusters

		Parameters
		----------
		X : pd.DataFrame
			self.raw_data
		clusters : Iterable[int]
			which clusters to keep

		Returns
		-------
		pd.Series[bool]
			len == X.shape[0]
			True if self.cluster[X.index] in clusters
			all True if 'cluster' not in self
		"""
		# Check and return all True if not clustered
		if 'cluster' not in self: return pd.Series(np.ones(X.shape[0]).astype(bool), index=X.index)
		
		# BEGIN VALIDATION
		if not self.raw_data.equals(X): raise ValueError('X must be self.raw_data')
		elif not all([i in self.layers['cluster'].values for i in clusters]): raise ValueError('All clusters must be valid cluster numbers')
		else: pass
		# END VALIDATION

		ret = pd.Series(np.zeros(X.shape[0]).astype(bool), index=X.index) # start with all False
		ret[self.layers['cluster'].index] = [i in clusters for i in self.layers['cluster'].values]
		return ret

	# END FUNCTION pick_clusters
	# END INTERNAL METHODS

	# BEGIN METHODS
	# BEGIN FUNCTION plot_qc
	def plot_qc(self, *, filtered : bool = True, paint : Optional[str] = None, xlabel : Union[str, Mapping[str, Any]] = 'Sum Counts', ylabel : Union[str, Mapping[str, Any]] = 'Unique Genes', zlabel : Union[str, Mapping[str, Any]] = 'Mitochondrial Fraction', ax : Optional[Axes3D] = None, verbose : bool = False, **kwargs) -> Tuple[Figure, Axes3D]:
		'''Plots all rows in QC space, with optional coloring by the expression of a gene

		Parameters
		----------
		*
		filtered : bool = True
			whether or not to plot only the filtered cells
		paint : str = None
			the gene/column to color the points by
			if None, no coloring is added, c is used if given
			if cmap not given, plt.cm.coolwarm is the default
			if norm not given, matplotlib.colors.Normalize is the default
		xlabel : str, dict = 'Sum Counts'
			the label for the x-axis of the graph
			if dict, the kwargs to pass to ax.set_xlabel
				use key 'xlabel' for the label itself
		ylabel : str, dict = 'Unique Genes'
			the label for the y-axis of the graph
			if dict, the kwargs to pass to ax.set_ylabel
				use key 'ylabel' for the label itself
		zlabel : str, dict = 'Mitochondrial Fraction'
			the label for the colorbar of the graph
			if dict, the kwargs to pass to cbar.set_label
			use key 'label' for the label itself
		ax = None
			the Axes3D on which to draw the plot
			if None (default), a new plt.figure is drawn and fig.add_subplots is called (see **kwargs)
			NOT TESTED
		verbose : bool = False
			whether or not to print what's going on
		**kwargs
			passed variously to plt.figure, fig.add_subplots, ax.scatter
			for fig.add_subplots: if n given, nrows/ncols/index is ignored
			extra kwargs ignored silently

		Special kwargs
		--------------
		facecolor, frameon
			passed to plt.figure
		axis_alpha, fc, axis_facecolor, frame_on, axis_zorder
			passed to fig.add_subplots
		alpha, zorder
			passed to ax.scatter

		Returns
		-------
		fig, ax
			the figure and axes of the resulting plot
		'''
		# BEGIN VALIDATION
		if not isinstance(filtered, bool): raise TypeError('filtered must be a bool')
		elif not all([isinstance(i, (str,dict)) for i in [xlabel, ylabel, zlabel]]): raise TypeError('All labels must be str or dict of kwargs')
		else: pass
		# END VALIDATION

		# Internal defaults
		if not all([i in kwargs for i in ['ncols', 'nrows', 'index']]) and not 'n' in kwargs: kwargs['n'] = '111'
		if 'n' in kwargs: [exec('del kwargs[i]') for i in ['ncols', 'nrows', 'index'] if i in kwargs]

		

		# BEGIN MAKE FIGURE/AXES
		if verbose: print('Preparing axes...')
		if not ax: # if not given, have to make one
			fig = plt.figure(**{i:kwargs[i] for i in utils.FIGURE_KWARGS if i in kwargs})

			# BEGIN PREPARE ADD_SUBPLOT KWARGS
			kw_axis = {i:kwargs[i] for i in utils.ADD_SUBPLOT_KWARGS if i in kwargs}
			if 'axis_facecolor' in kwargs: kw_axis['facecolor'] = kwargs['axis_facecolor']
			elif 'facecolor' in kw_axis: del kw_axis['facecolor']
			else: pass
			if 'axis_zorder' in kwargs: kw_axis['zorder'] = kwargs['axis_zorder']
			elif 'zorder' in kw_axis: del kw_axis['zorder']
			else: pass
			# END PREPARE ADD_SUBPLOT KWARGS

			# BEGIN MAKE AXES
			if 'n' in kwargs: ax = fig.add_subplot(kwargs['n'], projection='3d', **kw_axis)
			else: ax = fig.add_subplot(kwargs['nrows'], kwargs['ncols'], kwargs['index'], projection='3d', **kw_axis)
			# END MAKE AXES
		else: # make sure we're good
			if not isinstance(ax, Axes3D): raise TypeError('ax must be an Axes3D')
			fig = ax.figure
		# END MAKE FIGURE/AXES

		# BEGIN PREPARE paint
		if not paint is None:
			if verbose: print('Fetching ' + paint)
			if paint in self.columns: # make sure we got it
				kwargs['c'] = self['e', paint] if filtered else self.log(self.raw_data[paint] +1)
			else: warn(str(paint) + ' not in data.columns, ignoring @paint')

			# Internal defaults
			if 'cmap' not in kwargs: kwargs['cmap'] = plt.cm.coolwarm
			if 'norm' not in kwargs: kwargs['norm'] = Normalize()
		# END PREPARE paint

		# Internal default
		if 's' not in kwargs: kwargs['s'] = 4

		# Plot
		if verbose: print('Plotting...')
		ax.scatter(self['counts'] if filtered else self.counts, self['genes'] if filtered else self.genes,
					self['mito'] if filtered else self.mito, **{j:kwargs[j] for j in utils.SCATTER_KWARGS if j in kwargs})

		# BEGIN LABELING
		ax.set_xlabel(xlabel) if isinstance(xlabel, str) else ax.set_xlabel(**xlabel)
		ax.set_ylabel(ylabel) if isinstance(ylabel, str) else ax.set_ylabel(**ylabel)
		ax.set_zlabel(zlabel) if isinstance(zlabel, str) else ax.set_zlabel(**zlabel)
		# END LABELING

		return fig, ax
	# END FUNCTION plot_qc

	# BEGIN NORMALIZATIONS
	# BEGIN FUNCTION gf_icf
	def gf_icf(self, *, log : Callable[[pd.Series], pd.Series] = None, verbose : bool = False) -> 'Processor':
		"""Changes self.data into GF-ICF scores (see doi:10.3389/fgene.2019.00734)
		Replaces cell or housekeeping normalized if done
		
		After GF-ICF, self.expression will return the same values as self.data
		Additionally, any change to the filter will rerun this with the same values

		Parameters
		----------
		*
		log : callable = None
			the logarithm function to use in calculation
			if None, uses self.log
		verbose : bool = False
			whether or not to print what's going on

		Returns self for stacking
		"""
		if not self.normalized['gf-icf']: # only do if not already done
			# BEGIN VALIDATION
			if isinstance(self.normalized, frozendict): # drop out early
				warn('Data loaded already normalized, so cannot normalize again')
				return self
			if log is not None and not callable(log): raise TypeError('log must be callable')
			# END VALIDATION

			# Make sure we're good
			self.undo_normalize()

			# BEGIN CALCULATION
			if verbose: print('GF...')
			tf = (self.data / self.counts[:,None]).fillna(0) # nan if whole col is 0
			
			if verbose: print('ICF...')
			nt = self.data.astype(bool).sum(axis=0)
			if log is None: log = self.log
			idf = log((self.shape[0]+1)/nt +1)
			idf[idf == np.inf] = 0 # handle nt == 0 therefore idf == np.inf

			if verbose: print('Combining...')
			gf = tf * idf
			temp = pd.concat((self._data.loc[~self.idx], gf), sort=False).loc[self.raw_data.index]
			# END CALCULATION

			# SAVE
			self._data = temp
			self._normalized['gf-icf'] = True
			if verbose: print('Done')

		return self
	# END FUNCTION gf_icf

	# BEGIN FUNCTION cell_normalize
	def cell_normalize(self, *, verbose : bool = False) -> 'Processor':
		"""Changes self.data into TPM (transcripts per million)
		Replaces GF-ICF normalization if done

		Parameters
		----------
		*
		verbose : bool = False
			whether or not to print what's going on

		Returns self for stacking
		"""
		if not self.normalized['cell']: # only do if not done already
			if isinstance(self.normalized, frozendict): # drop out early
				warn('Data loaded already normalized, so cannot normalize again')
				return self
			elif self.normalized['gf-icf']:
				if verbose: print('Undoing GF-ICF...')
				self.undo_normalize(verbose=verbose) # make sure we're good
			else: self._layers = {} # undo_normalize() would do this for us

			# Calculate and save
			if verbose: print('Calculating TPM...')
			self._data = (10**6 * self.raw_data.T / self.raw_counts).T
			self._normalized['cell'] = True
		return self
	# END FUNCTION cell_normalize

	# BEGIN FUNCTION hk_normalize
	def hk_normalize(self, hk_genes : Union[str, Iterable[str]], *, verbose : bool = False) -> 'Processor':
		"""Normalizes each cell such that the mean across all housekeeping genes is constant
		Replaces GF-ICF normalization if done

		The housekeeping list from Eisenberg 2013 (doi:10.1016/j.tig.2013.05.010) is available for
			both homo and mus under HOUSEKEEPING[genus]

		Parameters
		----------
		hk_genes : Iterable, str
			the list of housekeeping gene symbols as they appear in the columns of self.data
			alternatively, the key in HOUSEKEEPING to pull the list from

			any genes in this list not present in the sample are silently ignored
			you can check for missing genes by comparing hk_genes and self.columns.intersection(hk_genes)
			if no genes overlap, no normalization is performed and a warning issued
		*
		verbose : bool = False
			whether or not to print what's going on

		Returns self for stacking
		"""
		if self.normalized['hk'] is False: # only do if not done
			if isinstance(self.normalized, frozendict): # drop out early
				warn('Data loaded already normalized, so cannot normalize again')
				return self
			# Handle str(hk_gens)
			if isinstance(hk_genes, str) and hk_genes in HOUSEKEEPING_SPECIES:
				if verbose: print('Loading ' + hk_genes + ' genes...')
				hk_genes = pd.read_csv(HOUSEKEEPING_SPECIES[hk_genes], header=None, squeeze=True).values.tolist(),

			# BEGIN VALIDATION
			if not isinstance(hk_genes, Iterable): raise TypeError('hk_genes must be an Iterable')
			elif len(hk_genes) == 0: raise ValueError('hk_genes must not be length 0')
			else: hk_genes = list(hk_genes)[:]
			# END VALIDATION

			# DOUBLE-CHECK
			genes = self.columns.intersection(hk_genes) # hk_genes is a list
			if len(genes) == 0: warn('No genes were found in self.data. No normalization can be performed.')
			else:
				# BEGIN VALIDATION
				if len(genes) != len(hk_genes):
					missing = [i for i in hk_genes if i not in genes]
					warn('The following genes were not found: ' + str(missing))

				if self.normalized['gf-icf']:
					if verbose: print('Undoing GF-ICF...')
					self.undo_normalize(verbose=verbose) # make sure we're good
				else: self._layers = {} # undo_normalize() would do this for us
				# END VALIDATION

				# Calculate and save
				if verbose: print('Calculating...')
				self._data = (self._data.T / self._data[genes].mean(axis=1)).T
				self._normalized['hk'] = hk_genes

		return self
	# END FUNCTION hk_normalize

	# BEGIN FUNCTION undo_normalize
	def undo_normalize(self, *, verbose : bool = False) -> 'Processor':
		"""Undo all normalization of data
		Does nothing if data were loaded normalized

		Parameters
		----------
		*
		verbose : bool = False
			whether or not to print what's going on

		Returns self for stacking
		"""
		if not isinstance(self._normalized, frozendict):
			if verbose: print('Resetting data from self.raw_data...')
			for k in self.normalized.keys(): self._normalized[k] = False # set all normalized False
			self._data = self.raw_data.copy() # reset _data
			self._layers = {} # self _layers
		else: warn('Data loaded already normalized, so cannot undo')

		return self
	# END FUNCTION undo_normalize
	# END NORMALIZATIONS

	# BEGIN FUNCTION finalize
	def finalize(self, *, verbose : bool = False) -> Sample:
		"""Convert Processor to Sample
		To use in finalizing your data as is

		Parameters
		----------
		*
		verbose : bool = False
			whether or not to print what's going on
		"""
		if verbose:
			print('Cells:' + self.shape[0])
			print('Genes:' + self.shape[1])
		return Sample(self.data, pca_=self.pca_, umap_=self.umap_, tsne_=self.tsne_, verbose=verbose)
	# BEGIN FUNCTION finalize

	# BEGIN FUNCTION shrink
	def shrink(self, *, verbose : bool = False) -> 'Processor':
		"""Shrink this Processor to the current data instead of raw_data
		Lowers memory use if only a small subset of cells is selected

		Parameters
		----------
		*
		verbose : bool = False
			whether or not to print what's going on
		"""
		# store for later
		if verbose: print('Storing current normalization...')
		normalized = self.normalized.copy()
		self.undo_normalize(verbose=verbose)
		data, counts, genes, div, reldiv, mito = self.data, self.counts, self.genes, self.div, self.reldiv, self.mito

		# BEGIN RESET NUMBERS
		if verbose: print('Resetting...')
		self._raw_data = data
		self._data = data.copy() # to keep separate
		self._raw_counts = counts
		self._raw_genes = genes
		self._raw_div = div
		self._raw_reldiv = reldiv
		self._raw_mito = mito
		# END RESET NUMBERS


		# BEGIN HANDLE Filter
		# self.filter works with raw data and so needs replacing
		if isinstance(self.normalized, frozendict): # data were given normalized
			self._filter = utils.Filter(self, [('raw_data', self._filter_clusters, 'cluster')])
		else: # data were raw counts
			self._filter = utils.Filter(self, [('raw_counts', ('linear', 'above'), 'counts'), ('raw_genes', ('linear', 'above'), 'genes'),
											('raw_mito', ('linear', 'below'), 'mito'), ('raw_counts', ('secant', 'above'), 'sec_counts'),
											('raw_reldiv', ('bimodal', 'above'), 'reldiv'), ('raw_data', self._filter_clusters, 'cluster')])
			# BEGIN REDO NORMALIZATION
			if verbose: print('Renormalizing if needed...')
			if normalized['gf-icf']: self.gf_icf(verbose=verbose)
			elif normalized['cell'] and normalized['hk'] is not False: self.cell_normalize(verbose=verbose).hk_normalize(normalized['hk'], verbose=verbose)
			elif normalized['cell']: self.cell_normalize(verbose=verbose)
			elif normalized['hk'] is not False: self.hk_normalize(normalized['hk'], verbose=verbose)
			else: pass
			# END REDO NORMALIZATION
		# END HANDLE Filter

		return self
	# END FUNCTION shrink
	# END METHODS
# END CLASS Processor






# BEGIN CLASS MultiProcessor
class MultiProcessor(Processor):
	"""For processing multiple samples together

	Parameters
	----------
	data : pd.DataFrame, Mapping[str, pd.DataFrame], Iterable[pd.DataFrame]
		the data to be processed
		can either be
			pd.DataFrame with a MultiIndex of (sample, cell) and columns are genes
			Mapping[sample : pd.DataFrame of cells (rows) by genes]
			Iterable[pd.DataFrame of cells (rows) by genes] with samples given as @samples
	*
	samples : Optional[Iterable] = None
		only used if data is an Iterable
		gives the sample names in the same order as the DataFrames in @data
	log : callable = np.log2
		the logarithm function to use
		given a pd.DataFrame, must return a pd.DataFrame with indices and columns unchanged
	mito_marker : str = 'mt-'
		the prefix used for mitochondrial gene names
		cAsE sEnSiTiVe
	normalized : str = None
		the normalization already applied to the data
		valid entries are 'cell', 'hk'/'housekeeping', 'hkcell', 'gf-icf'
		not case sensitive
	sparse : bool = True
		whether or not to use a sparse datatype
		handles the following kwargs internally:
			fill_value : float = 0
	minimal : bool = False
		whether to only store minimal data; self.layers will always be empty
		leads to a lot of calculating on the fly, increasing computation time
	only_overlap : bool = False
		whether or not to only use the overlapping genes
		if False, fills unmeasured genes with fill_value
	skip_validation : bool = False
		whether to skip validating as much as possible
	verbose : bool = False
		whether or not to print what's going on
	counts, genes, mito : Optional[Union[pd.Series, Mapping[str, pd.Series], Iterable[pd.Series]]] = None
		can give any combination of precomputed values
		each must be have the same index as data
			counts is the sum of counts per cell
			genes is the number of unique genes per cell
			mito is the mitochondrial fraction per cell
	"""
	# BEGIN DUNDER FUNCTIONS
	# BEGIN FUNCTION __init__
	def __init__(self, data : Union[pd.DataFrame, Mapping[str, pd.DataFrame], Iterable[pd.DataFrame]], *, samples : Optional[Iterable] = None, log : Callable[[pd.Series], pd.Series] = np.log2, mito_marker : str = 'mt-', normalized : Optional[str] = None, sparse : bool = True, fill_value : float = 0, minimal : bool = False, only_overlap : bool = False, skip_validation : bool = False, verbose : bool = False, counts : Optional[Union[pd.Series, Mapping[str, pd.Series], Iterable[pd.Series]]] = None, genes : Optional[Union[pd.Series, Mapping[str, pd.Series], Iterable[pd.Series]]] = None, mito : Optional[Union[pd.Series, Mapping[str, pd.Series], Iterable[pd.Series]]] = None, **kwargs) -> None:
		dtype = kwargs['dtype'] if 'dtype' in kwargs else 'float16'
		fill_value = kwargs['fill_value'] if 'fill_value' in kwargs else 0

		if not isinstance(skip_validation, bool) or not skip_validation:
			if verbose: print('Validating...')
			if not isinstance(data, pd.DataFrame) and not isinstance(data, Mapping) and isinstance(data, Iterable):
				if samples is None: raise ValueError('Must provide @samples if @data is Iterable')
				elif not len(samples) == len(data): raise ValueError('@samples and @data are not the same length')
				else: pass
				data = {s:d for s,d in zip(samples,data)}

			if isinstance(data, Mapping):
				if not all([isinstance(i, str) for i in data.keys()]): raise TypeError('All keys must be str')
				elif not all([isinstance(i, pd.DataFrame) for i in data.values()]):
					raise TypeError('All values must be pd.DataFrame. Given:' + ', '.join([str(type(i)) for i in data.values()]))
				else:
					if only_overlap:
						genes = [set(v.columns) for v in data.values()]
						for i in genes[1:]:
							genes[0] = genes[0].intersection(i)
						data = {k:v[genes[0]] for k,v in data.items()}

					data = pd.concat(data, axis=0, sort=True, copy=False).astype(dtype).fillna(fill_value)

			elif not isinstance(data, pd.DataFrame): raise TypeError('data must be pd.DataFrame, Mapping[str : pd.DataFrame], or Iterable[pd.DataFrame]')
			else: pass
		else:
			if not isinstance(data, pd.DataFrame) and not isinstance(data, Mapping) and isinstance(data, Iterable):
				data = {s:d for s,d in zip(samples,data)}

			if isinstance(data, Mapping):
				if only_overlap:
					genes = [set(v.columns) for v in data.values()]
					for i in genes[1:]:
						genes[0] = genes[0].intersection(i)
					data = {k:v[genes[0]] for k,v in data.items()}

				data = pd.concat(data, axis=0, sort=True, copy=False).astype(dtype).fillna(fill_value)
			elif not isinstance(data, pd.DataFrame): raise TypeError('data must be pd.DataFrame, Mapping[str : pd.DataFrame], or Iterable[pd.DataFrame]')
			else: pass

		if sparse: data = data.astype(pd.SparseDtype(dtype,fill_value=fill_value))
		data = data[data.columns[(data != 0).any()]]

		if not minimal:
			if counts is None or skip_validation: pass
			elif isinstance(counts, pd.Series):
				if not len(data.index.intersection(counts.keys())) == data.shape[0]: raise ValueError('counts must have the same index as data')
				else: pass
			elif isinstance(counts, Mapping):
				if not len(data.index.intersection(counts.keys())) == data.shape[0]: raise ValueError('counts and data must have the same keys')
				elif not all([isinstance(i, pd.Series) for i in counts.values()]): raise TypeError('all values in counts must be pd.Series')
				else: counts = pd.concat(counts, axis=0, copy=False)
			elif isinstance(counts, Iterable):
				if not all([isinstance(i, pd.Series) for i in counts]): raise TypeError('counts must be ')
				elif not len(counts) == data.shape[0]: raise ValueError('counts and data must be the same length')
				else: counts = pd.concat(data, axis=0, keys=samples, copy=False)
			else: raise TypeError('counts must be pd.DataFrame, Mapping[str : pd.DataFrame], or Iterable[pd.DataFrame]')

			if genes is None or skip_validation: pass
			elif isinstance(genes, pd.Series):
				if not len(data.index.intersection(counts.keys())) == data.shape[0]: raise ValueError('counts must have the same index as data')
				else: pass
			elif isinstance(genes, Mapping):
				if not len(data.index.intersection(genes.keys())) == data.shape[0]: raise ValueError('genes and data must have the same keys')
				elif not all([isinstance(i, pd.Series) for i in genes.values()]): raise TypeError('all values in genes must be pd.Series')
				else: genes = pd.concat(genes, axis=0, copy=False)
			elif isinstance(genes, Iterable):
				if not all([isinstance(i, pd.Series) for i in genes]): raise TypeError('genes must be ')
				elif not len(genes) == data.shape[0]: raise ValueError('genes and data must be the same length')
				else: genes = pd.concat(data, axis=0, keys=samples, copy=False)
			else: raise TypeError('genes must be pd.DataFrame, Mapping[str : pd.DataFrame], or Iterable[pd.DataFrame]')


			if mito is None or skip_validation: pass
			elif isinstance(mito, pd.Series):
				if not len(data.index.intersection(mito.keys())) == data.shape[0]: raise ValueError('mito must have the same index as data')
				else: pass
			elif isinstance(mito, Mapping):
				if not len(data.index.intersection(mito.keys())) == data.shape[0]: raise ValueError('mito and data must have the same keys')
				elif not all([isinstance(i, pd.Series) for i in mito.values()]): raise TypeError('all values in mito must be pd.Series')
				else: mito = pd.concat(mito, axis=0, copy=False)
			elif isinstance(mito, Iterable):
				if not all([isinstance(i, pd.Series) for i in mito]): raise TypeError('mito must be ')
				elif not len(mito) == data.shape[0]: raise ValueError('mito and data must be the same length')
				else: mito = pd.concat(data, axis=0, keys=samples, copy=False)
			else: raise TypeError('mito must be pd.DataFrame, Mapping[str : pd.DataFrame], or Iterable[pd.DataFrame]')

		super().__init__(data=data, counts=counts, genes=genes, mito=mito, log=log, mito_marker=mito_marker, normalized=normalized, minimal=minimal, skip_validation=skip_validation, verbose=verbose)

	# END FUNCTION __init__
	# END DUNDER FUNCTIONS
	
	# BEGIN CLASS METHODS
	# BEGIN CLASS METHOD from_csvs
	@classmethod
	def from_csvs(cls, files : Union[Iterable[Union[str,IO]], Mapping[str, Union[str,IO]]], *, samples : Optional[Iterable[str]] = None, log : Callable[[pd.DataFrame], pd.DataFrame] = np.log2, mito_marker : str = 'mt-', normalized : Optional[str] = None, sparse : bool = True, minimal : bool = False, skip_validation : bool = False, only_overlap : bool = False, verbose : bool = False, **kwargs) -> 'MultiProcessor':
		"""Loads multiple samples for processing from individual CSV files

		Parameters
		----------
		files : Union[Iterable[Union[str, IO]], Mapping[str, Union[str,IO]]]
			if Mapping, name: file
			elif Iterable[str], name = file.split('/')[-1].split('.')[0]
				eg. path/to/file.example.csv -> file
			elif Iterable[IO], name = file.__name__
		*
		samples : Optional[Iterable[str]] = None
			the names to give the samples, in the same order as in @files
			only used if files is a list
		log : callable = np.log2
			the logarithm function to use
			given a pd.DataFrame, must return a pd.DataFrame with indices and columns unchanged
		mito_marker : str = 'mt-'
			the prefix used for mitochondrial gene names
			cAsE sEnSiTiVe
		normalized : str = None
			the normalization already applied to the data
			valid entries are 'cell', 'hk'/'housekeeping', 'hkcell', 'gf-icf'
			not case sensitive
		sparse : bool = True
			whether or not to use a sparse datatype
			handles the following kwargs internally:
				sep : str = ','
				header : int, Iterable[int] = 0 # NotImplementedError
				index_col : int, Iterable[int] = (0,1)
				usecols : int, str, Iterable[int,str] = None
				dtype : type, str = 'float16'
				fill_value : float = 0
		minimal : bool = False
			whether to only store minimal data; self.layers will always be empty
			leads to a lot of calculating on the fly, increasing computation time
		skip_validation : bool = False
			whether to skip validating as much as possible
		only_overlap : bool = False
			whether or not to only use the overlapping genes
			if False, fills unmeasured genes with fill_value
		verbose : bool = False
			whether or not to print what's going on
		**kwargs
			passed to pd.read_csv if not sparse
			if not passed `index_col` defaults to (0,1)
		"""
		if not isinstance(skip_validation, bool) or not skip_validation:
			if verbose: print('Validating...')
			if not isinstance(sparse, bool): raise TypeError('sparse must be a bool')
			elif sparse and not isinstance(fill_value, (float, int)): raise TypeError('fill_value must be a float if sparse')
			elif not isinstance(minimal, bool): raise TypeError('minimal must be a bool')
			else: pass
			
			if isinstance(files, Mapping):
				if not all([isinstance(i, str) for i in files.keys()]): raise TypeError('All keys must be str')
				elif not all([isinstance(i,str) or hasattr(i,'readline') for i in files.values()]): raise TypeError('All values must be str or file-like')
				else: pass
			elif isinstance(files, Iterable):
				if not all([isinstance(i,str) or hasattr(i, 'readline') for i in files]): raise TypeError('All files must be str or file-like')
				files = {f:s for f,s in zip(files,samples)}
			else: raise TypeError('files must be a Mapping[str: str/file-like] or Iterable[str/file-like]')
		elif not isinstance(files, Mapping) and isinstance(files, Iterable): files = {s:f for s,f in zip(samples,files)}
		else: pass

		if 'index_col' not in kwargs: kwargs['index_col'] = (0,1)
		if sparse:
			sep = kwargs['delimiter'] if 'delimiter' in kwargs else kwargs['sep'] if 'sep' in kwargs else ','
			header = kwargs['header'] if 'header' in kwargs else [0]
			usecols = kwargs['usecols'] if 'usecols' in kwargs else None

			dtype = kwargs['dtype'] if 'dtype' in kwargs else 'float16'
			fill_value = kwargs['fill_value'] if 'fill_value' in kwargs else 0
			sparse_dtype = pd.SparseDtype(dtype,fill_value=fill_value)

			index_col = kwargs['index_col']
			if isinstance(index_col, Iterable): index_col = list(index_col)
			else: index_col = [index_col]

			dfs = {}
			for k,v in files.items():
				if verbose: print('Loading ' + k + '...')
				if not hasattr(v, 'readline'): f = open(v, 'r')
				else: f = v

				df = []
				temp = np.array(f.readline().rstrip().split(sep))
				if usecols: dat = [i for i in range(len(temp)) if i not in index_col and (i in usecols or temp[i] in usecols)]
				else: dat = [i for i in range(len(temp)) if i not in index_col]
				temp = pd.Series(temp[dat], name=tuple(temp[index_col].tolist()))

				for c in f:
					d = np.array(c.split(sep))
					df.append(pd.Series(d[dat], name=tuple(d[index_col].tolist())).astype(sparse_dtype))

				if not hasattr(v, 'readline'): f.close()
				df = pd.concat(df, axis=1, copy=False).T.astype(sparse_dtype)
				df.columns = temp
				if verbose: print('\tloaded size:', df.shape)

				if len(dfs):
					if only_overlap:
						df = df[dfs[list(dfs.keys())[0]].columns.intersection(df.columns)]
						dfs = {k:v[df.columns] for k,v in dfs.items()}
					else:
						col = set(dfs[list(dfs.keys())[0]].columns) - set(df.columns)
						cols = set(df.columns) - set(dfs[list(dfs.keys())[0]].columns)
						
						if len(col): df = pd.concat([df,pd.DataFrame(np.tile(fill_value, (df.shape[0],len(col))), index=df.index,columns=col,copy=False
													).astype(sparse_dtype)],sort=False, axis=1)
						if len(cols): dfs = {k:pd.concat([v,pd.DataFrame(np.tile(fill_value, (v.shape[0],len(cols))), index=v.index,columns=cols,copy=False
														 ).astype(sparse_dtype)], sort=False, axis=1) for k,v in dfs.items()}

				dfs[k] = df
				if verbose: print('\tkept size:', df.shape)

			if verbose:
				print('Cells:', {k:v.shape[0] for k,v in dfs.items()})
				print('Total cells:', sum([v.shape[0] for v in dfs.values()]))
				print('Genes:', dfs[list(dfs.keys())[0]].shape[1])
				print('Building structure...')
			return cls(dfs, log=log, mito_marker=mito_marker, minimal=minimal, skip_validation=skip_validation, verbose=verbose)
		else:
			dfs = {k:pd.read_csv(v, **{i:kwargs[i] for i in utils.CSV_KWARGS if i in kwargs}) for k,v in files.items()}
			if verbose:
				print('Cells:', {k:v.shape[0] for k,v in dfs.items()})
				print('Total cells:', sum([v.shape[0] for v in dfs.values()]))
				print('Genes:', dfs[list(dfs.keys())[0]].shape[1])
				print('Building structure...')
			return cls(dfs, log=log, mito_marker=mito_marker, minimal=minimal, skip_validation=skip_validation, only_overlap=only_overlap, verbose=verbose)
	# END CLASS METHOD from_csvs

	# BEGIN CLASS METHOD from_single_csv
	@classmethod
	def from_single_csv(cls, file : Union[str, IO], *, log : Callable[[pd.DataFrame], pd.DataFrame] = np.log2, mito_marker : str = 'mt-', normalized : Optional[str] = None, sparse : bool = True, minimal : bool = False, skip_validation : bool = False, verbose : bool = False, **kwargs) -> 'MultiProcessor':
		"""Loads multiple samples for processing from a single CSV file
		the first column is assumed to be the sample identifier and the second a unique identifier for each row
		ie `index_col=(0,1)` is passed to pd.read_csv if not otherwise given

		Parameters
		----------
		file : Union[str, IO]
			the file containing the data
			rows are cells, columns are genes
			column name should be gene symbols

			assumes first column is sample, second column is unique identifier
			otherwise, pass `index_col` kwarg for pd.read_csv
		*
		log : callable = np.log2
			the logarithm function to use
			given a pd.DataFrame, must return a pd.DataFrame with indices and columns unchanged
		mito_marker : str = 'mt-'
			the prefix used for mitochondrial gene names
			cAsE sEnSiTiVe
		normalized : str = None
			the normalization already applied to the data
			valid entries are 'cell', 'hk'/'housekeeping', 'hkcell', 'gf-icf'
			not case sensitive
		sparse : bool = True
			whether or not to use a sparse datatype
			handles the following kwargs internally:
				sep : str = ','
				header : int, Iterable[int] = 0 # NotImplementedError
				index_col : int, Iterable[int] = (0,1)
				usecols : int, str, Iterable[int,str] = None
				dtype : type, str = 'float16'
				fill_value : float = 0
		minimal : bool = False
			whether to only store minimal data; self.layers will always be empty
			leads to a lot of calculating on the fly, increasing computation time
		skip_validation : bool = False
			whether to skip validating as much as possible
		verbose : bool = False
			whether or not to print what's going on
		**kwargs
			passed to pd.read_csv if not sparse
		"""
		if not isinstance(skip_validation, bool) or not skip_validation:
			if verbose: print('Validating...')
			if not isinstance(sparse, bool): raise TypeError('sparse must be a bool')
			elif not isinstance(minimal, bool): raise TypeError('minimal must be a bool')
			elif not (isinstance(file,str) or hasattr(file, 'readline')): raise TypeError('file must be str or file-like')
			else: pass

		if verbose: print('Loading...')
		if 'index_col' not in kwargs: kwargs['index_col'] = (0,1)
		if sparse:
			sep = kwargs['delimiter'] if 'delimiter' in kwargs else kwargs['sep'] if 'sep' in kwargs else ','
			header = kwargs['header'] if 'header' in kwargs else [0]
			usecols = kwargs['usecols'] if 'usecols' in kwargs else None
			
			dtype = kwargs['dtype'] if 'dtype' in kwargs else 'float16'
			fill_value = kwargs['fill_value'] if 'fill_value' in kwargs else 0
			sparse_dtype = pd.SparseDtype(dtype,fill_value=fill_value)

			index_col = kwargs['index_col']
			if isinstance(index_col, Iterable): index_col = list(index_col)
			else: index_col = [index_col]

			df = []
			if not hasattr(file, 'readline'): f = open(file, 'r')
			else: f = file

			temp = np.array(f.readline().rstrip().split(sep))
			dat = [i for i in range(len(temp)) if i not in index_col]
			temp = pd.Series(temp[dat], name=tuple(temp[index_col].tolist()))

			for c in f:
				d = np.array(c.split(sep))
				df.append(pd.Series(d[dat], name=tuple(d[index_col].tolist())).astype(sparse_dtype))

			if not hasattr(file, 'readline'): f.close()
			df = pd.concat(df, axis=1, copy=False).T
			df.columns = temp

			if verbose:
				print('Cells:', df.shape[0])
				print('Genes:', df.shape[1])
				print('Building structure...')
			return cls(df, log=log, mito_marker=mito_marker, normalized=normalized, minimal=minimal, skip_validation=skip_validation, verbose=verbose)
		else:
			df = pd.read_csv(file, **{i:kwargs[i] for i in utils.CSV_KWARGS if i in kwargs})
			if verbose:
				print('Cells:', df.shape[0])
				print('Genes:', df.shape[1])
				print('Building structure...')
			return cls(df, log=log, mito_marker=mito_marker, normalized=normalized, minimal=minimal, skip_validation=skip_validation, verbose=verbose)
	# END CLASS METHOD from_single_csv

	# BEGIN CLASS METHOD from_many
	@classmethod
	def from_many(cls, processors : Union[Mapping[str, Processor], Iterable[Processor]], *, samples : Optional[Iterable[str]] = None, log : Callable[[pd.DataFrame], pd.DataFrame] = np.log2, mito_marker : str = 'mt-', normalized : Optional[str] = None, sparse : bool = True, minimal : bool = False, skip_validation : bool = False, verbose : bool = False) -> 'MultiProcessor':
		"""Create a MultiProcessor from multiple Processors

		Parameters
		----------
		processors : Mapping[str, Processor], Iterable[Processor]
			the sample Processors to condense into a single MultiProcessor
			uses .data, .counts, .genes, .mito
		*
		samples : Optional[Iterable[str]] = None
			the names to give the samples, in the same order as in @processors
			only used if processors is a list
		log : callable = np.log2
			the logarithm function to use
			given a pd.DataFrame, must return a pd.DataFrame with indices and columns unchanged
		mito_marker : str = 'mt-'
			the prefix used for mitochondrial gene names
			cAsE sEnSiTiVe
		normalized : str = None
			the normalization already applied to the data
			valid entries are 'cell', 'hk'/'housekeeping', 'hkcell', 'gf-icf'
			not case sensitive
		sparse : bool = True
			handles the following kwargs internally:
				dtype : type, str = 'float16'
				fill_value : float = 0
		minimal : bool = False
			whether to only store minimal data; self.layers will always be empty
			leads to a lot of calculating on the fly, increasing computation time
		skip_validation : bool = False
			whether to skip validating as much as possible
		verbose : bool = False
			whether or not to print what's going on
		"""
		if not isinstance(skip_validation, bool) or not skip_validation:
			if verbose: print('Validating...')
			if not isinstance(sparse, bool): raise TypeError('sparse must be a bool')
			elif sparse and not isinstance(fill_value, (float, int)): raise TypeError('fill_value must be a float if sparse')
			elif not isinstance(minimal, bool): raise TypeError('minimal must be a bool')
			else: pass

			if isinstance(samples, Iterable): samples = {n:s for n,s in zip(names,samples)}

			if isinstance(samples, Mapping) and not all([isinstance(v, Sample) for v in samples.values()]): raise ValueError('samples must be Mapping[str:Sample], or Iterable[Sample] with names given')
			else: raise TypeError('samples must be Mapping[str:Sample], or Iterable[Sample] with names given')
		elif isinstance(samples, Iterable): samples = {n:s for n,s in zip(names,samples)}
		else: pass

		if verbose: print('Gathering data...')
		if not sparse:
			return cls({k:v.data for k,v in processors.items()}, counts={k:v.counts for k,v in processors.items()} if not minimal else None,
				genes={k:v.genes for k,v in processors.items()} if not minimal else None, mito={k:v.mito for k,v in processors.items()} if not minimal else None,
				log=log, mito_marker=mito_marker, normalized=normalized, minimal=minimal, skip_validation=skip_validation, verbose=verbose)
		else:
			dtype = kwargs['dtype'] if 'dtype' in kwargs else 'float16'
			fill_value = kwargs['fill_value'] if 'fill_value' in kwargs else 0
			sparse_dtype = pd.SparseDtype(dtype,fill_value=fill_value)

			return cls({k:v.data.astype(sparse_dtype) for k,v in processors.items()}, counts={k:v.counts for k,v in processors.items()} if not minimal else None,
				genes={k:v.genes for k,v in processors.items()} if not minimal else None, mito={k:v.mito for k,v in processors.items()} if not minimal else None,
				log=log, mito_marker=mito_marker, normalized=normalized, minimal=minimal, skip_validation=skip_validation, verbose=verbose)
	# END CLASS METHOD from_many
	# END CLASS METHODS

	# BEGIN METHODS
	# BEGIN FUNCTION finalize
	def finalize(self, *, verbose : bool = False) -> Sample:
		"""Convert Processor to Sample
		To use in finalizing your data as is

		Parameters
		----------
		*
		verbose : bool = False
			whether or not to print what's going on
		"""
		if verbose:
			print('Cells:' + self.shape[0])
			print('Genes:' + self.shape[1])
		return MultiSample(self.data, pca_=self.pca_, umap_=self.umap_, tsne_=self.tsne_)
	# BEGIN FUNCTION finalize
	# END METHODS
# END CLASS MultiProcessor