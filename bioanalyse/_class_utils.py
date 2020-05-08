from typing import Any, Hashable, Iterable

from numpy import array, ndarray
from pandas import DataFrame, Series

# BEGIN FUNCTION _getitem
def _getitem(self, key) -> Any:
	"""Implements __getitem__(self, key) for arbitray self, key

	Special case handling implemented for:
		self is list, pd.Series, pd.DataFrame, np.ndarray
	"""

	# BEGIN list HANDLING
	if isinstance(self, list): # list[key]
		# BEGIN list[tuple] HANDLNIG
		if isinstance(key, tuple): # list[tuple] -> list[...]
			try: temp = self[key[0]] # for list[(key,...)] -> list[key]
			except Exception as e: raise KeyError('Key not found in ' + str(type(self)) + '[' + ', '.join([str(i) if not isinstance(i, str) else '\''+i+'\'' for i in key]) + ']') from e
			
			if len(key) == 1: return temp # we already got it!
			else:
				try:
					if len(key) == 2: return _getitem(temp, key[1])
					else: return _getitem(temp, key[1:])
				except KeyError as e: raise KeyError('Key not found in ' + str(type(self)) + '[' + ', '.join([str(i) if not isinstance(i,str) else '\''+i+'\'' for i in key[:1]]) + str(e).split('[')[1]) from e
		# END list[tuple] HANDLNIG

		# BEGIN DEAL WITH THE PROBLEMS OF LISTS
		else:
			try: return self[key] # do this becuase lists suck
			except Exception as e:
				try: return _getitem(array(self), key) # try it again as an array because they're better
				except KeyError as f: raise KeyError('Key not found in ' + str(type(self)) + '[' + ', '.join([str(i) if not isinstance(i,str) else '\''+i+'\'' for i in key]) + str(f).split('[')[1]) from f
		# END DEAL WITH THE PROBLEMS OF LISTS
	# END list HANDLING

	# BEGIN pd.Series HANDLING
	elif isinstance(self, Series):
		try: return self[key]
		except Exception as e: # Series[random list[str,int](index)], Series[[i,j],k]
			if isinstance(key, Iterable) and not isinstance(key, str): # some basic qualifications
				if all([i in self for i in key]): return pd.Series([self[i] for i in key], names=key) # assume self[i] if i in self; for self[random list[str](index)]
				elif all([isinstance(i, int) and len(self.index) + i > -1 and i < len(self.index) for i in key]): # allow all int [-len,len); for self[random list[int](index)]
					return pd.Series([self[self.index[i].unique()] for i in key], names=[self.index[i].unique() for i in key]) # can't fail, but can be unexpectedly long
				else:
					try: return [_getitem(self, i) for i in key] # for self[((i,j),k)] -> [self[i,j], self[k]] not [self[i,k], self[j,k]]
					except Exception as f: raise KeyError('Key not found in ' + str(type(self)) + '[' + (str(key[0]) if not isinstance(key[0], str) else '\''+key[0]+'\'') + str(f).split('[')[1]) from e
			else: raise KeyError('Key not found in ' + str(type(self)) + '[' + ', '.join([str(i) if not isinstance(i, str) else '\''+i+'\'' for i in key]) + ']') # make sure str are quoted 
	# END pd.Series HANDLING

	# BEGIN pd.Dataframe HANDLING
	elif isinstance(self, DataFrame): # DataFrame[...]
		# BEGIN pd.DataFrame[float] HANDLING
		if isinstance(key, float): # DataFrame[float]
			if key in self.index: return self.loc[key] # assumes .loc[key] if key in index; self[list[float](index)]
			else: # DataFrame [~index : int]
				try: return self[key] # just give it to pandas for everything else, but return it with a KeyError
				except Exception as e: raise KeyError('Key not found in ' + str(type(self)) + '[' + ', '.join([str(i) if not isinstance(i, str) else '\''+i+'\'' for i in key]) + ']') from e
		# END pd.DataFrame[float] HANDLING

		# BEGIN pd.DataFrame[int] HANDLING
		elif isinstance(key, int): # DataFrame[int]
			if key in self.index: return self.loc[key] # assumes .loc[key] if key in index; self[row : int]
			elif key in self: return self[key] # assumes self[key] if key in self; self[col : int]
			else: # DataFrame[list[int](random)]
				try: return self.iloc[key] # give it to iloc for anything else, but raise it with a KeyError
				except Exception as e: raise KeyError('Key not found in ' + str(type(self)) + '[' + (str(key) if not isinstance(key, str) else '\'' + key + '\'') + ']') from e
		# END pd.DataFrame[int] HANDLING

		# BEGIN pd.DataFrame[pd.Series] HANDLING
		elif isinstance(key, Series): # DataFrame[Series]
			if len(self.index.intersection(key.index)) == self.shape[0]: return self[key[self.index.intersection(key.index)]] # df[Series(all index)]; for df[col == 0] -> df[all rows for which Series[row] is True]
			elif len(self.columns.intersection(key.index)) == self.shape[1]: return self[self.columns[key[self.columns.intersection(key.index)]]] # df[Series(all columns)]; for df[row == 0] -> df[all cols for which Series[col] is True]
			else:
				try: return self[key] # just give it to pandas, but return it with a KeyError
				except Exception as e: raise KeyError('Key not found in ' + str(type(self)) + '[' + (str(key) if not isinstance(key, str) else '\'' + key + '\'') + ']') from e
		# END pd.DataFrame[pd.Series] HANDLING

		# BEGIN pd.DataFrame[np.ndarray] HANDLING
		elif isinstance(key, ndarray): # DataFrame[array]
			if key.ndim == 1: # DataFrame[1d-array]
				if key.shape[0] == self.shape[1] and all([i >= -self.shape[1] and i < self.shape[1] for i in key]): return self.T[key].T # for DataFrame[array(all int in range(-len(index),len(index)) )] -> DataFrame[self[i] for each i]
				else:
					try: return self[key] # just give it to pandas, but raise it with a KeyError
					except Exception as e: raise KeyError('Key not found in ' + str(type(self)) + '[' + (str(key) if not isinstance(key, str) else '\'' + key + '\'') + ']') from e
			else: # DataFrame[matrix]
				try: self[key] # just give to pandas, but raise it with a KeyError
				except Exception as e: raise KeyError('Key not found in ' + str(type(self)) + '[' + (str(key) if not isinstance(key, str) else '\'' + key + '\'') + ']') from e
		# END pd.DataFrame[np.ndarray] HANDLING

		# BEGIN pd.DataFrame[tuple] HANDLING
		elif isinstance(key, tuple): # DataFrame[tuple]

			# BEGIN pd.DataFrame[Any, Any] HANDLING
			if not isinstance(key, Series) and key in self.index: return self.loc[key] # DataFrame[row : tuple]
			elif not isinstance(key, Series) and key in self.columns: return self[key] # DataFrame[col : tuple]
			elif len(key) == 2: # DataFrame[row, col]
				try: return self.loc[key] # is it [str, str]?
				except Exception as e:
					try: return self.iloc[key] # is it [int, int]?
					except Exception as f:
						try: temp = self.loc[key[0]] # is it [str, col]
						except Exception as g:
							try: temp = self.iloc[key[0]] # is it [int, col]
							except Exception as h: raise KeyError('Key not found in ' + str(type(self)) + '[' + (str(key) if not isinstance(key, str) else '\'' + key + '\'') + ']') from h
							else: # temp is Series from [int, col]
								try: return temp[key[1]] # just let pandas handle it, but raise it as a KeyError
								except Exception as i: raise KeyError('Key not found in ' + str(type(self)) + '[' + (str(key) if not isinstance(key, str) else '\'' + key + '\'') + ']') from i
						else: # temp is Series from [str, col]
							try: return temp[key[1]] # just let pandas handle it, but raise it as a KeyError
							except Exception as h: raise KeyError('Key not found in ' + str(type(self)) + '[' + (str(key) if not isinstance(key, str) else '\'' + key + '\'') + ']') from h
			# END pd.DataFrame[Any, Any] HANDLING

			# BEGIN pd.DataFrame[Any, Any, ...] HANDLING
			else:
				if not isinstance(key, Series) and key in self.index: return self.loc[key] # DataFrame[row : tuple]
				elif not isinstance(key, Series) and key in self.columns: return self[key] # DataFrame[col : tuple]
				else:
					try: temp = self.loc[key[:2]] # is it [str, str, ...]?
					except Exception as e:
						try: temp = self.iloc[key[:2]] # is it [int, int, ...]?
						except Exception as f:
							try: temp = _getitem(temp, key[2])
							except KeyError as g: raise KeyError('Key not found in ' + str(type(self)) + '[' + ', '.join([str(i) if not isinstance(i,str) else '\''+i+'\'' for i in key[2:]]) + str(f).split('[')[1]) from g
							else:
								try: # not easy, just throw a recursion with a KeyError
									if len(key) == 3: return _getitem(temp, key[2])
									else: return _getitem(temp, key[2:])
								except KeyError as g: raise KeyError('Key not found in ' + str(type(self)) + '[' + ', '.join([str(i) if not isinstance(i,str) else '\''+i+'\'' for i in key[2:]]) + str(f).split('[')[1]) from g
						else:
							try:
								if len(key) == 3: return _getitem(temp, key[2]) # let the previously established code get it 
								else: return _getitem(temp, key[2:]) # let the previously established code get it 
							except KeyError as g:  raise KeyError('Key not found in ' + str(type(self)) + '[' + ', '.join([str(i) if not isinstance(i,str) else '\''+i+'\'' for i in key[2:]]) + str(f).split('[')[1]) from g
					else:
						try:
							if len(key) == 3: return _getitem(temp, key[2]) # let the previously established code get it
							else: return _getitem(temp, key[2:])  # let the previously established code
						except KeyError as f: raise KeyError('Key not found in ' + str(type(self)) + '[' + ', '.join([str(i) if not isinstance(i,str) else '\''+i+'\'' for i in key[2:]]) + str(f).split('[')[1]) from f
			# END pd.DataFrame[Any, Any, ...] HANDLING

		# BEGIN pd.DataFrame[ ~(float,int,Series,ndarray,tuple) ] HANDLING
		else:
			try: return self[key] # is it a col?
			except Exception as e:
				try: return self.loc[key] # is it a row?
				except Exception as f:
					try: return self.iloc[key] # is it a valid int?
					except Exception as g: raise KeyError('Key not found in ' + str(type(self)) + '[' + (str(key) if not isinstance(key, str) else '\'' + key + '\'') + ']') from g
		# END pd.DataFrame[ ~(float,int,Series,ndarray,tuple) ]  HANDLING

	# BEGIN np.ndarray HANDLING
	elif isinstance(self, ndarray):
		# BEGIN np.ndarray[tuple] HANDLING
		if isinstance(key, tuple):
			if len(key) <= self.ndim: # ndarray[idx : tuple]
				try: return self[key] # try the dumb thing
				except Exception as e: raise KeyError('Key not found in ' + str(type(self)) + '[' + (str(key) if not isinstance(key, str) else '\'' + key + '\'') + ']') from e
			else: # ndarray[idx : tuple, ...]
				try: temp = self[key[:self.ndim]] # try to get the dumb thing
				except Exception as e: raise KeyError('Key not found in ' + str(type(self)) + '[' + (str(key) if not isinstance(key, str) else '\'' + key + '\'') + ']') from e
				else:
					try: # now what have we got?
						if len(key) == self.ndim+1: return _getitem(temp, key[-1]) # use recursion!
						else: return _getitem(temp, key[self.ndim:]) # use recursion![slice]
					except KeyError as e: raise KeyError('Key not found in ' + str(type(self)) + '[' + ', '.join([str(i) if not isinstance(i,str) else '\''+i+'\'' for i in key[:self.ndim]]) + str(e).split('[')[1]) from e
		else: # np.ndarray[~tuple]
			try: return self[key] # try the dumb thing
			except Exception as e: raise KeyError('Key not found in ' + str(type(self)) + '[' + (str(key) if not isinstance(key, str) else '\'' + key + '\'') + ']') from e
	# END np.ndarray[~tuple] HANDLING

	# BEGIN OTHERS HANDLING
	else: # self is ~(list,Series,DataFrame,ndarray)
		if isinstance(key, tuple): # self[tuple]
			if key in self: return self[key] # assumes self[key] if key in self
			else: # self[...]
				try: temp = _getitem(self, key[0]) # use recursion!
				except KeyError as e: raise KeyError('Key not found in ' + str(type(self)) + '[' + (str(key) if not isinstance(key, str) else '\''+key[0]+'\'') + ']') from e
				else:
					if len(key) == 1: return temp # self[key,]
					else:
						try: # no trusting anything
							if len(key) == 2: # self[Any, Any]
								if isinstance(key[0], Iterable) and not isinstance(key[0], str): return [_getitem(i, key[1]) for i in temp] # self[Iterable, Any]
								else: return _getitem(temp, key[1]) # self[~Iterable, Any]
							else:
								if isinstance(key[0], Iterable) and not isinstance(key[0], str): return [_getitem(i, key[1:]) for i in temp] # self[Iterable, Any, ...]
								else: return _getitem(temp, key[1:]) # self[~Iterable, Any, ...]
						except KeyError as e: raise KeyError('Key not found in ' + str(type(self)) + '[' + ', '.join([str(i) if not isinstance(i,str) else '\''+i+'\'' for i in key[:1]]) + str(e).split('[')[1]) from e
		elif isinstance(key, Iterable) and not isinstance(key, str):
			try: return [_getitem(self, i) for i in key] # self[random Iterable]
			except KeyError as e: raise KeyError('Key not found in ' + str(type(self)) + '[' + (str(key) if not isinstance(key, str) else '\''+key+'\'') + ']') from e
		else:
			if hasattr(self, '__contains__') and isinstance(key, Hashable) and key in self: return self.__getitem__(key)
			elif isinstance(key, str) and hasattr(self, key) and not callable(getattr(self, key)): return getattr(self, key) # self.key exists
			else: raise KeyError('Key not found in ' + str(type(self)) + '[' + (str(key) if not isinstance(key, str) else '\'' + key + '\'') + ']')
	# END OTHERS HANDLING
# END FUNCTION _getitem