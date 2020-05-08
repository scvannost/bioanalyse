# Overall structure
There are generally two steps to scRNAseq analysis using `bioanalyse` - processing/filtering and data exploration/analysis.
The first is based around the `Processor` class, while the second is based around the `Sample` class.

# Loading data
Whether using processing or analysing, there are a few ways to load data into a bioanalyse structure.
1. `Sample.from_csv(` or `Processor.from_csv(`: for loading a single sample from a single comma-separated value file
2. `Sample.from_pickle(` or `Processor.from_pickle(`: for loading a single sample from a pickled `pandas.DataFrame`
3. `MultiSample.from_single_csv(` or `MultiProcessor.from_single_csv(`: for loading multiple samples from a single comma-separated value file
4. `MultiSample.from_csvs(` or `MultiProcessor.from_csvs(`: for loading multiple samples, each from their own single comma-separated value file

There is also one additional pair of constructor methods -- `MultiSample.from_many(` and `MultiProcessor.from_many(` -- which combine multiple `Sample` or `Processor` instances into a single structure.

For `Processor`s and `MultiProcessor`s, loaded data can be either raw count data (default, or with `normalized in [None, False]`), CPM normalized (with `normalized='cell'`), normalized to a set of housekeeping genes (`hk`/`housekeeping`), both (`hkcell`), or GF-ICF values (`gf-icf`).
All accepted value for `normalized` are NOT case-sensitive.
Data loaded already normalized cannot be further normalized, but raw count data can be normalized to the above forms using `cell_normalize`, `hk_normalize`, and `gf_icf`.
You can check data normalization with `.normalized`.  
`Sample` and `MultiSample` must be loaded with all desired normalization.

Additionally, an existing `pandas.DataFrame` can be incorporated into a bioanalyse structure using standard class instantiation, which allows for passing of more attributes than just data.

## Converting between types
`Processor`s and `MultiProcessor`s can be converted to their `Sample` counterpart using `.finalize(`, transferring as much information as possible about the filtered data.  
Additionally, they can also be shrunk using `.shrink(` to apply the current filtering onto the raw data, handling all normalization of `.data`.

## To come
More constructors are planned, including direct loading from: `.h5ad`; zipped `.csv`s via eg `.gz`, `.zip`, `.tar`; `scipy.sparse` matrices from `.npz`; dropest results from `.rds`.  
A `minimal : bool` kwarg for calculating as many attributes as possible on the fly and storing only the minimal data.

# Filtering
`Processor`s and `MultiProcessor`s come with a `.filter` attribute that allows for filtering cells based on many different schemes.
The default filters are:
1. `.filter.counts(` to remove cells below a given total counts
2. `.filter.genes(` to remove cells with fewer than a given number of genes detected
3. `.filter.mito(` to remove cells with over a certain fraction of mitochondrial counts
4. `.filter.sec_counts(` to run the secant method as described in Heiser 2019 bioRχiv (doi:10.1101/684340)
5. `.filter.reldiv(` to only keep cells in the second group of a bimodally distributed relative diversity
6. `.filter.clusters(` to keep cells only in the given clusters (see Clustering below)
7. `.filter.reset(` to remove all filtering currently applied
Only `clusters` and `reset` are available if the data was originally loaded unnormalized, as the others all rely on raw counts.

Changing the filtering clears `.layers`, `.pca_`, `.tsne_`, and `.umap_` -- and reruns GF-ICF normalization if done.

# Basic data access
`.data`, `.d`, `['data']`, and `['d']` all point to the data.
Furthermore `['data'/'d', a]` is implemented equivalent to `['d'][a]` or `['d'].iloc[a]`, and `['data'/'d', a, b]` as `['d'].loc[a,b]`, `['d'].iloc[a,b]`, `['d'].loc[a][b]`, or `['d'].iloc[a][b]`.  


Other basic data are accessible as `.shape`, total counts per cell as `.counts`, genes detected per cell as `.genes`, mitochondrial fractions as `.mito`, diversity = log1p(genes) as `.div`, relative diversity = zscore(div) as `.reldiv`.
All of these are also accessible via their `['<attrib>']` counterparts as well.
Data columns are available as `.columns`, and the index as `.index`.

For `Processor`s and `MultiProcessor`s, all mentioned above are for the filtered data.
The equivalents for the unfiltered data prepend `raw_` to the attribute name.
To get unfiltered data, only `.raw_data` and `['raw_data']` are available.  
Additionally, `.expression`, `.e`, `['expression']`, and `['e']` return a log-scaled expression matrix (if not GF-ICF normalized) filtered data, with `['expression'/'e', a[, b]]` implemented as for `['data'/'d']`.
For GF-ICF normalized data, `.e` is equivalent to `.d`.
No equivalent for raw expression is provided.

Calculated values are acessible by key in either `.layers` or directly on the structure -- ie `sample.layers['<attrib>'] == sample['<attrib>']`.
Examples include `pca`, `umap`, `tsne`, `cluster`.
These values remain once calculated and are returned by their generating functions as is until `.wipe_layer(` is called for that atttribute, even if different parameters are passed on the second call.
Additionally, `['pca_loadings']` returns the loadings of each gene on each PC.  
See below for more on each of these individually.

Finally, objects used to generate `pca`, `umap`, and `tsne` values are stored at `pca_`, `umap_`, and `tsne_` respectively.
These are not removed during a call to `.wipe_layer(` and are instead replaced upon the next call of the generating function.
`.pca_` is overriden on every call to `.init_pca(`.

# Basic data processing
## Normalization in (Multi)Processor
`.cell_normalize(` converts the data to CPM.  
`.hk_normalize(` normalizes the data to a given list of genes.
A preset list of genes from Eisenberg 2013 (doi:10.1016/j.tig.2013.05.010) can be loaded for several species listed in `HOUSEKEEPING_SPECIES`.  
`.gf_icf(` updates data with GF-ICF normalization modified from Gambardella 2019 (see doi:10.3389/fgene.2019.00734).
No L-normalization is given, as this removes the dependency on gene frequency (see Van Nostrand 2020 thesis).  
`.undo_normalize(` resets the data to unnormalized.

Current data normalization can be seen in `.normalized`.

## Dimensionality Reduction
PCA, UMAP, and TSNE of `.expression` are all implemented via `.pca(`, `.umap(`, and `.tsne(`.
UMAP and TSNE run on the PCA reductions, and will generate `.pca_` if needed.  
`.init_pca(` trains the PCA model without transforming the data and returns the `PCA` object saved at `.pca_`, overriding any existing instance.

Once run, the results are saved in `.layers` and returned as is, regardless of parameters.
To run again with different parameters, call `.wipe_layer(` first.

For UMAP and TSNE, `n_components` is passed to PCA, but is overriden by the special kwarg `pca_components` if passed.
To pass `n_components` for UMAP or TSNE, pass `umap_components` or `tsne_components` respectively.  
Alternately, `random_state` is passed to UMAP or TSNE, but overriden by the special kwargs `umap_random_state` and `tsne_random_state`.
To pass `random_state` for pca, pass `pca_random_state`.

### Key Parameters
For PCA, the key kwarg is `n_components`, which sets the number of PCs to keep if > 1 or the percent of the variance to keep if < 1.

For UMAP, the key kwarg is `n_neighbors`, which determines the size of the local neighborhood, with larger values yielding a more global view and smaller values yielding a more local view.
Internal defaults set this to the square-root of the number of filtered cells if not given.

For TSNE, the key kwargs are `learning_rate` and `perplexity`, which together determine the distribution of points and the effective number of neighbors.
These default to 1000 and the square-root of the number of filtered cells if not given.
See `sklearn.manifold.tsne` for more information.
Additionally, if the module `MultiCoreTSNE` is installed, `n_jobs` can be used to increase processing speed.
However, `MultiCoreTSNE` does not have `.transform(` implemented and is not saved as `.tsne_` as to require calling `.tsne(` directly every time.

### To come
Checking `.pca(`, `.umap(`, and `.tsne(` parameters against existing objects and rerun if passed different parameters.

# Clustering
`.cluster(` runs the Leiden algorithm, an extension of the Louvain algorithm, if module `leidenalg` is installed.
Otherwise uses `igraph.Graph.community_multilevel(` to run the Louvain algorithm itself.
Both algorithms use PCA transformation followed by UMAP's `fuzzy_simplicial_set(` as a distance/weights matrix.

If `leigenalg` is installed, `n_iterations` sets how many iterations of the Leiden algorithm are used and `seed` is passed to `leigenalg.find_partition(`.
The key parameter for Leigen clustering is `resolution` -- higher values means more smaller clusters, while lower values means fewer larger clusters.
Louvain clustering has no equivalent parameters.  
Notes: `random_state` is handled as in UMAP; `n_components` is passed to PCA; `n_neighbors` is passed to UMAP.

The cluster assigned for each cell is saved in `.layers['cluster']` (also accessible as `['cluster']` directly.)
No `.cluster_` object is created.

When plotted in UMAP space (see Visualization below), these clusters are generally nearly continguous but are not guaranteed to be as such unless the same random_state is passed for both.

## To come
Create `.cluster_` object with `.cluster_.transform(` returning the cluster identifications -- possibly based on the nearest cell in UMAP space.  
Pass existing `.pca_` or `.umap_` objects or parameters to `utils.cluster(`.

# Visualization
`.plot(` implements many custom visualization features, but is generally meant for "painting" information onto the cells.
It defaults to UMAP space for clean graphs, but also can use TSNE or PCA space via the `method` kwarg.
Information can also be grouped using the `by_cluster` kwarg using pandas groupby functions, eg `mean`, `median`, `std`.

You can paint on any gene/column of `.expression`, or `cluster` (which calls `.cluster(`  with applicable kwargs if not already clustered).  
`Processor` and `MultiProcessor` can also paint `.counts`, `.genes`, `.mito`, `.div`, and `.reldiv`.  

This function defaults to points of `s=1` due to the inherently large sizes of scRNAseq experiments.
If axes are not given in `ax`, new ones are created.

Special kwargs include:
* `c`: only used if `paint is None` (default)
* `xlabel`/`ylabel`: either str or dict
* `tsne_method`: passed to `.tsne(` as `method`
* `legend_...`: prefix for `edgecolor`, `framecolor`, and `frameon` for passing to `ax.legend(` instead of `plt.figure(`
* `title_...`: prefix for `alpha`, `c`, `color`, and `zorder` for passing to `ax.set_title(` instead of `ax.scatter(`
* `n_components`: equivalent to but overriden by `pca_components`; pass `umap_components` or `tsne_components` for those functions
* `aspect`: if painting 'cluster', then passed to `mpl_toolkits.axes_grid1.ImageGrid(` but overridden by `grid_aspect`; else, passed to `ax.colorbar(` but overriden by `cbar_aspect`
* `fontsize`/`loc`: if painting 'cluster', then passed to `ax.legend` but overridden by `legend_...`; else, passed to `ax.set_title` but overriden by `title_...`
* `norm`: if painting multiple, overriden by `matplotlib.colors.Normalize()`
* `random_state`: preference for `umap_random_state` or `tsne_random_state`, but for `pca_random_state` if method == 'pca'; always overriden by more specific

## QC Visualization
Additionally, `Processor` and `MultiProcessor` implement `.plot_qc(`, which plots `.counts` vs `.genes` vs `.mito` on an `mpl_toolkits.mplot3d.Axes3D`.
You can also paint onto these cells similar to `.plot(`.

# Advanced
## All available kwargs for each underlying function are listed as a constant in `utils`.
Namely, see:
* PCA_KWARGS, UMAP_KWARGS, TSNE_KWARGS
* CLUSTER_KWARGS
* FIGURE_KWARGS, IMAGEGRID_KWARGS, ADD_SUBPLOT_KWARGS, SUBPLOTS_KWARGS, SUBPLOTS_ADJUST_KWARGS
* COLLECTIONS_KWARGS, PLOT_KWARGS, SCATTER_KWARGS, COLORBAR_KWARGS
* TICK_KWARGS, TEXT_KWARGS, TITLE_KWARGS
* CSV_KWARGS

## Alternate Settings
Parameters `log` and `mito_marker` can be set in any constructor to change the default logarithm and mitochondrial gene prefix for an instance.
Defaults: `log = np.log2`, `mito_marker = 'mt-'`

## Filters
The `utils.Filter` is highly extensible, with many features not used in the simple implementation.
The filter is able to be wired into anything, as it takes any `target` and looks for its `data_key` attribute or element as the DataFrame to filter.
It also allows for notification back to its target via calling `target._notify(self)` if such method exists.  
The filtered result is available at `.result` and a boolean Series mask at `.idx`.

The list of currently applied filters is `.filters`, while available filters are at `.filter_options`.
Filters can be added using `.add_option((by, how[, name]))`, and removed by `.remove_option(name/by)`.
`by` is used as `name` if not given, and names of filters cannot conflict -- meaning two filters with the same `by` must have at least one `name` given.
Filters are added as lambda functions to one of the methods listed below.  
Note that if a filter option is removed, anything filtered out through that option remains filtered.


Several methods are already implemented for ease, namely:
* `external`: an arbitrary filter can be added by passing a callable `how` that takes `target[by]`, returns a boolean mask as a Series or array, and can also take any number of additional args and/or kwargs
* `bimodal`: assumes `target[by]` is bimodal and takes only one of those modes; which mode and the method of separation are configurable
* `linear`: a linear cutoff on `target[by]`, either above/below a single value or in between two values
* `secant`: uses the secant distance of `target[by]` to eliminate entries a certain percentile above/below the maximum

## Clustering
The clustering used by the main classes is  `utils.cluster(`. This can be called directly on any array or DataFarme, returning an array of cluster membership.

## Colors
4 different color-blind friendly palettes were lifted from scanpy, for up to 10, 20, 28, and 102 distinguishable colors.
`utils.colors(` takes an array, Series/DataFrame, Iterable, Mapping and returns an array, Series, list, or dict of rgb color values using the smallest color palette available.  
The color palettes are available directly as `utils.VEGA_10`, `utils.VEGA_20`, `utils.ZEILEIS_28`, and `utils.GODSNOT_102`.

Additionally, `utils.annot_color(` and `utils.annot_bbox(` returns annotation and bbox colors for higher viability against the given background color.  
Preferences for light/dark colors, light/dark bbox alpha, and luminence cutoff that control the return in `utils.ANNOT_PREF` can be set by `utils.set_annot_pref(`

## Extensions
`utils.SampleBase` provides an abstract class that can be inherited to give minimal functionality to any daughter class.
It requires the definition of a property method `data`, and implements basic `__getitem__` and `__contains__` methods.
