import matplotlib
matplotlib.use("PS")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb
import seaborn as sns
from seaborn.relational import *
from seaborn.axisgrid import (
    Grid, dedent, _facet_docs, LooseVersion, mpl,
    utils, product, FacetGrid)
from seaborn.relational import _ScatterPlotter, _LinePlotter


def getRasterFacetIdx(
        plotDF, y, row=None, col=None, hue=None):
    plotDF.loc[:, y + '_facetIdx'] = np.nan
    dummyG = sns.FacetGrid(
        plotDF,
        row=row, col=col)
    for name, group in dummyG.facet_data():
        if hue is None:
            subGroupBy = 'feature'
        else:
            subGroupBy = hue
        idxOffset = 0
        for subName, subGroup in group.groupby(subGroupBy):
            uniqueIdx = pd.unique(group[y])
            idxLookup = {
                uIdx: idx + idxOffset
                for idx, uIdx in enumerate(uniqueIdx)}
            plotDF.loc[group.index, y + '_facetIdx'] = (
                group[y].map(idxLookup))
            idxOffset += uniqueIdx.shape[0]
    return plotDF


def twin_relplot(
        x=None, y1=None, y2=None, hue=None, size=None, style=None, data=None,
        row=None, col=None, col_wrap=None, row_order=None, col_order=None,
        palette=None, hue_order=None, hue_norm=None, style_order=None,
        sizes1=None, size_order1=None, size_norm1=None,
        sizes2=None, size_order2=None, size_norm2=None,
        markers1=None, dashes1=None,
        markers2=None, dashes2=None,
        func1_kws={}, func2_kws={},
        query1=None, query2=None,
        legend="brief", kind1="scatter", kind2="scatter",
        height=5, aspect=1, facet1_kws=None, facet2_kws=None, **kwargs):

    if kind1 == "scatter":
        plotter1 = _ScatterPlotter
        func1 = scatterplot
        markers1 = True if markers1 is None else markers1

    elif kind1 == "line":
        plotter1 = _LinePlotter
        func1 = lineplot
        dashes1 = True if dashes1 is None else dashes1

    else:
        err = "Plot kind {} not recognized".format(kind1)
        raise ValueError(err)
    
    if kind2 == "scatter":
        plotter2 = _ScatterPlotter
        func2 = scatterplot
        markers2 = True if markers2 is None else markers2

    elif kind2 == "line":
        plotter2 = _LinePlotter
        func2 = lineplot
        dashes2 = True if dashes2 is None else dashes2

    else:
        err = "Plot kind {} not recognized".format(kind1)
        raise ValueError(err)

    # Use the full dataset to establish how to draw the semantics
    p1 = plotter1(
        x=x, y=y1, hue=hue, size=size, style=style, data=data,
        palette=palette, hue_order=hue_order, hue_norm=hue_norm,
        sizes=sizes1, size_order=size_order1, size_norm=size_norm1,
        markers=markers1, dashes=dashes1, style_order=style_order,
        legend=legend,
    )
    p2 = plotter2(
        x=x, y=y2, hue=hue, size=size, style=style, data=data,
        palette=palette, hue_order=hue_order, hue_norm=hue_norm,
        sizes=sizes2, size_order=size_order2, size_norm=size_norm2,
        markers=markers2, dashes=dashes2, style_order=style_order,
        legend=legend,
    )

    palette = p1.palette if p1.palette else None
    hue_order = p1.hue_levels if any(p1.hue_levels) else None
    hue_norm = p1.hue_norm if p1.hue_norm is not None else None
    style_order = p1.style_levels if any(p1.style_levels) else None

    sizes1 = p1.sizes if p1.sizes else None
    size_order1 = p1.size_levels if any(p1.size_levels) else None
    size_norm1 = p1.size_norm if p1.size_norm is not None else None

    markers1 = p1.markers if p1.markers else None
    dashes1 = p1.dashes if p1.dashes else None

    sizes2 = p2.sizes if p2.sizes else None
    size_order2 = p2.size_levels if any(p2.size_levels) else None
    size_norm2 = p2.size_norm if p2.size_norm is not None else None

    markers2 = p2.markers if p2.markers else None
    dashes2 = p2.dashes if p2.dashes else None

    plot_kws1 = dict(
        palette=palette, hue_order=hue_order, hue_norm=p1.hue_norm,
        sizes=sizes1, size_order=size_order1, size_norm=p1.size_norm,
        markers=markers1, dashes=dashes1, style_order=style_order,
        legend=False,
    )
    plot_kws1.update(kwargs)
    plot_kws1.update(func1_kws)
    if kind1 == "scatter":
        plot_kws1.pop("dashes")
    
    plot_kws2 = dict(
        palette=palette, hue_order=hue_order, hue_norm=p1.hue_norm,
        sizes=sizes2, size_order=size_order2, size_norm=p1.size_norm,
        markers=markers1, dashes=dashes1, style_order=style_order,
        legend=False,
    )
    plot_kws2.update(kwargs)
    plot_kws2.update(func2_kws)
    if kind2 == "scatter":
        plot_kws2.pop("dashes")

    # Set up the FacetGrid object
    facet1_kws = {} if facet1_kws is None else facet1_kws
    data1 = data.query(query1) if query1 is not None else data
    g1 = FacetGrid(
        data=data1, row=row, col=col, col_wrap=col_wrap,
        row_order=row_order, col_order=col_order,
        height=height, aspect=aspect, dropna=False,
        **facet1_kws
    )
    
    twin_axes = np.empty_like(g1.axes)
    for i, axList in enumerate(g1.axes):
        for j, ax in enumerate(axList):
            twin_axes[i, j] = ax.twinx()

    facet2_kws = {} if facet2_kws is None else facet2_kws
    data2 = data.query(query2) if query2 is not None else data
    g2 = FacetGridShadow(
        data=data2, fig=g1.fig, axes=twin_axes,
        row=row, col=col, col_wrap=col_wrap,
        row_order=row_order, col_order=col_order,
        height=height, aspect=aspect, dropna=False,
        **facet2_kws
    )
    
    # Draw the plot
    g1.map_dataframe(
        func1, x, y1,
        hue=hue, size=size, style=style,
        **plot_kws1)
    g2.map_dataframe(
        func2, x, y2,
        hue=hue, size=size, style=style,
        **plot_kws2)

    # Show the legend
    if legend:
        p1.add_legend_data(g1.axes.flat[0])
        if p1.legend_data:
            g1.add_legend(
                legend_data=p1.legend_data,
                label_order=p1.legend_order)
        p2.add_legend_data(g2.axes.flat[0])
        if p2.legend_data:
            g2.add_legend(
                legend_data=p2.legend_data,
                label_order=p2.legend_order)

    return g1


class FacetGridShadow(Grid):
    """Multi-plot grid for plotting conditional relationships."""
    def __init__(self, data, fig, axes, row=None, col=None, hue=None, col_wrap=None,
                 sharex=True, sharey=True, height=3, aspect=1, palette=None,
                 row_order=None, col_order=None, hue_order=None, hue_kws=None,
                 dropna=True, legend_out=True, despine=True,
                 margin_titles=False, xlim=None, ylim=None, subplot_kws=None,
                 gridspec_kws=None, size=None):

        MPL_GRIDSPEC_VERSION = LooseVersion('1.4')
        OLD_MPL = LooseVersion(mpl.__version__) < MPL_GRIDSPEC_VERSION

        # Handle deprecations
        if size is not None:
            height = size
            msg = ("The `size` paramter has been renamed to `height`; "
                   "please update your code.")
            warnings.warn(msg, UserWarning)

        # Determine the hue facet layer information
        hue_var = hue
        if hue is None:
            hue_names = None
        else:
            hue_names = utils.categorical_order(data[hue], hue_order)

        colors = self._get_palette(data, hue, hue_order, palette)

        # Set up the lists of names for the row and column facet variables
        if row is None:
            row_names = []
        else:
            row_names = utils.categorical_order(data[row], row_order)

        if col is None:
            col_names = []
        else:
            col_names = utils.categorical_order(data[col], col_order)

        # Additional dict of kwarg -> list of values for mapping the hue var
        hue_kws = hue_kws if hue_kws is not None else {}

        # Make a boolean mask that is True anywhere there is an NA
        # value in one of the faceting variables, but only if dropna is True
        none_na = np.zeros(len(data), np.bool)
        if dropna:
            row_na = none_na if row is None else data[row].isnull()
            col_na = none_na if col is None else data[col].isnull()
            hue_na = none_na if hue is None else data[hue].isnull()
            not_na = ~(row_na | col_na | hue_na)
        else:
            not_na = ~none_na

        # Compute the grid shape
        ncol = 1 if col is None else len(col_names)
        nrow = 1 if row is None else len(row_names)
        self._n_facets = ncol * nrow

        self._col_wrap = col_wrap
        if col_wrap is not None:
            if row is not None:
                err = "Cannot use `row` and `col_wrap` together."
                raise ValueError(err)
            ncol = col_wrap
            nrow = int(np.ceil(len(col_names) / col_wrap))
        self._ncol = ncol
        self._nrow = nrow

        # Calculate the base figure size
        # This can get stretched later by a legend
        # TODO this doesn't account for axis labels
        figsize = (ncol * height * aspect, nrow * height)

        # Validate some inputs
        if col_wrap is not None:
            margin_titles = False

        # Build the subplot keyword dictionary
        subplot_kws = {} if subplot_kws is None else subplot_kws.copy()
        gridspec_kws = {} if gridspec_kws is None else gridspec_kws.copy()
        if xlim is not None:
            subplot_kws["xlim"] = xlim
        if ylim is not None:
            subplot_kws["ylim"] = ylim

        # Initialize the subplot grid
        self.axes = axes
        
        if col_wrap is not None:
            # Now we turn off labels on the inner axes
            if sharex:
                for ax in self._not_bottom_axes:
                    for label in ax.get_xticklabels():
                        label.set_visible(False)
                    ax.xaxis.offsetText.set_visible(False)
            if sharey:
                for ax in self._not_left_axes:
                    for label in ax.get_yticklabels():
                        label.set_visible(False)
                    ax.yaxis.offsetText.set_visible(False)

        # Set up the class attributes
        # ---------------------------

        # First the public API
        self.data = data
        self.fig = fig
        self.axes = axes

        self.row_names = row_names
        self.col_names = col_names
        self.hue_names = hue_names
        self.hue_kws = hue_kws

        # Next the private variables
        self._nrow = nrow
        self._row_var = row
        self._ncol = ncol
        self._col_var = col

        self._margin_titles = margin_titles
        self._col_wrap = col_wrap
        self._hue_var = hue_var
        self._colors = colors
        self._legend_out = legend_out
        self._legend = None
        self._legend_data = {}
        self._x_var = None
        self._y_var = None
        self._dropna = dropna
        self._not_na = not_na

    __init__.__doc__ = dedent("""\
        Initialize the matplotlib figure and FacetGrid object.

        This class maps a dataset onto multiple axes arrayed in a grid of rows
        and columns that correspond to *levels* of variables in the dataset.
        The plots it produces are often called "lattice", "trellis", or
        "small-multiple" graphics.

        It can also represent levels of a third varaible with the ``hue``
        parameter, which plots different subets of data in different colors.
        This uses color to resolve elements on a third dimension, but only
        draws subsets on top of each other and will not tailor the ``hue``
        parameter for the specific visualization the way that axes-level
        functions that accept ``hue`` will.

        When using seaborn functions that infer semantic mappings from a
        dataset, care must be taken to synchronize those mappings across
        facets. In most cases, it will be better to use a figure-level function
        (e.g. :func:`relplot` or :func:`catplot`) than to use
        :class:`FacetGrid` directly.

        The basic workflow is to initialize the :class:`FacetGrid` object with
        the dataset and the variables that are used to structure the grid. Then
        one or more plotting functions can be applied to each subset by calling
        :meth:`FacetGrid.map` or :meth:`FacetGrid.map_dataframe`. Finally, the
        plot can be tweaked with other methods to do things like change the
        axis labels, use different ticks, or add a legend. See the detailed
        code examples below for more information.

        See the :ref:`tutorial <grid_tutorial>` for more information.

        Parameters
        ----------
        {data}
        row, col, hue : strings
            Variables that define subsets of the data, which will be drawn on
            separate facets in the grid. See the ``*_order`` parameters to
            control the order of levels of this variable.
        {col_wrap}
        {share_xy}
        {height}
        {aspect}
        {palette}
        {{row,col,hue}}_order : lists, optional
            Order for the levels of the faceting variables. By default, this
            will be the order that the levels appear in ``data`` or, if the
            variables are pandas categoricals, the category order.
        hue_kws : dictionary of param -> list of values mapping
            Other keyword arguments to insert into the plotting call to let
            other plot attributes vary across levels of the hue variable (e.g.
            the markers in a scatterplot).
        {legend_out}
        despine : boolean, optional
            Remove the top and right spines from the plots.
        {margin_titles}
        {{x, y}}lim: tuples, optional
            Limits for each of the axes on each facet (only relevant when
            share{{x, y}} is True.
        subplot_kws : dict, optional
            Dictionary of keyword arguments passed to matplotlib subplot(s)
            methods.
        gridspec_kws : dict, optional
            Dictionary of keyword arguments passed to matplotlib's ``gridspec``
            module (via ``plt.subplots``). Requires matplotlib >= 1.4 and is
            ignored if ``col_wrap`` is not ``None``.

        See Also
        --------
        PairGrid : Subplot grid for plotting pairwise relationships.
        relplot : Combine a relational plot and a :class:`FacetGrid`.
        catplot : Combine a categorical plot and a :class:`FacetGrid`.
        lmplot : Combine a regression plot and a :class:`FacetGrid`.

        Examples
        --------

        Initialize a 2x2 grid of facets using the tips dataset:

        .. plot::
            :context: close-figs

            >>> import seaborn as sns; sns.set(style="ticks", color_codes=True)
            >>> tips = sns.load_dataset("tips")
            >>> g = sns.FacetGrid(tips, col="time", row="smoker")

        Draw a univariate plot on each facet:

        .. plot::
            :context: close-figs

            >>> import matplotlib.pyplot as plt
            >>> g = sns.FacetGrid(tips, col="time",  row="smoker")
            >>> g = g.map(plt.hist, "total_bill")

        (Note that it's not necessary to re-catch the returned variable; it's
        the same object, but doing so in the examples makes dealing with the
        doctests somewhat less annoying).

        Pass additional keyword arguments to the mapped function:

        .. plot::
            :context: close-figs

            >>> import numpy as np
            >>> bins = np.arange(0, 65, 5)
            >>> g = sns.FacetGrid(tips, col="time",  row="smoker")
            >>> g = g.map(plt.hist, "total_bill", bins=bins, color="r")

        Plot a bivariate function on each facet:

        .. plot::
            :context: close-figs

            >>> g = sns.FacetGrid(tips, col="time",  row="smoker")
            >>> g = g.map(plt.scatter, "total_bill", "tip", edgecolor="w")

        Assign one of the variables to the color of the plot elements:

        .. plot::
            :context: close-figs

            >>> g = sns.FacetGrid(tips, col="time",  hue="smoker")
            >>> g = (g.map(plt.scatter, "total_bill", "tip", edgecolor="w")
            ...       .add_legend())

        Change the height and aspect ratio of each facet:

        .. plot::
            :context: close-figs

            >>> g = sns.FacetGrid(tips, col="day", height=4, aspect=.5)
            >>> g = g.map(plt.hist, "total_bill", bins=bins)

        Specify the order for plot elements:

        .. plot::
            :context: close-figs

            >>> g = sns.FacetGrid(tips, col="smoker", col_order=["Yes", "No"])
            >>> g = g.map(plt.hist, "total_bill", bins=bins, color="m")

        Use a different color palette:

        .. plot::
            :context: close-figs

            >>> kws = dict(s=50, linewidth=.5, edgecolor="w")
            >>> g = sns.FacetGrid(tips, col="sex", hue="time", palette="Set1",
            ...                   hue_order=["Dinner", "Lunch"])
            >>> g = (g.map(plt.scatter, "total_bill", "tip", **kws)
            ...      .add_legend())

        Use a dictionary mapping hue levels to colors:

        .. plot::
            :context: close-figs

            >>> pal = dict(Lunch="seagreen", Dinner="gray")
            >>> g = sns.FacetGrid(tips, col="sex", hue="time", palette=pal,
            ...                   hue_order=["Dinner", "Lunch"])
            >>> g = (g.map(plt.scatter, "total_bill", "tip", **kws)
            ...      .add_legend())

        Additionally use a different marker for the hue levels:

        .. plot::
            :context: close-figs

            >>> g = sns.FacetGrid(tips, col="sex", hue="time", palette=pal,
            ...                   hue_order=["Dinner", "Lunch"],
            ...                   hue_kws=dict(marker=["^", "v"]))
            >>> g = (g.map(plt.scatter, "total_bill", "tip", **kws)
            ...      .add_legend())

        "Wrap" a column variable with many levels into the rows:

        .. plot::
            :context: close-figs

            >>> att = sns.load_dataset("attention")
            >>> g = sns.FacetGrid(att, col="subject", col_wrap=5, height=1.5)
            >>> g = g.map(plt.plot, "solutions", "score", marker=".")

        Define a custom bivariate function to map onto the grid:

        .. plot::
            :context: close-figs

            >>> from scipy import stats
            >>> def qqplot(x, y, **kwargs):
            ...     _, xr = stats.probplot(x, fit=False)
            ...     _, yr = stats.probplot(y, fit=False)
            ...     plt.scatter(xr, yr, **kwargs)
            >>> g = sns.FacetGrid(tips, col="smoker", hue="sex")
            >>> g = (g.map(qqplot, "total_bill", "tip", **kws)
            ...       .add_legend())

        Define a custom function that uses a ``DataFrame`` object and accepts
        column names as positional variables:

        .. plot::
            :context: close-figs

            >>> import pandas as pd
            >>> df = pd.DataFrame(
            ...     data=np.random.randn(90, 4),
            ...     columns=pd.Series(list("ABCD"), name="walk"),
            ...     index=pd.date_range("2015-01-01", "2015-03-31",
            ...                         name="date"))
            >>> df = df.cumsum(axis=0).stack().reset_index(name="val")
            >>> def dateplot(x, y, **kwargs):
            ...     ax = plt.gca()
            ...     data = kwargs.pop("data")
            ...     data.plot(x=x, y=y, ax=ax, grid=False, **kwargs)
            >>> g = sns.FacetGrid(df, col="walk", col_wrap=2, height=3.5)
            >>> g = g.map_dataframe(dateplot, "date", "val")

        Use different axes labels after plotting:

        .. plot::
            :context: close-figs

            >>> g = sns.FacetGrid(tips, col="smoker", row="sex")
            >>> g = (g.map(plt.scatter, "total_bill", "tip", color="g", **kws)
            ...       .set_axis_labels("Total bill (US Dollars)", "Tip"))

        Set other attributes that are shared across the facetes:

        .. plot::
            :context: close-figs

            >>> g = sns.FacetGrid(tips, col="smoker", row="sex")
            >>> g = (g.map(plt.scatter, "total_bill", "tip", color="r", **kws)
            ...       .set(xlim=(0, 60), ylim=(0, 12),
            ...            xticks=[10, 30, 50], yticks=[2, 6, 10]))

        Use a different template for the facet titles:

        .. plot::
            :context: close-figs

            >>> g = sns.FacetGrid(tips, col="size", col_wrap=3)
            >>> g = (g.map(plt.hist, "tip", bins=np.arange(0, 13), color="c")
            ...       .set_titles("{{col_name}} diners"))

        Tighten the facets:

        .. plot::
            :context: close-figs

            >>> g = sns.FacetGrid(tips, col="smoker", row="sex",
            ...                   margin_titles=True)
            >>> g = (g.map(plt.scatter, "total_bill", "tip", color="m", **kws)
            ...       .set(xlim=(0, 60), ylim=(0, 12),
            ...            xticks=[10, 30, 50], yticks=[2, 6, 10])
            ...       .fig.subplots_adjust(wspace=.05, hspace=.05))

        """).format(**_facet_docs)

    def facet_data(self):
        """Generator for name indices and data subsets for each facet.

        Yields
        ------
        (i, j, k), data_ijk : tuple of ints, DataFrame
            The ints provide an index into the {row, col, hue}_names attribute,
            and the dataframe contains a subset of the full data corresponding
            to each facet. The generator yields subsets that correspond with
            the self.axes.flat iterator, or self.axes[i, j] when `col_wrap`
            is None.

        """
        data = self.data

        # Construct masks for the row variable
        if self.row_names:
            row_masks = [data[self._row_var] == n for n in self.row_names]
        else:
            row_masks = [np.repeat(True, len(self.data))]

        # Construct masks for the column variable
        if self.col_names:
            col_masks = [data[self._col_var] == n for n in self.col_names]
        else:
            col_masks = [np.repeat(True, len(self.data))]

        # Construct masks for the hue variable
        if self.hue_names:
            hue_masks = [data[self._hue_var] == n for n in self.hue_names]
        else:
            hue_masks = [np.repeat(True, len(self.data))]

        # Here is the main generator loop
        for (i, row), (j, col), (k, hue) in product(enumerate(row_masks),
                                                    enumerate(col_masks),
                                                    enumerate(hue_masks)):
            data_ijk = data[row & col & hue & self._not_na]
            yield (i, j, k), data_ijk

    def map(self, func, *args, **kwargs):
        """Apply a plotting function to each facet's subset of the data.

        Parameters
        ----------
        func : callable
            A plotting function that takes data and keyword arguments. It
            must plot to the currently active matplotlib Axes and take a
            `color` keyword argument. If faceting on the `hue` dimension,
            it must also take a `label` keyword argument.
        args : strings
            Column names in self.data that identify variables with data to
            plot. The data for each variable is passed to `func` in the
            order the variables are specified in the call.
        kwargs : keyword arguments
            All keyword arguments are passed to the plotting function.

        Returns
        -------
        self : object
            Returns self.

        """
        # If color was a keyword argument, grab it here
        kw_color = kwargs.pop("color", None)

        if hasattr(func, "__module__"):
            func_module = str(func.__module__)
        else:
            func_module = ""

        # Check for categorical plots without order information
        if func_module == "seaborn.categorical":
            if "order" not in kwargs:
                warning = ("Using the {} function without specifying "
                           "`order` is likely to produce an incorrect "
                           "plot.".format(func.__name__))
                warnings.warn(warning)
            if len(args) == 3 and "hue_order" not in kwargs:
                warning = ("Using the {} function without specifying "
                           "`hue_order` is likely to produce an incorrect "
                           "plot.".format(func.__name__))
                warnings.warn(warning)

        # Iterate over the data subsets
        for (row_i, col_j, hue_k), data_ijk in self.facet_data():

            # If this subset is null, move on
            if not data_ijk.values.size:
                continue

            # Get the current axis
            ax = self.facet_axis(row_i, col_j)

            # Decide what color to plot with
            kwargs["color"] = self._facet_color(hue_k, kw_color)

            # Insert the other hue aesthetics if appropriate
            for kw, val_list in self.hue_kws.items():
                kwargs[kw] = val_list[hue_k]

            # Insert a label in the keyword arguments for the legend
            if self._hue_var is not None:
                kwargs["label"] = utils.to_utf8(self.hue_names[hue_k])

            # Get the actual data we are going to plot with
            plot_data = data_ijk[list(args)]
            if self._dropna:
                plot_data = plot_data.dropna()
            plot_args = [v for k, v in plot_data.iteritems()]

            # Some matplotlib functions don't handle pandas objects correctly
            if func_module.startswith("matplotlib"):
                plot_args = [v.values for v in plot_args]

            # Draw the plot
            self._facet_plot(func, ax, plot_args, kwargs)

        # Finalize the annotations and layout
        self._finalize_grid(args[:2])

        return self

    def map_dataframe(self, func, *args, **kwargs):
        """Like ``.map`` but passes args as strings and inserts data in kwargs.

        This method is suitable for plotting with functions that accept a
        long-form DataFrame as a `data` keyword argument and access the
        data in that DataFrame using string variable names.

        Parameters
        ----------
        func : callable
            A plotting function that takes data and keyword arguments. Unlike
            the `map` method, a function used here must "understand" Pandas
            objects. It also must plot to the currently active matplotlib Axes
            and take a `color` keyword argument. If faceting on the `hue`
            dimension, it must also take a `label` keyword argument.
        args : strings
            Column names in self.data that identify variables with data to
            plot. The data for each variable is passed to `func` in the
            order the variables are specified in the call.
        kwargs : keyword arguments
            All keyword arguments are passed to the plotting function.

        Returns
        -------
        self : object
            Returns self.

        """

        # If color was a keyword argument, grab it here
        kw_color = kwargs.pop("color", None)

        # Iterate over the data subsets
        for (row_i, col_j, hue_k), data_ijk in self.facet_data():

            # If this subset is null, move on
            if not data_ijk.values.size:
                continue

            # Get the current axis
            ax = self.facet_axis(row_i, col_j)

            # Decide what color to plot with
            kwargs["color"] = self._facet_color(hue_k, kw_color)

            # Insert the other hue aesthetics if appropriate
            for kw, val_list in self.hue_kws.items():
                kwargs[kw] = val_list[hue_k]

            # Insert a label in the keyword arguments for the legend
            if self._hue_var is not None:
                kwargs["label"] = self.hue_names[hue_k]

            # Stick the facet dataframe into the kwargs
            if self._dropna:
                data_ijk = data_ijk.dropna()
            kwargs["data"] = data_ijk

            # Draw the plot
            self._facet_plot(func, ax, args, kwargs)

        # Finalize the annotations and layout
        self._finalize_grid(args[:2])

        return self

    def _facet_color(self, hue_index, kw_color):

        color = self._colors[hue_index]
        if kw_color is not None:
            return kw_color
        elif color is not None:
            return color

    def _facet_plot(self, func, ax, plot_args, plot_kwargs):

        # Draw the plot
        func(*plot_args, **plot_kwargs)

        # Sort out the supporting information
        self._update_legend_data(ax)
        self._clean_axis(ax)

    def _finalize_grid(self, axlabels):
        """Finalize the annotations and layout."""
        self.set_axis_labels(*axlabels)
        self.set_titles()
        self.fig.tight_layout()

    def facet_axis(self, row_i, col_j):
        """Make the axis identified by these indices active and return it."""

        # Calculate the actual indices of the axes to plot on
        if self._col_wrap is not None:
            ax = self.axes.flat[col_j]
        else:
            ax = self.axes[row_i, col_j]

        # Get a reference to the axes object we want, and make it active
        plt.sca(ax)
        return ax

    def despine(self, **kwargs):
        """Remove axis spines from the facets."""
        utils.despine(self.fig, **kwargs)
        return self

    def set_axis_labels(self, x_var=None, y_var=None):
        """Set axis labels on the left column and bottom row of the grid."""
        if x_var is not None:
            self._x_var = x_var
            self.set_xlabels(x_var)
        if y_var is not None:
            self._y_var = y_var
            self.set_ylabels(y_var)
        return self

    def set_xlabels(self, label=None, **kwargs):
        """Label the x axis on the bottom row of the grid."""
        if label is None:
            label = self._x_var
        for ax in self._bottom_axes:
            ax.set_xlabel(label, **kwargs)
        return self

    def set_ylabels(self, label=None, **kwargs):
        """Label the y axis on the left column of the grid."""
        if label is None:
            label = self._y_var
        for ax in self._left_axes:
            ax.set_ylabel(label, **kwargs)
        return self

    def set_xticklabels(self, labels=None, step=None, **kwargs):
        """Set x axis tick labels on the bottom row of the grid."""
        for ax in self.axes.flat:
            if labels is None:
                labels = [l.get_text() for l in ax.get_xticklabels()]
                if step is not None:
                    xticks = ax.get_xticks()[::step]
                    labels = labels[::step]
                    ax.set_xticks(xticks)
            ax.set_xticklabels(labels, **kwargs)
        return self

    def set_yticklabels(self, labels=None, **kwargs):
        """Set y axis tick labels on the left column of the grid."""
        for ax in self.axes.flat:
            if labels is None:
                labels = [l.get_text() for l in ax.get_yticklabels()]
            ax.set_yticklabels(labels, **kwargs)
        return self

    def set_titles(self, template=None, row_template=None,  col_template=None,
                   **kwargs):
        """Draw titles either above each facet or on the grid margins.

        Parameters
        ----------
        template : string
            Template for all titles with the formatting keys {col_var} and
            {col_name} (if using a `col` faceting variable) and/or {row_var}
            and {row_name} (if using a `row` faceting variable).
        row_template:
            Template for the row variable when titles are drawn on the grid
            margins. Must have {row_var} and {row_name} formatting keys.
        col_template:
            Template for the row variable when titles are drawn on the grid
            margins. Must have {col_var} and {col_name} formatting keys.

        Returns
        -------
        self: object
            Returns self.

        """
        args = dict(row_var=self._row_var, col_var=self._col_var)
        kwargs["size"] = kwargs.pop("size", mpl.rcParams["axes.labelsize"])

        # Establish default templates
        if row_template is None:
            row_template = "{row_var} = {row_name}"
        if col_template is None:
            col_template = "{col_var} = {col_name}"
        if template is None:
            if self._row_var is None:
                template = col_template
            elif self._col_var is None:
                template = row_template
            else:
                template = " | ".join([row_template, col_template])

        row_template = utils.to_utf8(row_template)
        col_template = utils.to_utf8(col_template)
        template = utils.to_utf8(template)

        if self._margin_titles:
            if self.row_names is not None:
                # Draw the row titles on the right edge of the grid
                for i, row_name in enumerate(self.row_names):
                    ax = self.axes[i, -1]
                    args.update(dict(row_name=row_name))
                    title = row_template.format(**args)
                    bgcolor = self.fig.get_facecolor()
                    ax.annotate(title, xy=(1.02, .5), xycoords="axes fraction",
                                rotation=270, ha="left", va="center",
                                backgroundcolor=bgcolor, **kwargs)

            if self.col_names is not None:
                # Draw the column titles  as normal titles
                for j, col_name in enumerate(self.col_names):
                    args.update(dict(col_name=col_name))
                    title = col_template.format(**args)
                    self.axes[0, j].set_title(title, **kwargs)

            return self

        # Otherwise title each facet with all the necessary information
        if (self._row_var is not None) and (self._col_var is not None):
            for i, row_name in enumerate(self.row_names):
                for j, col_name in enumerate(self.col_names):
                    args.update(dict(row_name=row_name, col_name=col_name))
                    title = template.format(**args)
                    self.axes[i, j].set_title(title, **kwargs)
        elif self.row_names is not None and len(self.row_names):
            for i, row_name in enumerate(self.row_names):
                args.update(dict(row_name=row_name))
                title = template.format(**args)
                self.axes[i, 0].set_title(title, **kwargs)
        elif self.col_names is not None and len(self.col_names):
            for i, col_name in enumerate(self.col_names):
                args.update(dict(col_name=col_name))
                title = template.format(**args)
                # Index the flat array so col_wrap works
                self.axes.flat[i].set_title(title, **kwargs)
        return self

    @property
    def ax(self):
        """Easy access to single axes."""
        if self.axes.shape == (1, 1):
            return self.axes[0, 0]
        else:
            err = ("You must use the `.axes` attribute (an array) when "
                   "there is more than one plot.")
            raise AttributeError(err)

    @property
    def _inner_axes(self):
        """Return a flat array of the inner axes."""
        if self._col_wrap is None:
            return self.axes[:-1, 1:].flat
        else:
            axes = []
            n_empty = self._nrow * self._ncol - self._n_facets
            for i, ax in enumerate(self.axes):
                append = (i % self._ncol and
                          i < (self._ncol * (self._nrow - 1)) and
                          i < (self._ncol * (self._nrow - 1) - n_empty))
                if append:
                    axes.append(ax)
            return np.array(axes, object).flat

    @property
    def _left_axes(self):
        """Return a flat array of the left column of axes."""
        if self._col_wrap is None:
            return self.axes[:, 0].flat
        else:
            axes = []
            for i, ax in enumerate(self.axes):
                if not i % self._ncol:
                    axes.append(ax)
            return np.array(axes, object).flat

    @property
    def _not_left_axes(self):
        """Return a flat array of axes that aren't on the left column."""
        if self._col_wrap is None:
            return self.axes[:, 1:].flat
        else:
            axes = []
            for i, ax in enumerate(self.axes):
                if i % self._ncol:
                    axes.append(ax)
            return np.array(axes, object).flat

    @property
    def _bottom_axes(self):
        """Return a flat array of the bottom row of axes."""
        if self._col_wrap is None:
            return self.axes[-1, :].flat
        else:
            axes = []
            n_empty = self._nrow * self._ncol - self._n_facets
            for i, ax in enumerate(self.axes):
                append = (i >= (self._ncol * (self._nrow - 1)) or
                          i >= (self._ncol * (self._nrow - 1) - n_empty))
                if append:
                    axes.append(ax)
            return np.array(axes, object).flat

    @property
    def _not_bottom_axes(self):
        """Return a flat array of axes that aren't on the bottom row."""
        if self._col_wrap is None:
            return self.axes[:-1, :].flat
        else:
            axes = []
            n_empty = self._nrow * self._ncol - self._n_facets
            for i, ax in enumerate(self.axes):
                append = (i < (self._ncol * (self._nrow - 1)) and
                          i < (self._ncol * (self._nrow - 1) - n_empty))
                if append:
                    axes.append(ax)
            return np.array(axes, object).flat

