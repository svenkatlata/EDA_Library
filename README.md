# edakit

`edakit` is a Python library designed for advanced exploratory data analysis (EDA). It provides a comprehensive set of tools for performing univariate, bivariate, multivariate, and time series analysis, as well as outlier detection, feature engineering, distribution comparison, and dimensionality reduction. The library is built to simplify the process of understanding and visualizing data, offering a range of methods and visualizations to extract valuable insights.

## Features

1. **Univariate Analysis**: Perform detailed analysis of individual columns, including histograms, box plots, density plots, and summary statistics for numerical data, and bar plots, pie charts, and frequency distributions for categorical data.
2. **Bivariate Analysis**: Analyze relationships between two variables with scatter plots, correlation coefficients, hexbin plots, box plots, violin plots, and more.
3. **Multivariate Analysis**: Explore interactions between multiple variables using pair plots, heatmaps, PCA, cross-tabulations, facet grids, and 3D scatter plots.
4. **Segmentation Analysis**: Segment data based on categorical variables or numerical ranges, and visualize using segmented bar plots, grouped box plots, histograms, density plots, and more.
5. **Time Series Analysis**: Analyze time series data for trends, seasonality, and autocorrelation with line plots, moving averages, seasonal decomposition, and autocorrelation plots.
6. **Outlier Detection**: Identify outliers using visual methods such as box plots and scatter plots, and statistical methods like Z-score and IQR-based detection.
7. **Missing Data Analysis**: Perform missing data analysis on a DataFrame by visualizing distributions before and after imputation.
8. **Feature Engineering Insights**: Evaluate feature importance and interactions using models like Random Forest, and generate plots to understand feature relationships.
9. **Distribution Comparison**: Compare distributions across different categories or numerical ranges using histograms, KDE plots, and statistical tests such as the Kolmogorov-Smirnov and Chi-square tests.
10. **Dimensionality Reduction**: Reduce dimensions using PCA, t-SNE, and UMAP, and visualize results with 2D and 3D plots.

## Installation

You can install `edakit` via pip:

```bash
pip install edakit
```

## Usage
Here's a brief overview of how to use edakit for various types of analysis.


### Univariate Analysis
Perform detailed univariate analysis on numerical and categorical data.
```python
import edakit as eda

# Perform univariate analysis
eda.univariate_analysis(df, 
                        numerical_columns=['col1', 'col2'], 
                        categorical_columns=['cat1'], 
                        kinds=['Hist', 'Boxplot', 'Barplot'])
```
#### Parameters:
- **df** (`pd.DataFrame`): The input dataframe.
- **numerical_columns** (`list of str`, optional): List of numerical columns to analyze.
- **categorical_columns** (`list of str`, optional): List of categorical columns to analyze.
- **kinds** (`list of str`, optional): Types of plots to generate. Options include:
  - `'Hist'`: Histograms for numerical data
  - `'Boxplot'`: Box plots for numerical data
  - `'Density'`: Density plots for numerical data
  - `'Summary'`: Detailed summary statistics for numerical data
  - `'Barplot'`: Bar plots for categorical data
  - `'Piechart'`: Pie charts for categorical data
  - `'Frequency'`: Detailed frequency distribution for categorical data
  - `'All'`: Generate all of the above plots
#### Returns:
- **None**


### Bivariate Analysis
Analyze relationships between pairs of variables.
```python
import edakit as eda

# Perform bivariate analysis
eda.bivariate_analysis(df, 
                        num_vs_num=[('col1', 'col2')], 
                        num_vs_cat=[('col1', 'cat1')], 
                        cat_vs_cat=[('cat1', 'cat2')],
                        kinds=['Scatter', 'Boxplot'])
```
#### Parameters
- **df** (`pd.DataFrame`): The input dataframe.
- **num_vs_num** (`list of tuples`, optional): Numerical vs. numerical columns to analyze.
- **num_vs_cat** (`list of tuples`, optional): Numerical vs. categorical columns to analyze.
- **cat_vs_cat** (`list of tuples`, optional): Categorical vs. categorical columns to analyze.
- **kinds** (`list of str`, optional): Types of plots to generate. Options include:
  - `'Scatter'`: Scatter plots for numerical vs. numerical data
  - `'Correlation'`: Correlation coefficients for numerical vs. numerical data
  - `'Hexbin'`: Hexbin plots for numerical vs. numerical data
  - `'Boxplot'`: Box plots for numerical vs. categorical data
  - `'Violin'`: Violin plots for numerical vs. categorical data
  - `'Strip'`: Strip plots for numerical vs. categorical data
  - `'Contingency'`: Contingency tables for categorical vs. categorical data
  - `'Stacked Bar'`: Stacked bar plots for categorical vs. categorical data
  - `'Mosaic'`: Mosaic plots for categorical vs. categorical data
  - `'All'`: Generate all of the above plots
#### Returns:
- **None**


### Multivariate Analysis
Explore interactions among multiple variables.
```python
import edakit as eda

# Perform multivariate analysis
eda.multivariate_analysis(df, 
                           numerical_columns=['col1', 'col2'], 
                           categorical_columns=['cat1'], 
                           mixed_columns=[('col1', 'cat1')],
                           kinds=['Pairplot', 'Heatmap Correlation'])
```
#### Parameters
- **df** (`pd.DataFrame`): The input dataframe.
- **numerical_columns** (`list of str`, optional): Numerical columns to analyze.
- **categorical_columns** (`list of str`, optional): Categorical columns to analyze.
- **mixed_columns** (`list of tuples`, optional): Columns for mixed data types to analyze.
- **kinds** (`list of str`, optional): Types of plots to generate. Options include:
  - `'Pairplot'`: Pair plots for numerical data
  - `'Heatmap Correlation'`: Heatmaps of correlation matrices
  - `'PCA'`: Principal Component Analysis (PCA) plots
  - `'Cross-tab'`: Cross-tabulations for categorical data
  - `'Heatmap Frequency'`: Heatmaps of frequency distributions for categorical data
  - `'Facet Grid'`: Facet grids for mixed data types
  - `'Parallel Coordinates'`: Parallel coordinates plots for mixed data types
  - `'3D Scatter'`: 3D scatter plots for numerical data
  - `'All'`: Generate all of the above plots
#### Returns:
- **None**


### Segmentation Analysis
Segment data based on categorical variables or numerical ranges.
```python
import edakit as eda

# Perform segmentation analysis
eda.segmentation_analysis(df, 
                           categorical_columns=['cat1'], 
                           numerical_columns=['col1'], 
                           segmentation_type='Categories', 
                           kinds=['Segmented Bar', 'Histogram'])
```
#### Parameters
- **df** (`pd.DataFrame`): The input dataframe.
- **categorical_columns** (`list of str`, optional): Categorical columns to analyze.
- **numerical_columns** (`list of str`, optional): Numerical columns to analyze.
- **segmentation_type** (`str`, optional): Type of segmentation. Options include:
  - `'Categories'`: Segment by categorical variables
  - `'Numerical Ranges'`: Segment by numerical ranges or bins
- **kinds** (`list of str`, optional): Types of plots to generate. Options include:
  - `'Segmented Bar'`: Segmented bar plots for categorical data
  - `'Grouped Box'`: Grouped box plots for categorical data
  - `'Histogram'`: Histograms for numerical ranges
  - `'Density'`: Density plots for numerical ranges
  - `'Boxplot'`: Box plots for numerical ranges
  - `'Facet Grid'`: Facet grids for numerical vs. categorical data
  - `'Mosaic'`: Mosaic plots for categorical data
  - `'Heatmap'`: Heatmaps for numerical ranges
  - `'Violin'`: Violin plots for numerical ranges
  - `'All'`: Generate all of the above plots
#### Returns:
- **None**


### Time Series Analysis
Analyze trends and patterns over time.
```python
import edakit as eda

# Perform time series analysis
eda.time_series_analysis(df, 
                         time_column='date', 
                         value_column='value', 
                         kinds=['Line', 'Moving Average', 'Seasonal Decomposition'])
```
#### Parameters
- **df** (`pd.DataFrame`): The input dataframe.
- **time_column** (`str`): Column containing time information (e.g., date).
- **value_column** (`str`): Column containing values to analyze.
- **kinds** (`list of str`, optional): Types of plots to generate. Options include:
  - `'Line'`: Line plots of the time series data
  - `'Moving Average'`: Moving averages of the time series data
  - `'Seasonal Decomposition'`: Seasonal decomposition of the time series data
  - `'Autocorrelation'`: Autocorrelation plots of the time series data
  - `'All'`: Generate all of the above plots
- **window_size** (`int`, optional): Window size for calculating moving averages. Default is 12.
#### Returns:
- **None**


### Outlier Detection
Identify outliers using visual and statistical methods.
```python
import edakit as eda

# Perform outlier detection
eda.outlier_detection(df, 
                      numerical_columns=['col1'], 
                      kinds=['Boxplot', 'Z-score', 'IQR'])
```
#### Parameters
- **df** (`pd.DataFrame`): The input dataframe.
- **numerical_columns** (`list of str`, optional): Numerical columns to analyze.
- **kinds** (`list of str`, optional): Types of outlier detection methods. Options include:
  - `'Boxplot'`: Box plots for visual detection of outliers
  - `'Scatter'`: Scatter plots for visual detection of outliers
  - `'Z-score'`: Z-score based outlier detection
  - `'IQR'`: IQR-based outlier detection
  - `'All'`: Generate all of the above plots
- **z_threshold** (`float`, optional): Threshold for Z-score based detection. Default is 3.
- **iqr_threshold** (`float`, optional): Threshold for IQR-based detection. Default is 1.5.
#### Returns:
- **None**


### Missing Data Analysis
Perform missing data analysis on a DataFrame by visualizing distributions before and after imputation. 
```python
import edakit as eda

# Perform missing data analysis and impute missing values using mean strategy
imputed_df = eda.missing_data_analysis(df, 
                                       impute_strategy='mean', 
                                       visualize=True)
```
#### Parameters
- **df** (`pd.DataFrame`): The input DataFrame containing missing values that you want to analyze and impute.
- **impute_strategy** (`str`, optional): The strategy for imputing missing values. The available options are:
  - `'mean'`: Replace missing values with the mean of the column.
  - `'median'`: Replace missing values with the median of the column.
  - `'most_frequent'`: Replace missing values with the most frequent value in the column.
  - `'constant'`: Replace missing values with a constant value specified by the user. (Note: The constant value is not specified in this function; if required, it should be implemented in the future version.)
  - Default is `'mean'`.
- **visualize** (`bool`, optional): Whether to visualize the results of the missing data analysis and imputation. If set to `True`, plots showing the distribution of missing values before and after imputation will be displayed. Default is `True`.
#### Returns
- **imputed_df** (`pd.DataFrame`): The DataFrame with missing values imputed according to the specified strategy.


### Feature Engineering Insights
Analyze feature importance and relationships between features.
```python
import edakit as eda

# Perform feature engineering insights
eda.feature_engineering_insights(df, 
                                 target_column='target', 
                                 numerical_columns=['col1', 'col2'], 
                                 categorical_columns=['cat1'], 
                                 kinds=['Feature Importance'])
```
#### Parameters
- **df** (`pd.DataFrame`): The input dataframe.
- **target_column** (`str`): Target column for feature importance analysis.
- **numerical_columns** (`list of str`, optional): Numerical columns to include in the analysis.
- **categorical_columns** (`list of str`, optional): Categorical columns to include in the analysis.
- **kinds** (`list of str`, optional): Types of feature engineering insights to generate. Options include:
  - `'Feature Importance'`: Feature importance plots using models like Random Forest
  - `'Feature Relationships'`: Feature interaction plots
  - `'All'`: Generate all of the above plots
#### Returns:
- **None**


### Distribution Comparison
Compare distributions across categories or numerical ranges.
```python
import edakit as eda

# Perform distribution comparison
eda.distribution_comparison(df, 
                            numerical_columns=['col1'], 
                            categorical_columns=['cat1'], 
                            compare_col='cat1', 
                            kinds=['Histogram', 'KDE'])
```
#### Parameters
- **df** (`pd.DataFrame`): The input dataframe.
- **numerical_columns** (`list of str`, optional): Numerical columns to analyze.
- **categorical_columns** (`list of str`, optional): Categorical columns to analyze.
- **compare_col** (`str`, optional): Column name to compare distributions across different categories or numerical ranges.
- **kinds** (`list of str`, optional): Types of analyses to perform. Options include:
  - `'Histogram'`: Overlay histograms for numerical data
  - `'KDE'`: Kernel Density Estimate (KDE) plots for numerical data
  - `'KS-Test'`: Kolmogorov-Smirnov test for numerical data
  - `'Chi2-Test'`: Chi-square test for categorical data
#### Returns:
- **None**


### Dimensionality Reduction
Reduce dimensions using PCA, t-SNE, and UMAP, and visualize the results.
```python
import edakit as eda

# Perform dimensionality reduction
eda.dimensionality_reduction(df, 
                              numerical_columns=['col1', 'col2'], 
                              n_components=2, 
                              plot_title="Dimensionality Reduction")
```
#### Parameters

- **df** (`pd.DataFrame`): The input dataframe.
- **numerical_columns** (`list of str`): Numerical columns for dimensionality reduction.
- **n_components** (`int`, optional): Number of components for reduction (default is 2).
- **random_state** (`int`, optional): Random state for reproducibility (default is 0).
- **plot_title** (`str`, optional): Title for the plots (default is "Dimensionality Reduction").
#### Returns:
- **None**


## Contributing
If you would like to contribute to `edakit`, please follow the standard open-source contribution guidelines. Fork the repository, make your changes, and submit a pull request.

## License
`edakit` is licensed under the MIT License. See the LICENSE file for more details.

## Contact
For any questions or feedback, please contact Venkat Lata.





