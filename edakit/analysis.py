import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import parallel_coordinates, autocorrelation_plot
from scipy.stats import skew, kurtosis, pearsonr, spearmanr, zscore, ks_2samp, chi2_contingency
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.mosaicplot import mosaic
from statsmodels.tsa.seasonal import seasonal_decompose


def univariate_analysis(df, numerical_columns=None, categorical_columns=None, kinds=None):
    """
    univariate_analysis
    ====================

    Perform detailed univariate analysis on numerical and categorical data.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    numerical_columns : list of str, optional
        List of numerical columns to analyze. If None, no numerical analysis is performed.
    categorical_columns : list of str, optional
        List of categorical columns to analyze. If None, no categorical analysis is performed.
    kinds : list of str, optional
        List of types of plots to generate. Options include:
        - 'Hist': Histograms for numerical data
        - 'Boxplot': Box plots for numerical data
        - 'Density': Density plots for numerical data
        - 'Summary': Detailed summary statistics for numerical data
        - 'Barplot': Bar plots for categorical data
        - 'Piechart': Pie charts for categorical data
        - 'Frequency': Detailed frequency distribution for categorical data
        - 'All': Generate all of the above plots

    Returns
    -------
    None
    """

    # Default kinds if not specified
    if kinds is None:
        kinds = []

    # Check if 'All' is in kinds
    if 'All' in kinds:
        kinds = ['Hist', 'Boxplot', 'Density', 'Summary', 'Barplot', 'Piechart', 'Frequency']

    # Numerical Data Analysis
    if numerical_columns:
        if 'Hist' in kinds:
            for column in numerical_columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[column], bins=30, kde=False)
                plt.title(f'Histogram of {column}')
                plt.xlabel(column)
                plt.ylabel('Frequency')
                plt.show()

        if 'Boxplot' in kinds:
            for column in numerical_columns:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=df[column])
                plt.title(f'Box Plot of {column}')
                plt.xlabel(column)
                plt.show()

        if 'Density' in kinds:
            for column in numerical_columns:
                plt.figure(figsize=(10, 6))
                sns.kdeplot(df[column], shade=True)
                plt.title(f'Density Plot of {column}')
                plt.xlabel(column)
                plt.ylabel('Density')
                plt.show()

        if 'Summary' in kinds:
            # Initialize an empty list to store summary statistics
            summary_stats = []

            for column in numerical_columns:
                data = df[column].dropna()
                stats = {
                    'Column': column,
                    'Mean': np.mean(data),
                    'Median': np.median(data),
                    'Mode': data.mode()[0] if not data.mode().empty else np.nan,
                    'Q1': np.percentile(data, 25),
                    'Q3': np.percentile(data, 75),
                    'IQR': np.percentile(data, 75) - np.percentile(data, 25),
                    'Variance': np.var(data, ddof=1),
                    'Standard Deviation': np.std(data, ddof=1),
                    'Skewness': skew(data),
                    'Kurtosis': kurtosis(data),
                    'Min': np.min(data),
                    'Max': np.max(data)
                }
                summary_stats.append(stats)

            # Create a DataFrame from the summary statistics
            summary_df = pd.DataFrame(summary_stats)

            # Plot the summary statistics table
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=summary_df.values,
                             colLabels=summary_df.columns,
                             rowLabels=summary_df['Column'],
                             cellLoc='center',
                             loc='center',
                             bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            plt.title('Detailed Summary Statistics')
            plt.show()

    # Categorical Data Analysis
    if categorical_columns:
        if 'Barplot' in kinds:
            for column in categorical_columns:
                plt.figure(figsize=(10, 6))
                sns.countplot(x=df[column])
                plt.title(f'Bar Plot of {column}')
                plt.xlabel(column)
                plt.ylabel('Count')
                plt.show()

        if 'Piechart' in kinds:
            for column in categorical_columns:
                plt.figure(figsize=(8, 8))
                size = df[column].value_counts()
                plt.pie(size, labels=size.index, autopct='%1.1f%%', startangle=140)
                plt.title(f'Pie Chart of {column}')
                plt.show()

        if 'Frequency' in kinds:
            # Initialize an empty list to store frequency distributions
            freq_distributions = []

            for column in categorical_columns:
                counts = df[column].value_counts()
                unique_values = df[column].unique()
                mode = df[column].mode()[0] if not df[column].mode().empty else np.nan

                # Prepare data for DataFrame
                freq_data = {
                    'Column': column,
                    'Mode': mode,
                    'Unique Values': ', '.join(map(str, unique_values)),
                    'Value Counts': str(counts.to_dict())
                }
                freq_distributions.append(freq_data)

            # Create a DataFrame from the frequency distributions
            freq_df = pd.DataFrame(freq_distributions)

            # Plot the frequency distribution table
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=freq_df.values,
                             colLabels=freq_df.columns,
                             rowLabels=freq_df['Column'],
                             cellLoc='center',
                             loc='center',
                             bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            plt.title('Detailed Frequency Distribution')
            plt.show()


def bivariate_analysis(df, num_vs_num=None, num_vs_cat=None, cat_vs_cat=None, kinds=None):
    """
    Bivariate Analysis
    ===================

    Perform bivariate analysis on numerical vs. numerical, numerical vs. categorical, and categorical vs. categorical
    data.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    num_vs_num : list of tuples, optional
        List of tuples where each tuple contains two numerical columns to analyze.
    num_vs_cat : list of tuples, optional
        List of tuples where each tuple contains a numerical column and a categorical column to analyze.
    cat_vs_cat : list of tuples, optional
        List of tuples where each tuple contains two categorical columns to analyze.
    kinds : list of str, optional
        List of types of plots to generate. Options include:
        - 'Scatter': Scatter plots for numerical vs. numerical data
        - 'Correlation': Correlation coefficients for numerical vs. numerical data
        - 'Hexbin': Hexbin plots for numerical vs. numerical data
        - 'Boxplot': Box plots for numerical vs. categorical data
        - 'Violin': Violin plots for numerical vs. categorical data
        - 'Strip': Strip plots for numerical vs. categorical data
        - 'Contingency': Contingency tables for categorical vs. categorical data
        - 'Stacked Bar': Stacked bar plots for categorical vs. categorical data
        - 'Mosaic': Mosaic plots for categorical vs. categorical data
        - 'All': Generate all of the above plots

    Returns
    -------
    None
    """

    # Default kinds if not specified
    if kinds is None:
        kinds = []

    # Check if 'All' is in kinds
    if 'All' in kinds:
        kinds = ['Scatter', 'Correlation', 'Hexbin', 'Boxplot', 'Violin', 'Strip', 'Contingency', 'Stacked Bar',
                 'Mosaic']

    # Numerical vs. Numerical Analysis
    if num_vs_num:
        for col1, col2 in num_vs_num:
            if 'Scatter' in kinds:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=df[col1], y=df[col2])
                plt.title(f'Scatter Plot of {col1} vs {col2}')
                plt.xlabel(col1)
                plt.ylabel(col2)
                plt.show()

            if 'Correlation' in kinds:
                if col1 in df.columns and col2 in df.columns:
                    pearson_corr, _ = pearsonr(df[col1].dropna(), df[col2].dropna())
                    spearman_corr, _ = spearmanr(df[col1].dropna(), df[col2].dropna())
                    print(f'\nCorrelation Coefficients between {col1} and {col2}:')
                    print(f'Pearson: {pearson_corr}')
                    print(f'Spearman: {spearman_corr}')

            if 'Hexbin' in kinds:
                plt.figure(figsize=(10, 6))
                plt.hexbin(df[col1], df[col2], gridsize=30, cmap='Blues')
                plt.colorbar(label='Count')
                plt.title(f'Hexbin Plot of {col1} vs {col2}')
                plt.xlabel(col1)
                plt.ylabel(col2)
                plt.show()

    # Numerical vs. Categorical Analysis
    if num_vs_cat:
        for num_col, cat_col in num_vs_cat:
            if 'Boxplot' in kinds:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=df[cat_col], y=df[num_col])
                plt.title(f'Box Plot of {num_col} by {cat_col}')
                plt.xlabel(cat_col)
                plt.ylabel(num_col)
                plt.show()

            if 'Violin' in kinds:
                plt.figure(figsize=(10, 6))
                sns.violinplot(x=df[cat_col], y=df[num_col])
                plt.title(f'Violin Plot of {num_col} by {cat_col}')
                plt.xlabel(cat_col)
                plt.ylabel(num_col)
                plt.show()

            if 'Strip' in kinds:
                plt.figure(figsize=(10, 6))
                sns.stripplot(x=df[cat_col], y=df[num_col], jitter=True)
                plt.title(f'Strip Plot of {num_col} by {cat_col}')
                plt.xlabel(cat_col)
                plt.ylabel(num_col)
                plt.show()

    # Categorical vs. Categorical Analysis
    if cat_vs_cat:
        for col1, col2 in cat_vs_cat:
            if 'Contingency' in kinds:
                contingency_table = pd.crosstab(df[col1], df[col2])
                chi2, p, _, _ = chi2_contingency(contingency_table)
                print(f'\nContingency Table and Chi-Square Test between {col1} and {col2}:')
                print(contingency_table)
                print(f'Chi-Square Statistic: {chi2}')
                print(f'p-value: {p}')

            if 'Stacked Bar' in kinds:
                contingency_table = pd.crosstab(df[col1], df[col2])
                contingency_table.div(contingency_table.sum(1), axis=0).plot(kind='bar', stacked=True, figsize=(10, 6))
                plt.title(f'Stacked Bar Plot of {col1} and {col2}')
                plt.xlabel(col1)
                plt.ylabel('Proportion')
                plt.show()

            if 'Mosaic' in kinds:
                plt.figure(figsize=(10, 6))
                mosaic(df, index=[col1, col2])
                plt.title(f'Mosaic Plot of {col1} and {col2}')
                plt.show()


def multivariate_analysis(df, numerical_columns=None, categorical_columns=None, mixed_columns=None, kinds=None):
    """
    Multivariate Analysis
    ====================

    Perform multivariate analysis on numerical, categorical, and mixed data types.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    numerical_columns : list of str, optional
        List of numerical columns to analyze. If None, no numerical analysis is performed.
    categorical_columns : list of str, optional
        List of categorical columns to analyze. If None, no categorical analysis is performed.
    mixed_columns : list of tuples, optional
        List of tuples where each tuple contains columns for mixed data types to analyze.
    kinds : list of str, optional
        List of types of plots to generate. Options include:
        - 'Pairplot': Pair plots for numerical data
        - 'Heatmap Correlation': Heatmaps of correlation matrices
        - 'PCA': Principal Component Analysis (PCA) plots
        - 'Cross-tab': Cross-tabulations for categorical data
        - 'Heatmap Frequency': Heatmaps of frequency distributions for categorical data
        - 'Facet Grid': Facet grids for mixed data types
        - 'Parallel Coordinates': Parallel coordinates plots for mixed data types
        - '3D Scatter': 3D scatter plots for numerical data
        - 'All': Generate all of the above plots

    Returns
    -------
    None
    """

    # Default kinds if not specified
    if kinds is None:
        kinds = []

    # Check if 'All' is in kinds and include all analyses
    if 'All' in kinds:
        kinds = ['Pairplot', 'Heatmap Correlation', 'PCA', 't-SNE', 'Cross-tab', 'Heatmap Frequency', 'Facet Grid',
                 'Parallel Coordinates', '3D Scatter']

    # Numerical Data Analysis
    if numerical_columns:
        if 'Pairplot' in kinds:
            sns.pairplot(df[numerical_columns])
            plt.title('Pair Plot of Numerical Variables')
            plt.show()

        if 'Heatmap Correlation' in kinds:
            corr_matrix = df[numerical_columns].corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Heatmap of Correlation Matrix')
            plt.show()

        if 'PCA' in kinds:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numerical_columns].dropna())
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            plt.figure(figsize=(10, 6))
            plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
            plt.title('PCA Plot')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.show()

        if 't-SNE' in kinds:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numerical_columns].dropna())
            tsne = TSNE(n_components=2, random_state=0)
            tsne_result = tsne.fit_transform(scaled_data)
            plt.figure(figsize=(10, 6))
            plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7)
            plt.title('t-SNE Plot')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.show()

    # Categorical Data Analysis
    if categorical_columns:
        if 'Cross-tab' in kinds:
            for column in categorical_columns:
                print(f'\nCross-tabulation for {column}:')
                print(pd.crosstab(df[column], df.index))

        if 'Heatmap Frequency' in kinds:
            for column in categorical_columns:
                freq_table = pd.crosstab(df[column], df.index)
                plt.figure(figsize=(8, 8))
                sns.heatmap(freq_table, annot=True, cmap='coolwarm', fmt='d')
                plt.title(f'Heatmap of Frequency Distribution for {column}')
                plt.show()

    # Mixed Data Types Analysis
    if mixed_columns:
        if 'Facet Grid' in kinds:
            for col1, col2 in mixed_columns:
                g = sns.FacetGrid(df, col=col1, row=col2)
                g.map(sns.scatterplot, col2)
                plt.show()

        if 'Parallel Coordinates' in kinds:
            if numerical_columns:
                df_mixed = df[numerical_columns + [col for col, _ in mixed_columns]]
                plt.figure(figsize=(12, 8))
                parallel_coordinates(df_mixed, class_column=numerical_columns[0])
                plt.title('Parallel Coordinates Plot')
                plt.show()

        if '3D Scatter' in kinds:
            if len(numerical_columns) >= 3:
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(df[numerical_columns[0]], df[numerical_columns[1]], df[numerical_columns[2]], alpha=0.7)
                ax.set_xlabel(numerical_columns[0])
                ax.set_ylabel(numerical_columns[1])
                ax.set_zlabel(numerical_columns[2])
                plt.title('3D Scatter Plot')
                plt.show()


def segmentation_analysis(df, categorical_columns=None, numerical_columns=None, segmentation_type=None, kinds=None):
    """
    Segmentation Analysis
    ====================

    Perform segmentation analysis on numerical and categorical data.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    categorical_columns : list of str, optional
        List of categorical columns to analyze. If None, no categorical analysis is performed.
    numerical_columns : list of str, optional
        List of numerical columns to analyze. If None, no numerical analysis is performed.
    segmentation_type : str, optional
        Type of segmentation to perform. Options include:
        - 'Categories': Segment by categorical variables
        - 'Numerical Ranges': Segment by numerical ranges or bins
    kinds : list of str, optional
        List of types of plots to generate. Options include:
        - 'Segmented Bar': Segmented bar plots for categorical data
        - 'Grouped Box': Grouped box plots for categorical data
        - 'Histogram': Histograms for numerical ranges
        - 'Density': Density plots for numerical ranges
        - 'Boxplot': Box plots for numerical ranges
        - 'Facet Grid': Facet grids for numerical vs. categorical data
        - 'Mosaic': Mosaic plots for categorical data
        - 'Heatmap': Heatmaps for numerical ranges
        - 'Violin': Violin plots for numerical ranges
        - 'All': Generate all of the above plots

    Returns
    -------
    None
    """

    # Default kinds if not specified
    if kinds is None:
        kinds = []

    # Check if 'All' is in kinds and include all analyses
    if 'All' in kinds:
        kinds = ['Segmented Bar', 'Grouped Box', 'Histogram', 'Density', 'Boxplot', 'Facet Grid', 'Mosaic', 'Heatmap',
                 'Violin']

    # Segmentation by Categories
    if segmentation_type == 'Categories' and categorical_columns:
        if 'Segmented Bar' in kinds:
            for column in categorical_columns:
                plt.figure(figsize=(10, 6))
                sns.countplot(x=df[column])
                plt.title(f'Segmented Bar Plot of {column}')
                plt.xlabel(column)
                plt.ylabel('Count')
                plt.show()

        if 'Grouped Box' in kinds and numerical_columns:
            for column in categorical_columns:
                for num_col in numerical_columns:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x=df[column], y=df[num_col])
                    plt.title(f'Grouped Box Plot of {num_col} by {column}')
                    plt.xlabel(column)
                    plt.ylabel(num_col)
                    plt.show()

        if 'Facet Grid' in kinds and numerical_columns:
            for num_col in numerical_columns:
                g = sns.FacetGrid(df, col=categorical_columns[0], col_wrap=4, height=4)
                g.map(sns.scatterplot, num_col)
                g.set_titles("{col_name}")
                plt.show()

        if 'Mosaic' in kinds and len(categorical_columns) >= 2:
            plt.figure(figsize=(10, 6))
            mosaic(df, index=categorical_columns)
            plt.title(f'Mosaic Plot of {", ".join(categorical_columns)}')
            plt.show()

    # Segmentation by Numerical Ranges
    if segmentation_type == 'Numerical Ranges' and numerical_columns:
        for column in numerical_columns:
            segments = pd.cut(df[column], bins=5)  # Customize bins as needed

            if 'Histogram' in kinds:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=df, x=column, hue=segments, multiple='stack', bins=30)
                plt.title(f'Histogram of {column} by Segments')
                plt.xlabel(column)
                plt.ylabel('Frequency')
                plt.show()

            if 'Density' in kinds:
                plt.figure(figsize=(10, 6))
                sns.kdeplot(data=df, x=column, hue=segments, common_norm=False)
                plt.title(f'Density Plot of {column} by Segments')
                plt.xlabel(column)
                plt.ylabel('Density')
                plt.show()

            if 'Boxplot' in kinds:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=segments, y=df[column])
                plt.title(f'Box Plot of {column} by Segments')
                plt.xlabel('Segment')
                plt.ylabel(column)
                plt.show()

            if 'Heatmap' in kinds:
                pivot_table = df.pivot_table(index=segments, values=column, aggfunc='count')
                plt.figure(figsize=(10, 6))
                sns.heatmap(pivot_table, annot=True, cmap='Blues', cbar=True)
                plt.title(f'Heatmap of {column} by Segments')
                plt.xlabel('Segment')
                plt.ylabel(column)
                plt.show()

            if 'Violin' in kinds:
                plt.figure(figsize=(10, 6))
                sns.violinplot(x=segments, y=df[column])
                plt.title(f'Violin Plot of {column} by Segments')
                plt.xlabel('Segment')
                plt.ylabel(column)
                plt.show()


def time_series_analysis(df, time_column, value_column, kinds=None, window_size=12):
    """
    Time Series Analysis
    ====================

    Perform time series analysis including trend analysis and seasonality/cyclic pattern analysis.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    time_column : str
        The name of the column containing time information (e.g., date).
    value_column : str
        The name of the column containing the values to analyze.
    kinds : list of str, optional
        List of types of plots to generate. Options include:
        - 'Line': Line plots of the time series data
        - 'Moving Average': Moving averages of the time series data
        - 'Seasonal Decomposition': Seasonal decomposition of the time series data
        - 'Autocorrelation': Autocorrelation plots of the time series data
        - 'All': Generate all of the above plots
    window_size : int, optional
        The window size for calculating moving averages. Default is 12.

    Returns
    -------
    None
    """

    # Default kinds if not specified
    if kinds is None:
        kinds = []

    # Check if 'All' is in kinds and include all analyses
    if 'All' in kinds:
        kinds = ['Line', 'Moving Average', 'Seasonal Decomposition', 'Autocorrelation']

    # Ensure time_column is in datetime format
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(by=time_column)

    # Time Series Analysis
    if 'Line' in kinds:
        plt.figure(figsize=(12, 6))
        plt.plot(df[time_column], df[value_column], label='Time Series')
        plt.title('Line Plot of Time Series Data')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    if 'Moving Average' in kinds:
        plt.figure(figsize=(12, 6))
        df['Moving Average'] = df[value_column].rolling(window=window_size).mean()
        plt.plot(df[time_column], df[value_column], label='Original')
        plt.plot(df[time_column], df['Moving Average'], label=f'{window_size}-Point Moving Average', color='orange')
        plt.title('Moving Average of Time Series Data')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    if 'Seasonal Decomposition' in kinds:
        plt.figure(figsize=(12, 8))
        result = seasonal_decompose(df.set_index(time_column)[value_column], model='additive')
        result.plot()
        plt.suptitle('Seasonal Decomposition of Time Series Data')
        plt.show()

    if 'Autocorrelation' in kinds:
        plt.figure(figsize=(12, 6))
        autocorrelation_plot(df[value_column])
        plt.title('Autocorrelation Plot of Time Series Data')
        plt.show()


def outlier_detection(df, numerical_columns=None, kinds=None, z_threshold=3, iqr_threshold=1.5):
    """
    Outlier Detection
    =================

    Perform outlier detection using visual and statistical methods.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    numerical_columns : list of str, optional
        List of numerical columns to analyze. If None, no numerical analysis is performed.
    kinds : list of str, optional
        List of types of outlier detection methods to apply. Options include:
        - 'Boxplot': Box plots for visual detection of outliers
        - 'Scatter': Scatter plots for visual detection of outliers
        - 'Z-score': Z-score based outlier detection
        - 'IQR': IQR-based outlier detection
        - 'All': Generate all of the above plots
    z_threshold : float, optional
        The threshold for Z-score based outlier detection. Default is 3.
    iqr_threshold : float, optional
        The threshold for IQR-based outlier detection. Default is 1.5.

    Returns
    -------
    None
    """

    # Default kinds if not specified
    if kinds is None:
        kinds = []

    # Check if 'All' is in kinds and include all analyses
    if 'All' in kinds:
        kinds = ['Boxplot', 'Scatter', 'Z-score', 'IQR']

    # Outlier Detection
    if numerical_columns:
        for column in numerical_columns:
            # Visual Methods
            if 'Boxplot' in kinds:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=df[column])
                plt.title(f'Box Plot of {column}')
                plt.xlabel(column)
                plt.show()

            if 'Scatter' in kinds:
                # Scatter plot only makes sense if there's another numerical column to compare with
                if len(numerical_columns) > 1:
                    for other_column in numerical_columns:
                        if other_column != column:
                            plt.figure(figsize=(10, 6))
                            plt.scatter(df[column], df[other_column], alpha=0.5)
                            plt.title(f'Scatter Plot of {column} vs {other_column}')
                            plt.xlabel(column)
                            plt.ylabel(other_column)
                            plt.show()

            # Statistical Methods
            if 'Z-score' in kinds:
                z_scores = zscore(df[numerical_columns], nan_policy='omit')
                abs_z_scores = np.abs(z_scores)
                outliers = df[abs_z_scores > z_threshold].any(axis=1)
                print(f'Z-score based outliers for {column}:')
                # Print rows where outliers are detected
                print(df[outliers])

            if 'IQR' in kinds:
                q1 = df[numerical_columns].quantile(0.25)
                q3 = df[numerical_columns].quantile(0.75)
                iqr = q3 - q1
                upper_wick = q3 + iqr_threshold * iqr
                lower_wick = q1 - iqr_threshold * iqr
                outliers = df[(df[numerical_columns] < lower_wick) | (df[numerical_columns] > upper_wick)].any(axis=1)
                print(f'IQR based outliers for {column}:')
                # Print rows where outliers are detected
                print(df[outliers])


def missing_data_analysis(df, impute_strategy='mean', visualize=True):
    """
    Perform missing data analysis on a DataFrame. This includes identifying patterns of missing values
    and visualizing distributions before and after imputation.

    Parameters ---------- df : pd.DataFrame The input dataframe with missing values. impute_strategy : str,
    optional The strategy used for imputation. Options are 'mean', 'median', 'most_frequent', and 'constant'. Default
    is 'mean'. visualize : bool, optional Whether to visualize the results. Default is True.

    Returns
    -------
    imputed_df : pd.DataFrame
        The dataframe with missing values imputed.
    """

    # Identify missing values
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    missing_summary = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percentage})

    # Display the missing summary
    print("Missing Data Summary:")
    print(missing_summary)

    if visualize:
        # Plot missing value heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
        plt.title('Missing Value Heatmap')
        plt.show()

    # Imputation
    imputer = SimpleImputer(strategy=impute_strategy)
    imputed_data = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(imputed_data, columns=df.columns)

    if visualize:
        # Plot distributions before and after imputation for numerical columns
        numerical_columns = df.select_dtypes(include=[np.number]).columns

        for column in numerical_columns:
            plt.figure(figsize=(12, 6))

            # Original distribution
            sns.histplot(df[column].dropna(), color='blue', label='Original', kde=True, stat='density', linewidth=0)

            # Imputed distribution
            sns.histplot(imputed_df[column], color='red', label='Imputed', kde=True, stat='density', linewidth=0)

            plt.title(f'Distribution of {column} Before and After Imputation')
            plt.xlabel(column)
            plt.ylabel('Density')
            plt.legend()
            plt.show()

    return imputed_df


def feature_engineering_insights(df, target_column, numerical_columns=None, categorical_columns=None, kinds=None):
    """
    Feature Engineering Insights
    ============================

    Perform analysis to understand feature importance and relationships between features.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    target_column : str
        The target column for feature importance analysis.
    numerical_columns : list of str, optional
        List of numerical columns to include in the analysis.
    categorical_columns : list of str, optional
        List of categorical columns to include in the analysis.
    kinds : list of str, optional
        List of types of feature engineering insights to generate. Options include:
        - 'Feature Importance': Feature importance plots using models like Random Forest
        - 'Feature Relationships': Feature interaction plots
        - 'All': Generate all of the above plots

    Returns
    -------
    None
    """

    # Default kinds if not specified
    if kinds is None:
        kinds = []

    # Check if 'All' is in kinds and include all analyses
    if 'All' in kinds:
        kinds = ['Feature Importance', 'Feature Relationships']

    # Prepare Data
    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=[object]).columns.tolist()

    # Encode categorical features if necessary
    le = LabelEncoder()
    for col in categorical_columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))

    # Define features and target
    X = df[numerical_columns + categorical_columns]
    y = df[target_column]

    if 'Feature Importance' in kinds:
        # Feature Importance
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        sns.barplot(x=importances[indices], y=X.columns[indices], palette='viridis')
        plt.title('Feature Importance from Random Forest')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.show()

    if 'Feature Relationships' in kinds:
        # Feature Relationships
        # Pair plots (only if fewer features to visualize)
        if len(numerical_columns + categorical_columns) <= 10:
            plt.figure(figsize=(12, 10))
            sns.pairplot(df[numerical_columns + categorical_columns + [target_column]], hue=target_column,
                         palette='viridis')
            plt.title('Pair Plot of Features')
            plt.show()

        # Interaction plots
        for col1 in numerical_columns:
            for col2 in numerical_columns:
                if col1 != col2:
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(x=df[col1], y=df[col2], hue=df[target_column], palette='viridis')
                    plt.title(f'Interaction Plot of {col1} vs {col2}')
                    plt.xlabel(col1)
                    plt.ylabel(col2)
                    plt.show()


def distribution_comparison(df, numerical_columns=None, categorical_columns=None, compare_col=None, kinds=None):
    """
    Perform distribution comparison on numerical and categorical data with visualizations and statistical tests.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    numerical_columns : list of str, optional
        List of numerical columns to analyze. If None, no numerical analysis is performed.
    categorical_columns : list of str, optional
        List of categorical columns to analyze. If None, no categorical analysis is performed.
    compare_col : str, optional
        Column name to compare distributions across different categories or numerical ranges.
    kinds : list of str, optional
        List of types of analyzes to perform. Options include:
        - 'Histogram': Overlay histograms for numerical data
        - 'KDE': Kernel Density Estimate (KDE) plots for numerical data
        - 'KS-Test': Kolmogorov-Smirnov test for numerical data
        - 'Chi2-Test': Chi-square test for categorical data

    Returns
    -------
    None
    """

    # Default kinds if not specified
    if kinds is None:
        kinds = []

    # Numerical Data Analysis
    if numerical_columns and compare_col:
        if 'Histogram' in kinds:
            plt.figure(figsize=(12, 6))
            for column in numerical_columns:
                sns.histplot(df, x=column, hue=compare_col, element='step', stat='density', common_norm=False)
                plt.title(f'Overlay Histograms of {column} by {compare_col}')
                plt.xlabel(column)
                plt.ylabel('Density')
                plt.legend(title=compare_col)
                plt.show()

        if 'KDE' in kinds:
            plt.figure(figsize=(12, 6))
            for column in numerical_columns:
                sns.kdeplot(data=df, x=column, hue=compare_col, common_norm=False)
                plt.title(f'KDE Plot of {column} by {compare_col}')
                plt.xlabel(column)
                plt.ylabel('Density')
                plt.legend(title=compare_col)
                plt.show()

        if 'KS-Test' in kinds:
            for column in numerical_columns:
                unique_groups = df[compare_col].unique()
                if len(unique_groups) == 2:
                    group1 = df[df[compare_col] == unique_groups[0]][column].dropna()
                    group2 = df[df[compare_col] == unique_groups[1]][column].dropna()
                    stat, p_value = ks_2samp(group1, group2)
                    print(f'Kolmogorov-Smirnov test for {column} between {unique_groups[0]} and {unique_groups[1]}:')
                    print(f'Statistic: {stat}, p-value: {p_value}')
                else:
                    print(f'Kolmogorov-Smirnov test requires exactly two groups. Found: {len(unique_groups)}')

    # Categorical Data Analysis
    if categorical_columns and compare_col:
        if 'Chi2-Test' in kinds:
            for column in categorical_columns:
                contingency_table = pd.crosstab(df[column], df[compare_col])
                chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
                print(f'Chi-square test for {column} against {compare_col}:')
                print(f'Chi2 Statistic: {chi2_stat}, p-value: {p_value}')
                print('Contingency Table:')
                print(contingency_table)


def dimensionality_reduction(df, numerical_columns, n_components=2, random_state=0, plot_title="Dimensionality "
                                                                                               "Reduction"):
    """
    Perform dimensionality reduction using PCA and t-SNE, and visualize the results.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    numerical_columns : list of str
        List of numerical columns to be used for dimensionality reduction.
    n_components : int, optional
        Number of components for PCA and t-SNE. Default is 2.
    random_state : int, optional
        Random state for reproducibility of t-SNE. Default is 0.
    plot_title : str, optional
        Title for the plots. Default is "Dimensionality Reduction".

    Returns
    -------
    None
    """
    # Ensure numerical columns are in the dataframe
    df_num = df[numerical_columns].dropna()

    # PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_result = pca.fit_transform(df_num)

    # t-SNE
    tsne = TSNE(n_components=n_components, random_state=random_state)
    tsne_result = tsne.fit_transform(df_num)

    # Plotting PCA Results
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    plt.title(f'{plot_title} - PCA')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')

    # Plotting t-SNE Results
    plt.subplot(1, 2, 2)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5)
    plt.title(f'{plot_title} - t-SNE')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    plt.tight_layout()
    plt.show()
