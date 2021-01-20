from matplotlib.pyplot import title
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.model_selection import KFold


# Remove rows by threshold
def remove_rows_by_feature(df, feature, limit=0, reverse=False):
    '''
        df - Pandas DataFrame
        limit - feature threshold
        reverse - remove above or below threshold
    '''
    df_ = df.copy()
    feature_values_to_delete = df_.groupby(feature)[feature].count()[df_.groupby(feature)[feature].count() < limit].index

    if reverse:
        return df_[df_[feature].isin(feature_values_to_delete)]
    else:
        return df_[~df_[feature].isin(feature_values_to_delete)]


def compare_features_in_df(df1=None, df2=None, feature=None, top_values=10, show_report=False, limit=50):
    '''
        Compare two features in Test and Train DataFrames
        df1 - Train DF
        df2 - Test DF
        feature - Column name to compare
        top_values - limit of value_counts
    '''
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[1].set_xlim(df1[feature].value_counts().nlargest(top_values).max(), 0)
    axes[1].yaxis.tick_right()
    fig.suptitle('Сравнение признака '+feature, fontsize=16)
    
    sns.barplot(y=df1[feature].value_counts().nlargest(top_values).index, 
                x=df1[feature].value_counts().nlargest(top_values), ax=axes[0]).set_title('Тренировочная выборка')
    
    sns.barplot(y=df2[feature].value_counts().nlargest(top_values).index, 
                x=df2[feature].value_counts().nlargest(top_values), ax=axes[1]).set_title('Тестовая выборка')
    
    if show_report:
        df1_unique = df1[feature].unique()
        df2_unique = df2[feature].unique()
        
        report_df1 = ' '.join(sorted(list(map(str,df1_unique))))
        report_df2 = ' '.join(sorted(list(map(str, df2_unique))))

        report2 = ' '.join(sorted(list(map(str, get_intersection_df(df1, df2, feature=feature).index))))

        print(f'Уникальные значения в df1 ({len(df1_unique)}):\n {report_df1[:limit]}\n')
        print(f'Уникальные значения в df2 ({len(df2_unique)}):\n {report_df2[:limit]}\n')
        print(f'Пересекаются следующие значения ({len(report2.split())}):\n', report2[:limit])
    

def get_intersection_df(df1=None, df2=None, feature=None, reverse=False) -> None:
    '''
        Get intersection on two dataframes by feature.
        
        df1 - PandasDataFrame
        df2 - PandasDataFrame
        feature - feature name
        reverse - ~intersection
    '''
    if df1 is None or df2 is None or feature is None:
        raise Exception('Fill params!')
    
    if reverse:   
        return df1[feature][~df1[feature].isin(df2[feature])].value_counts()
    else:
        return df1[feature][df1[feature].isin(df2[feature])].value_counts()
    

def plot_correlation(corr, neg=-0.4, pos=0.5):
    '''
        Plot heatmap matrix for correlation
            - corr - Pandas DF.corr()
    '''
  
    sns.heatmap(corr[(corr >= pos) | (corr <= neg)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True)
        

def heatmap_numeric_target_variable(df, dependent_variable):
    '''
    Takes df, a dependant variable as str
    Returns a heatmap of all independent variables' correlations with dependent variable 
    '''
    plt.figure(figsize=(8, 10))
    g = sns.heatmap(df.corr()[[dependent_variable]].sort_values(by=dependent_variable), 
                    annot=True, 
                    cmap='coolwarm', 
                    vmin=-1,
                    vmax=1) 
    return g


def eda_checks(df):
    '''
    Takes df
    Checks nulls
    '''
    if df.isnull().sum().sum() > 0:
        mask_total = df.isnull().sum().sort_values(ascending=False) 
        total = mask_total[mask_total > 0]

        mask_percent = df.isnull().mean().sort_values(ascending=False) 
        percent = mask_percent[mask_percent > 0] 

        missing_data = pd.concat([total, percent], axis=1, keys=['Всего', '%'])
    
        print(f'Общее количество и процент пропусков:\n {missing_data}')
    else: 
        print('Пропусков не обнаружено.')
        

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5, use_abs=True, target=None, filter_target=False):
    '''
    df - Pandas DataFrame
    n - number of return values
    use_abs - use absolute numbers
    target - feature name to return
    filter_target - use feature filter
    '''
    
    if use_abs:
        au_corr = df.corr().abs().unstack()
    else:
        au_corr = df.corr().unstack()
        
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    
    if filter_target and target is not None:
        return au_corr.filter(like=target, axis=0)[0:n]
    
    return au_corr[0:n]


# Посчитаем важность признаков при помощи однофакторного дисперсионного анализа (ANOVA)
def plot_fclassif(df, columns, target):
    '''
    sklean.feature_selection.f_classif wrapper
    ------------------------------------------
    df - Pandas DataFrame
    columns - numerical features
    target - Our target feature
    '''
    
    if df is not None and columns is not None and target is not None:
        imp_num = pd.Series(f_classif(df[columns], df[target])[0], index = columns)
        imp_num.sort_values(inplace = True)
        imp_num.plot(kind = 'barh')
    else:
        raise Exception('Fill params!')


def plot_mutual_info_classif(df, columns, target):
    '''
    sklean.feature_selection.mutual_info_classif
    ------------------------------------------
    df - Pandas DataFrame
    columns - binary, categorical features
    target - Our target feature
    '''
    if df is not None and columns is not None and target is not None:
        imp_cat = pd.Series(mutual_info_classif(df[columns], df[target],
                                            discrete_features =True), index = columns)
        imp_cat.sort_values(inplace = True)
        imp_cat.plot(kind = 'barh')
    else:
        raise Exception('Fill params!')
    

# Выводим распределение переменной и ее логарифма
def plot_dist_log(df, column, nrows=1, ncols=2, figsize=(15,15)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    plt.subplots_adjust(wspace = 0.5)
    axes = axes.flatten()
    axes[0].title.set_text(column)
    axes[1].title.set_text('np.log('+column+')')
    sns.distplot(df[column], ax=axes[0], hist=True, axlabel=False)
    sns.distplot(np.log(df[column] + 1), ax=axes[1], hist=True, axlabel=False)
    fig.suptitle(f'Распределение признака {column} и np.log({column})', fontsize=16, y=1.12)
    
    # Выводим дополнительную информацию о признаке
    feature_info = dict(count=df[column].count(), nan=df[column].isnull().sum(), 
                    min=df[column].min(), max=df[column].max(), median=df[column].median(), mean=df[column].mean())

    fi = pd.Series(feature_info)
    p = pd.DataFrame(fi, columns=[column])

    display(p)


# %% [code]
from sklearn.base import BaseEstimator, TransformerMixin

# Генератор стандартных фич
class FeatureGenerator(BaseEstimator, TransformerMixin):

    def __init__(self, feature_list = None, primitives=['mul']):
        self.feature_list = feature_list
        self.used_features = []
        self.primitives = primitives

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X_ = X.copy() # работаем с копией, чтобы не изменять оригинальный датасет
        self.used_features = []  # обнуляем список использованных фич перед каждым вызовом transform

        if self.feature_list is None:
            self.feature_list = X_.columns

        return self.feature_generator(X_)
    
    def feature_generator(self, X):
        # Перебираем наши признаки и генерируем новые
        for i, feature in enumerate(self.feature_list):
            # Добавяем признак в список использованных
            self.used_features.append(feature)
            # Генерируем новые признаки по списку оставшихся
            for feature_ in list(set(self.feature_list) - set(self.used_features)):
                if self.primitives == 'all':
                    self.primitives = ['mul', 'sum', 'sub', 'div', 'mean', 'std', 'median']
                for primitive in self.primitives:
                    feature_name = f'{feature}_{primitive}_{feature_}'
                    if primitive == 'mul':
                        X[feature_name] = X[feature] * X[feature_]
                    if primitive == 'sum':
                        X[feature_name] = X[feature] + X[feature_]
                    if primitive == 'sub':
                        X[feature_name] = X[feature] - X[feature_]
                    if primitive == 'div':
                        X[feature_name] = X[feature] / (X[feature_]+1)
                    if primitive == 'mean':
                        X[feature_name] = X[feature] / (X[feature_].mean() + 1)
                    if primitive == 'std':
                        X[feature_name] = X[feature] / (X[feature_].std() + 1)
                    if primitive == 'median':
                        X[feature_name] = X[feature] / (X[feature_].median() + 1)
                    
        return X
    
    
# Получаем бинарные признаки из признака, содержащего списки
def get_binary_dummies(data, column, prefix='dummy'):
    from sklearn.preprocessing import MultiLabelBinarizer

    # Бинарные метки
    mlb = MultiLabelBinarizer()
    expandedLabelData = mlb.fit_transform(data[column])
    labelClasses = mlb.classes_

    # Создаем датафрэйм бинарных признаков
    labels = []
    
    for label in labelClasses:
        labels.append(prefix+'_'+label)
    
    expandedLabels = pd.DataFrame(expandedLabelData, columns=labels)

    return expandedLabels


#Preprocess function
def preprocess_text(text, return_tokens=True):
    #Create lemmatizer and stopwords list
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              and token.strip() not in punctuation]
    
    text = " ".join(tokens)
    
    if return_tokens:
        return tokens
    else:
        return text
    
# TargetEncoder
class KFoldTargetEncoderTrain(BaseEstimator, TransformerMixin):

        def __init__(self, colnames,targetName,n_fold=5,verbosity=True,discardOriginal_col=False, random_state=42):

            self.colnames = colnames
            self.targetName = targetName
            self.n_fold = n_fold
            self.verbosity = verbosity
            self.discardOriginal_col = discardOriginal_col
            self.random_state = random_state

        def fit(self, X, y=None):
            return self


        def transform(self,X):

            assert(type(self.targetName) == str)
            assert(type(self.colnames) == str)
            assert(self.colnames in X.columns)
            assert(self.targetName in X.columns)

            mean_of_target = X[self.targetName].mean()
            kf = KFold(n_splits = self.n_fold, shuffle = True, random_state=self.random_state)



            col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
            X[col_mean_name] = np.nan

            for tr_ind, val_ind in kf.split(X):
                X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
    #             print(tr_ind,val_ind)
                X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)[self.targetName].mean())

            X[col_mean_name].fillna(mean_of_target, inplace = True)

            if self.verbosity:

                encoded_feature = X[col_mean_name].values
                print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,
                                                                                        self.targetName,
                                                                                        np.corrcoef(X[self.targetName].values, encoded_feature)[0][1]))
            if self.discardOriginal_col:
                X = X.drop(self.targetName, axis=1)
                

            return X
        
        
class KFoldTargetEncoderTest(BaseEstimator, TransformerMixin):
    
    def __init__(self,train,colNames,encodedName):
        
        self.train = train
        self.colNames = colNames
        self.encodedName = encodedName
        
        
    def fit(self, X, y=None):
        return self

    def transform(self,X):


        mean = self.train[[self.colNames,self.encodedName]].groupby(self.colNames).mean().reset_index() 
        
        dd = {}
        for index, row in mean.iterrows():
            dd[row[self.colNames]] = row[self.encodedName]

        
        X[self.encodedName] = X[self.colNames]
        X = X.replace({self.encodedName: dd})

        return X

# %% [code]
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def remove_outliers_iqr_column(df_in, col_name):
    '''
    df_in - Pandas DataFrame
    col_name - DataFrame column to detect outliers
    '''
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    
    return df_out


# IQR 
def remove_outliers_irq_dataset(df, iqr_detect=1.5):
    '''
    df - Pandas DataFrame
    iqr_detect - IQR to detect and filter outliers. Default: 1.5 
    '''
    
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    return df[~((df < (Q1 - iqr_detect * IQR)) |(df > (Q3 + iqr_detect * IQR))).any(axis=1)]

# Plot Boxplots
def plot_outliers(df, width=20, height=10):
    '''
    df - Pandas DataFrame
    width - Figure width
    height - Figure height
    '''
    fig = plt.figure(figsize=(width, height))
    sns.boxplot(x="variable", y="value", data=pd.melt(df))
    plt.show()
    

# Plot Boxplots between features
def plot_boxplots(df, feature, target, figsize=(10,5)):
    fig = plt.figure(figsize=figsize)
    plt.xticks(rotation=45)
    sns.boxplot(x = feature, y = target, data = df)
        

from math import log
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def log_loss(predicted, target):
    if len(predicted) != len(target):
        print('lengths not equal!')
        return
        
    target = [float(x) for x in target]   # make sure all float values
    predicted = [min([max([x,1e-15]),1-1e-15]) for x in predicted]  # within (0,1) interval
            
    return -(1.0/len(target))*sum([target[i]*log(predicted[i]) + 
                                               (1.0-target[i])*log(1.0-predicted[i]) 
                                               for i in range(len(target))])
    
    
# Отрисовать ROC кривую
def roc_auc_plot(y_true, y_pred_proba):
    '''
    Функция считает AUC и отрисовывает ROC кривую:
        y_true - истинное значение класса
        y_pred_proba - предсказанная вероятность класса [:, 1]
    '''
    # Посчитать значения ROC кривой и значение площади под кривой AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure()
    plt.plot([0, 1], label='Baseline', linestyle='--')
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.title('ROC AUC = %0.3f' % roc_auc, fontsize=15)
    plt.xlabel('False positive rate (FPR)', fontsize=15)
    plt.ylabel('True positive rate (TPR)', fontsize=15)
    plt.legend(fontsize=15, loc = 'lower right')