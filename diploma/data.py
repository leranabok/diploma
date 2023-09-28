import nltk
import re
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import talib
import yfinance
import gzip
import typing as tp
import pandas as pd
from tqdm.notebook import tqdm
from collections import defaultdict
import os
tickers = ['AAPL', 'TSLA', 'MSFT', 'AMZN', 'GOOG', 'V', 'NVDA']

wnl = WordNetLemmatizer()

def filter_out_non_text(text: str) -> str:
  filtered_text = re.sub(r'[^\w\s]', '', text) # прибрати пунктуацію
  filtered_text = repr(filtered_text.encode('utf-8'))[2:-1] # black magic (емодзі до строки) (у utf-8 емодзі має вигляд \x(від трьох цифр в 16-розрядній арифметиці))
  filtered_text = filtered_text.replace('\\n',' ')
  filtered_text = re.sub(r"won\'t", "will not", filtered_text)
  filtered_text = re.sub(r"can\'t", "can not", filtered_text)
  filtered_text = re.sub(r"n\'t", " not", filtered_text)
  filtered_text = re.sub(r"\'re", " are", filtered_text)
  filtered_text = re.sub(r"(he|He)\'s", "he is", filtered_text)
  filtered_text = re.sub(r"(she|She)\'s", "she is", filtered_text)
  filtered_text = re.sub(r"(it|It)\'s", "it is", filtered_text)
  filtered_text = re.sub(r"\'d", " would", filtered_text)
  filtered_text = re.sub(r"\'ll", " will", filtered_text)
  filtered_text = re.sub(r"\'t", " not", filtered_text)
  filtered_text = re.sub(r"[#@$]", "", filtered_text)
  filtered_text = re.sub(r"(\'ve|has)", " have", filtered_text)
  filtered_text = re.sub(r"\'m", " am", filtered_text)
  filtered_text = re.sub(r"\\x[a-f0-9]*", "", filtered_text) # прибрати емодзі
  filtered_text = filtered_text.split(' ')
  filtered_text = ' '.join([
      wnl.lemmatize(word.lower()) for word in filtered_text if word.isalpha() # lemmatizer. приводить слова до називного відмінку (іменники + прикметники) або інфінітиву (дієслова) в однині
      ])
  return filtered_text

def construct_vocabulary():
  res = []
  for word in all_vocab:
    if word != '':
        res.append(twitter_vectors[word])
  return np.vstack(res)



def read_tickers(tickers):
  """
  Read and process stock prices fro given list of tickers
  """
  df = dict()
  for ticker_name in tickers:
    ticker = yfinance.Ticker(ticker_name)
    history_prices = ticker.history('1d', start='2015-01-01', end='2020-07-01')[columns]
    df[ticker_name] = history_prices
  return df

def plot_tickers(tickers_dataset: pd.DataFrame, name: str = 'Stock Prices'):
  """
  Plot tickers from tickers dataset
  """
  number_of_plots = len(tickers_dataset.columns)
  fig, ax = plt.subplots(number_of_plots, figsize=(15, 10))
  fig.suptitle(name)
  for idx, ticker in enumerate(tickers_dataset.columns):
    if number_of_plots > 1:
      axis = ax[idx]
    else:
      axis = ax
    axis.plot(tickers_dataset.index.values.ravel(), tickers_dataset[ticker].values.ravel())
    axis.set(ylabel=ticker)
  plt.show()

def shift_data(tickers_dataset, y_col, days_shift):
  """
  Add to data some lags for given column
  """
  new_dataset = tickers_dataset.copy()
  for lag in range(1, days_shift + 1):
    new_dataset[f'{y_col}_-{lag}'] = new_dataset[y_col].shift(lag)

  return new_dataset

def compute_return_for_k_days(df: pd.DataFrame, k: int = 0):
  column_name = 'Return' + (f'{k}_days' if k else '')
  df[column_name] = df['Close'] - df['Open'].shift(-k)
def compute_future_return_for_k_days(df: pd.DataFrame, k: int = 0):
  column_name = 'Future_Return' + (f'{k}_days' if k else '')
  df[column_name] = df['Close'].shift(-k) - df['Open']


def compute_stoch(x: pd.DataFrame,
                  fastk_period: int = 14,
                  slowk_period: int = 3,
                  slowk_matype: int = 0,
                  slowd_period: int = 3,
                  slowd_matype: int = 0):
    slowk, slowd = talib.STOCH(x['High'].ffill(), x['Low'].ffill(), x['Close'].ffill(),
                               fastk_period=fastk_period,
                               slowk_period=slowk_period,
                               slowk_matype=slowk_matype,
                               slowd_period=slowd_period,
                               slowd_matype=slowd_matype)
    x['slowd'] = slowd
    x['slowk'] = slowk


def compute_bop(x: pd.DataFrame):
    x['BOP'] = talib.BOP(x['Open'],
                         x['High'],
                         x['Low'],
                         x['Close'])


def compute_cci(x: pd.DataFrame):
    x['CCI'] = talib.CCI(x['High'],
                         x['Low'],
                         x['Close'])


def compute_mfi(x: pd.DataFrame, timeperiod: int = 14):
    x['MFI'] = talib.MFI(x['High'],
                         x['Low'],
                         x['Close'],
                         x['Volume'],
                         timeperiod=timeperiod)


def compute_wma(x: pd.DataFrame, column: str):
    x['WMA_' + column] = talib.WMA(x[column].fillna(method='pad'))

def add_text(stock: str):
  tweets_about_stock = twts[twts.ticker_symbol == stock][['text']].groupby('Date').agg({'text': set})
  tweets_about_stock['text'] = tweets_about_stock['text'].map(list)
  dict_nasdaq[stock] = dict_nasdaq[stock].join(tweets_about_stock, how='left')
  dict_nasdaq[stock]['text'] = dict_nasdaq[stock]['text'].fillna('')

def construct_dataset(stock: str, column_lag_order_dict: tp.Dict[str, int], columns_to_keep: tp.List[str], target_feature: str) -> tp.Dict[str, tp.List[str]]:
  new_ds = dict_nasdaq[stock][columns_to_keep].copy()
  new_ds = new_ds.rename(columns={column: f'{stock}_{column}' for column in columns_to_keep})
  column_mapping = {column: [column] for column in new_ds.columns}
  target_feature_name = f"{stock}_{target_feature}"
  new_ds[target_feature_name] = dict_nasdaq[stock][target_feature]
  for column, lag_order in column_lag_order_dict.items():
    if lag_order=='auto':
      lag_order_temp = sum(sm.stattools.pacf(dict_nasdaq[stock][column].dropna()) >= 0.1)
    else:
      lag_order_temp = lag_order
    column_mapping.update({ f"{stock}_{column}": [f"{stock}_{column}_{lag}_days" for lag in range(lag_order_temp+1)]})
    for lag in range(lag_order_temp+1):
      new_ds[f"{stock}_{column}_{lag}_days"] = dict_nasdaq[stock][column].shift(lag)
  dict_nasdaq[stock] = new_ds.dropna()
  return column_mapping, target_feature_name

# TODO: перетворити pandas в tf.data.Dataset, виділити train та test (доробити для кількох ознак)
def convert_to_dataset(pandas_dataset: pd.DataFrame,
                       column_mapping: tp.Dict[tp.List[str], str],
                       target_feature_names: tp.List[str]) -> tf.data.Dataset: #ще раз запитати у Діми
  tf_datasets = []
  for subset in column_mapping.values():
    vals = pandas_dataset[subset].values
    shape = (*vals.shape, 1) # (entries, timesteps, 1)
    vals = vals.reshape(shape)
    tf_datasets.append(tf.data.Dataset.from_tensor_slices(vals))
  tf_datasets = tf.data.Dataset.zip(tuple(tf_datasets)).map(lambda *features: {feature_n: (feature if not(feature_n.split('_')[1].startswith('text')) else feature[:, 0]) for feature_n, feature in zip(column_mapping.keys(), features)})
  target_vals = pandas_dataset[target_feature_names]
  target_vals = tf.data.Dataset.from_tensor_slices(target_vals)
  tf_datasets = tf.data.Dataset.zip((tf_datasets, target_vals))
  return tf_datasets