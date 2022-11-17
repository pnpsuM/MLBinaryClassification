import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout

class Titanic():
  def __init__(self, path = f'datasets/'):
    print('Initializing...')
    self._original = None
    self._data = None
    self._scaled_data = None
    self._train = None
    self._test = None
    self._scaler = StandardScaler()
    self._encoder = LabelEncoder()
    self.ReadData(path)
    
  def ReadData(self, path, get = False, show_head =True):    
    df_train = pd.read_csv(path + "train.csv")
    df_test = pd.read_csv(path + "test.csv")
    df_train['Type'] = 'train'
    df_test['Type'] = 'test'
    
    print('Data Loaded.')
    
    self._data = pd.concat([df_train, df_test])
    if show_head:
      print(self._data.head())
    
    if get:
      return self._data
    
  def Preprocess(self, map, titles, get = False):
    self.TitleExtraction(map, titles)
    self.FamilySizeExtraction()
    self.IfChild()
    self.FillOut()
    self.FeatureEncoding()
    
    if get:
      return self._data
  
  def TitleExtraction(self, map : dict, titles : dict):    
    self._data['Title'] = self._data['Name']
    for name_string in self._data['Name']:
      self._data['Title'] = self._data['Name'].str.extract('([A-Za-z]+)\.', expand=True)
    
    self._data.replace({'Title': map}, inplace=True)  
    
    for title in titles:
      median_age = self._data.groupby('Title')['Age'].median()[titles.index(title)]
      self._data.loc[(self._data['Age'].isnull()) & (self._data['Title'] == title), 'Age'] = median_age
  
  def FamilySizeExtraction(self):    
    self._data['Family_Size'] = self._data['Parch'] + self._data['SibSp'] + 1

  def IfChild(self):    
    self._data.loc[:,'Child'] = 1
    self._data.loc[(self._data['Age'] >= 18),'Child'] = 0
    
  def FillOut(self):
    fa = self._data[self._data["Pclass"] == 3]
    self._data['Fare'].fillna(fa['Fare'].median(), inplace = True)
    
  def FeatureEncoding(self):
    # Encoding features
    target_col = ["Survived"]
    id_dataset = ["Type"]
    cat_cols   = self._data.nunique()[self._data.nunique() < 12].keys().tolist()
    cat_cols   = [x for x in cat_cols ]
    
    # numerical columns
    num_cols   = [x for x in self._data.columns if x not in cat_cols + target_col + id_dataset]
    # Binary columns with 2 values
    bin_cols   = self._data.nunique()[self._data.nunique() == 2].keys().tolist()
    # Columns more than 2 values
    categorical_cols = [i for i in cat_cols if i not in bin_cols]
    
    # Label encoding Binary columns
    for i in bin_cols :
        self._data[i] = self._encoder.fit_transform(self._data[i])
    # Duplicating columns for multi value columns
    self._data = pd.get_dummies(data = self._data, columns = categorical_cols )
    # Scaling Numerical columns
    self._scaled_data = self._scaler.fit_transform(self._data[num_cols])
    self._scaled_data = pd.DataFrame(self._scaled_data, columns = num_cols)
    # dropping original values merging scaled values for numerical columns
    self._original = self._data.copy()
    self._data = self._data.drop(columns = num_cols,axis = 1)
    self._data = self._data.merge(self._scaled_data, left_index = True,right_index = True,how = "left")
    self._data = self._data.drop(columns = ['PassengerId'],axis = 1)

    # Target = 1st column
    cols = self._data.columns.tolist()
    cols.insert(0, cols.pop(cols.index('Survived')))
    self._data = self._data.reindex(columns= cols)
    
  def GetXandY(self):
    # Cutting train and test
    self._train = self._data[self._data['Type'] == 1].drop(columns = ['Type'])
    self._test = self._data[self._data['Type'] == 0].drop(columns = ['Type'])

    # X and Y
    x_train = self._train.iloc[:, 1:20].as_matrix()
    y_train = self._train.iloc[:,0].as_matrix()
    x_test = self._test.iloc[:, 1:20].as_matrix()
    y_test = self._test.iloc[:, 0].as_matrix()
    
    self.data_dict = {'x_train' : x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
    
    return self.data_dict