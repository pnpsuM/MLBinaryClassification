import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class Titanic():
  def __init__(self, path = f'datasets/', **kwargs):
    print('Initializing...')
    self._original = None
    self._data = None
    self._scaled_data = None
    self._train = None
    self._test = None
    self._scaler = StandardScaler()
    self._encoder = LabelEncoder()
    self.ReadData(path, show_head = kwargs['show_head'])
    
  def ReadData(self, path, show_head =True, test_only = False, get = False):
    if not test_only:
      df_train = pd.read_csv(path + "train.csv")
      df_test = pd.read_csv(path + "test.csv")
      df_train['Type'] = 'train'
      df_test['Type'] = 'test'
      self._data = pd.concat([df_train, df_test])
      # Concatenate train and test data and save it in a Variable
    else:
      df_test = pd.read_csv(path + "test.csv")
      df_test['Type'] = 'test'
      self._data = df_test
      
    print('Data Loaded.')
    if show_head:
      print(self._data.head())
    
    if get:
      return self._data
    
  def Preprocess(self, map, titles, VERSION, prepath = f'preprocessed', get = False):
    # Make a definition of a func. elsewhere and add it here before FeatureEncoding()
    print("Data Preprocessing...")
    self.TitleExtraction(map, titles)
    self.FamilySizeExtraction()
    self.IfChild()
    self.FillOut()
    self.FamilySurvival()
    self._data = self._data.drop(columns = ['Age','Cabin','Name','Last_Name',
                                            'Parch', 'SibSp','Ticket'])
    self._data.to_csv(prepath + f"/preprocessed_{VERSION}.csv", index = False)
    self.FeatureEncoding()
    print("Done Preprocessing.")
    
    if get:
      return self._data
  
  def TitleExtraction(self, map : dict, titles : dict):    
    self._data['Title'] = self._data['Name']
    for name_string in self._data['Name']:
      self._data['Title'] = self._data['Name'].str.extract('([A-Za-z]+)\.', expand=True)
    
    self._data.replace({'Title': map}, inplace=True)  
    
    for title in titles:
      median_age = self._data.groupby('Title')['Age'].median()[titles.index(title)]
      self._data.loc[(self._data['Age'].isnull()) & (self._data['Title'] == title), 'Age'] =\
        median_age
  
  def FamilySizeExtraction(self):    
    self._data['Family_Size'] = self._data['Parch'] + self._data['SibSp'] + 1

  def IfChild(self):    
    self._data.loc[:,'Child'] = 1
    self._data.loc[(self._data['Age'] > 19),'Child'] = 0
    
  def FillOut(self):
    fa = self._data[self._data["Pclass"] == 3]
    self._data['Fare'].fillna(fa['Fare'].median(), inplace = True)
    
  def FamilySurvival(self):
    DEFAULT_SURVIVAL_VALUE = 0
    self._data['Last_Name'] = self._data['Name'].apply(lambda x: str.split(x, ",")[0])
    self._data['Family_Survival'] = DEFAULT_SURVIVAL_VALUE
    # Initialize Fam_Sur column with 0.5
    for _, group_df in self._data[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                                    'SibSp', 'Parch', 'Age', 'Type']].groupby(['Last_Name', 'Ticket']):
      # Same LN and Fare => Family Group, and makes df out of them
      if (len(group_df)) != 1:
        # When a family group is found
        cnt = group_df.loc[group_df['Type'] != 'test', 'Survived'].sum()
        cnt /= len(group_df.loc[group_df['Type'] != 'test'])
        for i, row in group_df.iterrows():
          pass_id = row['PassengerId']
          self._data.loc[self._data['PassengerId'] == pass_id, 'Family_Survival'] = cnt
          
    # DEFAULT_SURVIVAL_VALUE = 0.5
    # self._data['Last_Name'] = self._data['Name'].apply(lambda x: str.split(x, ",")[0])
    # self._data['Family_Survival'] = DEFAULT_SURVIVAL_VALUE
    # # Initialize Fam_Sur column with 0.5
    # for group, group_df in self._data[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
    #                                 'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Ticket']):
    #   # Same LN and Fare => Family Group, and makes df out of them
    #   if (len(group_df)) != 1:
    #     # When a family group is found
    #     for i, row in group_df.iterrows():
    #       smax = group_df.drop(i)['Survived'].max()
    #       smin = group_df.drop(i)['Survived'].min()
    #       pass_id = row['PassengerId']
    #       if (smax == 1.):
    #         self._data.loc[self._data['PassengerId'] == pass_id, 'Family_Survival'] = 1
    #       elif (smin == 0.):
    #         self._data.loc[self._data['PassengerId'] == pass_id, 'Family_Survival'] = 0
    # for _, group_df in self._data.groupby('Ticket'):
    #   if (len(group_df) != 1):
    #     for i, row in group_df.iterrows():
    #       if (row['Family_Survival'] == 0) or (row['Family_Survival'] == 0.5):
    #         smax = group_df.drop(i)['Survived'].max()
    #         smin = group_df.drop(i)['Survived'].min()
    #         pass_id = row['PassengerId']
    #         if (smax == 1.):
    #           self._data.loc[self._data['PassengerId'] == pass_id, 'Family_Survival'] = 1
    #         elif (smin == 0.):
    #           self._data.loc[self._data['PassengerId'] == pass_id, 'Family_Survival'] = 0
    
  def FeatureEncoding(self):
    # Encoding features
    target_col = ["Survived"]
    id_dataset = ["Type"]
    cat_cols   = self._data.nunique()[self._data.nunique() < 12].keys().tolist()
    cat_cols   = [x for x in cat_cols if x != 'Family_Survival' or x != 'Family_Size']
    print(cat_cols)
    
    # numerical columns
    num_cols   = [x for x in self._data.columns if (x not in cat_cols + target_col + \
      id_dataset)]
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
    x_train = self._train.iloc[:, 1:20].values.astype(np.float32)
    y_train = self._train.iloc[:, 0].values.astype(np.float32)
    x_test = self._test.iloc[:, 1:20].values.astype(np.float32)
    _ = self._test.iloc[:, 0].values.astype(np.float32)
    
    self.data_dict = {'x_train' : x_train, 'y_train': y_train, 'x_test': x_test}
    print("Returned Data Dictionary")
    
    return self.data_dict
  
  def GetTestXandY(self):
    self._test = self._data[self._data['Type'] == 0].drop(columns = ['Type'])
    x_test = self._test.iloc[:, 1:20].values.astype(np.float32)
    y_test = self._test.iloc[:, 0].values.astype(np.float32)
    
    self.data_dict = {'x_test': x_test, 'y_test': y_test}
    print("Returned Data Dictionary")
    
    return self.data_dict
    