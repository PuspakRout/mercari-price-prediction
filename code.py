import pandas as pd
import numpy as np
import string
from string import punctuation
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix,hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge, Lasso
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import joblib
porter = PorterStemmer()
stop = stopwords.words("english")

    
class Data:
    
    def __init__(self, df):
        self.df = self.create_df(df)
    
    def create_df (self, file, preprocess=True, preprocess_category_name=True):
        if preprocess:
            df = self.fill_missing_values(file)
            
            if hasattr(df,'price'):
                df = df[df.price !=0]
                df["price"] = np.log(df["price"] + 1)
            
            df["count"] = df["item_description"].apply(lambda x: len(str(x)))
            df["item_description"] = df["item_description"].apply(self.remove_punctuation)
            df["item_description"] = df["item_description"].apply(porter.stem)
            df["item_description"] = df["item_description"].apply(self.remove_stopwords)
            df["item_description"] = df["item_description"].apply(self.lowercase)
            df['shipping'] = df['shipping'].astype(str)
            df['item_condition_id'] = df['item_condition_id'].astype(str)
        if preprocess_category_name:
            df["category_main"],df["category_sub1"], df["category_sub2"] = zip(*df["category_name"].apply(self.preprocess_category))
        return df
        
    def fill_missing_values(self,df):
        df["category_name"].fillna(value = "other",inplace = True)
        df["brand_name"].fillna(value = "unknown",inplace = True)
        df["item_description"].fillna(value = "no description yet",inplace = True)
        return df

    def remove_punctuation(self,sentence):
        return sentence.translate(str.maketrans('','',string.punctuation))

    def remove_stopwords(self,x):
        x = ' '.join([i for i in x.lower().split(' ') if i not in stop])
        return x

    def lowercase(self,x):
        return x.lower()

    def preprocess_category(self,category_name):
        try:
            main,sub1,sub2 = category_name.split('/')
            return main,sub1,sub2
        except:
            return 'other','other','other'
        
        
class feature_engineering():
    def __init__(self,df):
        self.df_feature = self.count_vectorize(df)
        
    def count_vectorize (self, df):
        cv = CountVectorizer()
        tfidf = TfidfVectorizer(max_features = 55000, ngram_range = (1,2), stop_words = "english")
        
        X_name = cv.fit_transform(df["name"])
        print('X_name:',X_name.shape)
        joblib.dump(cv,'objects/X_name')
        
        X_main = cv.fit_transform(df["category_main"])
        print('X_main:',X_main.shape)
        joblib.dump(cv,'objects/X_main')
        
        X_sub1 = cv.fit_transform(df["category_sub1"])
        print('X_sub1:',X_sub1.shape)
        joblib.dump(cv,'objects/X_sub1')
        
        X_sub2 = cv.fit_transform(df["category_sub2"])
        print('X_sub2:',X_sub2.shape)
        joblib.dump(cv,'objects/X_sub2')
        
        X_brand = cv.fit_transform(df["brand_name"])
        print('X_brand:',X_brand.shape)
        joblib.dump(cv,'objects/X_brand')
        
        X_dummies = csr_matrix(pd.get_dummies(df[['item_condition_id', 'shipping']], sparse=True).values)
        print('X_dummies:',X_dummies.shape)
        
        X_description = tfidf.fit_transform(df["item_description"])
        print('X_description:',X_description.shape)
        joblib.dump(tfidf,'objects/X_description')
        
        df1 = hstack((X_name,X_main,X_sub1,X_sub2,X_brand,X_dummies,X_description)).tocsr()
        print('df1:',df1.shape)
        return df1
    
    

class modelcontainer():
    def __init__(self,models = []):
        self.models = models
        self.best_model = None
        self.predictions = None
        self.mean_mse = {}
        
    def add_model(self,model):
        self.models.append(model)
    
    def cross_validate (self,features,target,k=3):
        features_df = features
        target_col = target
        for model in self.models:
            mse = np.sqrt(-(cross_val_score(model,features_df,target_col,cv=k, scoring = "neg_mean_squared_error")))
            self.mean_mse[model] = np.mean(mse)
            
    def select_best_model (self):
        self.best_model = min(self.mean_mse, key = self.mean_mse.get)
        
    def best_model_fit(self, features, target):
        self.best_model.fit(features,target)
        
    def best_model_predict(self, features):
        self.best_model.predict(features)
        
    def print_summary(self):
        '''prints summary of models, best model, and feature importance'''
        print('\nModel Summaries:\n')
        for model in self.models:
            print( '\n', model, '- MSE', self.mean_mse[model])
        print('\n best model: \n', self.best_model)
        print('\n MSE of best model\n', self.mean_mse[self.best_model])
        
        
def model_training():
    ## Define the number of processors to be used for parallel processing
    num_procs = 4

    ## Set verbose level for models
    verbose_lvl = 0

    ## Define the input files
    train_file = 'train.tsv'

    ## Define variables
    cat_cols = ['item_condition_id', 'shipping']
    text_cols = ['name','category_name','brand_name','item_description']
    target_col = 'price'
    
    train_df = pd.read_csv(train_file, sep='\t')
    train_df = train_df.sample(frac = 0.07).reset_index(drop = True)
    
    data = Data(train_df)
    
    df_engineered = feature_engineering(data.df)
    
    feature_matrix = df_engineered.df_feature
    
    target = data.df.price
    
    #create model container
    models = modelcontainer()

    #create models -- hyperparameter tuning already done by hand for each model
    models.add_model(Ridge(solver = "auto", random_state = 40, fit_intercept = True))
    models.add_model(Lasso(fit_intercept = True, random_state = 40))
    models.add_model(lgb.LGBMRegressor(num_leaves=31, n_jobs=-1, learning_rate=0.1, n_estimators=500, random_state=42))
    
    models.cross_validate(feature_matrix,target, k=2)
    models.select_best_model()
    models.best_model_fit(feature_matrix,target)
    print('model features shape:',models.best_model.n_features_)
    models.print_summary()
    joblib.dump(models.best_model,'objects/model.obj')

def test_frame(df):
    data = Data(df)
    
    cv = joblib.load('objects/X_name')
    X_name = cv.transform(df["name"])
    print('test X_name:',X_name.shape)
    
    cv = joblib.load('objects/X_main')
    X_main = cv.transform(df["category_main"])
    print('test X_main:',X_main.shape)
    
    cv = joblib.load('objects/X_sub1')
    X_sub1 = cv.transform(df["category_sub1"])
    print('test X_sub1:',X_sub1.shape)
    
    cv = joblib.load('objects/X_sub2')
    X_sub2 = cv.transform(df["category_sub2"])
    print('test X_sub2:',X_sub2.shape)
    
    cv = joblib.load('objects/X_brand')
    X_brand = cv.transform(df["brand_name"])
    print('test X_brand:',X_brand.shape)
    
    #dummy creation
    columns = ['item_condition_id_1', 'item_condition_id_2', 'item_condition_id_3', 'item_condition_id_4', 'item_condition_id_5', 'shipping_0', 'shipping_1']
    X_dummy_df = pd.DataFrame(np.zeros([1,7]), columns=columns)
    X_dummy_df['item_condition_id_' + str(df.item_condition_id.iloc[0])].iloc[0] = int(df.item_condition_id.iloc[0])
    X_dummy_df['shipping_' + str(df.shipping.iloc[0])].iloc[0] = int(df.shipping.iloc[0])
    X_dummies = csr_matrix(X_dummy_df.values)
    print('test X_dummies:',X_dummies.shape)
    
    tfidf = joblib.load('objects/X_description')
    X_description = tfidf.transform(df["item_description"])
    print('test X_description:',X_description.shape)
    
    df1 = hstack((X_name,X_main,X_sub1,X_sub2,X_brand,X_dummies,X_description)).tocsr()
    print('test df1:',df1.shape)
    return df1