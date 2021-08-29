import pickle

import pandas as pd
import numpy as np



#Transformation function
def Transformations(featureDF, targetDF):
    # Load the PreProcessing Objects needed for Transformations
    #Import training features
    import pathlib
    path_to_read_model = 'C:\\Users\\koriv\\Desktop\\MachineLearning_DataScience\\Hands_On_Machine_Learning\\my_env_codebasics\\My_CODE\\ML-projects\\advanced_House_Price_Prediction-Regression\\preProcessingObjects_list' #Path of current working Directory
    with open(path_to_read_model + '\\preProcessingObjects_list.pkl', 'rb') as f:
        preProcessingObjects_list = pickle.load(f)
    location_stats_greater_than_10 = preProcessingObjects_list[0]
    
    
    # Feature Selection via Business knoledge. Society is dependednt on location
    featureDF.drop(['society'], axis=1, inplace=True)
    
    # Duplicate elimination
    featureDF.drop_duplicates(inplace=True)
    # Update y matrix based X
    ## since we've removed some data from X, we need to pass on these updations to y as well, as y doesn't know some of its corresponding X's have been deleted.
    targetDF = targetDF[featureDF.index]
    
    # convert cat-col of total_sqft to float with null values also
    def convert_sqft_to_num(curr_tuple):
        try:
            tokens = curr_tuple.split('-')
            if len(tokens) == 2:
                return (float(tokens[0])+float(tokens[1]))/2
            return float(curr_tuple)
        except:
            return np.NaN
    
    featureDF['total_sqft'] = featureDF.total_sqft.apply(convert_sqft_to_num)
    
    # convert cat-col of availability to make both Immediate Possession and Ready To Move same and take only month, include null also.
    def convert_availability(curr_tuple):
        try:
            curr_tuple = curr_tuple.lower()
            if curr_tuple == 'ready to move' or curr_tuple == 'immediate possession':
                return 'available_currently'
            tokens = curr_tuple.split('-')
            if len(tokens) == 2:
                return tokens[1].strip()
        except:
            return np.NaN
        
    featureDF['availability'] = featureDF.availability.apply(convert_availability)
    
    # convert cat-col of size to number by the frist value with null value also
    def convert_size_to_num(curr_tuple):
        try:
            tokens = curr_tuple.split(' ')
            return float(tokens[0])
        except:
            return np.NaN
        
    featureDF['size'] = featureDF['size'].apply(convert_size_to_num)
    
    # Convert location to reduce the unique values in the column
    featureDF['location'] = featureDF['location'].apply(lambda x: x if (x in location_stats_greater_than_10) else 'other')
    
    # Update y matrix based X
    ## since we've removed some data from X, we need to pass on these updations to y as well, as y doesn't know some of its corresponding X's have been deleted.
    targetDF = targetDF[featureDF.index]
    
    return featureDF, targetDF
	

# Outlier Removal function
def outlierRemoval(featureDF, targetDF):
    import json
    outlier_Dict_DirPath = 'C:\\Users\\koriv\\Desktop\\MachineLearning_DataScience\\Hands_On_Machine_Learning\\my_env_codebasics\\My_CODE\\ML-projects\\advanced_House_Price_Prediction-Regression\\preProcessingObjects_list'
    with open(outlier_Dict_DirPath+'\\outlier_Dict.json') as json_file:
        outlier_Dict = json.load(json_file)
    
    for column in outlier_Dict:
        lower_limit =outlier_Dict[column]['lower_limit']
        upper_limit =outlier_Dict[column]['upper_limit']
        print("column: %s, lower_limit: %s, upper_limit: %s"%(column, lower_limit, upper_limit))
        featureDF = featureDF[((featureDF[column]>lower_limit)&(featureDF[column]<upper_limit))| (featureDF[column].isna())]
    
    # Update y matrix based X
    ## since we've removed some data from X, we need to pass on these updations to y as well, as y doesn't know some of its corresponding X's have been deleted.
    targetDF = targetDF[featureDF.index]
    
    return featureDF, targetDF  
	
	
