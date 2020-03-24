# -*- coding: utf-8 -*-
"""
#--======================================= Code Header ==============================================================
#--Code Description: Python script to trend IRS dataset
#--Code Creation Date: 2020/03/17
#--Author: Pavithran R
#--Version : 3.0
#--Changes in 2.0: 1. Added function to read and manipulate 2011-2016 files
#                  2. Added loop for trending attributes
#--Changes in 3.0: Added steps to remove duplicates (zipcode = 0)
#-- =================================  End of Code Header ===========================================================
"""
import numpy as np
import pandas as pd
import os

input_path = 'C:\Users\Pavithran.R\Documents\GP\Wrangling tasks\IRS assignment\Input files'
output_path = 'C:\Users\Pavithran.R\Documents\GP\Wrangling tasks\IRS assignment\Output files'

####################### defining required columns based on kyle's file ##################

req_cols = ['STATE', 'zipcode', 'agi_stub','N1','MARS1', 'MARS2', 'MARS4', 'NUMDEP', 'A00100', 'N02650', 'A02650',
            'N00200', 'A00200', 'N00900', 'A00900', 'N02300', 'A02300', 'N18425', 'A18425', 'N18500', 'A18500',
            'N09600', 'A09600', 'N09400', 'A09400', 'N06500', 'A06500']

#converting them into uppercase so that it matches with the IRS colnames               
req_cols = map(str.upper, req_cols)

####################### reading 2017 input file and filtering for required columns ####################### 

irs_2017 = pd.read_csv(input_path + '\\' +'17zpallagi.csv')
irs_2017.columns = irs_2017.columns.str.upper()
irs_2017_req = irs_2017.filter(req_cols)

# renaming 'N1' of 2017 to 'n1' to avoid conflicts while filtering for columns
irs_2017_req = irs_2017_req.rename(columns = {'N1': 'n1'})

####################### reading 2011-2016 input file and filtering for required columns ##################

# loop to read, filter for required columns, add suffix of the year and join to form master dataset
file_names = os.listdir(input_path)
file_names.pop()

# reversing the sequence of file names in-order to maintain descending order while merging with 2017 data
file_names.sort(reverse = True) 

# initializing base data on which the loop performs joins
irs_11_16 = irs_2017_req.iloc[:, [0,1,2]].drop_duplicates(keep = 'first', inplace = False)

for x in range(0,len(file_names)):
    
    irs_intermediate = pd.read_csv(input_path + '\\' + file_names[x])  
    
    irs_intermediate.columns = irs_intermediate.columns.str.upper()    
    irs_intermediate = irs_intermediate.filter(req_cols)
    
    # renaming 'N1' of 2017 to 'n1' to avoid conflicts while filtering for columns
    irs_intermediate = irs_intermediate.rename(columns = {'N1': 'n1'})
    
    # adding year suffix
    irs_intermediate = irs_intermediate.add_suffix('_' + file_names[x][:2])
    
    # renaming first 3 level columns to maintain uniformity
    irs_intermediate.rename(columns = {irs_intermediate.columns[0]: irs_2017_req.columns[0], irs_intermediate.columns[1]: irs_2017_req.columns[1], irs_intermediate.columns[2]: irs_2017_req.columns[2]}, inplace = True)
    
    irs_11_16 = irs_11_16.merge(irs_intermediate, on = ['STATE', 'ZIPCODE','AGI_STUB'], how = 'left')

# joining 2017's data to 2011-2016 joined dataset
irs_master_ds = irs_2017_req.merge(irs_11_16, on = ['STATE', 'ZIPCODE', 'AGI_STUB'], how = 'left')

# removing zipcode = 0 as it is invalid entry and is repeating across states
irs_master_ds = irs_master_ds[irs_master_ds['ZIPCODE'] != 0]
irs_master_ds.shape[0] == irs_2017_req.shape[0] #FALSE

# removing duplicate rows (3 rows)
irs_master_ds = irs_master_ds.drop_duplicates(keep = 'first', inplace = False)

# creating duplicate df to have just the attributes
irs_master_ds_attributes = irs_master_ds

######################### calculating percentages of each metric column for 2017 ########################

# loop to calculate percentage of total for all the variables of 2017
for x in range(3, len(irs_master_ds.columns)):
    
    tot = irs_master_ds.groupby(['STATE', 'ZIPCODE'])[irs_master_ds.columns[x]].sum().reset_index()
    
    col_name = irs_master_ds.columns[x] + '_tot'
    tot.rename(columns = {irs_master_ds.columns[x]: col_name}, inplace = True)
    
    irs_master_ds = irs_master_ds.merge(tot, on = ['STATE', 'ZIPCODE'])
    col_name_2 = irs_master_ds.columns[x] + '_perc'
    
    irs_master_ds[col_name_2] = irs_master_ds[irs_master_ds.columns[x]]/irs_master_ds[col_name] * 100

# Selecting 3 levels and percentage columns (removing the absolute and total columns)
base_levels = irs_master_ds.iloc[:, [0,1,2]].drop_duplicates(keep = 'first', inplace = False)
irs_master_ds_perc = irs_master_ds.filter(regex = '_perc')
irs_master_ds_perc = pd.concat([base_levels, irs_master_ds_perc], axis=1)

# converting NAN to zero in both base files and resetting index
irs_master_ds_attributes.fillna(0, inplace=True)
irs_master_ds_perc.fillna(0, inplace=True)

irs_master_ds_attributes.reset_index(drop = True, inplace = True)
irs_master_ds_perc.reset_index(drop = True, inplace = True)

######################### trending based on previous years' data #######################################

# Defining a function to calculate and iterate the YOY trend
#df = irs_master_ds_attributes
def trend_generator(df):
    # loop runs for # cols of 2017 data
    for x in range(3,len(irs_2017_req.columns)):
        
        # fixing the first column so that it stays as the 3rd variable after deleting 3rd variable of previous iteration
        first_var = df.columns[3]
        iteration_ds = df.filter(regex = first_var.split('_')[0])
        iteration_ds.reset_index(drop = True, inplace = True)
        
        iteration_ds['non_zeros'] = iteration_ds.gt(0).sum(axis=1)
        iteration_ds['denom'] = np.where(iteration_ds['non_zeros'] == 0, 0, iteration_ds.non_zeros - 1)
        
        variables_list = list(iteration_ds.columns)
        df = df.drop([first_var], axis = 1)
        
        yoy_change_sum = 0

        for y in range(0, len(iteration_ds.columns)-3):
            
            yoy_change = np.where(np.logical_or(iteration_ds[variables_list[y]] == 0, iteration_ds[variables_list[y+1]] == 0), 0, 
                                  np.where(iteration_ds.shape[1] == 1, 0, 
                                           (iteration_ds[variables_list[y]] - iteration_ds[variables_list[y+1]])/iteration_ds[variables_list[y+1]]))
            
            yoy_change_sum = yoy_change_sum + yoy_change
            
        yoy_change_sum = pd.DataFrame(yoy_change_sum)#.reset_index(drop = TRUE, inplace = TRUE)
        yoy_change_sum.rename(columns = {0: first_var + '_sum'}, inplace = True)
        
        iteration_ds = pd.concat([iteration_ds, yoy_change_sum], axis=1)
        
        iteration_ds[first_var + '_trend'] = np.where(iteration_ds['denom'] == 0, 0, iteration_ds[first_var + '_sum']/iteration_ds['denom'])
        
        df = pd.concat([df, iteration_ds[first_var + '_trend']], axis=1)
    
    df_trend =  pd.concat([base_levels, df.filter(regex = '_trend')], axis=1)
    return df_trend
        
# function call
attributes_yoy_trends = trend_generator(irs_master_ds_attributes)
perc_yoy_trends = trend_generator(irs_master_ds_perc)

# filtering for only 2017 columns
irs_master_ds_attributes_req = irs_master_ds_attributes.iloc[:, 0:27]
irs_master_ds_perc_req = irs_master_ds_perc.iloc[:, 0:27]

# joining all 4 data frames (master datasets and trends for both attributes and percentages)
pivot_ads = pd.merge(irs_master_ds_attributes_req, attributes_yoy_trends, on = ['STATE', 'ZIPCODE', 'AGI_STUB'], how = 'left')
pivot_ads = pd.merge(pivot_ads, irs_master_ds_perc_req, on = ['STATE', 'ZIPCODE', 'AGI_STUB'], how = 'left')
pivot_ads = pd.merge(pivot_ads, perc_yoy_trends, on = ['STATE', 'ZIPCODE', 'AGI_STUB'], how = 'left')

######################### pivoting and reiterating for all the variables ########################

# Approach:
# 1. creating a data set with first 4 columns
# 2. delete 4 column
# 3. pivot and prefix column name
# 4. reiterate the same with new set of 4 columns
# 5. loop to join all the intermediate tables

# global variable - taking all the zipcodes to make a base pivot table
irs_pivot_base = pivot_ads.iloc[:, [1]].drop_duplicates(keep = 'first', inplace = False)

# loop starts
for x in range(3,len(pivot_ads.columns)):
    iteration_file = pivot_ads.iloc[:, [0,1,2,3]]
    
    fourth_col = pivot_ads.columns[3]
    pivot_ads = pivot_ads.drop([fourth_col], axis = 1)
    
    irs_pivot = pd.pivot_table(iteration_file, values= fourth_col, index = "ZIPCODE", columns = "AGI_STUB").reset_index()
    pivot_col = iteration_file.columns[3]
    irs_pivot = irs_pivot.add_prefix(pivot_col + '_')

    first_col = irs_pivot.columns[0]
    irs_pivot = irs_pivot.rename(columns = {first_col: 'ZIPCODE'})

    irs_pivot_base = irs_pivot_base.merge(irs_pivot, on = 'ZIPCODE', how = 'left')

irs_pivot_base.to_csv(output_path + '\\' + 'IRS_pivoted.csv', index = False)

#split = np.array_split(irs_pivot_base, 4)
#number = ['1','2','3','4']
#for x in range(0, len(split)):
#    split[x].to_csv(output_path + '\\' + 'split_' + number[x] +'_IRS_pivoted.csv', index = False)
