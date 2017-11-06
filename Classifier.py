import pandas as p # Importing Pandas library for using the data frame
"""
@author:Sanket Agarwal
Python module for the decision tree which will classify the test data
"""
import csv  # Importing the csv library for reading the csv filefrom HW.HW_05C_DecTree_Writer_for_Students.HW_05C_Agarwal_Sanket_Trainer import traning_data_dec_tree_creation  # Importing the HW_05C_Agarwal_Sanket_Trainer file
import pickle # Importing Pickle library to store dictionary object 


def decision_tree(df):
    with open('decision_tree.pickle', 'rb') as handle:
        node = pickle.load(handle)

    column_header = list(df)
    column_header.append('Class')
    data_list = df.values.tolist()

    for row in data_list:
        row.append(predict(node, row))
    with open('HW_05C_Agarwal_Sanket_MyClassifications.csv', 'w') as csvfile:

        fieldnames = column_header
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data_list:
            writer.writerow({column_header[0]: row[0], column_header[1]: row[1], column_header[2]: row[2], column_header[3]: row[3],column_header[4]: row[4],column_header[5]: row[5], column_header[6]: 'whippet' if row[6]==0 else 'Greyhound'})


def predict(node, row):
    if row[node['best_index']] < node['best_value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def main():
    file_name = input('Enter the filename for classifying the data: ')
    df = p.read_csv(file_name)
    decision_tree(df)
if __name__ == "__main__":
    main()
