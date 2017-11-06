"""
@author:Sanket Agarwal
Python code for training the model and making the decision trees
"""
import numpy as np  # Importing numpy library for numerical calculations
import pandas as p  # Importing Pandas library for using the data frame
import pickle  # Importing Pickle library to store dictionary object


def traning_data_dec_tree_creation(df):
    """
    Function to call other important functions to build the tree, print the tree, and checking accuracy
    :param df: The dataframe to store the data
    :return: The decision tree node
    """

    # converting the data frame to list
    data_list = df.values.tolist()

    for row in data_list:
        if row[-1] == 'Whippet':  # whippet : 0 Greyhound = 1
            row[-1] = 0
        else:
            row[-1] = 1

    # calling the build_tree() method to build the tree
    tree = build_tree(data_list)

    # Printing the tree
    print_tree(tree)

    # calling the function to make a program which will be a classifier
    decision_tree_program_creation()

    # Making a pickle object of the decision tree object to save it
    with open('decision_tree.pickle', 'wb') as handle:
        pickle.dump(tree, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #  Checking Accuracy of the decision tree
    df_frac = df.sample(frac=.7) # Taking 70% of random  data for validation purpose

    data_chunk = df_frac.values.tolist()
    for row in data_chunk:
        if row[-1] == 'Whippet':  # whippet : 0 Greyhound = 1
            row[-1] = 0
        else:
            row[-1] = 1
    for row in data_chunk:
        row.append(predict_for_validating_accuracy(tree, row))
    acc_count = 0
    for row in data_chunk:
        if row[-1] == row[-2]:
            acc_count += 1
    accuracy = float(acc_count/len(data_chunk))
    print("Accuracy for the decision tree is: "+ str(accuracy*100))

    return tree


def predict_for_validating_accuracy(node, row):
    """
    This function is helping in validating the decision tree model
    :param node: The decision tree node
    :param row: Row of data
    :return: The classification 0 or 1
    """
    if row[node['best_index']] < node['best_value']:
        if isinstance(node['left'], dict):
            return predict_for_validating_accuracy(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict_for_validating_accuracy(node['right'], row)
        else:
            return node['right']


def calculate_gini(groups, classes):
    """
    This function calculates the weighted gini index
    :param groups: Groups of left and right sub tree
    :param classes: list of classes --> [0,1]
    :return: Weighted Gini Index
    """
    gini = 0

    total_no_instances = float(sum([len(row) for row in groups]))
    for group in groups:
        size = float(len(group))
        if size == 0.0:
            continue
        part_gini = 0.0

        for cls in classes:
            l = [row[-1] for row in group]
            part = float(l.count(cls)/size)
            part_gini += part * part

        part_gini = 1- part_gini

        gini += (size/total_no_instances) * part_gini

    return gini


def split(index, data, thres_val):
    """
    Splitting the data based on the threshold value in left and right
    :param index: Index of attribute
    :param data: The Tree Data
    :param thres_val: Threshold value
    :return: left and right subtree
    """
    left, right = [], []

    for row in data:
        if row[index] <= thres_val:
            left.append(row)
        else:
            right.append(row)

    return left, right


def get_split(data):
    """
    This function gets the split by performing the necessary steps
    :param data: The data
    :return: The decision node
    """

    # List of the class --> [0,1]
    classes = list(set([row[-1] for row in data]))

    # Initializing the variables
    best_gini, best_value, best_index, best_groups = 999, 999, 999 , None

    for index in range(len(data[0])-1):

        for row in data:
            groups = split(index, data, row[index])
            gini = calculate_gini(groups, classes)

            if gini < best_gini:
                best_gini, best_value, best_index, best_groups = gini, row[index] , index, groups

    return {'best_gini':best_gini,'best_value': best_value, 'best_index':best_index, 'best_groups': best_groups}


def terminal_output(group):
    """
    This function calculates the majority class
    :param group: Group of left and right sub tree
    :return: Majority class
    """

    cls = [row[-1] for row in group]
    return max(cls, key=cls.count)


def root_split(node, max_depth, depth):
    """
    This function makes the actual decision trees by calling the required methods created
    :param node: The root of the decisoin tree
    :return: None
    """

    left_tree, right_tree = node['best_groups']
    del(node['best_groups'])

    # Pruning by taking a maximum depth
    if depth >= max_depth:
        node['left'] = terminal_output(left_tree) if len(left_tree) > 0 else []
        node['right'] = terminal_output(right_tree) if len(right_tree) > 0 else []
        return

    if len(left_tree) == 0 or len(right_tree) == 0:
        node['left'] = node['right'] = terminal_output(left_tree + right_tree)
        return
    else:
        node['left'] = get_split(left_tree)
        root_split(node['left'], max_depth, depth+1)

        node['right'] = get_split(right_tree)
        root_split(node['right'],max_depth, depth+1)


def build_tree(data):
    """
    This function the best root
    :param data: The entire data
    :return: Root of the decision tree
    """
    max_depth = 6
    depth = 1
    root = get_split(data)
    root_split(root, max_depth, depth)
    return root


def decision_tree_program_creation():
    """
    This function will write another python program for the classification task
    :return: None
    """
    # Prologue
    f = open('HW_05C_Agarwal_Sanket_Classifier.py', 'w')
    f.write("import pandas as p # Importing Pandas library for using the data frame\n")
    f.write('"""\n')
    f.write("@author:Sanket Agarwal\n")
    f.write("Python module for the decision tree which will classify the test data\n")
    f.write('"""\n')
    f.write("import csv  # Importing the csv library for reading the csv file")
    f.write("from HW.HW_05C_DecTree_Writer_for_Students.HW_05C_Agarwal_Sanket_Trainer import traning_data_dec_tree_creation  # Importing the HW_05C_Agarwal_Sanket_Trainer file\n")
    f.write("import pickle # Importing Pickle library to store dictionary object \n\n\n")

    #  main body
    f.write('def decision_tree(df):\n')
    f.write("    with open('decision_tree.pickle', 'rb') as handle:\n")
    f.write("        node = pickle.load(handle)\n\n")
    f.write("    column_header = list(df)\n")
    f.write("    column_header.append('Class')\n")
    f.write('    data_list = df.values.tolist()\n\n')

    f.write("    for row in data_list:\n")
    f.write("        row.append(predict(node, row))\n")
    f.write("    with open('HW_05C_Agarwal_Sanket_MyClassifications.csv', 'w') as csvfile:\n\n")
    f.write("        fieldnames = column_header\n")
    f.write("        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)\n")
    f.write("        writer.writeheader()\n")
    f.write("        for row in data_list:\n")
    f.write("            writer.writerow({column_header[0]: row[0], column_header[1]: row[1], column_header[2]: row[2]"
            ", column_header[3]: row[3],column_header[4]: row[4],column_header[5]: row[5], column_header[6]: 'whippet' if row[6]==0 else 'Greyhound'})")
    f.write("\n\n\n")

    f.write("def predict(node, row):\n")
    f.write("    if row[node['best_index']] < node['best_value']:\n")
    f.write("        if isinstance(node['left'], dict):\n")
    f.write("            return predict(node['left'], row)\n ")
    f.write("       else:\n")
    f.write("            return node['left']\n")
    f.write("    else:\n")
    f.write("        if isinstance(node['right'], dict):\n")
    f.write("            return predict(node['right'], row)\n ")
    f.write("       else:\n")
    f.write("            return node['right']\n\n")

    # main function creation
    f.write('def main():\n')
    f.write("    file_name = input('Enter the filename for classifying the data: ')\n")
    f.write("    df = p.read_csv(file_name)\n")
    f.write("    decision_tree(df)\n")
    f.write('if __name__ == "__main__":\n')
    f.write("    main()\n")


def print_tree(node, depth=0):
    """
    This function prints the decision tree
    :param node: The decision tree node
    :param depth: The depth of the tree
    :return: None
    """
    if isinstance(node, dict):
        print('%s[Attr%d < %.3f]' % ((depth * ' ', (node['best_index']), node['best_value'])))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * ' ', node)))


def main():
    """
    The main function
    :return: None
    """

    file_name = input("Enter the filename for the training data: ")
    df = p.read_csv(file_name)

    # Calling the method which would call all other methods
    traning_data_dec_tree_creation(df)

if __name__ == "__main__":
    main()