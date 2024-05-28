#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold

# set the random state for reproducibility 
import numpy as np
np.random.seed(401)
convert = {"SGDF" : 0, "Gauss" : 1, "RFC": 2, "MLP" : 3, "ADA" : 4}#used to reference classifiers in loops
convertBack = ["SGDF", "Gauss", "RFC", "MLP", "ADA"] #likewise^

def SGDFmat(X_train, X_test, y_train, y_test):
    """
    return the confusion matrix for SGDF on given training/testing input
    """
    SGDF = SGDClassifier()
    SGDF.fit(X_train, y_train)
    SGDF_predicted = SGDF.predict(X_test)
    return confusion_matrix(y_test, SGDF_predicted)

def Gaussmat(X_train, X_test, y_train, y_test):
    """
        return the confusion matrix for Gaussian on given training/testing input
        """
    Gauss = GaussianNB()
    Gauss.fit(X_train, y_train)
    Gauss_predicted = Gauss.predict(X_test)
    return confusion_matrix(y_test, Gauss_predicted)

def RFCmat(X_train, X_test, y_train, y_test):
    """
        return the confusion matrix for random forest on given training/testing input
        """
    RFC = RandomForestClassifier()  # rapid
    RFC.fit(X_train, y_train)
    RFC_predicted = RFC.predict(X_test)
    return confusion_matrix(y_test, RFC_predicted)

def MLPmat(X_train, X_test, y_train, y_test):
    """
        return the confusion matrix for MLP on given training/testing input
        """
    MLP = MLPClassifier()
    MLP.fit(X_train, y_train)
    MLP_predicted = MLP.predict(X_test)
    return confusion_matrix(y_test, MLP_predicted)

def ADAmat(X_train, X_test, y_train, y_test):
    """
        return the confusion matrix for ADA on given training/testing input
        """
    ADA = AdaBoostClassifier()
    ADA.fit(X_train, y_train)
    ADA_predict = ADA.predict(X_test)
    return confusion_matrix(y_test, ADA_predict)

def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    size = len(C)
    num, denom = 0.0, 0.0
    for i in range(0, size):
        num += C[i][i]
        for j in range(0, size):
            denom += C[i][j]
    return float(num/denom)

def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    recalls = []
    size = len(C)
    for i in range(0, size):
        num = C[i][i]
        denom = 0
        for j in range(0, size):
            denom += C[i][j]
        recalls.append(float(num/denom))
    return recalls


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    p = []
    size = len(C)
    for i in range(0, size):
        num = C[i][i]
        denom = 0
        for j in range(0, size):
            denom += C[j][i]
        p.append(float(num / denom))
    return p


def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''
    y_train = y_train.ravel()#warnings for array shape?

    #find confusion matrix for each method
    SGDF_confusion = SGDFmat( X_train, X_test, y_train, y_test)
    Gauss_confusion = Gaussmat( X_train, X_test, y_train, y_test)
    RFC_confusion = RFCmat( X_train, X_test, y_train, y_test)
    MLP_confusion = MLPmat( X_train, X_test, y_train, y_test)
    ADA_confusion = ADAmat( X_train, X_test, y_train, y_test)

    #load dict for easier output
    classifiers = {"SGDF" : SGDF_confusion, "Gauss" : Gauss_confusion, "RFC": RFC_confusion, "MLP" : MLP_confusion, "ADA" : ADA_confusion}
    acc_max = [-1, "SGDF"]
    weighted_max = [-1, "SGDF"]
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        # For each classifier, compute results and write the following output:
        for key in classifiers:
            acc = accuracy(classifiers[key])
            outf.write(f'Results for {key}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in recall(classifiers[key])]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in precision(classifiers[key])]}\n')
            outf.write(f'\tConfusion Matrix: \n{classifiers[key]}\n\n')

            #find accuracy max
            if acc > acc_max[0]:
                acc_max[0] = acc
                acc_max[1] = key

            #weighted sum where the three factors are equally weighted
            if acc + (sum([x for x in recall(classifiers[key])])/5.0) + (sum([x for x in precision(classifiers[key])])/5.0) > weighted_max[0]:
                weighted_max[0] = acc + (sum([x for x in recall(classifiers[key])])/5.0) + (sum([x for x in precision(classifiers[key])])/5.0)
                weighted_max[1] = key

    return convert[weighted_max[1]]


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    y_train = y_train.ravel()
    training_values = [1000, 5000, 10000, 15000, 20000]
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        # the following output:
        #     outf.write(f'{num_train}: {accuracy:.4f}\n'))
        for value in training_values:  #slicing from front of list, might cause closer scores.
            x_slice = X_train[0:value]
            y_slice = y_train[0:value]

            con_mat = None
            if iBest == 0:
                con_mat = SGDFmat(x_slice, X_test, y_slice, y_test)
            if iBest == 1:
                con_mat = Gaussmat(x_slice, X_test, y_slice, y_test)
            if iBest == 2:
                con_mat = RFCmat(x_slice, X_test, y_slice, y_test)
            if iBest == 3:
                con_mat = MLPmat(x_slice, X_test, y_slice, y_test)
            if iBest == 4:
                con_mat = ADAmat(x_slice, X_test, y_slice, y_test)

            outf.write(f'{value}: {accuracy(con_mat):.4f}\n')

    X_1k = X_train[0:value]
    y_1k = y_train[0:value]
    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier    (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    #selector stuff
    y_train = y_train.ravel()
    selector5 = SelectKBest(f_classif, 5)
    X5_new = selector5.fit_transform(X_train, y_train)
    pp5 = selector5.pvalues_
    X5_new_test = selector5.fit_transform(X_test, y_test)
    selector5small = SelectKBest(f_classif, 5)
    X5_new_small = selector5small.fit_transform(X_1k, y_1k)
    pp5_small = selector5small.pvalues_

    selector50 = SelectKBest(f_classif, 50)
    pp50 = selector50.pvalues_

    pp5_to_print = []
    pp50_to_print = []

    pp5_indices = []
    pp5_small_indices = []

    #processing probs for printing
    for i in range(0, 5):
        min_prob = 999999
        min_index = -1
        for j in range(len(pp5)):
            if pp5[j] < min_prob:
                min_prob = pp5[j]
                min_index = j
        pp5_indices.append(min_index)
        pp5[min_index] = 99999
        pp5_to_print.append(min_prob)

    for i in range(0, 50):
        min_prob = 999999
        min_index = -1
        for j in range(len(pp50)):
            if pp50[j] < min_prob:
                min_prob = pp50[j]
                min_index = j
        pp50[min_index] = 99999
        pp50_to_print.append(min_prob)

    for i in range(0, 5):
        min_prob = 99999
        min_index = -1
        for j in range(len(pp5_small)):
            if pp5_small[j] < min_prob:
                min_prob = pp5_small[j]
                min_index = j
        pp5_small_indices.append(min_index)
        pp5_small[min_index] = 99999

    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.
        
        # for each number of features k_feat, write the p-values for
        # that number of features:
        k_feat = 5
        p_values = pp5_to_print
        outf.write(f'{k_feat} p-values: {[format(pval) for pval in p_values]}\n')
        k_feat = 50
        p_values = pp50_to_print
        outf.write(f'{k_feat} p-values: {[format(pval) for pval in p_values]}\n')

        con_mat32 = None
        con_mat1 = None
        if i == 0:
            con_mat32 = SGDFmat(X5_new, X5_new_test, y_train, y_test)
            con_mat1 = SGDFmat(X5_new_small, X5_new_test, y_1k, y_test)
        if i == 1:
            con_mat32 = Gaussmat(X5_new, X5_new_test, y_train, y_test)
            con_mat1 = Gaussmat(X5_new_small, X5_new_test, y_1k, y_test)
        if i == 2:
            con_mat32 = RFCmat(X5_new, X5_new_test, y_train, y_test)
            con_mat1 = RFCmat(X5_new_small, X5_new_test, y_1k, y_test)
        if i == 3:
            con_mat32 = MLPmat(X5_new, X5_new_test, y_train, y_test)
            con_mat1 = MLPmat(X5_new_small, X5_new_test, y_1k, y_test)
        if i == 4:
            con_mat32 = ADAmat(X5_new, X5_new_test, y_train, y_test)
            con_mat1 = ADAmat(X5_new_small, X5_new_test, y_1k, y_test)

        accuracy_1k = accuracy(con_mat1)
        accuracy_full = accuracy(con_mat32)

        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        feature_intersection = [x for x in pp5_small_indices if x in pp5_indices]
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        top_5 = pp5_indices
        outf.write(f'Top-5 at higher: {top_5}\n')
        pass


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''

    Xall = np.concatenate((X_train, X_test), axis=0)
    Yall = np.concatenate((y_train, y_test), axis=0)


    fold = KFold(n_splits=5, shuffle=True)

    classifier_total_acc = {"SGDF": [], "Gauss": [], "RFC": [], "MLP": [], "ADA": []}


    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        iteration = 1
        for train_index, test_index in fold.split(Xall):
            # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
            # for each fold:
            kfold_accuracies = []
            for classfier in convertBack:
                xT = Xall[train_index]
                xTst = Xall[test_index]
                yT = Yall[train_index]
                yTst = Yall[test_index]
                conf_matrix = eval(classfier+'mat(xT, xTst, yT, yTst)')
                acc = accuracy(conf_matrix)
                classifier_total_acc[classfier].append([acc])
                kfold_accuracies.append(acc)
            outf.write(f'fold number #{iteration}\n')
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
            iteration += 1

        v1 = np.array(classifier_total_acc["SGDF"])
        v2 = classifier_total_acc["Gauss"]
        v3 = classifier_total_acc["RFC"]     
        v4 = classifier_total_acc["MLP"]
        v5 = np.array(classifier_total_acc["ADA"])

        p_values = []
        p_values.append(ttest_rel(v1, v5)[1][0])
        p_values.append(ttest_rel(v2, v5)[1][0])
        p_values.append(ttest_rel(v3, v5)[1][0])
        p_values.append(ttest_rel(v4, v5)[1][0])
        outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    
    # TODO: load data and split into train and test.
    # TODO : complete each classification experiment, in sequence.
    raw = np.load(args.input)['arr_0']
    Xall = raw[0:, 0:173]
    Yall = raw[0:, 173:174]

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xall, Yall, test_size= .2)


    #a,b = class32(args.output_dir, Xtrain, Xtest, Ytrain, Ytest, 4)

    #class33(args.output_dir, Xtrain, Xtest, Ytrain, Ytest, 4, a ,b)

    class34(args.output_dir, Xtrain, Xtest, Ytrain, Ytest, 4)