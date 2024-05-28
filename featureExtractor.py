#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import numpy as np
import argparse
import json
import os
import csv
import html
import statistics
import re

leftID = {}
rightID = {}
centerID = {}
altID = {}
leftData = np.load('Left_feats.dat.npy')
rightData = np.load('Right_feats.dat.npy')
centerData = np.load('Center_feats.dat.npy')
altData = np.load('Alt_feats.dat.npy')

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}



bristolDict = {}
warrinerDict = {}
leftID = []
rightID = []
centerID = []
altID = []
leftData = np.load('Left_feats.dat.npy')
rightData = np.load('Right_feats.dat.npy')
centerData = np.load('Center_feats.dat.npy')
altData = np.load('Alt_feats.dat.npy')
def build_da_books():
    """
    Loads all relevant csv and text files, assumes that all files are in local directory
    to function properly!!!!
    """
    cdir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    warriner = open(os.path.join(cdir, 'Ratings_Warriner_et_al.csv'))

    reader = csv.reader(warriner)

    first = True
    for row in reader:
        if first:
            first = False
        else:
            warrinerDict[row[1]] = (float(row[2]), float(row[5]), float(row[8]))
    warriner.close()

    bristol = open(os.path.join(cdir, 'BristolNorms+GilhoolyLogie.csv'))
    reader = csv.reader(bristol)
    first = True
    for row in reader:
        if first:
            first = False
        else:
            bristolDict[row[1]] = row[2:7]
    bristol.close()

    a = open(os.path.join(cdir, 'Alt_IDs.txt'))
    reader = csv.reader(a)
    for row in reader:
        altID.append(row[0])
    a.close()

    a = open(os.path.join(cdir, 'Left_IDs.txt'))
    reader = csv.reader(a)
    for row in reader:
        leftID.append(row[0])
    a.close()

    a = open(os.path.join(cdir, 'Right_IDs.txt'))
    reader = csv.reader(a)
    for row in reader:
        rightID.append(row[0])
    a.close()
    a = open(os.path.join(cdir, 'Center_IDs.txt'))
    reader = csv.reader(a)
    for row in reader:
        centerID.append(row[0])
    a.close()

def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    # TODO: Extract features that rely on capitalization.
    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    # TODO: Extract features that do not rely on capitalization.
    arr = np.zeros(174)
    arr[0] = len(re.findall(r"[A-Z]{3,}\/", comment))
    #bring all words to lower case and create a list of tokens for easier mathching
    re.sub(r" \S+\/", lambda x: str(x.group()).lower(),comment)
    wordlist = re.findall(r"[A-z]+\/", comment)
    newlst = []
    for word in wordlist:
        new = word[:-1].lower()
        newlst.append(new)
    wordlist = newlst
    #list of non punctuation tokens stored in wordlist

    AoATotal = 0
    IMGTotal = 0
    FAMTotal = 0
    VTotal = 0
    ATotal = 0
    DTotal = 0

    AoAlst = []
    IMGlst = []
    FAMlst = []
    Vlst = []
    Alst = []
    Dlst = []
    bristolCount = 0
    warrinerCount = 0
    token_total_len = 0
    for word in wordlist:
        if word in FIRST_PERSON_PRONOUNS:
            arr[1] += 1
        if word in SECOND_PERSON_PRONOUNS:
            arr[2] += 1
        if word in THIRD_PERSON_PRONOUNS:
            arr[3] += 1
        if word[0:2] == "wh":
            arr[12] += 1
        if word in SLANG:
            arr[13] += 1
        if re.match(r"[A-z]", word):
            token_total_len += len(word)

        if word in bristolDict: #adding found word's score to features
            bristolCount += 1
            AoATotal += float(bristolDict[word][1])
            AoAlst.append(float(bristolDict[word][1]))
            IMGTotal += float(bristolDict[word][2])
            IMGlst.append(float(bristolDict[word][2]))
            FAMTotal += float(bristolDict[word][3])
            FAMlst.append(float(bristolDict[word][3]))

        if word in warrinerDict:
            warrinerCount += 1
            VTotal += warrinerDict[word][0]
            Vlst.append(warrinerDict[word][0])
            ATotal += warrinerDict[word][1]
            Alst.append(warrinerDict[word][1])
            DTotal += warrinerDict[word][2]
            Dlst.append(warrinerDict[word][2])

    adjust = 0
    if comment[-3:] == ".\n":
        adjust = 1
    wordlen = float(len(re.findall(r"\S+\/", comment)))
    if len(wordlist) == 0:
        return arr
    arr[4] = len(re.findall(r"\/CC ", comment))
    arr[5] = len(re.findall(r"\/VBD ", comment))
    arr[6] = len(re.findall(r"\/MD \S+\/VB", comment))# len(re.findall(r"\/TO \S+\/VB", comment)) + # + len(re.findall(r"\/MD \S+\/RB \S+\/VB", comment))
    arr[7] = len(re.findall(r"\/\,", comment))
    arr[8] = len(re.findall(r"\/NFP ", comment))
    arr[9] = len(re.findall(r"\/NN ", comment)) + len(re.findall(r"\/NNS ", comment))
    arr[10] = len(re.findall(r"\/NNP ", comment)) + len(re.findall(r"\/NNPS ", comment))
    arr[11] = len(re.findall(r"\/RB ", comment)) + len(re.findall(r"\/RBR ", comment)) + len(re.findall(r"\/RBS ", comment))
    arr[14] = (float(len(re.findall(r"\/", comment)))) / max(float(adjust + len(re.findall(r"\/\.", comment)) + len(re.findall(r"\/NFP", comment)) + len(re.findall(r"\/\:", comment))), 1)
    arr[15] = float(token_total_len)/float(len(wordlist))
    arr[16] = len(re.findall(r"\/\.", comment)) + len(re.findall(r"\/NFP", comment))
    if(bristolCount > 0):
        arr[17] = AoATotal / bristolCount
        arr[18] = IMGTotal / bristolCount
        arr[19] = FAMTotal / bristolCount
    if len(AoAlst) > 1:
        arr[20] = statistics.stdev(AoAlst)#left 0 if no words are found, likewise below
    if len(IMGlst) > 1:
        arr[21] = statistics.stdev(IMGlst)
    if len(FAMlst) > 1:
        arr[22] = statistics.stdev(FAMlst)
    if (warrinerCount > 0):
        arr[23] = VTotal / warrinerCount
        arr[24] = ATotal / warrinerCount
        arr[25] = DTotal / warrinerCount
    if len(Vlst) > 1:
        arr[26] = statistics.stdev(Vlst)
    if len(Alst) > 1:
        arr[27] = statistics.stdev(Alst)
    if len(Dlst) > 1:
        arr[28] = statistics.stdev(Dlst)

    return arr


def extract2(feat, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feat: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feat : numpy Array, a 173-length vector of floating point features (this
        function adds feature 30-173). This should be a modified version of
        the parameter feats.
    '''
    i = -1
    arr = np.zeros(174)
    if comment_class == 'Left':
        i = leftID.index(comment_id)
        arr[29:173] = leftData[i]
    if comment_class == 'Right':
        i = rightID.index(comment_id)
        arr[29:173] = rightData[i]
    if comment_class == 'Center':
        i = centerID.index(comment_id)
        arr[29:173] = centerData[i]
    if comment_class == 'Alt':
        i = altID.index(comment_id)
        arr[29:173] = altData[i]

    for i in range(29,173):
        feat[i] = arr[i]
    return feat


def main(args):
    #Declare necessary global variables here. 

    #Load data
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 174))

    # TODO: Call extract1 for each datatpoint to find the first 29 features. 
    # Add these to feats.
    # TODO: Call extract2 for each feature vector to copy LIWC features (features 30-173)
    # into feats. (Note that these rely on each data point's class,
    # which is why we can't add them in extract1).

    #dict for converting category to number for feature 174
    conv = {
        "Left": 0,
        "Center": 1,
        "Right" : 2,
        "Alt"   : 3
    }

    build_da_books()
    counter = 0
    for thing in data:
        body = thing['body']
        cat = thing['cat']
        id = thing['id']

        arr = extract1(body)
        arr = extract2(arr, cat, id)
        arr[173] = conv[cat] #fill in the 'solution' to our feature vector
        feats[counter] = arr
        counter += 1
    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)

