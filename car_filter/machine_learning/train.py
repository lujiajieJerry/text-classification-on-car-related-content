from nltk.metrics import BigramAssocMeasures
import itertools
from nltk.probability import FreqDist, ConditionalFreqDist
from sklearn import svm, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import codecs
import random
import re

train_filename = 'data/train_cut.txt'
predict_filename = 'data/test_cut.txt'

def load_data(file):
    reader = open(file, 'r', encoding='utf-8')
    posWords = []
    negWords = []

    for line in reader.readlines():
        lable, sentence = line.strip().split('---')
        words = sentence.split(' ')
        for word in words:
            if lable == '1':
                posWords.append(word)
            if lable == '0':
                negWords.append(word)
            # if lable == '1':
            #     neuWords.append(word)
    return posWords, negWords

def create_word_scores(posWords, negWords):
    posWords_set = set(posWords)
    negWords_set = set(negWords)
    # neuWords_set = set(neuWords)

    # posWords = list(itertools.chain(*posWords))
    # negWords = list(itertools.chain(*negWords))
    # neuWords = list(itertools.chain(*neuWords))

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in posWords:
        word_fd[word] += 1
        cond_word_fd['pos'][word] += 1
    for word in negWords:
        word_fd[word] += 1
        cond_word_fd['neg'][word] += 1

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    # neu_word_count = cond_word_fd['neu'].N()

    total_word_count = pos_word_count + neg_word_count

    word_neg_scores = {}
    word_pos_scores = {}

    for word in posWords_set:
        freq = word_fd[word]
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        word_pos_scores[word] = pos_score

    for word in negWords_set:
        freq = word_fd[word]
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_neg_scores[word] = neg_score



    return [word_pos_scores,word_neg_scores]

def find_best_words(posWords, negWords, numbers):
    word_scores = create_word_scores(posWords, negWords)

    best_words_dict = {}
    for number in numbers:
        best_pos_vals = sorted(word_scores[0].items(), key=lambda x: (-x[1], x[0]))[:number[0]]
        # best_nue_vals = sorted(word_scores[2].items(), key=lambda x: (-x[1], x[0]))[:number[1]]
        best_neg_vals = sorted(word_scores[1].items(), key=lambda x: (-x[1], x[0]))[:number[1]]

        best_words = set([w for w, z in best_neg_vals])
        best_words = best_words.union(set([w for w, z in best_pos_vals]))
        # best_words = best_words.union(set([w for w, z in best_nue_vals]))
        key = "_".join([str(int(n)) for n in number])
        best_words_dict[key] = best_words
    return best_words_dict



def classify_svm_train(tfvectorizer, train_samples, train_labels):
    print("train the support vector machine classifier")
    C = 1.0

    X_train = tfvectorizer.transform(train_samples)
    y_train = train_labels

    svm_classifier = svm.SVC(kernel='linear', gamma=0.7, C=C, decision_function_shape="ovo")
    svm_classifier.fit(X_train, y_train)
    return svm_classifier


def classify_nb_train(tfvectorizer, train_samples, train_labels):
    print("train the naive-bayes classifier")
    nb_classifier = BernoulliNB()
    X_train = tfvectorizer.transform(train_samples)
    # print(X_train)
    Y_train = train_labels
    nb_classifier.fit(X_train, Y_train)
    return nb_classifier

def classify_GBDT_train(tfvectorizer, train_samples, train_labels):
    print("train the GBDT classifier")

    X_train = tfvectorizer.transform(train_samples)
    # print(X_train)
    Y_train = train_labels
    gbdt = GradientBoostingClassifier(random_state=10)
    gbdt.fit(X_train, Y_train)
    return gbdt

def classify_LR_train(tfvectorizer, train_samples, train_labels):
    print("train the LR classifier")

    X_train = tfvectorizer.transform(train_samples)
    # print(X_train)
    Y_train = train_labels
    lr_classfier = LogisticRegression()
    lr_classfier.fit(X_train, Y_train)
    return lr_classfier

def classify_KNN_train(tfvectorizer, train_samples, train_labels):
    print("train the KNN classifier")

    X_train = tfvectorizer.transform(train_samples)
    # print(X_train)
    Y_train = train_labels
    knn_classfier = KNeighborsClassifier()
    knn_classfier.fit(X_train, Y_train)
    return knn_classfier

def classify_DT_train(tfvectorizer, train_samples, train_labels):
    print("train the DT classifier")

    X_train = tfvectorizer.transform(train_samples)
    # print(X_train)
    Y_train = train_labels
    dt_classfier = DecisionTreeClassifier()
    dt_classfier.fit(X_train, Y_train)
    return dt_classfier

def classify_RF_train(tfvectorizer, train_samples, train_labels):
    print("train the RF classifier")

    X_train = tfvectorizer.transform(train_samples)
    # print(X_train)
    Y_train = train_labels
    rf_classfier = RandomForestClassifier()
    rf_classfier.fit(X_train, Y_train)
    return rf_classfier

def read_train_docs(filename):
    print("read train documents files")
    train_samples = []
    with open(filename, 'r', encoding='utf-8') as lines:
        for line in lines:
            line = line.strip()
            item = line.split("---")
            train_samples.append(item)
    return train_samples
def read_predict_docs(filename):
    print("read predict documents from files")
    predict_samples = []
    with open(filename, 'r', encoding='utf-8') as lines:
        for line in lines:
            line = line.strip()
            item = line.split("---")
            predict_samples.append(item)
    return predict_samples

def write_result(model,texts,labels_test,labels_predict):
    filename = model+"(result).txt"
    file = codecs.open(filename, 'w', encoding='utf-8')
    file.write("真实label---预测label---测试语料\n")
    for i in range(len(labels_predict)):
        file.write(str(labels_test[i]) + "---" + str(labels_predict[i]) + "---" + texts[i].replace(" ", "") + "\n")
    file.close()


# ----------------------process vote----------------------------
def predict_and_vote():
    print("vote by multiple classifier")

    train_documents = read_train_docs(train_filename)
    predict_documents = read_predict_docs(predict_filename)

    train_samples = [i[1] for i in train_documents]
    train_labels = [i[0] for i in train_documents]

    test_samples = [i[1] for i in predict_documents]

    test_labels = [i[0] for i in predict_documents]
    # print(test_samples)
    # print(test_labels)

    train_docs = []
    for i in range(len(train_samples)):
        train_docs.append(["xxx", train_labels[i], train_samples[i]])

    numbers = [[5000, 2000]]
    posWords, negWords = load_data(train_filename)
    best_words_dict = find_best_words(posWords, negWords, numbers)

    key_feature = "_".join([str(int(n)) for n in numbers[0]])
    best_words = best_words_dict[key_feature]
    vocabs = {}

    for x in best_words:
        vocabs[x] = len(vocabs)

    tfvectorizer = TfidfVectorizer(use_idf=True, vocabulary=vocabs, analyzer=lambda s: s.split(' '))
    tfvectorizer.fit(train_samples)
    
#***************************** Naive Bayes ******************************************
    nb_classifier = classify_nb_train(tfvectorizer, train_samples, train_labels)

    NB_probability = open("data/NB_train.txt", "w", encoding='utf-8')

    documents_len = len(train_samples)
    for i in range(documents_len):
        train_vector = tfvectorizer.transform([train_samples[i]])
        nb_des = nb_classifier.predict_proba(train_vector)

        str_nb_prob = [str(j) for j in nb_des[0]]
        nb_prob_str = " ".join(str_nb_prob)

        NB_probability.write(train_labels[i] + ' ' + nb_prob_str + '\n')
    NB_probability.close()

    NB_probability = open("data/NB_test.txt", "w", encoding='utf-8')
    documents_len = len(test_samples)
    nb_res_set = []
    for i in range(documents_len):
        test_vector = tfvectorizer.transform([test_samples[i]])
        nb_res = nb_classifier.predict(test_vector)
        nb_res_set.append(nb_res[0])

        nb_des = nb_classifier.predict_proba(test_vector)
        str_nb_prob = [str(j) for j in nb_des[0]]
        nb_prob_str = " ".join(str_nb_prob)
        NB_probability.write(test_labels[i] + ' ' + nb_prob_str + '\n')
    # print(nb_res_set)
    NB_probability.close()
    print("Naive Bayes:")
    print(metrics.classification_report(test_labels, np.array(nb_res_set), digits=4))
    write_result("Bayes", test_samples, test_labels, nb_res_set)

    # print(tfvectorizer)
    # print()

    svm_classifier = classify_svm_train(tfvectorizer, train_samples, train_labels)

    SVM_probability = open("data/SVM_train.txt", "w", encoding='utf-8')
    documents_len = len(train_samples)
    for i in range(documents_len):
        train_vector = tfvectorizer.transform([train_samples[i]])
        svm_des = svm_classifier.decision_function(train_vector)
        str_svm_prob = [str(j) for j in svm_des]
        svm_prob_str = " ".join(str_svm_prob)
        SVM_probability.write(train_labels[i] + ' ' + svm_prob_str + '\n')
    SVM_probability.close()

    SVM_probability = open("data/SVM_test.txt", "w", encoding='utf-8')
    documents_len = len(test_samples)

    svm_res_set = []
    for i in range(documents_len):
        test_vector = tfvectorizer.transform([test_samples[i]])
        svm_res = svm_classifier.predict(test_vector)
        svm_res_set.append(svm_res[0])

        svm_des = svm_classifier.decision_function(test_vector)
        str_svm_prob = [str(j) for j in svm_des]
        svm_prob_str = " ".join(str_svm_prob)
        SVM_probability.write(test_labels[i] + ' ' + svm_prob_str + '\n')

    SVM_probability.close()
    print("Support vector Machines:")
    print(metrics.classification_report(test_labels, np.array(svm_res_set), digits=4))
    write_result("SVM", test_samples, test_labels, svm_res_set)

#**************************** GBDT *******************************************
    gbdt_classifier = classify_GBDT_train(tfvectorizer, train_samples, train_labels)

    gbdt_probability = open("data/gbdt_train.txt", "w", encoding='utf-8')
    documents_len = len(train_samples)
    for i in range(documents_len):
        train_vector = tfvectorizer.transform([train_samples[i]])
        gbdt_des = gbdt_classifier.decision_function(train_vector.todense())
        str_gbdt_prob = [str(j) for j in gbdt_des]
        gbdt_prob_str = " ".join(str_gbdt_prob)
        gbdt_probability.write(train_labels[i] + ' ' + gbdt_prob_str + '\n')
    gbdt_probability.close()

    gbdt_probability = open("data/gbdt_test.txt", "w", encoding='utf-8')
    documents_len = len(test_samples)

    gbdt_res_set = []
    for i in range(documents_len):
        test_vector = tfvectorizer.transform([test_samples[i]])
        gbdt_res = gbdt_classifier.predict(test_vector.todense())
        gbdt_res_set.append(gbdt_res[0])


        gbdt_des = gbdt_classifier.decision_function(test_vector.todense())
        str_gbdt_prob = [str(j) for j in gbdt_des]
        gbdt_prob_str = " ".join(str_gbdt_prob)
        gbdt_probability.write(test_labels[i] + ' ' + gbdt_prob_str + '\n')
    # print(gbdt_res_set)
    gbdt_probability.close()
    print("Gradient Boosting Decision Tree:")
    print(metrics.classification_report(test_labels, np.array(gbdt_res_set), digits=4))
    write_result("GBDT", test_samples, test_labels, gbdt_res_set)
#**************************** Logistic Regression *******************************************
    lr_classifier = classify_LR_train(tfvectorizer, train_samples, train_labels)
    #
    lr_probability = open("data/lr_train.txt", "w", encoding='utf-8')
    documents_len = len(train_samples)
    for i in range(documents_len):
        train_vector = tfvectorizer.transform([train_samples[i]])
        lr_des = lr_classifier.decision_function(train_vector.todense())
        str_lr_prob = [str(j) for j in lr_des]
        lr_prob_str = " ".join(str_lr_prob)
        lr_probability.write(train_labels[i] + ' ' + lr_prob_str + '\n')
    lr_probability.close()
    #
    lr_probability = open("data/lr_test.txt", "w", encoding='utf-8')
    documents_len = len(test_samples)

    lr_res_set = []
    for i in range(documents_len):
        test_vector = tfvectorizer.transform([test_samples[i]])
        lr_res = lr_classifier.predict(test_vector.todense())
        lr_res_set.append(lr_res[0])

        lr_des = lr_classifier.decision_function(test_vector.todense())
        str_lr_prob = [str(j) for j in lr_des]
        lr_prob_str = " ".join(str_lr_prob)
        lr_probability.write(test_labels[i] + ' ' + lr_prob_str + '\n')
    lr_probability.close()
    print("Logistic Regression:")
    print(metrics.classification_report(test_labels, np.array(lr_res_set), digits=4))
    write_result("LR", test_samples, test_labels, lr_res_set)

# **************************** k-NearestNeighbor *******************************************
    knn_classifier = classify_KNN_train(tfvectorizer, train_samples, train_labels)

    knn_probability = open("data/knn_train.txt", "w", encoding='utf-8')
    documents_len = len(train_samples)
    for i in range(documents_len):
        train_vector = tfvectorizer.transform([train_samples[i]])
        knn_des = knn_classifier.predict_proba(train_vector)

        str_knn_prob = [str(j) for j in knn_des[0]]
        knn_prob_str = " ".join(str_knn_prob)

        knn_probability.write(train_labels[i] + ' ' + knn_prob_str + '\n')
    knn_probability.close()

    knn_probability = open("data/knn_test.txt", "w", encoding='utf-8')
    documents_len = len(test_samples)
    knn_res_set = []
    for i in range(documents_len):
        test_vector = tfvectorizer.transform([test_samples[i]])
        knn_res = knn_classifier.predict(test_vector)
        knn_res_set.append(knn_res[0])

        knn_des = knn_classifier.predict_proba(test_vector)
        str_knn_prob = [str(j) for j in knn_des[0]]
        knn_prob_str = " ".join(str_knn_prob)
        knn_probability.write(test_labels[i] + ' ' + knn_prob_str + '\n')
    # print(knn_res_set)
    knn_probability.close()

    print("k-NearestNeighbor:")
    print(metrics.classification_report(test_labels, np.array(knn_res_set), digits=4))
    write_result("KNN", test_samples, test_labels, knn_res_set)

# **************************** Decision Tree *******************************************
    dt_classifier = classify_DT_train(tfvectorizer, train_samples, train_labels)

    dt_probability = open("data/dt_train.txt", "w", encoding='utf-8')
    documents_len = len(train_samples)
    for i in range(documents_len):
        train_vector = tfvectorizer.transform([train_samples[i]])
        dt_des = dt_classifier.predict_proba(train_vector)

        str_dt_prob = [str(j) for j in dt_des[0]]
        dt_prob_str = " ".join(str_dt_prob)

        dt_probability.write(train_labels[i] + ' ' + dt_prob_str + '\n')
    dt_probability.close()

    dt_probability = open("data/dt_test.txt", "w", encoding='utf-8')
    documents_len = len(test_samples)
    dt_res_set = []
    for i in range(documents_len):
        test_vector = tfvectorizer.transform([test_samples[i]])
        dt_res = dt_classifier.predict(test_vector)
        dt_res_set.append(dt_res[0])

        dt_des = dt_classifier.predict_proba(test_vector)
        str_dt_prob = [str(j) for j in dt_des[0]]
        dt_prob_str = " ".join(str_dt_prob)
        dt_probability.write(test_labels[i] + ' ' + dt_prob_str + '\n')
    # print(dt_res_set)
    dt_probability.close()

    print("Decision Tree:")
    print(metrics.classification_report(test_labels, np.array(dt_res_set), digits=4))
    write_result("DT", test_samples, test_labels, dt_res_set)

# **************************** Random Forest *******************************************
    rf_classifier = classify_RF_train(tfvectorizer, train_samples, train_labels)

    rf_probability = open("data/rf_train.txt", "w", encoding='utf-8')
    documents_len = len(train_samples)
    for i in range(documents_len):
        train_vector = tfvectorizer.transform([train_samples[i]])
        rf_des = rf_classifier.predict_proba(train_vector)

        str_rf_prob = [str(j) for j in rf_des[0]]
        rf_prob_str = " ".join(str_rf_prob)

        rf_probability.write(train_labels[i] + ' ' + rf_prob_str + '\n')
    rf_probability.close()

    rf_probability = open("data/rf_test.txt", "w", encoding='utf-8')
    documents_len = len(test_samples)
    rf_res_set = []
    for i in range(documents_len):
        test_vector = tfvectorizer.transform([test_samples[i]])
        rf_res = rf_classifier.predict(test_vector)
        rf_res_set.append(rf_res[0])

        rf_des = rf_classifier.predict_proba(test_vector)
        str_rf_prob = [str(j) for j in rf_des[0]]
        rf_prob_str = " ".join(str_rf_prob)
        rf_probability.write(test_labels[i] + ' ' + rf_prob_str + '\n')
    # print(rf_res_set)
    rf_probability.close()

    print("Random Forest:")
    print(metrics.classification_report(test_labels, np.array(rf_res_set), digits=4))
    write_result("RF", test_samples, test_labels, rf_res_set)


if __name__ == '__main__':
    predict_and_vote()
