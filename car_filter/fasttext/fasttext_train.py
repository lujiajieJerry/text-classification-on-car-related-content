#!/usr/bin/python
# #coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import logging
import fasttext
import codecs


if __name__=='__main__':
    # model = fasttext.skipgram('data.txt', 'model')
    # print model.words

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # classifier = fasttext.supervised("data/train_cut.txt", "news_fasttext.model", label_prefix="__label__")

    # load训练好的模型
    classifier = fasttext.load_model('news_fasttext.model.bin', label_prefix='__label__')
    result = classifier.test("data/test_cut.txt")
    print result.precision
    # print result.recall

    labels_right = []
    texts = []
    with codecs.open("data/test_cut.txt",'r',encoding='utf-8') as fr:
        lines = fr.readlines()
    for line in lines:
        labels_right.append(line.split("\t")[1].rstrip().replace("__label__", "").replace("\n",""))
        texts.append(line.split("\t")[0])
        # print labels_right
        # print texts
    #     break


    labels_predict = [e[0] for e in classifier.predict(texts)]  # 预测输出结果为二维形式
    text_labels = list(set(labels_right))
    text_predict_labels = list(set(labels_predict))
    print labels_right
    print labels_predict


    A = dict.fromkeys(text_labels, 0)  # 预测正确的各个类的数目
    B = dict.fromkeys(text_labels, 0)  # 测试数据集中各个类的数目
    C = dict.fromkeys(text_predict_labels, 0)  # 预测结果中各个类的数目

    for i in range(0, len(labels_right)):
        B[labels_right[i]] += 1
        C[labels_predict[i]] += 1
        if labels_right[i] == labels_predict[i]:
            A[labels_right[i]] += 1

    print "测试集中正例数为:　"+str(B['1'])+" 负例数:　"+str(B['0'])
    print "预测结果的正例数为:　" + str(C['1']) + " 负例数:　" + str(C['0'])
    print "预测正确的正例数为:　" + str(A['1']) + " 负例数:　" + str(A['0'])
    # 计算准确率，召回率，F值
    for key in B:
        p = float(A[key]) / float(C[key])
        r = float(A[key]) / float(B[key])
        f = p * r * 2 / (p + r)
        print ("%s:\tp:%f\tr：%f\tf1:%f" % (key, p, r, f))

    file = codecs.open('result(fasttext).txt', 'w', encoding='utf-8')
    file.write("真实label---预测label---测试语料\n")
    wrong_result = codecs.open('wrong_result(tgrocery).txt', 'w', encoding='utf-8')
    for i in range(len(labels_predict)):
        file.write(str(labels_right[i]) + "---" + str(labels_predict[i]) + "---" + texts[i].replace(" ","") + "\n")
    file.close()


