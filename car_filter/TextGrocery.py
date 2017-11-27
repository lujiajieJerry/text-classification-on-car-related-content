#!/usr/bin/python
#coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import xlrd
import codecs
import numpy
from tgrocery import Grocery

def read_xlxs(filename,sheet):
        workbook = xlrd.open_workbook(filename)
        booksheet = workbook.sheet_by_name(sheet)
        p = list()
        for row in range(1,booksheet.nrows):
                row_data = []
                for col in range(booksheet.ncols):
                        cel = booksheet.cell(row, col)
                        val = cel.value
                        row_data.append(val)
                p.append(row_data)
        # print(p)
        return p

def corpus_split(datalist1,datalist2):
        positive_cases = []
        negative_cases = []
        dispute_cases = []
        for item in datalist1:
                # print(item)
                if item[1] == '与汽车有关':
                        positive_cases.append(item)
                elif item[1]== '与汽车无关':
                        negative_cases.append(item)
                else:
                        dispute_cases.append(item)
        print(len(positive_cases),len(negative_cases),len(dispute_cases))

def write_file(datalist, filename):

        data = []
        file = codecs.open(filename,'w',encoding='utf-8')
        for item in datalist:
                sample = []
                if item[1].encode('utf-8') == '与汽车有关':
                        file.write("1---"+ str(item[2])+"\n")
                        sample.append("1")
                        sample.append(str(item[2]))

                        data.append(sample)
                elif item[1].encode('utf-8')== '与汽车无关':
                        file.write("0---"+ str(item[2])+"\n")
                        sample.append("0")
                        sample.append(str(item[2]))

                        data.append(sample)
        file.close()

        return data


def test_split(datalist):
        '''提取类标'''
        corpus = []
        label = []
        for item in datalist:
                corpus.append(item[1])
                label.append(item[0])

        return corpus,label

def tgrocery_train(train_data,test_data):
        '''model预测'''
        print("训练语料总数为:　" + str(len(train_data)))
        test_corpus, test_label = test_split(test_data)

        grocery = Grocery('TextGrocery')
        print("start training......")
        grocery.train(train_data)
        grocery.save()
        new_grocery = Grocery('TextGrocery')
        new_grocery.load()

        predict_label = []
        for sample in test_corpus:
                label = new_grocery.predict(sample)

                predict_label.append(str(label))
        # print(predict_label)
        return test_corpus,test_label,predict_label

def evalution(test_corpus,test_label,predict_label):

        TP=FP=TN=FN=0
        # print len(test_corpus),len(test_label),len(predict_label)
        file = codecs.open('result(tgrocery).txt','w',encoding='utf-8')
        file.write("真实label---预测label---测试语料\n")
        wrong_result=codecs.open('wrong_result(tgrocery).txt','w',encoding='utf-8')
        for i in range(len(test_corpus)):

                file.write(test_label[i]+"---"+str(predict_label[i])+"---"+test_corpus[i]+"\n")

                if str(test_label[i])=="1" and str(predict_label[i])=="1":
                        TP = TP+1
                elif str(test_label[i])=="0" and str(predict_label[i])=="1":
                        FP = FP+1
                        wrong_result.write(test_label[i] + "---" + str(predict_label[i]) + "---" + test_corpus[i] + "\n")
                elif str(test_label[i])=="1" and str(predict_label[i])=="0":
                        FN = FN+1
                        wrong_result.write(test_label[i] + "---" + str(predict_label[i]) + "---" + test_corpus[i] + "\n")
                elif str(test_label[i])=="0" and str(predict_label[i])=="0":
                        TN = TN+1

        file.close()
        print("测试语料总数为:　"+str(len(test_corpus)))
        print("分类正确的语句个数为:　"+str(TP+TN))
        accuracy = float(TP+TN)/(TP+FP+FN+TN)
        print("分类的准确率accuracy为:　"+str(accuracy))

        pos_precision = float(TP)/(TP+FP)
        print("正例的精确率precision为:　" + str(pos_precision))

        pos_recall = float(TP)/(TP+FN)
        print("正例的召回率recall rate为:　" + str(pos_recall))

        pos_F1 = (2*pos_precision*pos_recall)/(pos_precision+pos_recall)
        print("正例的F1值为:　" + str(pos_F1))

        neg_precision = float(TN) / (TN+ FN)
        print("负例的精确率precision为:　" + str(neg_precision))

        neg_recall = float(TN) / (FP + TN)
        print("负例的召回率recall rate为:　" + str(neg_recall))

        neg_F1 = (2 * neg_precision * neg_recall) / (neg_precision + neg_recall)
        print("负例的F1值为:　" + str(neg_F1))

if __name__=='__main__':
        datalist1=read_xlxs('data/显式实体情感标注任务1回收.xlsx','Sheet1')
        datalist2 = read_xlxs('data/显式实体情感标注第二批数据任务1回收.xlsx','Sheet1')
        datalist3 = read_xlxs('data/显式实体情感标注任务第三批数据任务1回收.xlsx', 'Sheet1')
        datalist4 = read_xlxs('data/显式实体情感标注任务第五批数据_大通训练数据_.xlsx', 'Sheet1')

        print(len(datalist1),len(datalist2),len(datalist3))
        train_datalist = []
        for item in datalist1:
                train_datalist.append(item)
        for item in datalist2:
                train_datalist.append(item)
        for item in datalist3:
                train_datalist.append(item)

        # print len(train_datalist)
        # corpus_split(datalist1,datalist2)
        # print(len(datalist1), len(datalist2))
        train_data = write_file(train_datalist,'data/train.txt')
        test_data = write_file(datalist4,'data/test.txt')

        print(len(train_data),len(test_data))
        #
        test_corpus, test_label, predict_label=tgrocery_train(train_data,test_data)
        evalution(test_corpus, test_label, predict_label)