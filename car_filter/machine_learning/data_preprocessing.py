import jieba
import  codecs

train_file = 'data/train.txt'
test_file = 'data/test.txt'

def cut_word(filename,cut_filename):
    file = codecs.open(filename,'r',encoding='utf-8')
    line = file.readline()
    cut_file = codecs.open(cut_filename,'w',encoding='utf-8')
    while line:
        # print(line)
        label = line.split("---")[0]
        info = line.split("---")[1]

        w = jieba.cut(info.strip().replace(" ", ""), cut_all=False)
        p = ' '.join(w)

        cut_file.write(label+"---"+p+"\n")
        line=file.readline()

    file.close()
    cut_file.close()





if __name__=='__main__':
    train_file = 'data/train.txt'
    test_file = 'data/test.txt'
    cut_word(train_file,'data/train_cut.txt')
    cut_word(test_file, 'data/test_cut.txt')
