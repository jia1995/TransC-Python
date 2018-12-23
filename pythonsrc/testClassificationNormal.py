import numpy as np
import math
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s %(funcName)s %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class testForNormal():
    def __init__(self):
        self.dim = 100
        self.test_num = 0
        self.relation_num = 0
        self.valid_num = 0
        self.entity_num = 0
        self.dataSet = 'YAGO39K'
        self.valid = True
        self.getMinMax = False

    def sqr(self, x):
        return x * x

    def check(self, h, t, r):
        tmp=[(self.entity_vec[h][i] + self.relation_vec[r][i]) for i in range(self.dim)]
        dis = sum([math.fabs(tmp[i] - self.entity_vec[t][i]) for i in range(self.dim)])

        if self.getMinMax:
            if dis > int(self.max_min_relation[r][0]):
                self.max_min_relation[r][0] = dis
            if dis < int(self.max_min_relation[r][1]):
                self.max_min_relation[r][1] = dis
        return dis < self.delta_relation[r]

    def prepare(self, final_test = False):
        logger.info('Start prepare')
        if self.valid:
            fin = open("../data/" + self.dataSet + "/Valid/triple2id_negative.txt")
            fin_right = open("../data/" + self.dataSet + "/Valid/triple2id_positive.txt")
            self.valid_num = int(fin_right.readline())
            self.valid_num = int(fin.readline())
        else:
            fin = open("../data/" + self.dataSet + "/Test/triple2id_negative.txt")
            fin_right = open("../data/" + self.dataSet + "/Test/triple2id_positive.txt")
            self.test_num = int(fin_right.readline())
            self.test_num = int(fin.readline())
        fin_relation = open("../data/" + self.dataSet + "/Train/relation2id.txt")
        self.relation_num = int(fin_relation.readline())
        fin_relation.close()
        fin_entity = open("../data/" + self.dataSet + "/Train/instance2id.txt")
        self.entity_num = int(fin_entity.readline())
        fin_entity.close()

        if not final_test:
            self.delta_relation = [0 for i in range(self.relation_num)]
        self.max_min_relation = [[-1,1000000] for j in range(self.relation_num)]

        inputSize = self.valid_num if self.valid else self.test_num
        self.right_triple = np.zeros([inputSize, 3],dtype='int32')
        self.wrong_triple = np.zeros([inputSize, 3],dtype='int32')
        for i in range(inputSize):
            x=fin.readline()
            m = [int(j) for j in x[:-1].split(' ')]
            self.wrong_triple[i][0] = m[0]
            self.wrong_triple[i][1] = m[1]
            self.wrong_triple[i][2] = m[2]
            m = [int(j) for j in fin_right.readline()[:-1].split(' ')]
            self.right_triple[i][0] = m[0]
            self.right_triple[i][1] = m[1]
            self.right_triple[i][2] = m[2]
        fin.close()
        fin_right.close()

        f1 = open("../vector/" + self.dataSet + "/entity2vec.vec")
        f2 = open("../vector/" + self.dataSet + "/relation2vec.vec")
        self.entity_vec = [np.array(i[:-2].split('\t'),dtype='float32').tolist() for i in f1]
        self.relation_vec = [np.array(i[:-2].split('\t'),dtype='float32').tolist() for i in f2]
        logger.info('finish prepare')

    def test(self):
        TP, TN, FP, FN = 0, 0, 0, 0
        ans = [[0 for j in range(4)] for i in range(self.relation_num)]
        inputSize: int = self.valid_num if self.valid else self.test_num
        for i in range(inputSize):
            if self.check(self.right_triple[i][0], self.right_triple[i][1], self.right_triple[i][2]):
                TP += 1
                ans[self.right_triple[i][2]][0] += 1
            else:
                FN += 1
                ans[self.right_triple[i][2]][1] += 1
            if not self.check(self.wrong_triple[i][0], self.wrong_triple[i][1], self.wrong_triple[i][2]):
                TN += 1
                ans[self.wrong_triple[i][2]][2] += 1
            else:
                FP += 1
                ans[self.wrong_triple[i][2]][3] += 1
        if self.valid:
            returnAns = [0 for i in range(self.relation_num)]
            for i in range(self.relation_num):
                if  (ans[i][0] + ans[i][1] + ans[i][2] + ans[i][3]) is 0:
                    returnAns[i] = 0
                else:
                    returnAns[i] = (ans[i][0] + ans[i][2]) * 100 / (ans[i][0] + ans[i][1] + ans[i][2] + ans[i][3])
            return returnAns
        else:
            print('Triple classification:')
            print('accuracy: {:.4f}%'.format((TP + TN) * 100 / (TP + TN + FP + FN)))
            print('precision: {:.4f}%'.format(TP * 100 / (TP + FP)))
            print('recall: {:.4f}%'.format(TP * 100 / (TP + FN)))
            p = TP * 100 / (TP + FP)
            r = TP * 100 / (TP + FN)
            print('F1-score: {:.4f}%'.format(2 * p * r / (p + r)))
            return None

    def runValid(self):
        self.getMinMax = True
        self.test()
        self.getMinMax = False
        best_delta_relation = [0 for i in range(self.relation_num)]
        best_ans_relation = [0 for i in range(self.relation_num)]
        for i in range(100):
            for j in range(self.relation_num):
                self.delta_relation[j] = self.max_min_relation[j][1] + (
                            self.max_min_relation[j][0] - self.max_min_relation[j][1]) * i / 100
            ans = self.test()
            for k in range(self.relation_num):
                if ans[k] is not 0 and ans[k] > best_ans_relation[k]:
                    best_ans_relation[k] = ans[k]
                    best_delta_relation[k] = self.delta_relation[k]
        self.delta_relation = best_delta_relation.copy()
        #for i in range(self.relation_num):
            #self.delta_relation[i] = best_delta_relation[i]
        self.valid = False
        self.prepare(final_test = True)
        self.test()

def main(args):
    tCforNormal = testForNormal()
    tCforNormal.dataSet = args['data']
    tCforNormal.dim = args['dim']
    print('data:',args['data'])
    print('dimension:',args['dim'])
    tCforNormal.prepare()
    tCforNormal.runValid()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="This is a Test classification for normal")
    parser.add_argument('-data', default='YAGO39K', nargs=1, type=str)
    parser.add_argument('-dim', nargs=1, default=100, type=int)
    args = vars(parser.parse_args())
    main(args)
