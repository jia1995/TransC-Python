import math
import numpy as np
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s %(funcName)s %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class testForIsA():
    def __init__(self):
        self.dim = 100
        self.sub_test_num = 0
        self.ins_test_num = 0
        self.concept_num = 0
        self.entity_num = 0
        self.delta_ins = 0
        self.delta_sub = 0
        self.valid = True
        self.mix = False
        self.dataSet = "YAGO39K"
        self.ins_right: list = []
        self.ins_wrong: list = []
        self.sub_right: list = []
        self.sub_wrong: list = []

    # - 平方
    def sqr(self, x):
        return x ** 2

    # - 检查子类
    def checkSubClass(self, concept1: int, concept2: int):
        dis = 0
        for i in range(self.dim):
            dis += self.sqr(self.concept_vec[concept1][i] - self.concept_vec[concept2][i])
        if (math.sqrt(dis) < math.fabs(self.concept_r[concept1] - self.concept_r[concept2])) and (self.concept_r[concept1] < self.concept_r[concept2]):
            return True
        if math.sqrt(dis) < (self.concept_r[concept1] + self.concept_r[concept2]):
            tmp = (self.concept_r[concept1] + self.concept_r[concept2] - math.sqrt(dis)) / self.concept_r[concept1]
            if tmp > self.delta_sub:
                return True
        return False

    # 检查实例
    def checkInstance(self, instance: int, concept: int):
        dis = 0
        for i in range(self.dim):
            dis += self.sqr(self.entity_vec[instance][i] - self.concept_vec[concept][i])
        if math.sqrt(dis) < self.concept_r[concept]:
            return True
        tmp = self.concept_r[concept] / math.sqrt(dis)
        return tmp > self.delta_ins

    # 准备
    def prepare(self):
        logger.info('Start prepare')
        if self.valid:
            if self.mix:
                fin = open('../data/' + self.dataSet + '/M-Valid/instanceOf2id_negative.txt')
                fin_right = open('../data/' + self.dataSet + '/M-Valid/instanceOf2id_positive.txt')
            else:
                fin = open('../data/' + self.dataSet + '/Valid/instanceOf2id_negative.txt')
                fin_right = open('../data/' + self.dataSet + '/Valid/instanceOf2id_positive.txt')
        else:
            if self.mix:
                fin = open('../data/' + self.dataSet + '/M-Test/instanceOf2id_negative.txt')
                fin_right = open('../data/' + self.dataSet + '/M-Test/instanceOf2id_positive.txt')
            else:
                fin = open('../data/' + self.dataSet + '/Test/instanceOf2id_negative.txt')
                fin_right = open('../data/' + self.dataSet + '/Test/instanceOf2id_positive.txt')
        self.ins_test_num = int(fin.readline())
        self.ins_test_num = int(fin_right.readline())
        self.ins_wrong = [[int(i[:-1].split(' ')[0]),int(i[:-1].split(' ')[1])] for i in fin]
        self.ins_right = [[int(i[:-1].split(' ')[0]),int(i[:-1].split(' ')[1])] for i in fin_right]
        fin.close()
        fin_right.close()
        if self.valid:
            if self.mix:
                fin = open('../data/' + self.dataSet + '/M-Valid/subClassOf2id_negative.txt')
                fin_right = open('../data/' + self.dataSet + '/M-Valid/subClassOf2id_positive.txt')
            else:
                fin = open('../data/' + self.dataSet + '/Valid/subClassOf2id_negative.txt')
                fin_right = open('../data/' + self.dataSet + '/Valid/subClassOf2id_positive.txt')
        else:
            if self.mix:
                fin = open('../data/' + self.dataSet + '/M-Test/subClassOf2id_negative.txt')
                fin_right = open('../data/' + self.dataSet + '/M-Test/subClassOf2id_positive.txt')
            else:
                fin = open('../data/' + self.dataSet + '/Test/subClassOf2id_negative.txt')
                fin_right = open('../data/' + self.dataSet + '/Test/subClassOf2id_positive.txt')
        self.sub_test_num = int(fin.readline())
        self.sub_test_num = int(fin_right.readline())
        self.sub_wrong = [[int(i[:-1].split(' ')[0]),int(i[:-1].split(' ')[1])] for i in fin]
        self.sub_right = [[int(i[:-1].split(' ')[0]),int(i[:-1].split(' ')[1])] for i in fin_right]
        fin.close()
        fin_right.close()

        fin_num = open("../data/" + self.dataSet + "/Train/instance2id.txt")
        self.entity_num = int(fin_num.readline())
        fin_num.close()
        fin_num = open("../data/" + self.dataSet + "/Train/concept2id.txt")
        self.concept_num = int(fin_num.readline())
        fin_num.close()
        f1 = open('../vector/' + self.dataSet + '/entity2vec.vec')
        f2 = open('../vector/' + self.dataSet + '/concept2vec.vec')
        self.entity_vec = [np.array(i[:-2].split('\t'),dtype='float32').tolist() for i in f1]
        self.concept_vec = [[] for i in range(self.concept_num)]
        self.concept_r = [0 for i in range(self.concept_num)]
        for i in range(self.concept_num):
            self.concept_vec[i] = np.array([f2.readline()[:-2].split('\t')], dtype='float32').tolist()[0]
            self.concept_r[i] = np.array([f2.readline()[:-1]], dtype='float32')[0]
        logger.info('End prepare')

    def test(self):
        TP_ins, TN_ins, FP_ins, FN_ins = 0, 0, 0, 0
        TP_sub, TN_sub, FP_sub, FN_sub = 0, 0, 0, 0
        TP_ins_map, TN_ins_map, FP_ins_map, FN_ins_map = {}, {}, {}, {}
        for i in range(self.ins_test_num):
            if self.checkInstance(int(self.ins_right[i][0]), int(self.ins_right[i][1])):
                TP_ins += 1
                if self.ins_right[i][1] in TP_ins_map:
                    TP_ins_map[self.ins_right[i][1]] = TP_ins_map[self.ins_right[i][1]] + 1
                else:
                    TP_ins_map[self.ins_right[i][1]] = 1
            else:
                FN_ins += 1
                if self.ins_right[i][1] in FN_ins_map:
                    FN_ins_map[self.ins_right[i][1]] = FN_ins_map[self.ins_right[i][1]] + 1
                else:
                    FN_ins_map[self.ins_right[i][1]] = 1
            if not self.checkInstance(int(self.ins_wrong[i][0]), int(self.ins_wrong[i][1])):
                TN_ins += 1
                if self.ins_wrong[i][1] in TN_ins_map:
                    TN_ins_map[self.ins_wrong[i][1]] = TN_ins_map[self.ins_wrong[i][1]] + 1
                else:
                    TN_ins_map[self.ins_wrong[i][1]] = 1
            else:
                FP_ins += 1
                if self.ins_wrong[i][1] in FP_ins_map:
                    FP_ins_map[self.ins_wrong[i][1]] = FP_ins_map[self.ins_wrong[i][1]] + 1
                else:
                    FP_ins_map[self.ins_wrong[i][1]] = 1
        for i in range(self.sub_test_num):
            if self.checkSubClass(int(self.sub_right[i][0]), int(self.sub_right[i][1])):
                TP_sub += 1
            else:
                FN_sub += 1
            if not self.checkSubClass(int(self.sub_wrong[i][0]), int(self.sub_wrong[i][1])):
                TN_sub += 1
            else:
                FP_sub += 1
        if self.valid:
            ins_ans = (TP_ins + TN_ins) * 100 / (TP_ins + TN_ins + FN_ins + FP_ins)
            sub_ins = (TP_sub + TN_sub) * 100 / (TP_sub + TN_sub + FP_sub + FN_sub)
            #print(ins_ans, sub_ins)
            return ins_ans, sub_ins
        else:
            print('instanceOf triple classification:')
            print('accuracy: {:.2f} %'.format((TP_ins + TN_ins) * 100 / (TP_ins + TN_ins + FN_ins + FP_ins)))
            print('precision: {:.2f} %'.format(TP_ins* 100 / (TP_ins + FP_ins)))
            print('recall: {:.2f} %'.format(TP_ins * 100 / (TP_ins + FN_ins)))
            p: float = TP_ins * 100 / (TP_ins + FP_ins)
            r: float = TP_ins * 100 / (TP_ins + FN_ins)
            print('F1-score: {:.2f} %'.format( 2 * p * r / (p + r)))
            print()
            print('subClassOf triple classification:')
            print('accuracy: {:.2f} %'.format((TP_sub + TN_sub) * 100 / (TP_sub + TN_sub + FN_sub + FP_sub)))
            print('precision: {:.2f} %'.format(TP_sub* 100 / (TP_sub + FP_sub)))
            print('recall: {:.2f}%'.format(TP_sub * 100 / (TP_sub + FN_sub)))
            p: float = TP_sub * 100 / (TP_sub + FP_sub)
            r: float = TP_sub * 100 / (TP_sub + FN_sub)
            print('F1-score: {:.2f} %'.format( 2 * p * r / (p + r)))
            return 0,0

    def runValid(self):
        ins_best_answer, ins_best_delta, sub_best_answer, sub_best_delta = 0.0, 0.0, 0.0, 0.0
        for i in range(101):
            f = i
            f /= 100
            self.delta_ins = f
            self.delta_sub = f * 2
            ans1, ans2 = self.test()
            if ans1 > ins_best_answer:
                ins_best_answer = ans1
                ins_best_delta = f
            if ans2 > sub_best_answer:
                sub_best_answer = ans2
                sub_best_delta = f * 2
        print('delta_ins is {}. The best ins accuracy on valid data is {:.2f}%'.format(ins_best_delta, ins_best_answer))
        print('delta_sub is {}. The best sub accuracy on valid data is {:.2f}%'.format(sub_best_delta, sub_best_answer))
        print()
        self.delta_ins = ins_best_delta
        self.delta_sub = sub_best_delta
        self.valid = False
        self.prepare()
        self.test()

def main(args):
    tCisA = testForIsA()
    tCisA.dataSet = args['data']
    tCisA.mix = args['mix']
    tCisA.dim = args['dim']
    print('data =', tCisA.dataSet)
    print('mix =', tCisA.mix)
    print('dimension =', tCisA.dim)
    tCisA.prepare()
    tCisA.runValid()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="This is a Test classification for isA")
    parser.add_argument('-data', default='YAGO39K', nargs=1, type=str)
    parser.add_argument('-mix', nargs=1, default='False', type=bool)
    parser.add_argument('-dim', nargs=1, default=100, type=int)
    args = vars(parser.parse_args())
    main(args)
