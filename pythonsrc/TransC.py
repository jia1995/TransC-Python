import numpy as np
import math
import random
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s %(funcName)s %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

pi = 3.1415926535897932384626433832795
L1Flag = True
bern = False
ins_cut = 8
sub_cut = 8
dataSet = 'YAGO39K'

def rand(min, max):
    return random.uniform(min, max)

def normal(x, miu, sigma):
    return 1.0 / math.sqrt(2 * pi) / sigma * math.exp(-1 * (x - miu) * (x - miu) / (2 * sigma * sigma))

def randN(miu, sigma, min, max):
    x=0
    while True:
        x = rand(min, max)
        y = normal(x, miu, sigma)
        dScope = rand(0.0, normal(miu, miu, sigma))
        if dScope <= y:
            break
    return x

def vecLen(a):
    res = 0
    for i in a:
        res+=i*i
    res = math.sqrt(res)
    return res

def sqr(x):
    return x * x

def norm(a):
    x = vecLen(a)
    if x > 1:
        a = np.divide(a, x)
    return a

def normR(r):
    if r > 1:
        r = 1
    return r

def randMax(x):
    res = (random.randint(0,x-1) * random.randint(0,x-1)) % x
    return res

relation_num, entity_num, concept_num, triple_num = 0, 0, 0, 0
left_num = {} #dict int-double 
right_num = {} #dict int-double 
concept_instance = [] #2-dimension int array 
instance_concept = [] #2-dimension int array 
instance_brother = [] #2-dimension int array 
sub_up_concept = [] #2-dimension int array 
up_sub_concept = [] #2-dimension int array 
concept_brother = [] #2-dimension int array 
left_entity = {} #dict int-dict{int:int} 
right_entity = {} #dict int-dict{int:int} 

class Train():
    
    def __init__(self):
        self.ok = {}  # key is a 1*2 tuple, value is a dict{int-int} 
        self.subClassOf_ok = {}  # key is a 1*2 tuple, value is an int 
        self.instanceOf_ok = {}  # key is a 1*2 tuple, value is an int 
        self.subClassOf = []  # element is a 1*2 list 
        self.instanceOf = []  # element is a 1*2 list 
        self.__fb_h = [] #1d int array 
        self.__fb_l = [] #1d int array 
        self.__fb_r = [] #1d int array 
        self.__n = 0
        self.__res = 0
        self.__rate, self.__margin, self.__margin_instance, self.__margin_subclass = 0, 0, 0, 0
        self.__trainSize = 0
        self.__relation_vec = [] #2d float array 
        self.__entity_vec = [] #2d float array 
        self.__concept_vec = [] #2d float array 
        self.__relation_tmp = [] #2d float array 
        self.__entity_tmp = [] #2d float array 
        self.__concept_tmp = [] #2d float array 
        self.__concept_r = [] #1d float array 
        self.__concept_r_tmp = [] #1d float array 
    
    def addHrt(self, x, y, z):
        self.__fb_h.append(x)
        self.__fb_r.append(z)
        self.__fb_l.append(y)
        self.ok[(x, z)] = {y:1}
    
    def addSubClassOf(self, sub, parent):
        self.subClassOf.append([sub, parent])
        self.subClassOf_ok[(sub, parent)] = 1
    
    def addInstanceOf(self, instance, concept):
        self.instanceOf.append([instance, concept])
        self.instanceOf_ok[(instance, concept)] = 1
    
    def setup(self, n_in, rate_in, margin_in, margin_ins, margin_sub):
        logger.info('Start train setup')
        global instance_brother,instance_concept,concept_instance,concept_brother,relation_num,entity_num,concept_num,up_sub_concept,sub_up_concept
        self.__n, self.__rate, self.__margin, self.__margin_instance, self.__margin_subclass = n_in, rate_in, margin_in, margin_ins, margin_sub
        for i in range(instance_concept.__len__()):
            for j in range(instance_concept[i].__len__()):
                for k in range(concept_instance[instance_concept[i][j]].__len__()):
                    if concept_instance[instance_concept[i][j]][k] != i:
                        instance_brother[i].append(concept_instance[instance_concept[i][j]][k])
        for i in range(sub_up_concept.__len__()):
            for j in range(sub_up_concept[i].__len__()):
                for k in range(up_sub_concept[sub_up_concept[i][j]].__len__()):
                    if up_sub_concept[sub_up_concept[i][j]][k] != i:
                        concept_brother[i].append(up_sub_concept[sub_up_concept[i][j]][k])
        self.__relation_vec = np.zeros([relation_num, self.__n])
        self.__entity_vec = np.zeros([entity_num, self.__n])
        self.__relation_tmp = np.zeros([relation_num, self.__n])
        self.__entity_tmp = np.zeros([entity_num, self.__n])
        self.__concept_vec = np.zeros([concept_num, self.__n])
        self.__concept_tmp = np.zeros([concept_num, self.__n])
        self.__concept_r = np.arange(concept_num,dtype='float32')
        self.__concept_r_tmp = np.arange(concept_num,dtype='float32')
        logger.info('start randN')
        for i in range(relation_num):
            self.__relation_vec[i] = [randN(0, 1 / self.__n, -6 / math.sqrt(self.__n), 6 / math.sqrt(self.__n)) for ii in range(self.__n)]
        for i in range(entity_num):
            self.__entity_vec[i] = [randN(0, 1 / self.__n, -6 / math.sqrt(self.__n), 6 / math.sqrt(self.__n)) for ii in range(self.__n)]
            self.__entity_vec[i] = norm(self.__entity_vec[i])
        for i in range(concept_num):
            self.__concept_vec[i] = [randN(0, 1 / self.__n, -6 / math.sqrt(self.__n), 6 / math.sqrt(self.__n)) for ii in range(self.__n)]
            self.__concept_vec[i] = norm(self.__concept_vec[i])
        self.__concept_r = [rand(0,1) for i in range(concept_num)]
        self.__trainSize = self.__fb_h.__len__() + self.instanceOf.__len__() + self.subClassOf.__len__()
    
    def doTrain(self):
        nbatches = 100
        nepoch = 1000
        batchSize = self.__trainSize / nbatches
        logger.info('Start Training')
        for epoch in range(nepoch):
            self.__res = 0
            for batch in range(nbatches):
                self.__relation_tmp = self.__relation_vec.copy()
                self.__entity_tmp = self.__entity_vec.copy()
                self.__concept_tmp = self.__concept_vec.copy()
                self.__concept_r_tmp = self.__concept_r.copy()
                for k in range(int(batchSize)):
                    i = randMax(self.__trainSize)
                    if i < self.__fb_r.__len__():
                        cut = 10 - int(epoch * 8 / nepoch)
                        self.__trainHLR(i, cut)
                    elif i < self.__fb_r.__len__()+self.instanceOf.__len__():
                        cut = 10 - int(epoch * ins_cut / nepoch)
                        self.__trainInstanceOf(i, cut)
                    else:
                        cut = 10 - int(epoch * sub_cut / nepoch)
                        self.__trainSubClassOf(i, cut)
                self.__relation_vec = self.__relation_tmp.copy()
                self.__entity_vec = self.__entity_tmp.copy()
                self.__concept_vec = self.__concept_tmp.copy()
                self.__concept_r = self.__concept_r_tmp.copy()
            if epoch % 1 is 0:
                logger.info('epoch : {} - res : {:.2f}'.format(epoch, self.__res))
            if epoch % 500 is 0 or epoch is nepoch - 1:
                f2 = open("../vector/" + dataSet + "/relation2vec.vec", 'w')
                f3 = open("../vector/" + dataSet + "/entity2vec.vec", 'w')
                f4 = open("../vector/" + dataSet + "/concept2vec.vec", 'w')
                for i in range(relation_num):
                    for ii in range(self.__n):
                        f2.write('%.6f'%self.__relation_vec[i][ii])
                        f2.write('\t')
                    f2.write('\n')
                for i in range(entity_num):
                    for ii in range(self.__n):
                        f3.write('%.6f'%self.__entity_vec[i][ii])
                        f3.write('\t')
                    f3.write('\n')
                for i in range(concept_num):
                    for ii in range(self.__n):
                        f4.write('%.6f'%self.__concept_vec[i][ii])
                        f4.write('\t')
                    f4.write('\n')
                    f4.write('%.6f'%self.__concept_r[i])
                    f4.write('\n')
                f2.close()
                f3.close()
                f4.close()
    
    def __trainHLR(self, i, cut):
        global entity_num,instance_brother,right_num,left_num,bern
        j = 0
        pr = 1000 * right_num[self.__fb_r[i]] / (right_num[self.__fb_r[i]] + left_num[self.__fb_r[i]]) if bern else 500
        if random.randint(0,999) < pr:
            while True:
                if instance_brother[self.__fb_l[i]].__len__() > 0:
                    if random.randint(0,9) < cut:
                        j = randMax(entity_num)
                    else:
                        tmp_num = instance_brother[self.__fb_l[i]].__len__()
                        j = random.randint(0,tmp_num-1) if tmp_num>1 else 0
                        j = instance_brother[self.__fb_l[i]][j]
                else:
                    j = randMax(entity_num)
                if ((self.__fb_h[i], self.__fb_r[i]) not in self.ok) or (not self.ok[(self.__fb_h[i], self.__fb_r[i])].__contains__(j)):
                    break
            self.__doTrainHLR(self.__fb_h[i], self.__fb_l[i], self.__fb_r[i], self.__fb_h[i], j, self.__fb_r[i])
        else:
            while True:
                if instance_brother[self.__fb_l[i]].__len__() > 0:
                    if random.randint(0,9) < cut:
                        j = randMax(entity_num)
                    else:
                        tmp_num = instance_brother[self.__fb_h[i]].__len__()
                        j = random.randint(0,tmp_num-1) if tmp_num>1 else 0
                        j = instance_brother[self.__fb_h[i]][j]
                else:
                    j = randMax(entity_num)
                if ((j, self.__fb_r[i]) not in self.ok) or( not self.ok[(j, self.__fb_r[i])].__contains__(self.__fb_l[i])):
                    break
            self.__doTrainHLR(self.__fb_h[i], self.__fb_l[i], self.__fb_r[i], j, self.__fb_l[i], self.__fb_r[i])
        self.__relation_tmp[self.__fb_r[i]] = norm(self.__relation_tmp[self.__fb_r[i]])
        self.__entity_tmp[self.__fb_h[i]] = norm(self.__entity_tmp[self.__fb_h[i]])
        self.__entity_tmp[self.__fb_l[i]] = norm(self.__entity_tmp[self.__fb_l[i]])
        self.__entity_tmp[j] = norm(self.__entity_tmp[j])
    
    def __trainInstanceOf(self, i, cut):
        global concept_brother,entity_num,instance_brother,concept_num
        i = i - self.__fb_h.__len__()
        j = 0
        if random.randint(0,1) == 0:
            while True:
                if instance_brother[self.instanceOf[i][0]].__len__() > 0:
                    if random.randint(0,9) < cut:
                        j = randMax(entity_num)
                    else:
                        tmp_num = instance_brother[self.instanceOf[i][0]].__len__()
                        j = random.randint(0,tmp_num-1) if tmp_num>1 else 0
                        j = instance_brother[self.instanceOf[i][0]][j]
                else:
                    j = randMax(entity_num)
                if not self.instanceOf_ok.__contains__((j, self.instanceOf[i][1])):
                    break
            self.__doTrainInstanceOf(self.instanceOf[i][0], self.instanceOf[i][1], j, self.instanceOf[i][1])
            self.__entity_tmp[j] = norm(self.__entity_tmp[j])
        else:
            while True:
                if concept_brother[self.instanceOf[i][1]].__len__() > 0:
                    if random.randint(0,9)< cut:
                        j = randMax(concept_num)
                    else:
                        tmp_num = concept_brother[self.instanceOf[i][1]].__len__()
                        j = random.randint(0,tmp_num-1) if tmp_num>1 else 0
                        j = concept_brother[self.instanceOf[i][1]][j]
                else:
                    j = randMax(concept_num)
                if not self.instanceOf_ok.__contains__((self.instanceOf[i][0], j)):
                    break
            self.__doTrainInstanceOf(self.instanceOf[i][0], self.instanceOf[i][1], self.instanceOf[i][0], j)
            self.__concept_tmp[j] = norm(self.__concept_tmp[j])
            self.__concept_r_tmp[j] = normR(self.__concept_r_tmp[j])
        self.__entity_tmp[self.instanceOf[i][0]] = norm(self.__entity_tmp[self.instanceOf[i][0]])
        self.__concept_tmp[self.instanceOf[i][1]] = norm(self.__concept_tmp[self.instanceOf[i][1]])
        self.__concept_r_tmp[self.instanceOf[i][1]] = normR(self.__concept_r_tmp[self.instanceOf[i][1]])
    
    def __trainSubClassOf(self, i, cut):
        global concept_brother,concept_num
        i = i - self.__fb_h.__len__() - self.instanceOf.__len__()
        j = 0
        if random.randint(0,1) == 0:
            while True:
                if concept_brother[self.subClassOf[i][0]].__len__() > 0:
                    if random.randint(0,9) < cut:
                        j = randMax(concept_num)
                    else:
                        tmp_num = concept_brother[self.subClassOf[i][0]].__len__()
                        j = random.randint(0,tmp_num-1) if tmp_num>1 else 0
                        j = concept_brother[self.subClassOf[i][0]][j]
                else:
                    j = randMax(concept_num)
                if not self.subClassOf_ok.__contains__((j, self.subClassOf[i][1])):
                    break
            self.__doTrainSubClassOf(self.subClassOf[i][0], self.subClassOf[i][1], j, self.subClassOf[i][1])
        else:
            while True:
                if concept_brother[self.subClassOf[i][1]].__len__() > 0:
                    if random.randint(0,9) < cut:
                        j = randMax(concept_num)
                    else:
                        tmp_num = concept_brother[self.subClassOf[i][1]].__len__()
                        j = random.randint(0,tmp_num-1) if tmp_num>1 else 0
                        j = concept_brother[self.subClassOf[i][1]][j]
                else:
                    j = randMax(concept_num)
                if not self.subClassOf_ok.__contains__((self.subClassOf[i][0], j)):
                    break
            self.__doTrainSubClassOf(self.subClassOf[i][0], self.subClassOf[i][1], self.subClassOf[i][0], j)
        self.__concept_tmp[self.subClassOf[i][0]] = norm(self.__concept_tmp[self.subClassOf[i][0]])
        self.__concept_tmp[self.subClassOf[i][1]] = norm(self.__concept_tmp[self.subClassOf[i][1]])
        self.__concept_tmp[j] = norm(self.__concept_tmp[j])
        self.__concept_r_tmp[self.subClassOf[i][0]] = normR(self.__concept_r_tmp[self.subClassOf[i][0]])
        self.__concept_r_tmp[self.instanceOf[i][1]] = normR(self.__concept_r_tmp[self.subClassOf[i][1]])
        self.__concept_r_tmp[j] = normR(self.__concept_r_tmp[j])
    
    def __doTrainHLR(self, e1_a, e2_a, rel_a, e1_b, e2_b, rel_b):
        sum1 = self.__calcSumHLT(e1_a, e2_a, rel_a)
        sum2 = self.__calcSumHLT(e1_b, e2_b, rel_b)
        if sum1 + self.__margin > sum2:
            self.__res += (self.__margin + sum1 - sum2)
            self.__gradientHLR(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b)
    
    def __doTrainInstanceOf(self, e_a, c_a, e_b, c_b):
        sum1 = self.__calcSumInstanceOf(e_a, c_a)
        sum2 = self.__calcSumInstanceOf(e_b, c_b)
        if (sum1 + self.__margin_instance) > sum2:
            self.__res += (self.__margin_instance + sum1 - sum2)
            self.__gradientInstanceOf(e_a, c_a, e_b, c_b)
    
    def __doTrainSubClassOf(self, c1_a, c2_a, c1_b, c2_b):
        sum1 = self.__calcSumSubClassOf(c1_a, c2_a)
        sum2 = self.__calcSumSubClassOf(c1_b, c2_b)
        if (sum1 + self.__margin_subclass) > sum2:
            self.__res += (self.__margin_subclass + sum1 - sum2)
            self.__gradientSubClassOf(c1_a, c2_a, c1_b, c2_b)
    
    def __calcSumHLT(self, e1, e2, rel):
        global L1Flag
        sum = 0
        if L1Flag:
            for ii in range(self.__n):
                sum += math.fabs(self.__entity_vec[e2][ii] - self.__entity_vec[e1][ii] - self.__relation_vec[rel][ii])
        else:
            for ii in range(self.__n):
                sum += sqr(self.__entity_vec[e2][ii] - self.__entity_vec[e1][ii] - self.__relation_vec[rel][ii])
        return sum
    
    def __calcSumInstanceOf(self, e, c):
        dis = 0
        for i in range(self.__n):
            dis += sqr(self.__entity_vec[e][i] - self.__concept_vec[c][i])
        if dis < sqr(self.__concept_r[c]):
            return 0
        return dis - sqr(self.__concept_r[c])
    
    def __calcSumSubClassOf(self, c1, c2):
        dis = 0
        for i in range(self.__n):
            dis += sqr(self.__concept_vec[c1][i] - self.__concept_vec[c2][i])
        if math.sqrt(dis) < math.fabs(self.__concept_r[c1] - self.__concept_r[c2]):
            return 0
        return dis - sqr(self.__concept_r[c2]) + sqr(self.__concept_r[c1])
    
    def __gradientHLR(self, e1_a, e2_a, rel_a, e1_b, e2_b, rel_b):
        global L1Flag
        for ii in range(self.__n):
            x = 2 * (self.__entity_vec[e2_a][ii] - self.__entity_vec[e1_a][ii] - self.__relation_vec[rel_a][ii])
            if L1Flag:
                x = 1 if x > 0 else -1
            self.__relation_tmp[rel_a][ii] -= -1 * self.__rate * x
            self.__entity_tmp[e1_a][ii] -= -1 * self.__rate * x
            self.__entity_tmp[e2_a][ii] += -1 * self.__rate * x
            x = 2 * (self.__entity_vec[e2_b][ii] - self.__entity_vec[e1_b][ii] - self.__relation_vec[rel_b][ii])
            if L1Flag:
                x = 1 if x > 0 else -1
            self.__relation_tmp[rel_b][ii] -= self.__rate * x
            self.__entity_tmp[e1_b][ii] -= self.__rate * x
            self.__entity_tmp[e2_b][ii] += self.__rate * x
    
    def __gradientInstanceOf(self, e_a, c_a, e_b, c_b):
        dis = 0
        for i in range(self.__n):
            dis += sqr(self.__entity_vec[e_a][i] - self.__concept_vec[c_a][i])
        if dis > sqr(self.__concept_r[c_a]):
            for j in range(self.__n):
                x = 2 * (self.__entity_vec[e_a][j] - self.__concept_vec[c_a][j])
                self.__entity_tmp[e_a][j] -= x * self.__rate
                self.__concept_tmp[c_a][j] -= -1 * x * self.__rate
            self.__concept_r_tmp[c_a] -= -2 * self.__concept_r[c_a] * self.__rate
        dis = 0
        for i in range(self.__n):
            dis += sqr(self.__entity_vec[e_b][i] - self.__concept_vec[c_b][i])
        if dis > sqr(self.__concept_r[c_b]):
            for j in range(self.__n):
                x = 2 * (self.__entity_vec[e_b][j] - self.__concept_vec[c_b][j])
                self.__entity_tmp[e_b][j] += x * self.__rate
                self.__concept_tmp[c_b][j] += -1 * x * self.__rate
            self.__concept_r_tmp[c_b] += -2 * self.__concept_r[c_b] * self.__rate
    
    def __gradientSubClassOf(self, c1_a, c2_a, c1_b, c2_b):
        dis = 0
        for i in range(self.__n):
            dis += sqr(self.__concept_vec[c1_a][i] - self.__concept_vec[c2_a][i])
        if math.sqrt(dis) > math.fabs(self.__concept_r[c1_a] - self.__concept_r[c2_a]):
            for j in range(self.__n):
                x = 2 * (self.__concept_vec[c1_a][i] - self.__concept_vec[c2_a][i])
                self.__concept_tmp[c1_a][i] -= x * self.__rate
                self.__concept_tmp[c2_a][i] -= -x * self.__rate
            self.__concept_r_tmp[c1_a] -= 2 * self.__concept_r[c1_a] * self.__rate
            self.__concept_r_tmp[c2_a] -= -2 * self.__concept_r[c2_a] * self.__rate
        dis = 0
        for i in range(self.__n):
            dis += sqr(self.__concept_vec[c1_b][i] - self.__concept_vec[c2_b][i])
        if math.sqrt(dis) > math.fabs(self.__concept_r[c1_b] - self.__concept_r[c2_b]):
            for j in range(self.__n):
                x = 2 * (self.__concept_vec[c1_b][i] - self.__concept_vec[c2_b][i])
                self.__concept_tmp[c1_b][i] += x * self.__rate
                self.__concept_tmp[c2_b][i] += -x * self.__rate
            self.__concept_r_tmp[c1_b] += 2 * self.__concept_r[c1_b] * self.__rate
            self.__concept_r_tmp[c2_b] += -2 * self.__concept_r[c2_b] * self.__rate

train = Train()

def prepare():
    logger.info('Start prepare')
    global entity_num, relation_num, concept_num, triple_num,bern,train,left_entity,right_entity,concept_brother,concept_instance,instance_brother,instance_concept,sub_up_concept,up_sub_concept
    f1 = open("../data/" + dataSet + "/Train/instance2id.txt")
    f2 = open("../data/" + dataSet + "/Train/relation2id.txt")
    f3 = open("../data/" + dataSet + "/Train/concept2id.txt")
    f_kb = open("../data/" + dataSet + "/Train/triple2id.txt")
    entity_num = int(f1.readline())
    relation_num = int(f2.readline())
    concept_num = int(f3.readline())
    triple_num = int(f_kb.readline())
    h, t, l = 0, 0, 0
    while True:
        m = f_kb.readline().split(' ')
        if m.__len__() is not 3:
            break
        h, t, l = [int(i) for i in m]
        train.addHrt(h, t, l)
        if bern:
            if l not in left_entity:
                left_entity[l] = {h:0}
            if l not in right_entity:
                right_entity[l] = {t:0}
            left_entity[l][h] += 1
            right_entity[l][t] += 1
    f_kb.close()
    f1.close()
    f2.close()
    f3.close()
    instance_concept = [[] for i in range(entity_num)]
    concept_instance = [[] for i in range(concept_num)]
    sub_up_concept = [[] for i in range(concept_num)]
    up_sub_concept = [[] for i in range(concept_num)]
    concept_brother = [[] for i in range(concept_num)]
    instance_brother = [[] for i in range(entity_num)]
    if bern:
        for i in range(relation_num):
            sum1 = 0
            sum2 = 0
            for it in left_entity[i]:
                sum1 += 1
                sum2 += left_entity[i][it]
            left_num[i] = sum2 / sum1
        for i in range(relation_num):
            sum1 = 0
            sum2 = 0
            for it in right_entity[i]:
                sum1 += 1
                sum2 += right_entity[i][it]
            right_num[i] = sum2 / sum1
    instanceOf_file = open("../data/" + dataSet + "/Train/instanceOf2id.txt")
    subClassOf_file = open("../data/" + dataSet + "/Train/subClassOf2id.txt")
    a, b = 0, 0
    while True:
        x = instanceOf_file.readline().split(' ')
        if x.__len__() is not 2:
            break
        a, b = [int(i) for i in x]
        train.addInstanceOf(a, b)
        instance_concept[a].append(b)
        concept_instance[b].append(a)
    while True:
        x = subClassOf_file.readline().split(' ')
        if x.__len__() is not 2:
            break
        a, b = [int(i) for i in x]
        train.addSubClassOf(a, b)
        sub_up_concept[a].append(b)
        up_sub_concept[b].append(a)

def main(args):
    rate = 0.001
    print('vector dimension =', args['dim'])
    print('learing rate =', rate)
    print('margin =', args['margin'])
    print('margin_ins =', args['margin_ins'])
    print('margin_sub =', args['margin_sub'])
    print('L1 Flag =', args['l1flag'])
    print('bern =', args['bern'])
    L1Flag = args['l1flag']
    bern = args['bern']
    prepare()
    train.setup(args['dim'], rate, args['margin'], args['margin_ins'], args['margin_sub'])
    train.doTrain()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="This is The TransC")
    parser.add_argument('-data', default='YAGO39K', nargs=1, type=str)
    parser.add_argument('-dim', nargs=1, default=100, type=int)
    parser.add_argument('-margin', nargs=1, default=1, type=float)
    parser.add_argument('-margin_ins', nargs=1, default=0.4, type=float)
    parser.add_argument('-margin_sub', nargs=1, default=0.3, type=float)
    parser.add_argument('-l1flag', nargs=1, default=True, type=bool)
    parser.add_argument('-bern', nargs=1, default=False, type=bool)
    args = vars(parser.parse_args())
    #args = {'data':'YAGO39K','dim':100,'margin':1,'margin_ins':0.4,'margin_sub':0.3,'l1flag':True,'bern':False}
    main(args)
