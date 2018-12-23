import numpy as np
import math
from numba import jit
import logging
from multiprocessing import Pool, Manager,Array,Process
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s %(funcName)s %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
relationTotal, entityTotal,testTotal,trainTotal,validTotal,tripleTotal,threads = 0,0,0,0,0,0,8
dataSet = "YAGO39K"
dimension = 100

class Triple():
    def __init__(self,h,r,t):
        self.h = h
        self.r = r
        self.t = t

testList = []
tripleList = []
relationVec = []
testTotalR = [0 for i in range(1500)]
entityRelVec = []
entityVec = []

def init():
    logger.info('Start init')
    global relationTotal,relationVec,entityTotal,testTotal,trainTotal,validTotal,tripleTotal,testTotalR,entityVec,testList,tripleList
    fin = open("../data/" + dataSet + "/Train/relation2id.txt")
    relationTotal = int(fin.readline())
    fin.close()
    relationVec = [0.0 for i in range(relationTotal*dimension)]
    fin = open("../data/" + dataSet + "/Train/instance2id.txt")
    entityTotal = int(fin.readline())
    fin.close()
    entityVec = [0.0 for i in range(entityTotal*dimension)]
    f_kb1 = open("../data/" + dataSet + "/Test/triple2id_positive.txt")
    f_kb2 = open("../data/" + dataSet + "/Train/triple2id.txt")
    f_kb3 = open("../data/" + dataSet + "/Valid/triple2id_positive.txt")
    testTotal = int(f_kb1.readline())
    trainTotal = int(f_kb2.readline())
    validTotal = int(f_kb3.readline())
    tripleTotal = testTotal+trainTotal+validTotal

    for i in range(testTotal):
        line = f_kb1.readline().split(' ')
        h,t,r = [int(i) for i in line]
        x = Triple(h=h,r=r,t=t)
        testList.append(x)
        tripleList.append(x)
        testTotalR[r] +=1

    for i in range(trainTotal):
        line = f_kb2.readline().split(' ')
        h, t, r = [int(i) for i in line]
        x = Triple(h=h, r=r, t=t)
        tripleList.append(x)

    for i in range(validTotal):
        line = f_kb3.readline().split(' ')
        h, t, r = [int(i) for i in line]
        x = Triple(h=h, r=r, t=t)
        tripleList.append(x)
    f_kb1.close()
    f_kb2.close()
    f_kb3.close()
    tripleList.sort(key=lambda x:(x.h,x.r,x.t))
    logger.info('End init')

def prepare():
    logger.info('Start prepare')
    global entityTotal,entityVec,relationTotal,relationVec,entityRelVec,dimension
    fin = open("../vector/" + dataSet + "/entity2vec.vec")
    entityVec = [j for i in fin for j in np.array(i[:-2].split('\t'),dtype='float32').tolist()]
    fin.close()
    fin = open("../vector/" + dataSet + "/relation2vec.vec")
    relationVec = [j for i in fin for j in np.array(i[:-2].split('\t'),dtype='float32').tolist()]
    fin.close()
    entityRelVec= [entityVec[i*dimension+k] for i in range(entityTotal) for j in range(relationTotal) for k in range(dimension)]
    logger.info('End prepare')

def calcSum(e1, e2, rel):
    global relationTotal,dimension,entityVec,entityRelVec,relationVec
    res = 0.0
    last1 = e1 * relationTotal * dimension + rel * dimension
    last2 = e2 * relationTotal * dimension + rel * dimension
    lastr = rel*dimension
    res=sum([math.fabs(entityRelVec[last1+i]+relationVec[lastr+i]-entityRelVec[last2+i]) for i in range(dimension)])
    return res

def find(h,t,r):
    global tripleList,tripleTotal
    lef = 0
    rig = tripleTotal-1
    while lef+1<rig:
        mid = (lef+rig)//2
        if (tripleList[mid]. h < h) or (tripleList[mid]. h == h and tripleList[mid]. r < r) or (tripleList[mid]. h == h and tripleList[mid]. r == r and tripleList[mid]. t < t):
            lef = mid
        else:
            rig = mid
    x = Triple(h,r,t)
    if (tripleList[lef].h == h and tripleList[lef].r == r and tripleList[lef].t == t) or (tripleList[rig].h == h and tripleList[rig].r == r and tripleList[rig].t == t):
        return True
    return False

def testMode(id,l_filter_tot,r_filter_tot,l_filter_tot1,r_filter_tot1,l_filter_tot3,r_filter_tot3,l_filter_tot5,r_filter_tot5,l_tot,r_tot,l_tot1,r_tot1,l_tot3,r_tot3,l_tot5,r_tot5,l_filter_rank,r_filter_rank,l_rank,r_rank,l_filter_rank_dao, r_filter_rank_dao,l_rank_dao,r_rank_dao):
    global testTotal,threads,testList,entityTotal,logger
    print('Process {} open'.format(id))
    lef = testTotal/threads*id
    rig = testTotal/threads*(id+1) if id < threads-1 else testTotal
    for i in range(int(lef),int(rig)):
        if divmod(i,100)[1] is 0:
            logger.info('{} Handle {} test'.format(id,i))
        h,t,r = testList[i].h,testList[i].t,testList[i].r
        #print(h,t,r)
        minimal = calcSum(h,t,r)
        l_filter_s,l_s,r_filter_s,r_s = 0,0,0,0
        for j in range(entityTotal):
            if j != h:
                value = calcSum(j,t,r)
                if value < minimal:
                    l_s+=1
                    if not find(j,t,r):
                        l_filter_s+=1
            if j != t:
                value = calcSum(h, j, r)
                if value < minimal:
                    r_s += 1
                    if not find(h, j, r):
                        r_filter_s += 1
        if l_filter_s <1:
            l_filter_tot1[id]+=1
            l_filter_tot3[id]+=1
            l_filter_tot5[id]+=1
            l_filter_tot[id] += 1
        elif l_filter_s<3:
            l_filter_tot3[id]+=1
            l_filter_tot5[id]+=1
            l_filter_tot[id] += 1
        elif l_filter_s<5:
            l_filter_tot5[id]+=1
            l_filter_tot[id] += 1
        elif l_filter_s <10:
            l_filter_tot[id] += 1
        if l_s<1:
            l_tot1[id]+=1
            l_tot3[id]+=1
            l_tot5[id]+=1
            l_tot[id]+=1
        elif l_s<3:
            l_tot3[id]+=1
            l_tot5[id]+=1
            l_tot[id]+=1
        elif l_s<5:
            l_tot5[id]+=1
            l_tot[id]+=1
        elif l_s<10:
            l_tot[id]+=1
        if r_filter_s <1:
            r_filter_tot1[id]+=1
            r_filter_tot3[id]+=1
            r_filter_tot5[id]+=1
            r_filter_tot[id]+=1
        elif r_filter_s <3:
            r_filter_tot3[id]+=1
            r_filter_tot5[id]+=1
            r_filter_tot[id]+=1
        elif r_filter_s <5:
            r_filter_tot5[id]+=1
            r_filter_tot[id]+=1
        elif r_filter_s < 10:
            r_filter_tot[id]+=1
        if r_s<1:
            r_tot1[id]+=1
            r_tot3[id]+=1
            r_tot5[id]+=1
            r_tot[id]+=1
        elif r_s<3:
            r_tot3[id]+=1
            r_tot5[id]+=1
            r_tot[id]+=1
        elif r_s<5:
            r_tot5[id]+=1
            r_tot[id]+=1
        elif r_s<10:
            r_tot[id]+=1
        l_filter_rank_dao[id]+=1/(l_filter_s + 1)
        r_filter_rank_dao[id] += 1 / (r_filter_s + 1)
        l_rank_dao[id] += 1 / (l_s + 1)
        r_rank_dao[id] += 1 / (r_s + 1)

        l_filter_rank[id] += l_filter_s
        r_filter_rank[id] += r_filter_s
        l_rank[id] += l_s
        r_rank[id] += r_s
    print('Process {} close()'.format(id))


def test():
    logger.info('Start test')
    global relationTotal,threads
    l_filter_tot,r_filter_tot,l_filter_tot1,r_filter_tot1,l_filter_tot3,r_filter_tot3,l_filter_tot5,r_filter_tot5,l_tot,r_tot,l_tot1,r_tot1,l_tot3,r_tot3,l_tot5,r_tot5,l_filter_rank,r_filter_rank,l_rank,r_rank,l_filter_rank_dao, r_filter_rank_dao,l_rank_dao,r_rank_dao=Array('i',[0 for i in range(threads)]),Array('i',[0 for i in range(threads)]),Array('i',[0 for i in range(threads)]),Array('i',[0 for i in range(threads)]),Array('i',[0 for i in range(threads)]),Array('i',[0 for i in range(threads)]),Array('i',[0 for i in range(threads)]),Array('i',[0 for i in range(threads)]),Array('i',[0 for i in range(threads)]),Array('i',[0 for i in range(threads)]),Array('i',[0 for i in range(threads)]),Array('i',[0 for i in range(threads)]),Array('i',[0 for i in range(threads)]),Array('i',[0 for i in range(threads)]),Array('i',[0 for i in range(threads)]),Array('i',[0 for i in range(threads)]),Array('f',[0.0 for i in range(threads)]),Array('f',[0.0 for i in range(threads)]),Array('f',[0.0 for i in range(threads)]),Array('f',[0.0 for i in range(threads)]),Array('f',[0.0 for i in range(threads)]),Array('f',[0.0 for i in range(threads)]),Array('f',[0.0 for i in range(threads)]),Array('f',[0.0 for i in range(threads)])
    
    thread_s = [Process(target=testMode,args=(i,l_filter_tot,r_filter_tot,l_filter_tot1,r_filter_tot1,l_filter_tot3,r_filter_tot3,l_filter_tot5,r_filter_tot5,l_tot,r_tot,l_tot1,r_tot1,l_tot3,r_tot3,l_tot5,r_tot5,l_filter_rank,r_filter_rank,l_rank,r_rank,l_filter_rank_dao, r_filter_rank_dao,l_rank_dao,r_rank_dao)) for i in range(threads)]
    for i in thread_s:
        i.start()
    for i in thread_s:
        i.join()
    lft0 = sum(l_filter_tot)
    lft1 = sum(l_filter_tot1)
    lft3 = sum(l_filter_tot3)
    lft5 = sum(l_filter_tot5)
    rft0 = sum(r_filter_tot)
    rft1 = sum(r_filter_tot1)
    rft3 = sum(r_filter_tot3)
    rft5 = sum(r_filter_tot5)
    lt = sum(l_tot)
    lt1 = sum(l_tot1)
    lt3 = sum(l_tot3)
    lt5 = sum(l_tot5)
    rt = sum(r_tot)
    rt1 = sum(r_tot1)
    rt3 = sum(r_tot3)
    rt5 = sum(r_tot5)
    lfr = sum(l_filter_rank)
    rfr = sum(r_filter_rank)
    lr = sum(l_rank)
    rr = sum(r_rank)
    lfrd = sum(l_filter_rank_dao)
    rfrd = sum(r_filter_rank_dao)
    lrd = sum(l_rank_dao)
    rrd = sum(r_rank_dao)
    print(testTotal)
    print(lft0,rft0)
    print(lft1,rft1)
    print(lft3 ,rft3)
    print(lft5,rft5)
    print(lt,rt)
    print(lt1,rt1)
    print(lt3,rt3)
    print(lt5,rt5)
    print(lfr,rfr)
    print(lr,rr)
    print(lfrd,rfrd)
    print(lrd,rrd)
        
    print("metric:\t\t MRR \t MR \t hit@10 \t hit@5  \t hit@3  \t hit@1 ")
    print("averaged(raw):\t{:.3f} \t {:.1f} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f} ".format(
            (lrd/ testTotal + rrd / testTotal)/2,
            (lr / testTotal + rr / testTotal)/2,
            (lt / testTotal + rt / testTotal)/2,
            (lt5 / testTotal + rt5 / testTotal)/2,
            (lt3 / testTotal + rt3 / testTotal)/2,
            (lt1/ testTotal + rt1 / testTotal)/2))
    print("averaged(filter):\t{:.3f} \t {:.1f} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f} ".format(
            (lfrd / testTotal + rfrd / testTotal)/2,
            (lfr / testTotal + rfr / testTotal)/2,
            (lft0 / testTotal + rft0/ testTotal)/2,
            (lft5 / testTotal + rft5 / testTotal)/2,
            (lft3 / testTotal + rft3 / testTotal)/2,
            (lft1 / testTotal + rft1 / testTotal)/2))
    logger.info('End test')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="This is  Test Link Predication")
    parser.add_argument('-data', default='YAGO39K', nargs=1, type=str)
    parser.add_argument('-dim', nargs=1, default=100, type=int)
    parser.add_argument('-threads', nargs=1, default=8, type=int)
    args = vars(parser.parse_args())
    dataSet = args['data']
    dimension = args['dim']
    threads = args['threads']
    print('data:{}\ndim:{}\nthreads:{}'.format(dataSet,dimension,threads))
    init()
    prepare()
    test()
