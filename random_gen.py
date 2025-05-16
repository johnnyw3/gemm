import random, sys

def gen_matrix(fname, n):
    '''
    '''
    res = []

    for idx_x in range(n):
        for idx_y in range(n):
            res.append(random.randint(0, 100))


    with open(fname, "w") as fp:
        fp.write("%d\n" % (n))
        for num in res:
            fp.write("%d " % (num))

if __name__ == '__main__':
    gen_matrix(sys.argv[1], int(sys.argv[2]))
