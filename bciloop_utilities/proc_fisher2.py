import numpy as np


# Input:
#   - P             data matrix []
#   - Pk            label vector (Only two classes allowed[samplesx1])
#   - NSTD          Outliers removal. Number of standard deviation to
#                   consider a sample as outlier [default: empty list]
#   - do_balance    Balancing classes [default: false]
#
#   It returns a vector F of fisher score. The vector is in the format
#   [(channelsxfrequencies) x 1]

def proc_fisher2(P, Pk, nstd=[], do_balance=False):

    classes = np.unique(Pk)
    nclasses = len(classes)

    if(nclasses != 2):
        raise ValueError("The number of classes must be two.")

    if(np.shape(P)[0] != len(Pk)):
        raise ValueError("First dimension of P and length of Pk must be the same.")

    F = np.zeros((np.shape(P)[1], np.shape(P)[2]))

    # for all bands
    for bId in range(np.shape(P)[2]):
        #for all csp dim
        for cId in range(np.shape(P)[1]):

            #get data for given class and given feature (channel)
            cdata1 = P[np.any([Pk == classes[0]], axis=0), cId, bId] 
            cdata2 = P[np.any([Pk == classes[1]], axis=0), cId, bId]

            # If rmsize is provided, remove outliersper class
            #if not nstd:
                #to do

            #computing mean and standard deviation of each class
            m1 = np.mean(cdata1)
            s1 = np.std(cdata1)

            m2 = np.mean(cdata2)
            s2 = np.std(cdata2)

            #computing feature score for the given feature
            F[cId, bId] = np.absolute(m2-m1) / np.sqrt(s1**2 + s2**2)

    return F
