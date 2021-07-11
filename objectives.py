import tensorflow as tf
from keras import backend as K
import numpy as np


def cca_loss(outdim_size, use_all_singular_values, batch_size, use_cluster, beta):
    """
    The main loss function (inner_cca_objective) is wrapped in this function due to
    the constraints imposed by Keras on objective functions
    """

    def inner_cca_objective_1loop(y_true, y_pred):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        It is implemented by Theano tensor operations, and does not work on Tensorflow backend
        y_true is just ignored
        """

        r1 = 1e-4
        r2 = 1e-4
        #o1 = o2 = y_pred.get_shape().as_list()[1]//2
        o1 = o2 = int(y_pred.get_shape()[1]//2)      # m x (o1+o2)
        # print(y_pred.get_shape())

        m = batch_size
        # m = tf.shape(y_pred)[0]
        # print("[%d, %d, %d]" % (m, o1+o2, y_pred.get_shape()[1]))
        H1 = y_pred[:, 0:o1]
        H2 = y_pred[:, o1:o1+o2]
        H1bar = H1 - tf.reduce_mean(H1,0,keep_dims=True)
        H2bar = H2 - tf.reduce_mean(H2,0,keep_dims=True)

        SigmaHat11 = (1.0/(m-1)) * tf.matmul(tf.transpose(H1bar), H1bar) + r1 * K.eye(o1)
        SigmaHat22 = (1.0/(m-1)) * tf.matmul(tf.transpose(H2bar), H2bar) + r2 * K.eye(o2)
        
        if (use_cluster):
            y_pred_c = tf.concat([H1bar, H2bar, y_pred[:,o1+o2:]], 1)
            # one-loop
            FEATDIM = o1
            CATDIM  = o1+o2
            NUMCAT  = 10
        
            cs = tf.zeros(shape=[FEATDIM,FEATDIM], dtype=tf.float32)  # tf.Variable(tf.constant([0, 0]))
            cat= tf.constant(0, dtype=tf.int32)   # tf.Variable(tf.constant(0))
            count = tf.constant(0, dtype=tf.int32)   # tf.Variable(tf.constant(0))
            def condition(cat, dat, cs, count):
                return cat<NUMCAT
            def body(cat, dat, cs, count):
                N = batch_size
                flag = tf.equal(dat[:N,CATDIM], tf.cast(cat,tf.float32))
                size = tf.reduce_sum(tf.cast(flag, tf.int32))
        
                index = tf.where(flag)
                index = tf.squeeze(index)
                xi = tf.gather(dat[:,:FEATDIM], index)
                xj = tf.gather(dat[:,FEATDIM:CATDIM], index)
        
                # xiext= tf.repeat(xi, [size], axis=0)
                xi2 = tf.reshape(tf.transpose(xi), [1, tf.multiply(size,FEATDIM)])
                xi3 = tf.tile(xi2, [size,1])
                xi4 = tf.reshape(xi3, [tf.cast(tf.multiply(FEATDIM,size), tf.int32), tf.cast(size, tf.int32)])
                xiext = tf.reshape(tf.transpose(xi4), [tf.cast(tf.multiply(size,size), tf.int32), tf.cast(FEATDIM, tf.int32)])
        
                # xjext= tf.tile(xj, [size,1])
                xj2 = tf.reshape(xj, [1, tf.multiply(size,FEATDIM)])
                xj3 = tf.tile(xj2, [size,1])
                xjext = tf.reshape(xj3, [tf.cast(tf.multiply(size,size), tf.int32), tf.cast(FEATDIM,tf.int32)])
        
                cs += tf.matmul(tf.transpose(xiext), xjext)
                count += size*size     
                # loop control
                cat= tf.add(cat,1)
                return cat, dat, cs, count
            result = tf.while_loop(condition, body, [cat, y_pred_c, cs, count])
            SigmaHat12_c = tf.div(result[2], tf.cast(result[3], tf.float32))

            SigmaHat12_r = (1.0/(m-1)) * tf.matmul(tf.transpose(H1bar), H2bar)
##            print "pairwise, category-base;",SigmaHat12_r, SigmaHat12_c
            SigmaHat12 = beta * SigmaHat12_r + (1-beta) * SigmaHat12_c

        else:
            SigmaHat12 = (1.0/(m-1)) * tf.matmul(tf.transpose(H1bar), H2bar)


        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = tf.self_adjoint_eig(SigmaHat11)
        [D2, V2] = tf.self_adjoint_eig(SigmaHat22)
        SigmaHat11RootInv = tf.matmul(tf.matmul(V1, tf.diag(D1 ** -0.5)), tf.transpose(V1))
        SigmaHat22RootInv = tf.matmul(tf.matmul(V2, tf.diag(D2 ** -0.5)), tf.transpose(V2))

        Tval = tf.matmul(tf.matmul(SigmaHat11RootInv,SigmaHat12), SigmaHat22RootInv)

        if use_all_singular_values:
            # all singular values are used to calculate the correlation
            corr = tf.sqrt(tf.trace(tf.matmul(tf.transpose(Tval), Tval)))
        else:
            # just the top outdim_size singular values are used
            [U, V] = tf.self_adjoint_eig(tf.matmul(tf.transpose(Tval), Tval))
            idx = (U.argsort())[0:outdim_size]
            corr = tf.reduce_sum(tf.sqrt(U[idx]))

        return -corr

    return inner_cca_objective_1loop

def linear_cca(H1, H2, outdim_size):
    """
    An implementation of linear CCA
    # Arguments:
        H1 and H2: the matrices containing the data for view 1 and view 2. Each row is a sample.
        outdim_size: specifies the number of new features
    # Returns
        A and B: the linear transformation matrices 
        mean1 and mean2: the means of data for both views
    """
    r1 = 1e-4
    r2 = 1e-4

    m = H1.shape[0]
    o = H1.shape[1]

    mean1 = numpy.mean(H1, axis=0)
    mean2 = numpy.mean(H2, axis=0)
    H1bar = H1 - numpy.tile(mean1, (m, 1))
    H2bar = H2 - numpy.tile(mean2, (m, 1))

    SigmaHat12 = (1.0 / (m - 1)) * numpy.dot(H1bar.T, H2bar)
    SigmaHat11 = (1.0 / (m - 1)) * numpy.dot(H1bar.T, H1bar) + r1 * numpy.identity(o)
    SigmaHat22 = (1.0 / (m - 1)) * numpy.dot(H2bar.T, H2bar) + r2 * numpy.identity(o)

    [D1, V1] = numpy.linalg.eigh(SigmaHat11)
    [D2, V2] = numpy.linalg.eigh(SigmaHat22)
    SigmaHat11RootInv = numpy.dot(numpy.dot(V1, numpy.diag(D1 ** -0.5)), V1.T)
    SigmaHat22RootInv = numpy.dot(numpy.dot(V2, numpy.diag(D2 ** -0.5)), V2.T)

    Tval = numpy.dot(numpy.dot(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

    [U, D, V] = numpy.linalg.svd(Tval)
    V = V.T
    A = numpy.dot(SigmaHat11RootInv, U[:, 0:outdim_size])
    B = numpy.dot(SigmaHat22RootInv, V[:, 0:outdim_size])
    D = D[0:outdim_size]

    return A, B, mean1, mean2