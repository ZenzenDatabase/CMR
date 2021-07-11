import numpy as np

def load_fold_ids(fold_path):
    fold_file = open(fold_path)
    return fold_file.read().splitlines()

def linear_cca(H1, H2, cat, use_cluster, outdim_size, beta):
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

    mean1 = np.mean(H1, axis=0)
    mean2 = np.mean(H2, axis=0)
    H1bar = H1 - np.tile(mean1, (m, 1))
    H2bar = H2 - np.tile(mean2, (m, 1))

    SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar.T, H1bar) + r1 * np.identity(H1.shape[1])
    SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar.T, H2bar) + r2 * np.identity(H2.shape[1])
    #print m, o, H1bar.shape, H2bar.shape, SigmaHat11.shape, SigmaHat22.shape
    if (use_cluster):
        FEATDIM = o # 1-demension
        NUMCAT  = 10 # classes
        corr = np.zeros([FEATDIM,FEATDIM], dtype=np.float32)
        num  = 0
        for c in range(NUMCAT):
            flag = (cat==c)
            flags = np.array([int(fg) for fg in flag])
            xi = H1bar[flags,:]     # use centered value
            xj = H2bar[flags,:]
            size = xi.shape
            xiext= np.repeat(xi, [size[0]], axis=0)
            xjext= np.tile(xj, (size[0], 1))
            corr += np.matmul(np.transpose(xiext), xjext)
            num += size[0]*size[0]

        SigmaHat12_c = corr/num
        SigmaHat12_r = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)
        SigmaHat12 = beta * SigmaHat12_r + (1-beta) * SigmaHat12_c
    else:
        SigmaHat12 = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)


    [D1, V1] = np.linalg.eig(SigmaHat11)
    [D2, V2] = np.linalg.eig(SigmaHat22)
    SigmaHat11RootInv = np.dot(np.dot(V1, np.diag(D1 ** -0.5)), V1.T)
    SigmaHat22RootInv = np.dot(np.dot(V2, np.diag(D2 ** -0.5)), V2.T)

    Tval = np.dot(np.dot(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

    [U, D, V] = np.linalg.svd(Tval)
    V = V.T
    A = np.dot(SigmaHat11RootInv, U[:, 0:outdim_size])
    B = np.dot(SigmaHat22RootInv, V[:, 0:outdim_size])
    D = D[0:outdim_size]

    return A, B, mean1, mean2

def linearccca_emb(train_audio_rgb_label, test_audio_rgb_label, output_size, beta):
    audioTrain, rgbTrain, labelTrain = train_audio_rgb_label
    audioTest,  rgbTest,  labelTest  = test_audio_rgb_label

    print("Linear CCA started!")
    w0, w1, m0, m1 = linear_cca(audioTrain, rgbTrain, labelTrain, False, output_size, beta)
    np.savez('Model_Cc_CCA.npz', w0=w0, w1=w1, m0=m0, m1=m1)
    print("Linear CCA ended!")

    data_num = len(audioTest)
    audioTest -= m0.reshape([1, -1]).repeat(data_num, axis=0)
    audioFeatTest = np.dot(audioTest, w0)
    rgbTest -= m1.reshape([1, -1]).repeat(data_num, axis=0)
    rgbFeatTest = np.dot(rgbTest, w1)
    print("Embedding the train data!", audioFeatTest.shape, rgbFeatTest.shape, labelTest.shape)
    return (audioFeatTest, rgbFeatTest), labelTest

def cross_fold_generate(path):
    file_list = os.listdir(path)

    for i in range(1):
        # train
        train_emb = "train_{0:02d}s".format(i)
        ftr = h5py.File(path + train_emb + ".h5", 'r')
        train_audio = ftr["featN1"]
        train_rgb = ftr["featN2"]
        train_lab = ftr["testLs"]
        train_lab = np.asarray([int(labb) for labb in train_lab])

        # test
        test_emb = "test_{0:02d}s".format(i)
        ff = h5py.File(path + test_emb + ".h5", 'r')
        test_audio, test_rgb = ff["featN1"], ff["featN2"]
        test_lab = ff["testLs"]
        print("train, test shape like {},{},{}; {},{},{}".format(train_audio.shape, train_rgb.shape, train_lab.shape,
                                                                 test_audio.shape, test_rgb.shape, test_lab.shape))
        print(type(train_audio), type(train_rgb), type(train_lab), type(test_audio), type(test_rgb), type(test_lab))
        yield np.array(train_audio, dtype=np.float32), np.array(train_rgb, dtype=np.float32), np.array(train_lab,
                                                                                                       dtype=np.float32), np.array(
            test_audio, dtype=np.float32), np.array(test_rgb, dtype=np.float32), np.array(test_lab, dtype=np.float32)

i=0
for data_xy in cross_fold_generate(path):
    train_audio, train_rgb, train_lab, test_audio, test_rgb, test_lab = data_xy
    trained_anchor = Model(inputs=anchor_input, outputs=encoded_anchor)
    trained_pos_neg = Model(inputs=positive_input, outputs=encoded_positive)

    trained_anchor.load_weights("encoded_anchor"+str(i)+".hdf5")
    trained_pos_neg.load_weights("encoded_pos_neg"+str(i)+".hdf5")

    test_audio = trained_anchor.predict(test_audio)
    test_rgb = trained_pos_neg.predict(test_rgb)
    train_audio = trained_anchor.predict(train_audio)
    train_rgb = trained_pos_neg.predict(train_rgb)

    train_audio_rgb_label = train_audio, train_rgb, train_lab
    test_audio_rgb_label  = test_audio,  test_rgb,  test_lab
    output_size = 10
    beta        = 0.2
    (audioFeatTest, rgbFeatTest), labelTest = linearccca_emb(train_audio_rgb_label, test_audio_rgb_label, output_size, beta)

    print("The output of test:", test_audio.shape, test_rgb.shape)
    views = test_audio, test_rgb
    corr_matrix = utils.corr_compute(views, tag="cosine")
    result_list = utils.metric(corr_matrix, test_lab)
    print(result_list)
