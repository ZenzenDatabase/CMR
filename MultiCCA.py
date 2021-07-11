import numpy as np
import keras
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Flatten, Input, Dense, Lambda, Dropout, LSTM, TimeDistributed
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from objectives import cca_loss
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint

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
    print("m={}, o={}; mean1={}, mean2={}; mean1 sample:{}, mean2 sample:{}".format(m, o, mean1.shape, mean2.shape, mean1[0], mean2[0]))
    SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar.T, H1bar) + r1 * np.identity(H1.shape[1])
    SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar.T, H2bar) + r2 * np.identity(H2.shape[1])
    print('SigmaHat11.shape, SigmaHat22.shape', SigmaHat11.shape, SigmaHat22.shape)
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
            xiext= np.repeat(xi, [size[0]], axis=0).astype('float32')
            xjext= np.tile(xj, (size[0], 1)).astype('float32')
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

############### model list  ################################
def myconcat1(List):
    x,y = List
    return tf.concat([x, y], 1)

def build_mlp_net(layer_sizes, input_size, reg_par):
    for l_id, ls in enumerate(layer_sizes):
        if l_id == 0:
            input_dim = input_size
            inputs = Input(shape=(input_dim,))
            x = BatchNormalization(name='')(inputs)
        
        if l_id == len(layer_sizes)-1:
            activation = 'linear'
        else:
            activation = 'sigmoid'
        x = Dense(ls, activation=activation, kernel_regularizer=l2(reg_par))(x)
    model = Model(inputs,x)
    return model

################# [Start SDCCA] #######################################
def creat_model_dcca(layer_sizes1, layer_sizes2, input_size1, input_size2, learning_rate, reg_par, outdim_size, use_all_singular_values, beta):
    """
    builds the whole model
    the structure of each sub-network is defined in build_mlp_net,
    and it can easily get substituted with a more efficient and powerful network like CNN
    """
    view1_model = build_mlp_net(layer_sizes1, input_size1, reg_par)
    view2_model = build_mlp_net(layer_sizes2, input_size2, reg_par)
    in_a = Input(shape=(input_size1,))
    in_b = Input(shape=(input_size2,))
    in_c = Input(shape=(1,))
    out_a= view1_model(in_a)
    out_b= view2_model(in_b)
    concat1 = Lambda(myconcat1)([out_a,out_b])
    concat2 = Lambda(myconcat1)([concat1,in_c])
    model = Model([in_a, in_b, in_c], concat2, name='all_model')
    model_optimizer = RMSprop(lr=0.001)
    model.compile(loss=cca_loss(outdim_size, True, 200, True, beta), optimizer=model_optimizer)
    return model

def train_model_dcca(model, data1, data2, data3, output_size, beta):
    train_set_x1, valid_set_x1, test_set_x1 = data1[0], data1[1], data1[2]
    train_set_x2, valid_set_x2, test_set_x2 = data2[0], data2[1], data2[2]
    train_set_x3, valid_set_x3, test_set_x3 = data3[0], data3[1], data3[2]
    checkpointer = ModelCheckpoint(filepath="temp_weights.h5", verbose=1, save_best_only=True, save_weights_only=True)
    model.fit([train_set_x1, train_set_x2, train_set_x3], np.zeros(len(train_set_x1)) ,epochs=1, \
              batch_size=200, validation_split=0.0, validation_data=None, shuffle=False)
    
    results = model.evaluate([train_set_x1, train_set_x2, train_set_x3], np.zeros(len(train_set_x1)), batch_size=200, verbose=1)
    print('loss on Train data: ', results)
    results = model.evaluate([test_set_x1, test_set_x2, test_set_x3], np.zeros(len(test_set_x1)), batch_size=200, verbose=1)
    print('loss on test data: ', results)
    results = model.evaluate([valid_set_x1, valid_set_x2, valid_set_x3], np.zeros(len(valid_set_x1)), batch_size=200, verbose=1)
    print('loss on Valid data: ', results)
    return model

def test_model_dcca(model, data1, data2, data3, output_size, beta, apply_linear_cca):
    new_data = []
    for k in range(3):
        pred_out = model.predict([data1[k], data2[k], data3[k]])
        r = int(pred_out.shape[1] / 2)
        print("pred_out:", pred_out.shape, r)
        new_data.append([pred_out[:, :r], pred_out[:, r:r*2]])

    if apply_linear_cca:
        w = [None, None]
        m = [None, None]
        print("Linear CCA started!")
        print(new_data[0][0].shape, new_data[0][1].shape, data3[0].shape)
        w[0], w[1], m[0], m[1] = linear_cca(new_data[0][0], new_data[0][1], data3[0], True, output_size, beta)
        print("Linear CCA ended!")

        for k in range(3):
            data_num = len(new_data[k][0])
            for v in range(2):
                new_data[k][v] -= m[v].reshape([1, -1]).repeat(data_num, axis=0)
                new_data[k][v] = np.dot(new_data[k][v], w[v])
                
    return new_data
################# [End SDCCA] #######################################

# sdcca
def create_model(output_size, beta):
    Output = 10
    def model_audio():
        inputs = Input(shape=(128,))
        x = BatchNormalization(name='')(inputs)
        x = Dense(128, activation='sigmoid')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='sigmoid')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='sigmoid')(x)
        x = Dropout(0.2)(x)
        x = Dense(Output, activation='linear')(x)
        model_audio = Model(inputs,x)
        return model_audio
    model_audio = model_audio()
    def model_rgb():
        inputs = Input(shape=(1024,))
        x = BatchNormalization(name='bn1')(inputs)
        x = Dense(256, activation='sigmoid')(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation='sigmoid')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='sigmoid')(x)
        x = Dropout(0.2)(x)
        x = Dense(Output, activation='linear')(x)
        model_rgb = Model(inputs,x)
        return model_rgb
    model_rgb = model_rgb()
    in_a = Input(shape=(128,))
    in_b = Input(shape=(1024,))
    in_c = Input(shape=(1,))
    out_a= model_audio(in_a)
    out_b= model_rgb(in_b)
    concat1 = Lambda(myconcat1)([out_a,out_b])
    concat2 = Lambda(myconcat1)([concat1,in_c])
    model = Model([in_a, in_b, in_c], concat2, name='all_model')
    model_optimizer = RMSprop(lr=0.001)
    model.compile(loss=linear_cca(output_size, False), optimizer=model_optimizer)
    return model

def train_model(data1, data2, data3, output_size, beta):
    train_set_x1, valid_set_x1, test_set_x1 = data1[0], data1[1], data1[2]
    train_set_x2, valid_set_x2, test_set_x2 = data2[0], data2[1], data2[2]
    train_set_x3, valid_set_x3, test_set_x3 = data3[0], data3[1], data3[2]
    model = creat_model_dcca(layer_sizes1, layer_sizes2, input_shape1, input_shape2, learning_rate, reg_par, outdim_size, use_all_singular_values, beta)
    print(model.summary())
    model.fit([train_set_x1, train_set_x2, train_set_x3], np.zeros(len(train_set_x1)) ,epochs=100, batch_size=200, validation_split=0.0,\
              validation_data=None, shuffle=False)
    filename = "weight.hdf5"
    model.save_weights(filename)
    return model

def load_model(output_size, beta):
    model = create_model(output_size, beta)
    model.load_weights("weight.hdf5")
    return model

def test_model(data1, data2, data3, trainDccaModel, output_size, beta):
    model = load_model(output_size, beta)
    new_data = []
    for k in range(3):
        pred_out = model.predict([data1[k], data2[k], data3[k]])

        r = int(pred_out.shape[1] / 2)
        print("pred_out:", pred_out.shape, r)
        new_data.append([pred_out[:, :r], pred_out[:, r:r*2]])
    print('new_data',np.array(new_data).shape)
    w = [None, None]
    m = [None, None]
    if (trainDccaModel):
        print("Linear SDCCA started!")
        w[0], w[1], m[0], m[1] = linear_cca(new_data[0][0], new_data[0][1], data3[0], True, output_size, beta)
        w0, w1, m0, m1 = w[0], w[1], m[0], m[1]
        np.savez("Model_SDCCA.npz", w0=w0, w1=w1, m0=m0, m1=m1)
        print("Linear SDCCA ended!")
    else:
        model = np.load("Model_SDCCA.npz")
        w[0] = model['w0']
        w[1] = model['w1']
        m[0] = model['m0']
        m[1] = model['m1']
    for k in range(3):
        data_num = len(new_data[k][0])
        for v in range(2):
            new_data[k][v] -= m[v].reshape([1, -1]).repeat(data_num, axis=0)
            new_data[k][v] = np.dot(new_data[k][v], w[v])
    return new_data

## dcca
def creating_model(output_size, beta):
    Output = 10
    print("Loading dcca model ......")
    def model_audio():
        inputs = Input(shape=(128,))
        x = BatchNormalization(name='tanh')(inputs)
        x = Dense(100, activation='sigmoid')(x)
        x = Dropout(0.1)(x)
        x = Dense(100, activation='sigmoid')(x)
        x = Dropout(0.1)(x)
        x = Dense(100, activation='sigmoid')(x)
        x = Dropout(0.1)(x)
        x = Dense(Output, activation='linear')(x)
        model_audio = Model(inputs,x)
        return model_audio
    model_audio = model_audio()
    def model_rgb():
        inputs = Input(shape=(1024,))
        x = BatchNormalization(name='bn1')(inputs)
        x = Dense(100, activation='sigmoid')(x)
        x = Dropout(0.1)(x)
        x = Dense(100, activation='sigmoid')(x)
        x = Dropout(0.1)(x)
        x = Dense(100, activation='sigmoid')(x)
        x = Dropout(0.1)(x)
        x = Dense(Output, activation='linear')(x)
        model_rgb = Model(inputs,x)
        return model_rgb
    model_rgb = model_rgb()
    in_a = Input(shape=(128,))
    in_b = Input(shape=(1024,))
    out_a= model_audio(in_a)
    out_b= model_rgb(in_b)
    concat1 = Lambda(myconcat1)([out_a,out_b])
    model = Model([in_a, in_b], concat1)
    model_optimizer = RMSprop(lr=0.0001)
    model.compile(loss=cca_loss(output_size, False, 200, False, beta), optimizer=model_optimizer)
    return model

def training_model(data1, data2, output_size, beta):
    train_set_x1, valid_set_x1, test_set_x1 = data1[0], data1[1], data1[2]
    train_set_x2, valid_set_x2, test_set_x2 = data2[0], data2[1], data2[2]

    model = creating_model(output_size, beta)
    model.summary()
    model.fit([train_set_x1, train_set_x2], np.zeros(len(train_set_x1)) ,epochs=50, batch_size=200, validation_split=0.0, validation_data=None, shuffle=False)
    filename = "weights.hdf5"
    model.save_weights(filename)
    results = model.evaluate([test_set_x1, test_set_x2], np.zeros(len(test_set_x1)), batch_size=100, verbose=1)
    print('loss on test data: ', results)
    return model

def loading_model(output_size, beta):
    model = creating_model(output_size, beta)
    model.load_weights("weights.hdf5")
    return model

def testing_model(data1, data2, output_size, beta):
    model = loading_model(output_size, beta) 
    new_data = []
    for k in range(3):
        pred_out = model.predict([data1[k], data2[k]])
        print("uuuuuuuuuuuui", len(pred_out), len(pred_out[0]))
        r = int(np.array(pred_out).shape[1] / 2)
        new_data.append([pred_out[:, :r], pred_out[:, r:r*2]])
    return new_data

### seg_lstm_cca
def created_model(output_size, beta):
    Output = 64
    def model_audio():
        inputs = Input(shape=(10,128))
        x = BatchNormalization(name='bn1')(inputs)
        x = LSTM(512, return_sequences=True)(x)
        x = Dropout(0.1)(x)
        x = LSTM(512, return_sequences=True)(x)
        x = Dropout(0.1)(x)
        x = TimeDistributed(Dense(128, activation="softmax"))(x)
        x = Flatten()(x)
        x = Dense(Output, activation="sigmoid")(x)
        model_audio = Model(inputs,x)
        return model_audio
    model_audio = model_audio()
    
    def model_rgb():
        inputs = Input(shape=(10,1024))
        x = BatchNormalization(name='bn2')(inputs)
        x = LSTM(784, return_sequences=True)(x)
        x = Dropout(0.1)(x)
        x = LSTM(784, return_sequences=True)(x)
        x = Dropout(0.1)(x)
        x = TimeDistributed(Dense(256, activation="softmax"))(x)
        x = Flatten()(x)
        x = Dense(Output, activation="sigmoid")(x)
        model_rgb = Model(inputs,x)
        return model_rgb
    model_rgb = model_rgb()
    in_a, in_b, in_c = Input(shape=(10,128)), Input(shape=(10,1024)), Input(shape=(1,))
    out_a, out_b = model_audio(in_a), model_rgb(in_b)
    concat1 = Lambda(myconcat1)([out_a,out_b])
    concat2 = Lambda(myconcat1)([concat1,in_c])
    model = Model([in_a, in_b, in_c], concat2, name='all_model')
    model_optimizer = RMSprop(lr=0.001)
    model.compile(loss=cca_loss(output_size, True, 200, True, beta), optimizer=model_optimizer)
    return model

def trained_model(data1, data2, data3, output_size, beta):
    train_set_x1, valid_set_x1, test_set_x1 = data1[0], data1[1], data1[2]
    train_set_x2, valid_set_x2, test_set_x2 = data2[0], data2[1], data2[2]
    train_set_x3, valid_set_x3, test_set_x3 = data3[0], data3[1], data3[2]
    model = created_model(output_size, beta)
    model.fit([train_set_x1, train_set_x2, train_set_x3], np.zeros(len(train_set_x1)) ,epochs=50, batch_size=200, validation_split=0.0, validation_data=None, shuffle=False)
    filename = "weighting.hdf5"
    model.save_weights(filename)
    test_results = model.evaluate([test_set_x1, test_set_x2, test_set_x3], np.zeros(len(test_set_x1)), batch_size=100, verbose=1)
    print('loss on test data: ', test_results)
    return model

def loaded_model(output_size, beta):
    model = created_model(output_size, beta)
    model.load_weights("weight.hdf5")
    return model

def tested_model(data1, data2, data3, trainDccaModel, output_size, beta):
    model = loaded_model(output_size, beta)
    new_data = []
    for k in range(3):
        pred_out = np.array(model.predict([data1[k], data2[k], data3[k]]))
        r = int(pred_out.shape[1] / 2)
        new_data.append([pred_out[:, :r], pred_out[:, r:r*2]])
    w = [None, None]
    m = [None, None]
    if (trainDccaModel):
        print("Linear CCA started!")
        w[0], w[1], m[0], m[1] = linear_cca(new_data[0][0], new_data[0][1], data3[0], False, output_size, beta)
        w0, w1, m0, m1 = w[0], w[1], m[0], m[1]
        np.savez("Model_DCCA_CCA.npz", w0=w0, w1=w1, m0=m0, m1=m1)
        print("Linear CCA ended!")
    else:
        model = np.load("Model_DCCA_CCA.npz")
        w[0] = model['w0']
        w[1] = model['w1']
        m[0] = model['m0']
        m[1] = model['m1']
    for k in range(3):
        data_num = len(new_data[k][0])
        for v in range(2):
            new_data[k][v] -= m[v].reshape([1, -1]).repeat(data_num, axis=0)
            new_data[k][v] = np.dot(new_data[k][v], w[v])
    return new_data


####################### Triplet moel ########################
def build_model(num_users, num_items, latent_dim):

    positive_item_input = Input((1, ), name='positive_item_input')
    negative_item_input = Input((1, ), name='negative_item_input')

    # Shared embedding layer for positive and negative items
    item_embedding_layer = Embedding(
        num_items, latent_dim, name='item_embedding', input_length=1)

    user_input = Input((1, ), name='user_input')

    positive_item_embedding = Flatten()(item_embedding_layer(
        positive_item_input))
    negative_item_embedding = Flatten()(item_embedding_layer(
        negative_item_input))
    user_embedding = Flatten()(Embedding(
        num_users, latent_dim, name='user_embedding', input_length=1)(
            user_input))

    loss = merge(
        [positive_item_embedding, negative_item_embedding, user_embedding],
        mode=bpr_triplet_loss,
        name='loss',
        output_shape=(1, ))

    model = Model(
        input=[positive_item_input, negative_item_input, user_input],
        output=loss)
    model.compile(loss=identity_loss, optimizer=Adam())

    return model
