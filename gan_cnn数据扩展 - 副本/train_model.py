
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam,RMSprop
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
import os
import numpy as np


class CGAN():
    def __init__(self):
        # 输入shape
        self.img_rows =41
        self.channels = 1
        self.img_shape = (self.img_rows,self.channels)
        # 分40类
        self.num_classes = 40
        self.latent_dim =100
        # adam优化器
        doptimizer =Adam(0.00001,clipvalue=1,decay=1e-8)
        ganoptimizer =Adam(0.00001,clipvalue=1, decay=1e-8)
        # 判别模型
        losses = ['binary_crossentropy']
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
                                   optimizer=doptimizer,
                                   metrics=['accuracy'])

        # 生成模型
        self.generator = self.build_generator()

        # conbine是生成模型和判别模型的结合
        # 判别模型的trainable为False
        # 用于训练生成模型
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])
        self.discriminator.trainable = False
        valid= self.discriminator([img,label])

        self.combined = Model([noise, label],valid)
        self.combined.compile(loss=losses,
                              optimizer=ganoptimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        #你自己加卷积试试，效果不好我没实验了
        model.add(Reshape((128,2)))
        model.add(Conv1D(128,2))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(64,2))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(32,2))
        model.add(MaxPooling1D(3))
        model.add(Flatten())
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        # 输入一个数字，将其转换为固定尺寸的稠密向量
        # 输出维度是self.latent_dim
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        # 将正态分布和索引对应的稠密向量相乘
        noise = Input(shape=(self.latent_dim,))
        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()
        model.add(Dense(128,input_shape=(41,)))
        #你自己加卷积试试，效果不好我没实验了
        # model.add(Reshape((64,2)))
        # model.add(Conv1D(32,2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=(41,1))  # 输入 （28，28，1）
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(40, np.prod((41,1)))(label))
        flat_img = Flatten()(img)
        model_input = multiply([flat_img, label_embedding])
        validity = model(model_input)  # 把 label 和 G(z) embedding 在一起，作为 model 的输入
        #label = Dense(self.num_classes, activation="softmax")(features)

        return Model([img,label],validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # 载入数据库
        X_train=pd.read_csv('./data/train_x.csv')
        #X_train=X_train.iloc[:1000,:]
        X_train=np.array(X_train)

        y_train=pd.read_csv('./data/train_y.csv')
        #y_train = y_train.iloc[:1000, :]
        y_train=np.array(y_train)
        y_train=[np.argmax(m) for m in y_train]
        y_train=np.array(y_train)

        # 归一化
        #X_train = np.expand_dims(X_train, axis=3)
        X_train= X_train.reshape(X_train.shape[0],X_train.shape[1],1)

        y_train = y_train.reshape(-1, 1)
        #print(y_train)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        aclist=[]
        dllist=[]
        gllist=[]
        for epoch in range(epochs):

            # --------------------- #
            #  训练鉴别模型
            # --------------------- #
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]
            # ---------------------- #
            #   生成正态分布的输入
            # ---------------------- #
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            #print([noise.shape, labels.shape])
            gen_imgs = self.generator.predict([noise,labels])
            d_loss_real = self.discriminator.train_on_batch([imgs,labels],valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs,labels],fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # --------------------- #
            #  训练生成模型
            # --------------------- #
            sampled_labels = np.random.randint(0, 40, (batch_size, 1))
            g_loss = self.combined.train_on_batch([noise, sampled_labels],valid)

            aclist.append(d_loss[1])
            dllist.append(d_loss[0])
            gllist.append(g_loss)

            if epoch % 100 == 0:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
                self.discriminator.save('./model/d_model')
                self.generator.save('./model/g_model')
            if epoch % sample_interval == 0:
                pca_img=imgs.reshape(imgs.shape[0],imgs.shape[1])
                pca_gen = gen_imgs.reshape(gen_imgs.shape[0],gen_imgs.shape[1])
                pca1 = PCA(n_components=2)
                pca2 = PCA(n_components=2)
                sc_img=pca1.fit_transform(pca_img)
                sc_gen=pca2.fit_transform(pca_gen)
                sc_img=pd.DataFrame(sc_img)
                sc_gen= pd.DataFrame(sc_gen)
                # print(pca_img)
                # print(pca_gen)
                # print(sc_img)
                # print(sc_gen)
                plt.scatter(sc_img[0],sc_img[1],label='真实')
                plt.scatter(sc_gen[0], sc_gen[1],label='生成')
                plt.legend()
                plt.savefig('./images/{}.png'.format(epoch))
                plt.cla()
                # plt.show()
        plt.figure()
        plt.plot([n for n in range(epochs)],aclist)
        plt.title('d_acc')
        plt.savefig('d_acc.png')

        plt.figure()
        plt.plot([n for n in range(epochs)],dllist)
        plt.title('d_loss')
        plt.savefig('d_loss.png')

        plt.figure()
        plt.plot([n for n in range(epochs)],gllist)
        plt.title('g_loss')
        plt.savefig('g_loss.png')
if __name__ == '__main__':
    if not os.path.exists("./images"):
        os.makedirs("./images")
    cgan = CGAN()
    cgan.train(epochs=25000, batch_size=256, sample_interval=200)