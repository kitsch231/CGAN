import random
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

def get_data(nums):
    gmodel=load_model('./model/g_model')
    noise = np.random.normal(0, 1, (nums,100))
    rannum=[random.randint(0,39) for x in range(nums)]
    label=np.array(rannum).reshape(-1, 1)
    print([noise.shape,label.shape])
    gen=gmodel.predict([noise,label])
    data=gen.reshape(nums,41)
    data=pd.DataFrame(data)
    data[41]=label
    print(data.info())
    data.to_csv('生成数据.csv',index=None)
    return data

get_data(500000)