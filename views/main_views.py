from flask import Flask, Blueprint, request
import pandas as pd
import os

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from torch.autograd import Variable

bp = Blueprint("main", __name__, url_prefix="/")

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes=num_classes
        self.num_layers=num_layers
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.seq_length=seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        self.fc_1 = nn.Linear(hidden_size, 256)
        self.fc = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        print("x: ", x)
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out

@bp.route("dust_info")
def dust_info():

    temperature = request.args.get('temperature')
    precipitation = request.args.get('precipitation')
    wind_speed = request.args.get('wind_speed')
    humidity = request.args.get('humidity')

    test = pd.DataFrame({
    '평균기온(°C)':[temperature],
    '일강수량(mm)':[precipitation],
    '평균 풍속(m/s)':[wind_speed],
    '평균 상대습도(%)':[humidity]
    })

    print(test.info())
    test = test.astype("float64")
    print(test.info())

    ss = StandardScaler()

    X_act = test
    X_act_ss = ss.fit_transform(X_act)
    X_act_tensors = torch.Tensor(X_act_ss)
    X_act_tensors_f = torch.reshape(X_act_tensors, (X_act_tensors.shape[0], 1, X_act_tensors.shape[1]))

    lr = 0.0001

    input_size = 4
    hidden_size = 8
    num_layers = 1

    num_classes = 2

    path = os.getcwd().replace('\\', '/') + '/pybo/static/models/10m_test.h5'
    
    model = LSTM(num_classes, input_size, hidden_size, num_layers, 365)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.train()

    predict = model(X_act_tensors_f).tolist()
    dict = {'PM10':int(predict[0][0]*7*100), 'PM2.5':int(predict[0][1]*1.55*100)}

    return dict
