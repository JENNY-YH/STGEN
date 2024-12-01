# -*- coding: utf-8 -*-
import os
import math
import random
import time
import shutil
import pickle
import logging
import datetime as dt
import numpy as np
import osmnx as ox
import pandas as pd
import os.path as osp
import networkx as nx
import xgboost as xgb
import matplotlib.pyplot as plt
import copy

import torch
import torch_geometric
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from torch import Tensor
from torch.nn import Parameter
from torch_geometric.io import read_npz
from torch_geometric.nn import Node2Vec
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform, zeros
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size
from torch_geometric.data import Data, DataLoader, InMemoryDataset, download_url

from sklearn.model_selection import GridSearchCV  # 新增导入

from pylab import cm
from matplotlib import colors
from IPython.display import clear_output
from xgboost.sklearn import XGBClassifier
from typing import Union, Tuple, Callable, Optional
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score

np.random.seed(7)
torch.manual_seed(7)
device = torch.device('cuda:0')

# 字典映射
us_state_to_abbrev = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC", "American Samoa": "AS",
    "Guam": "GU", "Northern Mariana Islands": "MP", "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM", "U.S. Virgin Islands": "VI",
}

us_abbrev_to_state = dict(map(reversed, us_state_to_abbrev.items()))

def read_npz(path):
    with np.load(path, allow_pickle=True) as f:
        return parse_npz(f)

import datetime as dt

def parse_npz(f):
    crash_time = f['crash_time']
    x = torch.from_numpy(f['x']).to(torch.float)
    coords = torch.from_numpy(f['coordinates']).to(torch.float)
    edge_attr = torch.from_numpy(f['edge_attr']).to(torch.float)
    cnt_labels = torch.from_numpy(f['cnt_labels']).to(torch.long)
    occur_labels = torch.from_numpy(f['occur_labels']).to(torch.long)
    edge_attr_dir = torch.from_numpy(f['edge_attr_dir']).to(torch.float)
    edge_attr_ang = torch.from_numpy(f['edge_attr_ang']).to(torch.float)
    severity_labels = torch.from_numpy(f['severity_8labels']).to(torch.long)
    edge_index = torch.from_numpy(f['edge_index']).to(torch.long).t().contiguous()

    crash_hours = []
    for sublist in crash_time:
        for item in sublist:
            if isinstance(item, list):
                for subitem in item:
                    if subitem:
                        try:
                            dt_obj = dt.datetime.strptime(subitem.split('.')[0], '%Y-%m-%d %H:%M:%S')
                            crash_hours.append(dt_obj.hour)
                        except ValueError as e:
                            print(f"Error parsing datetime: {subitem}, error: {e}")
            else:
                if item:
                    try:
                        dt_obj = dt.datetime.strptime(item.split('.')[0], '%Y-%m-%d %H:%M:%S')
                        crash_hours.append(dt_obj.hour)
                    except ValueError as e:
                        print(f"Error parsing datetime: {item}, error: {e}")

    crash_hours = np.array(crash_hours, dtype=int)

    if len(crash_hours) < x.size(0):
        padding = np.zeros(x.size(0) - len(crash_hours), dtype=int)
        crash_hours = np.concatenate([crash_hours, padding])
    elif len(crash_hours) > x.size(0):
        crash_hours = crash_hours[:x.size(0)]

    crash_time_counts = np.bincount(crash_hours)
    time_weights = crash_time_counts[crash_hours] / crash_time_counts.max()
    time_weights = torch.tensor(time_weights, dtype=torch.float).unsqueeze(1)

    return Data(x=x, y=occur_labels, severity_labels=severity_labels, edge_index=edge_index,
                edge_attr=edge_attr, edge_attr_dir=edge_attr_dir, edge_attr_ang=edge_attr_ang,
                coords=coords, cnt_labels=cnt_labels, crash_time=crash_hours, time_weights=time_weights)

def train_test_split_stratify(dataset, train_ratio, val_ratio, class_num):
    labels = dataset[0].y
    train_mask = torch.zeros(size=labels.shape, dtype=bool)
    val_mask = torch.zeros(size=labels.shape, dtype=bool)
    test_mask = torch.zeros(size=labels.shape, dtype=bool)
    for i in range(class_num):
        stratify_idx = np.argwhere(labels.numpy() == i).flatten()
        np.random.shuffle(stratify_idx)
        split1 = int(len(stratify_idx) * train_ratio)
        split2 = split1 + int(len(stratify_idx) * val_ratio)
        train_mask[stratify_idx[:split1]] = True
        val_mask[stratify_idx[split1:split2]] = True
        test_mask[stratify_idx[split2:]] = True
    highest = pd.DataFrame(labels).value_counts().head().iloc[0]
    return train_mask, val_mask, test_mask

class TRAVELDataset(InMemoryDataset):
    url = 'D:/zyh/travel-main/TAP-city/{}.npz'

    def __init__(self, root: str, name: str, transform=None, pre_transform=None):
        self.name = name.lower()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        local_path = self.url.format(self.name)
        print(f"当前城市名称：{self.name}")
        dest_path = os.path.join(self.raw_dir, self.raw_file_names)
        if not os.path.exists(dest_path):
            if not os.path.exists(self.raw_dir):
                os.makedirs(self.raw_dir)
            shutil.copy(local_path, dest_path)

    def process(self):
        data = read_npz(self.raw_paths[0])
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}Full()'

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

@torch.no_grad()
def test(model, data):
    model.eval()
    logits, measures = model().detach(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        mea = f1_score(data.y[mask].cpu(), pred.cpu(), average='binary')
        measures.append(mea)

    label_pred = logits.max(1)[1]
    mask = data.test_mask
    scores = logits[mask][:, 1]
    pred = logits[mask].max(1)[1]
    test_y = data.y[mask]

    test_acc = pred.eq(test_y).sum().item() / mask.sum().item()
    test_map = average_precision_score(test_y.cpu(), scores.cpu())
    test_auc = roc_auc_score(test_y.cpu(), scores.cpu())
    return measures, label_pred, test_acc, test_map, test_auc

log_folder = 'D:/zyh/travel-main/log_train'
os.makedirs(log_folder, exist_ok=True)
current_time = dt.datetime.now().strftime('%Y%m%d_%H%M')
log_file = os.path.join(log_folder, f'training_log_{current_time}.txt')

logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])



def train_loop(model, data, optimizer, num_epochs, model_name='', city_name=''):
    epochs, train_measures, valid_measures, test_measures, test_accs, test_maps, test_aucs = [], [], [], [], [], [], []
    best_train_f1, best_valid_f1, best_test_f1, best_test_auc, best_test_acc, best_test_map = 0, 0, 0, 0, 0, 0
    logging.info(f'当前城市：{city_name}')
    for epoch in range(num_epochs):  # 遍历所有训练周期
        train(model, data, optimizer)  # 训练模型
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        measures, label_pred, test_acc, test_map, test_auc = test(model, data)  # 测试模型
        train_mea, valid_mea, test_mea = measures  # 获取训练、验证和测试的F1得分

        # 更新最佳指标
        if train_mea > best_train_f1:
            best_train_f1 = train_mea
        if valid_mea > best_valid_f1:
            best_valid_f1 = valid_mea
        if test_mea > best_test_f1:
            best_test_f1 = test_mea
        if test_auc > best_test_auc:
            best_test_auc = test_auc
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        if test_map > best_test_map:
            best_test_map = test_map

        epochs.append(epoch)  # 记录当前周期
        train_measures.append(train_mea)  # 记录训练F1得分
        valid_measures.append(valid_mea)  # 记录验证F1得分
        test_measures.append(test_mea)  # 记录测试F1得分
        test_aucs.append(test_auc)  # 记录测试AUC
        test_accs.append(test_acc)  # 记录测试准确率
        test_maps.append(test_map)  # 记录测试MAP

        logging.info(
            f'Epoch: {epoch + 1:03d}, Train F1: {train_mea:.4f}, Val F1: {valid_mea:.4f}, Test F1: {test_mea:.4f}, '
            f'Test AUC: {test_auc:.4f}, Test ACC: {test_acc:.4f}, Test MAP: {test_map:.4f}')
        logging.info(
            f'Best so far - Train F1: {best_train_f1:.4f}, Val F1: {best_valid_f1:.4f}, Test F1: {best_test_f1:.4f},'
            f' Test AUC: {best_test_auc:.4f}, Test ACC: {best_test_acc:.4f}, Test MAP: {best_test_map:.4f}')


        if epoch == 400:  # 每2000个周期显示一次结果
            fig, (ax1, ax) = plt.subplots(1, 2, figsize=(30, 12))  # 创建子图
            gdf_pred['label'] = label_pred.cpu().numpy()  # 获取预测标签
            for i in range(class_num):  # 遍历所有类别
                G = nx.MultiGraph()  # 创建多重图
                G.add_nodes_from(gdf_pred[gdf_pred['label'] == i].index)  # 添加节点
                sub1 = nx.draw(G, pos=pos_dict, ax=ax1, node_color=color_ls[i], node_size=10)  # 绘制节点

            ax.text(1, 1, log.format(epoch, train_measures[-1], valid_measures[-1], test_measures[-1]),
                    fontsize=18)  # 添加文本
            ax.plot(epochs, train_measures, "r", epochs, valid_measures, "g", epochs, test_measures, "b")  # 绘制曲线
            ax.set_ylim([0, 1])  # 设置y轴范围
            ax.legend(["train", "valid", "processed"])  # 添加图例
            ax1.legend(["Negative", "Positive"])  # 添加图例
            ax1.set_title(city_name + ' ' + model_name, y=-0.01)  # 设置标题

            # 保存图片到log-train文件夹
            save_dir = 'D:/zyh/travel-main/log_train'
            os.makedirs(save_dir, exist_ok=True)
            timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M')
            save_path = os.path.join(save_dir, f'SuperGEN_{city_name}_{timestamp}.png')
            plt.savefig(save_path)
            plt.close(fig)  # 关闭图形
            print(f"Saved figure: {save_path}")


    final_test_mea = best_test_f1
    final_test_auc = best_test_auc  # 最佳测试AUC
    final_test_acc = best_test_acc  # 最佳测试准确率
    final_test_map = best_test_map  # 最佳测试MAP

    print('BEST F1 {:.5f} | AUC {:.5f} | Test Acc {:.5f} | MAP {:.5f}'.format(final_test_mea, final_test_auc, final_test_acc,
                                                                         final_test_map))  # 打印最终结果

    logging.info('Final results:')
    logging.info(
        'BEST F1 {:.5f} | AUC {:.5f} | Test Acc {:.5f} | MAP {:.5f}'.format(final_test_mea, final_test_auc, final_test_acc,
                                                                       final_test_map))  # 打印最终结果

    return (round(final_test_mea * 100, 2), round(final_test_auc * 100, 2), round(final_test_acc * 100, 2),
            round(final_test_map * 100, 2))  # 返回最终结果



def draw_with_labels(df_nodes, model_name='processed'):
    plt.figure(figsize=(6, 5))
    for i in range(class_num):
        G = nx.MultiGraph()
        G.add_nodes_from(df_nodes[df_nodes['label'] == i].index)
        nx.draw(G, pos=pos_dict, node_color=color_ls[i], node_size=3, label=i)
    plt.legend(labels=["Negative", "Positive"], loc="upper right", fontsize='small')
    plt.title(model_name, y=-0.01)
    plt.show()

d = 16
p = 0.5
all_res = []
color_ls = []
class_num = 2
num_epochs = 401  # 轮数
file_path = 'exp/'
cmap = cm.get_cmap('cool', class_num)
for i in range(class_num):
    rgba = cmap(i)
    color_ls.append(colors.rgb2hex(rgba))

class STGENConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, nn: Callable, aggr: str = 'add',
                 root_weight: bool = True, bias: bool = True, **kwargs):
        super(STGENConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.in_channels_l = in_channels[0]
        self.in_channels_r = in_channels[1]

        if root_weight:
            self.root = Parameter(torch.Tensor(self.in_channels_r, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        if self.root is not None:
            uniform(self.root.size(0), self.root)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None and self.root is not None:
            out += torch.matmul(x_r, self.root)

        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        # 这里要确保输入的维度正确
        inputs = torch.cat([x_j, edge_attr], dim=1)
        return self.nn(inputs)

    def __repr__(self):
        return '{}({}, {}, aggr="{}", nn={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.aggr, self.nn)


class STGEN(torch.nn.Module):
    def __init__(self, input_dim, dim=d):
        super(STGEN, self).__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_attr_all.size(-1), dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim)
        )
        convdim = 8

        self.conv1 = STGENConv(dim, convdim, self._create_nn(dim + dim, convdim))
        self.conv2 = STGENConv(convdim, dataset.num_classes, self._create_nn(convdim + dim, dataset.num_classes))
        self.bn1 = nn.BatchNorm1d(convdim)
        # self.fc = nn.Linear(dataset.num_classes * 2, dataset.num_classes)
        self.fc = nn.Linear(convdim + dataset.num_classes, dataset.num_classes)
        self.time_weights_fc = nn.Linear(1, dim)

    def _create_nn(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self):
        x, edge_index, edge_attr = self.node_encoder(data.x), data.edge_index, self.edge_encoder(edge_attr_all)
        time_weights = self.time_weights_fc(data.time_weights)
        x = x * time_weights
        x1 = F.relu(self.conv1(x, edge_index, edge_attr))
        x1 = self.bn1(x1)
        x1 = F.dropout(x1, p=p, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class GEN(torch.nn.Module):
    def __init__(self, dim=d):
        super(GEN, self).__init__()
        self.node_encoder = nn.Linear(data.x.size(-1), dim)
        self.edge_encoder = nn.Linear(edge_attr_all.size(-1), dim)
        self.conv1 = pyg_nn.GENConv(dim, dim)
        self.conv2 = pyg_nn.GENConv(dim, dim)
        self.fc1 = nn.Linear(dim, dataset.num_classes)
        self.time_weights_fc = nn.Linear(1, dim)

    def forward(self):
        x, edge_index, edge_attr = self.node_encoder(data.x), data.edge_index, self.edge_encoder(edge_attr_all)
        time_weights = self.time_weights_fc(data.time_weights)
        x = x * time_weights
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=p, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

def generate_node2vec_embeddings(edge_index, num_nodes, embedding_dim):
    model = Node2Vec(edge_index, embedding_dim=embedding_dim, walk_length=10, context_size=5, walks_per_node=10, num_negative_samples=1, sparse=True).to(device)
    loader = model.loader(batch_size=128, shuffle=False, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            pos_rw, neg_rw = pos_rw.to(device), neg_rw.to(device)
            optimizer.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    for epoch in range(1, 10):
        loss = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    embeddings = model()
    return embeddings


# time

for e in [('Miami', 'Florida'), ('Los Angeles', 'California'), ('Orlando', 'Florida'),
          ('Dallas', 'Texas'), ('Houston', 'Texas'), ('New York', 'New York')]:
    city_name, state_abbrev = e[0].lower().replace(" ", "_"), us_state_to_abbrev[e[1]].lower()
    city_format = e[0] + ' (' + us_state_to_abbrev[e[1]] + ')'
    if os.path.exists(file_path + city_name + '_' + state_abbrev + '/processed'):
        shutil.rmtree(file_path + city_name + '_' + state_abbrev + '/processed')
    dataset = TRAVELDataset(file_path, city_name + '_' + state_abbrev)
    data = dataset[0]
    class_num = dataset.num_classes

    # 训练、验证和测试集的划分比例为 70%, 20% 和 10%
    data.train_mask, data.val_mask, data.test_mask = train_test_split_stratify(dataset, train_ratio=0.6, val_ratio=0.2, class_num=class_num)
    sc = MinMaxScaler()
    data.x[data.train_mask] = torch.tensor(sc.fit_transform(data.x[data.train_mask]), dtype=torch.float)
    data.x[data.val_mask] = torch.tensor(sc.transform(data.x[data.val_mask]), dtype=torch.float)
    data.x[data.test_mask] = torch.tensor(sc.transform(data.x[data.test_mask]), dtype=torch.float)

    edge_attr_all = MinMaxScaler().fit_transform(data.edge_attr.cpu())
    edge_attr_all = torch.tensor(edge_attr_all).float().to(device)

    coords = data.coords.numpy()
    gdf_pred = pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1], 'label': data.y.numpy()})
    zip_iterator = zip(gdf_pred.index, gdf_pred[['x', 'y']].values)
    pos_dict = dict(zip_iterator)

    X_train, X_test, y_train, y_test = data.x[data.train_mask].cpu().numpy(), data.x[data.test_mask].cpu().numpy(), \
                                       data.y[data.train_mask].cpu().numpy(), data.y[data.test_mask].cpu().numpy()

    data = data.to(device)

    # 注意方向和角度边缘特征已在数据集中预先计算
    component_dir = np.concatenate((data.edge_attr.cpu(), data.edge_attr_dir.cpu()), axis=1)
    component_ang = np.concatenate((data.edge_attr.cpu(), data.edge_attr_ang.cpu()), axis=1)
    component_dir = StandardScaler().fit_transform(component_dir)
    component_ang = StandardScaler().fit_transform(component_ang)
    data.component_dir = torch.tensor(component_dir).float().to(device)
    data.component_ang = torch.tensor(component_ang).float().to(device)


    node_embeddings = generate_node2vec_embeddings(data.edge_index, num_nodes=data.x.size(0), embedding_dim=d)
    node_embeddings = node_embeddings.cpu().detach().numpy()


    start_time = time.time()
    input_dim = data.x.size(-1)
    model = GEN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0078, weight_decay=5e-4)
    res = train_loop(model, data, optimizer, num_epochs, 'STGEN', city_name)
    t = round(time.time() - start_time, 2)
    all_res.append((city_format,) + ('STGEN',) + res + (t,))
    print("Execution time: %.4f seconds" % t)

df = pd.DataFrame(all_res, columns=['City', 'Method', 'F1', 'AUC', 'Acc', 'MAP', 'Time'])
print('# datasets:', df.shape[0] // len(df.Method.unique()))
print(df)
