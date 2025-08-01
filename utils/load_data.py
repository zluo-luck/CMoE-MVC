import scipy
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import normalize

def set_dataloader(args, feature_list, labels):
    num=0
    for feature in feature_list:
        num+=1
    print('Number of views:', num)
    if num == 2:
        dataset = TensorDataset(feature_list[0],feature_list[1], labels)
        train_data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle = False)
        # test_data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle = False)
    elif num== 3:
        dataset = TensorDataset(feature_list[0],feature_list[1],feature_list[2], labels)
        train_data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle = False)
        # test_data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle = False)
    elif num== 4:
        dataset = TensorDataset(feature_list[0], feature_list[1], feature_list[2], feature_list[3], labels)
        train_data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle = False)
        # test_data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle = False)
    elif num == 6:
        dataset = TensorDataset(feature_list[0],feature_list[1],feature_list[2],feature_list[3],feature_list[4],feature_list[5], labels)
        train_data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle = False)
        # test_data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle = False)
    return train_data_loader

def load_data(args, device):
    print(args.path)
    data = scipy.io.loadmat(args.path + args.DS + '.mat')
    try:
        features = data['X']
    except:
        features1 = data['X1']
        features2 = data['X2']
        features = [[]]
        features[0].append(features1)
        features[0].append(features2)

    feature_list = []
    adj_list = []
    try:
        for key in ('truth', 'Y', 'gnd', 'truelabel'):
            if key in data:
                labels = data[key].flatten()
                break
    except:
        raise KeyError("None of 'truth', 'Y', or 'gnd' found in data")
    labels = labels - min(set(labels))
    labels = torch.from_numpy(labels).long()
    num_classes = len(np.unique(labels))

    for i in range(features.shape[1]):
        # print("Loading the data of" + str(i) + "th view")
        features[0][i] = normalize(features[0][i])
        feature = features[0][i]
        if scipy.sparse.isspmatrix_csr(feature):
            feature = feature.todense()
            print("sparse")
        feature = torch.from_numpy(feature).float().to(device)

        feature_list.append(feature)

    return feature_list, labels