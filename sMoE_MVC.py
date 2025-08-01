from tqdm import tqdm

from utils.load_data import *
from model.model import *
from model.Trainer import *
import warnings
import scipy

warnings.filterwarnings("ignore")

from utils.args import arg_parse

import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.evaluate_embedding import cluster_acc
import scipy

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = arg_parse()

    log_interval = 1
    lr = args.lr
    
    pre_dim = args.pre_dim
    expert_num=args.expert_num
    k = args.k # top k experts
    DS=args.DS
    feature_list, label = load_data(args, device)
    y = label
    input_dim_set = []
    for feature in feature_list:
        input_dim_set.append(feature.shape[1])
    num_view=len(feature_list)
    dataloader = set_dataloader(args, feature_list, label)
    n_cluster =n_label= len(np.unique(label))
    cluster_dim = args.cluster_emb
    Best_acc = 0
    Best_model = None

    print('================')
    print('Dataset:', DS)
    print('lr: {}'.format(lr))
    print('clutering embedding dimension: {}'.format(args.cluster_emb))
    print('================')

    iter = 10
    if args.eval == True:
        ACCList = np.zeros((iter, 1))
        NMIList = np.zeros((iter, 1))
        ARIList = np.zeros((iter, 1))
        ACC_MEAN = np.zeros((1, 2))
        NMI_MEAN = np.zeros((1, 2))
        ARI_MEAN = np.zeros((1, 2))
        for it in range(iter):
            model = MV_MOE(num_view, input_dim_set, pre_dim, n_label, cluster_dim, expert_num, k, device, args).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            trainer = MV_MOE_Trainer(args, model, optimizer, num_view, args.lamda, args.gamma, device)

            filename = './weight/' + args.DS + '.pth'
            model.load_state_dict(torch.load(filename, map_location=device)['model'])
            model.eval()
            emb, y = model.get_results(dataloader, num_view)
            
            kmeans = KMeans(n_clusters=n_cluster, n_init=100)
            y_pred = kmeans.fit_predict(emb.cpu().detach().numpy())
            y=y.numpy()
            acc = cluster_acc(y, y_pred)
            nmi = nmi_score(y, y_pred)
            ari = ari_score(y, y_pred)
            ACCList[it, :] = acc
            NMIList[it, :] = nmi
            ARIList[it, :] = ari
        ACC_MEAN[0, :] = np.around([np.mean(ACCList), np.std(ACCList)], decimals=4)
        NMI_MEAN[0, :] = np.around([np.mean(NMIList), np.std(NMIList)], decimals=4)
        ARI_MEAN[0, :] = np.around([np.mean(ARIList), np.std(ARIList)], decimals=4)

        with open('./result/' + args.DS + '_result_eval.txt', 'a') as f:
            f.write(args.DS + '_Result:' + '\n')
            f.write('ACC_MEAN:' + str(ACC_MEAN[0][0]*100)+'('+str(ACC_MEAN[0][1]*100) + ')\n')
            f.write('NMI_MEAN:' + str(NMI_MEAN[0][0]*100)+'('+str(NMI_MEAN[0][1]*100) + ')\n')
            f.write('ARI_MEAN:' + str(ARI_MEAN[0][0]*100)+'('+str(ARI_MEAN[0][1]*100) + ')\n')
            f.write('\n')

            print('Test ACC: {}'.format(ACC_MEAN))
            print('Test NMI: {}'.format(NMI_MEAN))
            print('Test ARI: {}'.format(ARI_MEAN))
    else:
        para_set = [0.0001, 0.001, 0.01, 0.1, 1, 10]
        for o in range(len(para_set)):
            args.lamda = para_set[o]
            for gamma in range(len(para_set)):
                args.gamma = para_set[gamma]
                ACCList = np.zeros((iter, 1))
                NMIList = np.zeros((iter, 1))
                ARIList = np.zeros((iter, 1))
                ACC_MEAN = np.zeros((1, 2))
                NMI_MEAN = np.zeros((1, 2))
                ARI_MEAN = np.zeros((1, 2))
                for it in range(iter):
                    model = MV_MOE(num_view, input_dim_set, pre_dim, n_label, cluster_dim, expert_num, k, device, args).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    trainer = MV_MOE_Trainer(args, model, optimizer, num_view, args.lamda, args.gamma, device)
                    # Training the model
                    acc, nmi, ari = trainer.train(dataloader, args.lr, args.epochs)
                    if acc > Best_acc:
                        Best_acc = acc
                        Best_model = model
                    ACCList[it, :] = acc
                    NMIList[it, :] = nmi
                    ARIList[it, :] = ari
                ACC_MEAN[0, :] = np.around([np.mean(ACCList), np.std(ACCList)], decimals=4)
                NMI_MEAN[0, :] = np.around([np.mean(NMIList), np.std(NMIList)], decimals=4)
                ARI_MEAN[0, :] = np.around([np.mean(ARIList), np.std(ARIList)], decimals=4)

                with open('./result/' + args.DS + '_result_train.txt', 'a') as f:
                    f.write(args.DS + '_Result:' + '\n')
                    f.write('Loss 1 :' +str(args.lamda)+ '\n')
                    f.write('Loss 2:' +str(args.gamma)+ '\n')
                    f.write('ACC_MEAN:' + str(ACC_MEAN[0][0]*100)+'('+str(ACC_MEAN[0][1]*100) + ')\n')
                    f.write('NMI_MEAN:' + str(NMI_MEAN[0][0]*100)+'('+str(NMI_MEAN[0][1]*100) + ')\n')
                    f.write('ARI_MEAN:' + str(ARI_MEAN[0][0]*100)+'('+str(ARI_MEAN[0][1]*100) + ')\n')
                    f.write('current_Best_acc: ' +str(Best_acc)+' \n')
                    f.write('\n')

