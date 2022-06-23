import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, RedditDataset, KarateClubDataset, CoraFullDataset

from gcn import GCN
#from gcn_mp import GCN
#from gcn_spmv import GCN

np.set_printoptions(threshold=np.inf)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        _, _, _, _, logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def main(args):
    # load and preprocess dataset
    if args['dataset'] == 'cora':
        data = CoraGraphDataset()
    elif args['dataset'] == 'citeseer':
        data = CiteseerGraphDataset()
    elif args['dataset'] == 'pubmed':
        data = PubmedGraphDataset()
    elif args['dataset'] == 'reddit':
        data = RedditDataset()
    elif args['dataset'] == 'karate':
        data = KarateClubDataset()
    elif args['dataset'] == 'cora_full':
        data = CoraFullDataset()
    elif args['dataset'] == 'muta':
        data = dgl.data.MUTAGDataset()
        category = data.predict_category
    else:
        raise ValueError('Unknown dataset: {}'.format(args['dataset']))

    g = data[0]
    if args['gpu'] < 0:
        to_cuda = False
    else:
        to_cuda = True
        g = g.to(args['gpu'])

    if args['dataset'] == 'muta':
        features = g.nodes[category].data['feat']
        labels = g.nodes[category].data['label']
        train_mask = g.nodes[category].data['train_mask']
        val_mask = g.nodes[category].data['val_mask']
        test_mask = g.nodes[category].data['test_mask']
    
    else:
        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']


    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.int().sum().item(),
              val_mask.int().sum().item(),
              test_mask.int().sum().item()))

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # # normalization
    # degs = g.in_degrees().float()
    # norm = torch.pow(degs, -0.5)
    # norm[torch.isinf(norm)] = 0
    # if cuda:
    #     norm = norm.cuda()
    # g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    # print(features.numpy())
    model = GCN(g,
                in_feats,
                args['n_hidden'],
                n_classes,
                args['n_layers'],
                features,
                args['cache'],
                to_cuda,
                F.relu,
                args['dropout'])

    if to_cuda:
        model.to(args['gpu'])
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    # initialize graph
    dur = []
    agg_time = []
    agg_time_layer1 = []
    agg_time_layer2 = []
    comp_time_layer1 = []
    comp_time_layer2 = []
    comp_time = []
    # start = time.time()
    Accuracy = []
    epoch_time = []

    for epoch in range(args['n_epochs']):
        start = time.time()
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        if args['cache'] == False and to_cuda == True:
            features.to(args['gpu'])
        t_agg_layer, t_comp_layer, t_agg, t_comp, logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        agg_time.append(t_agg)
        comp_time.append(t_comp)
        agg_time_layer1.append(t_agg_layer[0])
        agg_time_layer2.append(t_agg_layer[1])
        comp_time_layer1.append(t_comp_layer[0])
        comp_time_layer2.append(t_comp_layer[1])
        epoch_time.append(time.time() - start)

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))
        Accuracy.append(acc)
    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))
    print("Time Cost {:.4f}".format(time.time() - start))
    print("Test Accuracy {:.4f} | Time Cost {:.4f} | Aggregation (l1: {:.4f}, l2: {:.4f}, all: {:.4f}) | Reduce (l1: {:.4f}, l2: {:.6f}, all: {:.4f})".format(
        acc, np.mean(epoch_time), np.mean(agg_time_layer1), np.mean(agg_time_layer2), np.mean(agg_time),
        np.mean(comp_time_layer1), np.mean(comp_time_layer2), np.mean(comp_time)
    ))
    if args['save']:
        df = pd.DataFrame(np.array(Accuracy))
        df.to_csv('{}_cache_{}.csv'.format(args['dataset'], args['cache']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    # register_data_args(parser)
    parser.add_argument("--dataset", type=str, default="citeseer",
            help="Datasets:('cora', 'pumbed', 'reddit')")
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=20,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument("--cache", type=bool, default=False,
            help="Cache the aggregated content for the first layer")    
    parser.add_argument("--save", type=bool, default=False,
            help="Save accuracy to csv file")    
    args = vars(parser.parse_args())

    print(args)

    main(args)