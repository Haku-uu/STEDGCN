import util
import argparse
import torch
from model import STEDGCN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from datetime import datetime
#PEMSBAY PEMS08 PEMS04 PEMS03 PEMS08_60 Urban_60 Urban PEMS03_60 METRLA
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="")
parser.add_argument("--data", type=str, default="PEMS03", help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="input_dim")
parser.add_argument("--channels", type=int, default=128, help="number of nodes")
parser.add_argument("--num_nodes", type=int, default=170, help="number of nodes")
parser.add_argument("--input_len", type=int, default=12, help="input_len")
parser.add_argument("--output_len", type=int, default=12, help="out_len")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument(
    "--weight_decay", type=float, default=0.0001, help="weight decay rate"
)
parser.add_argument('--checkpoint', type=str,
                    default='log/2025-02-08-8_04_11-PEMS03/best_model.pth', help='')
parser.add_argument('--plotheatmap', type=str, default='True', help='')
args = parser.parse_args()

def main():

    device = torch.device(args.device)


    model = STEDGCN(
            device, args.input_dim, args.channels, args.num_nodes, args.input_len, args.output_len, args.dropout
        )
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    print('model load successfully')

    dataloader = util.load_dataset(
        args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

   
    amae = []
    amape = []
    awmape = []
    armse = []

    for i in range(args.output_len):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])

    


if __name__ == "__main__":
    main()
