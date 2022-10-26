import torch
import torch.nn as nn
from dataloader import FlyingChairs
import argparse
from full_nn import KPA
import flow_vis
import os
from PIL import Image
def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--shuffle', action='store_false')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--H', type=int, default=48)
    parser.add_argument('--W', type=int, default=64)
    parser.add_argument('--iter', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20220903)
    parser.add_argument('--pretrain', action='store_false')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train = FlyingChairs('FlyingChairs')
    train_set = torch.utils.data.DataLoader(train, batch_size = args.batch, shuffle = args.shuffle, num_workers = args.num_workers)
    model = KPA(args)
    if args.pretrain:
        model.load_state_dict(torch.load('weight/epoch=2.pt'))

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    criterion = nn.MSELoss()

    #if os.path.exists('./loss.txt'):
        #os.remove('./loss.txt')

    for epoch in range(args.epochs):
        total_loss = 0
        for idx, (data, flow) in enumerate(train_set):

            torch.cuda.empty_cache()

            data = data.to(device)
            flow = flow.to(device)

            pre = model( data )
            loss = criterion(flow, pre)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if(idx % 200 == 0):
                print(f'epoch {epoch} [{idx}/{args.iter}]  loss: {loss.item():.3f}')

            if idx == args.iter:
                break

        avg_loss = total_loss/args.iter
        print(f'epoch {epoch} avg. loss: {avg_loss:.3f}')

        with open('./loss.txt', 'a') as f:
            f.write(str(avg_loss)+',')

        flow_color = flow_vis.flow_to_color(pre[0].permute(1,2,0).cpu().detach().numpy(), convert_to_bgr=False)
        flow_color = Image.fromarray(flow_color)
        flow_color.save("flow/epoch="+str(epoch)+".jpg")


        torch.save(model.state_dict(), 'weight/epoch=' + str(epoch) + '.pt')

if __name__ == "__main__":
    main()
