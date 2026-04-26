import os, argparse, torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="datasets/detector_data")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])
    train = datasets.ImageFolder(os.path.join(args.data_root,"train"), transform)
    test = datasets.ImageFolder(os.path.join(args.data_root,"test"), transform)
    print("Classes:", train.class_to_idx)
    tl = DataLoader(train,32,shuffle=True,num_workers=2)
    vl = DataLoader(test,32,shuffle=False,num_workers=2)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features,1)
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    best=0

    for e in range(args.epochs):
        model.train(); tot=0; cor=0; s=0
        for x,y in tqdm(tl, desc=f"Epoch {e+1}/{args.epochs}", ncols=100):
            x,y=x.to(device),y.to(device).float().unsqueeze(1)
            opt.zero_grad()
            with autocast():
                out=model(x); loss=loss_fn(out,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot+=loss.item(); pred=(torch.sigmoid(out)>0.5).float()
            cor+=(pred==y).sum().item(); s+=y.size(0)
        acc=100*cor/s
        print(f"Train acc {acc:.2f}% loss {tot/len(tl):.4f}")

        model.eval(); c=t=0
        with torch.no_grad():
            for x,y in vl:
                x,y=x.to(device),y.to(device).float().unsqueeze(1)
                out=model(x); pred=(torch.sigmoid(out)>0.5).float()
                c+=(pred==y).sum().item(); t+=y.size(0)
        val=100*c/t
        print(f"Val acc {val:.2f}%")
        if val>best:
            best=val; os.makedirs("results_detector",exist_ok=True)
            torch.save(model.state_dict(),"results_detector/best_detector.pth")
            print("💾 Saved best model")
    print("✅ Done. Best val acc:",best)

if __name__=="__main__":
    main()
