import argparse
import os

import pandas as pd
import sagemaker
import torch
import torch.nn as nn
import torch.optim as optim
from sagemaker.feature_store.feature_group import FeatureGroup
from torch.nn import functional as F

ENVIRONMENT_NAME = os.environ["ENVIRONMENT_NAME"]


def get_feature_store_data(feature_group_name, limit: int) -> pd.DataFrame:
    sess = sagemaker.Session()
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=sess)
    query = feature_group.athena_query()
    query_string = query.query_string(select_columns=["columns", "you", "need"], limit=limit)
    query.run(query_string=query_string, output_location=f"s3://{sess.default_bucket()}/query_results/")
    query.wait()
    return query.as_dataframe()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature_group_name", type=str, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    train_data: pd.DataFrame = get_feature_store_data(args.feature_group_name, limit=args.limit)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
    )

    model.train()
    for epoch in range(args.epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        print(f"Train Epoch: {epoch} \tLoss: {loss.item()}")

    model_dir = os.environ["SM_MODEL_DIR"]
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))


if __name__ == "__main__":
    main()
