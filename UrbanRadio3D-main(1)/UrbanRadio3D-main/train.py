import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image


# ===================== 只读取 Delay 文件夹的 GIF =====================
class UrbanRadio3DDataset(Dataset):
    def __init__(self, root_path):
        self.root = root_path
        self.img_dir = os.path.join(self.root, "Delay")

        self.files = [
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(".gif")
        ]

        print("找到图片总数：", len(self.files))
        if len(self.files) == 0:
            raise RuntimeError("没有图片！")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        path = os.path.join(self.img_dir, f)

        # 读取GIF → 转成模型可用的数据
        img = Image.open(path).convert("L").resize((64, 64))
        arr = np.array(img).astype(np.float32) / 255.0

        # 构造 3D 数据格式 (1, 64, 64, 1)
        arr = arr[None, ..., None]
        return torch.from_numpy(arr)


# ===================== 简易 3D 模型 =====================
class Simple3DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(8, 1, 3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.conv2(x)


# ===================== 训练 =====================
def main():
    data_root = r"D:\UrbanRadio3D-main"
    device = "cpu"
    batch_size = 1
    epochs = 10

    dataset = UrbanRadio3DDataset(data_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Simple3DModel().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("开始训练！")
    for epoch in range(epochs):
        total_loss = 0
        bar = tqdm(loader, desc=f"Epoch {epoch + 1}")

        for x in bar:
            x = x.to(device)
            optimizer.zero_grad()
            y = model(x)
            loss = loss_fn(y, x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.3f}")

        print(f"Epoch {epoch + 1} 平均损失: {total_loss / len(loader):.4f}")


if __name__ == "__main__":
    main()