import torch
device = torch.device('cuda')

torch.manual_seed(0)

src = torch.rand((2048, 1, 512))
print('src = ',src)

tgt = torch.rand((2048, 20, 512))
print('tgt = ',tgt)


dataset = torch.utils.data.TensorDataset(src, tgt)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

model = torch.nn.Transformer(batch_first=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
model.train()
model = model.to(device)

for epoch in range(10):
    for source, targets in loader:
        source = source.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        output = model(source, targets)
        loss = criterion(output, targets)
        print('Epoch = ',epoch)
        print('loss = ', loss)
        loss.backward()
        optimizer.step()
