import torch
import torch.nn as nn

import datasets
import config
import models
from practices import utils


device = utils.device()

model = models.create_resnet50(device=device, pretrained=False)

optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
criterion = nn.CrossEntropyLoss()


for epoch in range(config.EPOCHS):
    epoch = epoch + 1
    model.train()
    optimizer.zero_grad()
    validation_loss = training_loss = 0.0
    for i, data in enumerate(datasets.training_dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        training_loss += loss.item()

    with torch.no_grad():
        model.eval()
        for x, y in datasets.validation_dataloader:
            x, y = x.to(device), y.to(device)
            y_validate = model(x)
            loss = criterion(y_validate, y)
            validation_loss += loss.item()
    print(f"Epoch{epoch:03d}, training_loss: {training_loss:.3f}, validation_loss: {validation_loss:.3f}")

