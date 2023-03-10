import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import u_net
import utils

# to ensure reproducible training/validation split
random.seed(42)

# directorys with data and to store training checkpoints and logs
DATA_DIR = Path("C:/Users/stijn/Desktop/school/TUe/2022-2023/8DM20 - Capita Selecta Image Analysis/8DM20 - Project/Data")
CHECKPOINTS_DIR = Path.cwd() / "segmentation_model_weights"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOGDIR = "segmentation_runs"
device = torch.device("cuda:0")

# training settings and hyperparameters
NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [64, 64]
BATCH_SIZE = 32
N_EPOCHS = 100
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
TOLERANCE = 0.01  # for early stopping

print(f'GPU initialized {torch.cuda.get_device_properties(None)}')

# find patient folders in training directory
# excluding hidden folders (start with .)
patients = [
    path
    for path in DATA_DIR.glob("*")
    if not any(part.startswith(".") for part in path.parts)
]
random.shuffle(patients)

# split in training/validation after shuffling
partition = {
    "train": patients[:-NO_VALIDATION_PATIENTS],
    "validation": patients[-NO_VALIDATION_PATIENTS:],
}

# load training data and create DataLoader with batching and shuffling
dataset = utils.ProstateMRDataset(partition["train"], IMAGE_SIZE)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# load validation data
valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# initialise model, optimiser, and loss function
loss_function = torch.nn.functional.mse_loss
unet_model = u_net.UNet(enc_chs=(1, 64, 128, 256), dec_chs=(256, 128, 64, 32), num_classes=1)
optimizer = torch.optim.SGD(unet_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

minimum_valid_loss = 10  # initial validation loss
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)  # tensorboard summary

# training loop
for epoch in range(N_EPOCHS):
    print(f"Starting epoch {epoch} of {N_EPOCHS}")
    if (device.type == 'cuda'):
        unet_model = unet_model.cuda()

    current_train_loss = 0.0
    current_valid_loss = 0.0

    for batch_idx, (input, target) in enumerate(dataloader):
        if (device.type == 'cuda'):
            input = input.cuda()
            target = target.cuda()

        optimizer.zero_grad()

        result = unet_model(input)

        loss = loss_function(result,target.float())
        loss.backward()
        optimizer.step()

        current_train_loss = current_train_loss + loss.item()

        if batch_idx % 100 == 0:
            print(f"Training batch {batch_idx} of {len(dataloader)} training loss: {loss.item():.5f}")

    # evaluate validation loss
    with torch.no_grad():
        unet_model.eval()
        for batch_idx, (input, target) in enumerate(valid_dataloader):
            if (device.type == 'cuda'):
                input = input.cuda()
                target = target.cuda()

            result = unet_model(input)

            loss = loss_function(result, target)

            current_valid_loss = current_valid_loss + loss.item()

        unet_model.train()

    # write to tensorboard log
    writer.add_scalar("Loss/train", current_train_loss / len(dataloader), epoch)
    writer.add_scalar(
        "Loss/validation", current_valid_loss / len(valid_dataloader), epoch
    )

    # if validation loss is improving, save model checkpoint
    # only start saving after 10 epochs
    if (current_valid_loss / len(valid_dataloader)) < minimum_valid_loss + TOLERANCE:
        minimum_valid_loss = current_valid_loss / len(valid_dataloader)
        if epoch > 9:
            torch.save(
                unet_model.cpu().state_dict(),
                CHECKPOINTS_DIR / f"u_net_{epoch}.pth",
            )
