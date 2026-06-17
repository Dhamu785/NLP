import os
import shutil
import random
import numpy as np 
from config import CONFIG

import torch.nn as nn
import torch as t

from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
progress = Progress(
    SpinnerColumn(spinner_name='dots', style='green'),
    TextColumn('[bold blue]{task.description}'),
    BarColumn(complete_style='bright_green', finished_style='green', style='grey35'),
    TextColumn('[yellow]{task.percentage:3.0f}%'),
    TextColumn('[red] Loss: {task.fields[loss]}'),
    TextColumn('[yellow] Elasped:'),
    TimeElapsedColumn(),
    TextColumn('[cyan] Remaining:'),
    TimeRemainingColumn()
)

design = CONFIG()

sav_loc = '.\\runs'
if not os.path.exists(sav_loc):
    os.mkdir(sav_loc)

def train(rnn, train_data, epochs=design.epochs, batch_size=design.batch_size, 
            lr=design.lr, loss=design.loss, optimizer=design.optimizer, resume=False):
    all_loss = []
    start_epoch = 1
    rnn.to(design.device)
    print(f"Model loaded on = {next(rnn.parameters()).device}")
    optim = optimizer(rnn.parameters(), lr=lr)
    if resume:
        ckpt = t.load(os.path.join(sav_loc, 'latest.pt'), map_location=design.device, weights_only=False)
        rnn.load_state_dict(ckpt['model_state_dict'])
        optim.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']+1
        all_loss = ckpt['all_loss']
        batch_size = ckpt['batch_size']
        print(f"Resume training...")

    current_loss = 0
    rnn.train()

    with progress:
        epoch_task = progress.add_task("Learning :", total=(epochs+1)-start_epoch, loss=0.0)
        for epoch in range(start_epoch, epochs+1):
            rnn.zero_grad()

            batches = list(range(len(train_data)))
            random.shuffle(batches)
            batches = np.array_split(batches, len(batches) // batch_size)
            batch_task = progress.add_task(f"Epoch [{epoch}/{epochs}]", total=len(batches), loss=0.0)
            for idx, batch in enumerate(batches):
                batch_loss = 0
                optim.zero_grad()
                for i in batch:
                    (lbl_tensor, txt_tensor, lbl, txt) = train_data[i]
                    output = rnn(txt_tensor.to(design.device))
                    ls = loss(output, lbl_tensor.to(design.device))
                    batch_loss += ls

                batch_loss.backward()
                nn.utils.clip_grad_norm_(rnn.parameters(), 3)
                optim.step()

                avg_batch_loss = batch_loss.item() / len(batch)
                current_loss += avg_batch_loss
                progress.update(batch_task, advance=1, description=f"Epoch [{epoch}/{epochs}]", loss=f"{avg_batch_loss:.4f}")

            progress.remove_task(batch_task)
            all_loss.append(current_loss/len(batches)) 
            current_loss = 0

            # Save the checkpoints
            checkpoint = {
                            'epoch': epoch, 'loss': all_loss[-1], 
                            # Model
                            'model_state_dict': rnn.state_dict(),
                            # optimizer
                            'optimizer_state_dict' : optim.state_dict(),
                            'lr' : lr,
                            'batch_size' : batch_size,
                            'all_loss' : all_loss
                            }
            t.save(checkpoint, './runs/latest.pt')
            progress.update(epoch_task, advance=1, loss=f"{all_loss[-1]:.4f}")

    return all_loss
