import random
import numpy as np 
from config import CONFIG

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

def train(rnn, train_data, epochs=design.epochs, batch_size=design.batch_size, 
            report_every=design.report_every, lr=design.lr, loss=design.loss, 
            optimizer=design.optimizer):
    current_loss = 0
    all_loss = []
    rnn.to(design.device)
    print(f"Model loaded on = {next(rnn.parameters()).device}")
    rnn.train()
    optim = optimizer(rnn.parameters(), lr=lr)

    with progress:
        epoch_task = progress.add_task("Learning :", total=epochs, loss=0.0)
        for epoch in range(1, epochs+1):
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
                optim.step()

                avg_batch_loss = batch_loss.item() / len(batch)
                current_loss += avg_batch_loss
                progress.update(batch_task, advance=1, description=f"Epoch [{epoch}/{epochs}]", loss=f"{avg_batch_loss:.4f}")

            progress.remove_task(batch_task)
            all_loss.append(current_loss/len(batches)) 
            # if epoch % report_every == 0:
            #     print(f"{epoch}/{epochs} | loss = {all_loss[-1]}")
            current_loss = 0
            progress.update(epoch_task, advance=1, loss=f"{all_loss[-1]:.4f}")

    return all_loss
