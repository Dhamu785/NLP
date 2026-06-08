from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    SpinnerColumn
)
import time

progress = Progress(
    SpinnerColumn(spinner_name='dots', style='green'),
    TextColumn("[bold blue]{task.description}"),
    BarColumn(
        complete_style="bright_green",
        finished_style="green",
        style="grey35",
    ),
    TextColumn("[yellow]{task.percentage:>3.0f}%"),
    TextColumn("[red] Loss: {task.fields[loss]}"),
    TextColumn("[green] Acc: {task.fields[acc]}"),
    TextColumn("[yellow] Elapsed:"),
    TimeElapsedColumn(),
    TextColumn("[cyan] Remaining:"),
    TimeRemainingColumn(),
    
)

epochs = 100
dataset_size = 10
with progress:
    task = progress.add_task("Learning :", total=100, loss=0.000, acc=0.9)

    for epoch in range(epochs):
        batch_task = progress.add_task(f"EPOCH [{epoch+1}/{epochs}]", total=dataset_size, loss=0.000, acc=0.9)

        for batch in range(dataset_size):
            progress.update(batch_task, advance=1, description=f"EPOCH [{epoch+1}/{epochs}]", loss=f"{0.1234:.4f}", acc=f"{0.1234:.4f}")
            time.sleep(0.001)

        progress.remove_task(batch_task)
        progress.update(task, advance=1, loss=f"{0.1234:.4f}", acc=f"{0.1234:.4f}")
