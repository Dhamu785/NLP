import random
import numpy as np 
from config import CONFIG

design = CONFIG()

def train(rnn, train_data, epochs=design.epochs, batch_size=design.batch_size, 
            report_every=design.report_every, lr=design.lr, loss=design.loss, 
            optimizer=design.optimizer):
    current_loss = 0
    all_loss = []

    rnn.train()
    optim = optimizer(rnn.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        rnn.zero_grad()

        batches = list(range(len(train_data)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) // batch_size)

        for idx, batch in enumerate(batches):
            batch_loss = 0
            optim.zero_grad()
            for i in batch:
                (lbl_tensor, txt_tensor, lbl, txt) = train_data[i]
                output = rnn(txt_tensor)
                ls = loss(output, lbl_tensor)
                batch_loss += ls

            batch_loss.backward()
            optim.step()
        
            current_loss += batch_loss.item() / len(batch)

        all_loss.append(current_loss/len(batches)) 
        if epoch % report_every == 0:
            print(f"{epoch}/{epochs} | loss = {all_loss[-1]}")
        current_loss = 0

    return all_loss
