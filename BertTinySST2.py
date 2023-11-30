from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
from datasets import load_dataset
import numpy as np
import evaluate
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from tqdm.notebook import tqdm
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

train_dataset = load_dataset("sst2", split="train")
test_dataset = load_dataset("sst2", split="validation")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", max_length=128)

BATCH_SIZE = 32
MAX_PHYSICAL_BATCH_SIZE = 8

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, eps=1e-8)

EPOCHS = 3
LOGGING_INTERVAL = 5000
EPSILON = 7.5
DELTA = 1 / len(train_dataloader)

def accuracy(preds, labels):
    return (preds == labels).mean()


def evaluate(model):   
   model.eval()

   loss_arr = []
   accuracy_arr = []

   def prepare_inputs(batch):
       inputs = tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
       inputs['labels'] = batch['label']
       return inputs

   for batch in test_dataloader:
       inputs = prepare_inputs(batch)
       outputs = model(**inputs)

       with torch.no_grad():
           loss, logits = outputs[:2]
           preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
           labels = batch['label'].detach().cpu().numpy()

           loss_arr.append(loss.item())
           accuracy_arr.append(accuracy(preds, labels))

   model.train()
   return np.mean(loss_arr), np.mean(accuracy_arr)

MAX_GRAD_NORM = 0.1

privacy_engine = PrivacyEngine()

model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_dataloader,
    target_delta=DELTA,
    target_epsilon=EPSILON, 
    epochs=EPOCHS,
    max_grad_norm=MAX_GRAD_NORM,
)
for epoch in range(1, EPOCHS+1):
    losses = []

    with BatchMemoryManager(
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
        data_loader=train_dataloader,
        optimizer=optimizer
    ) as memory_safe_data_loader:
        for step, batch in enumerate(tqdm(memory_safe_data_loader)):
            optimizer.zero_grad()
          
            outputs = model(**tokenizer(batch['sentence'], padding=True, truncation=True, max_length=128, return_tensors="pt")) # output = loss, logits, hidden_states, attentions

            loss = outputs[0]
            loss.backward()
            losses.append(loss.item())

            optimizer.step()

            if step > 0 and step % LOGGING_INTERVAL == 0:
                train_loss = np.mean(losses)
                eps = privacy_engine.get_epsilon(DELTA)

                eval_loss, eval_accuracy = evaluate(model)

                print(
                  f"Epoch: {epoch} | "
                  f"Step: {step} | "
                  f"Train loss: {train_loss:.3f} | "
                  f"Eval loss: {eval_loss:.3f} | "
                  f"Eval accuracy: {eval_accuracy:.3f} | "
                  f"ɛ: {eps:.2f}"
                )
