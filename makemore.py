"""
you give this script some words (one per line) and it will generate more things like it.
uses super state of the art Transformer AI tech
this code is intended to be super hackable. tune it to your needs.

Changes from minGPT:
- I removed the from_pretrained function where we init with GPT2 weights
- I removed dropout layers because the models we train here are small,
  it's not necessary to understand at this stage and at this scale.
- I removed weight decay and all of the complexity around what parameters are
  and are not weight decayed. I don't believe this should make a massive
  difference at the scale that we operate on here.
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
import torch.multiprocessing as mp

from src.models import (
    Bigram,
    BoW,
    MLP,
    ModelConfig,
    RNN,
    Transformer,
    DataParallel
)
from src.optimizers import (
    AdamW,
    Optimizer,
    get_optimizer
)


# -----------------------------------------------------------------------------
# helper functions for evaluating and sampling from the model

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def print_samples(args, model, train_dataset, test_dataset, num=10):
    """ samples from the model and pretty prints the decoded samples """
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    top_k = args.top_k if args.top_k != -1 else None
    steps = train_dataset.get_output_length() - 1 # -1 because we already start with <START> token (index 0)
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')
    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
        # token 0 is the <STOP> token, so we crop the output sequence at that point
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        # separately track samples that we have and have not seen before
        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)
    print('-'*80)
    for lst, desc in [(train_samples, 'in train'), (test_samples, 'in test'), (new_samples, 'new')]:
        print(f"{len(lst)} samples that are {desc}:")
        for word in lst:
            print(word)
    print('-'*80)

@torch.inference_mode()
def evaluate(model, dataset, args, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss

@torch.no_grad()
def gradnorm(model: nn.Module) -> float:
    """
    Given a PyTorch model, computes the average of the gradnorm across all parameters.
    """
    grad_norms = []
    for p in model.parameters():
      grad_norms.append(p.grad.norm())
    return grad_norms


# -----------------------------------------------------------------------------
# helper functions for creating the training and test Datasets that emit words

class CharDataset(Dataset):

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}
        self.itos = {i:s for s,i in self.stoi.items()} # inverse mapping

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1 # all the possible characters and special 0 token

    def get_output_length(self):
        return self.max_word_length + 1 # <START> token followed by words

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ''.join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 # index -1 will mask the loss at the inactive locations
        return x, y

def create_datasets(input_file):

    # preprocessing of the input text file
    with open(input_file, 'r') as f:
        data = f.read()
    words = data.splitlines()
    words = [w.strip() for w in words] # get rid of any leading or trailing white space
    words = [w for w in words if w] # get rid of any empty strings
    chars = sorted(list(set(''.join(words)))) # all the possible characters
    max_word_length = max(len(w) for w in words)
    print(f"number of examples in the dataset: {len(words)}")
    print(f"max word length: {max_word_length}")
    print(f"number of unique characters in the vocabulary: {len(chars)}")
    print("vocabulary:")
    print(''.join(chars))

    # partition the input data into a training and the test set
    test_set_size = min(1000, int(len(words) * 0.1)) # 10% of the training set, or up to 1000 examples
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")

    # wrap in dataset objects
    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)

    return train_dataset, test_dataset

class InfiniteDataLoader:
    """
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration: # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch



def train(args, model: nn.Module, optimizer: Optimizer, batch_loader: InfiniteDataLoader, train_dataset: Dataset, test_dataset: Dataset, writer: SummaryWriter):
    # training loop
    best_loss = None
    step = 0
    while True:

        t0 = time.time()

        # get the next batch, ship to device, and unpack it to input and target
        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        # feed into the model
        logits, loss = model(X, Y)

        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # wait for all CUDA work on the GPU to finish then calculate iteration time taken
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        # logging
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

        # evaluate the model
        test_loss, train_loss = None, None
        if step > 0 and step % 500 == 0:
            train_loss = evaluate(model, train_dataset, args, batch_size=100, max_batches=10)
            test_loss  = evaluate(model, test_dataset, args,  batch_size=100, max_batches=10)
            gradnorms = gradnorm(model)

            # calculate gradnorm statistics
            avg_gradnorm = 0 if not gradnorms else sum(gradnorms) / len(gradnorms)
            max_gradnorm = max(gradnorms)
            min_gradnorm = min(gradnorms)
            total_gradnorm = sum(gradnorms)

            writer.add_scalar("Loss/train", train_loss, step)
            writer.add_scalar("Loss/test", test_loss, step)
            writer.add_scalar("Gradnorm/average", avg_gradnorm, step)
            writer.add_scalar("Gradnorm/max", max_gradnorm, step)
            writer.add_scalar("Gradnorm/min", min_gradnorm, step)
            writer.add_scalar("Gradnorm/total", total_gradnorm, step)

            # only our optimizers emit the lr_norm. The torch optimizers do not.
            if args.optimizer in ["adagrad", "rmsprop", "adam", "adamw"] and not args.use_torch_optim:
                lr_norms = optimizer.lr_norms()

                # calculate learning rate statistics
                lrnorm_avg = 0 if not lr_norms else sum(lr_norms) / len(lr_norms)
                lrnorm_max = max(lr_norms)
                lrnorm_min = min(lr_norms)
                lrnorm_total = sum(lr_norms)

                writer.add_scalar("LRNorm/average", lrnorm_avg, step)
                writer.add_scalar("LRNorm/max", lrnorm_max, step)
                writer.add_scalar("LRNorm/min", lrnorm_min, step)
                writer.add_scalar("LRNorm/total", lrnorm_total, step)


            writer.flush()
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
            # save the model to disk if it has improved
            if best_loss is None or test_loss < best_loss:
                out_path = os.path.join(args.work_dir, "model.pt")
                print(f"test loss {test_loss} is the best so far, saving model to {out_path}")
                torch.save(model.state_dict(), out_path)
                best_loss = test_loss

        # write out the results
        gradnorms = gradnorm(model)

        # calculate gradnorm statistics
        avg_gradnorm = 0 if not gradnorms else sum(gradnorms) / len(gradnorms)
        max_gradnorm = max(gradnorms)
        min_gradnorm = min(gradnorms)
        total_gradnorm = sum(gradnorms)

        writer.add_scalar("Gradnorm/average", avg_gradnorm, step)
        writer.add_scalar("Gradnorm/max", max_gradnorm, step)
        writer.add_scalar("Gradnorm/min", min_gradnorm, step)
        writer.add_scalar("Gradnorm/total", total_gradnorm, step)
        # test loss will only be set when evlaluated
        writer.add_scalar("Loss/train-batch", loss, step)
        if test_loss is not None:
            writer.add_scalar("Loss/test-dataset", test_loss, step)
            writer.add_scalar("Loss/train-dataset", train_loss, step)

        # only our optimizers emit the lr_norm. The torch optimizers do not.
        if args.optimizer in ["adagrad", "rmsprop", "adam", "adamw"] and not args.use_torch_optim:
            lr_norms = optimizer.lr_norms()

            # calculate learning rate statistics
            lrnorm_avg = 0 if not lr_norms else sum(lr_norms) / len(lr_norms)
            lrnorm_max = max(lr_norms)
            lrnorm_min = min(lr_norms)
            lrnorm_total = sum(lr_norms)

            writer.add_scalar("LRNorm/average", lrnorm_avg, step)
            writer.add_scalar("LRNorm/max", lrnorm_max, step)
            writer.add_scalar("LRNorm/min", lrnorm_min, step)
            writer.add_scalar("LRNorm/total", lrnorm_total, step)



        # sample from the model
        if step > 0 and step % 200 == 0:
            print_samples(args, model, train_dataset, test_dataset, num=10)

        step += 1
        # termination conditions
        if args.max_steps >= 0 and step >= args.max_steps:
            break

def setup_optimizer(args, model: nn.Module):
    # init optimizer
    Optim = get_optimizer(args.optimizer, args.use_torch_optim)
    match args.optimizer:
        case "sgd":
            optimizer = Optim(params=model.parameters(), lr=args.learning_rate, momentum=args.momentum, nesterov=args.nesterov)
        case "adagrad":
            optimizer = Optim(params=model.parameters(), lr=args.learning_rate)
        case "rmsprop":
            optimizer = Optim(params=model.parameters(), lr=args.learning_rate, alpha=args.alpha, momentum=args.momentum)
        case "adam":
            optimizer = Optim(params=model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
        case "adamw":
            optimizer = Optim(params=model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        case _:
            raise ValueError(f'invalid optimizer selected: {args.optimizer}')
    
    return optimizer


def setup_model(type: str, config: ModelConfig, device: torch.device, work_dir: str, resume: bool, sample_only: bool):
    if type == 'transformer':
        model = Transformer(config)
    elif type == 'bigram':
        model = Bigram(config)
    elif type == 'mlp':
        model = MLP(config)
    elif type == 'rnn':
        model = RNN(config, cell_type='rnn')
    elif type == 'gru':
        model = RNN(config, cell_type='gru')
    elif type == 'bow':
        model = BoW(config)
    else:
        raise ValueError(f'model type {type} is not recognized')
    model.to(device)
    print(f"model #params: {sum(p.numel() for p in model.parameters())}")

    if resume or sample_only: # note: if we sample-only then we also assume we are resuming
        print("resuming from existing model in the workdir")
        model.load_state_dict(torch.load(os.path.join(work_dir, 'model.pt')))

    return model


def mock_dp(config: ModelConfig, dataset: CharDataset):
    BATCH = 128
    SPLIT = 4
    torch.manual_seed(42)
    models = []
    for _ in range(SPLIT):
        models.append(setup_model("mlp", config=config, device=torch.device("cpu"), resume=False, sample_only=False, work_dir="./out"))

    torch.manual_seed(42)
    model_seq = setup_model("mlp", config=config, device=torch.device("cpu"), resume=False, sample_only=False, work_dir="./out")

    # load dataset
    loader1 = InfiniteDataLoader(dataset, batch_size=BATCH, pin_memory=True)
    loader2 = InfiniteDataLoader(dataset, batch_size=BATCH, pin_memory=True)

    # sync the model gradients
    with torch.no_grad():
        # ensure model 1 and model 2 do not have identical params
        model0 = models[0]
        for model in models[1:]:
            for pref, p in zip(model0.parameters(), model.parameters()):
                if pref.data.equal(p.data):
                    print(pref.data)
                    print(p.data)
                assert not pref.data.equal(p.data), "model parameters will not be the same"
        
        ref_model = models[0]
        for model in models[1:]:
            for p, pref in zip(model.parameters(), ref_model.parameters()):
                p.copy_(pref)

        # ensure it was successful
        for model in models[0:]:
            for p, pref in zip(model.parameters(), ref_model.parameters()):
                assert p.equal(pref), "model parameters must be equal"
        
        print("model parameters were found to be the same!")

    # assert that the data loading behavior is identical
    torch.manual_seed(42)
    control_inputs, control_targets = loader1.next()
    torch.manual_seed(42)
    inputs2, targets2 = loader2.next()

    print(control_inputs, inputs2)
    print('==============')

    assert (control_inputs == inputs2).all(), "inputs1 and inputs2 from data loader must be the same"
    assert (control_targets == targets2).all(), "targets1 and targets2 from data loader must be the same"

    # first we need to calculate a forward pass on model seq, our baseline
    logits_seq, loss_seq = model_seq(control_inputs, targets=control_targets)

    # next we calculate the same on the other 2 models, just to double-check we get the same values
    logits1, loss1 = model_seq(inputs2, targets=control_targets)
    logits2, loss2 = model_seq(inputs2, targets=control_targets)

    assert logits1.allclose(logits_seq) and logits2.allclose(logits_seq), "logits must the identical for all models"
    assert loss1.allclose(loss_seq) and loss2.allclose(loss_seq), "loss must the identical for all models"

    # next lets split up the second tensor and process each portion in parallel on each model
    print(inputs2.shape, targets2.shape)

    B, T = inputs2.shape
    N = SPLIT
    Bn = B // N

    print(f"{B=}, {N=}, {T=}, {Bn=}")

    # now we operate on inputs2
    batch_inputs = torch.split(inputs2, Bn, dim=0)
    batch_targets = torch.split(targets2, Bn, dim=0)

    assert len(batch_inputs) == len(batch_targets) == N, f"batch split must equal to {N}: {len(batch_inputs)=}, {len(batch_targets)=}"

    # for checking batch inputs
    stacked_inputs = torch.stack(batch_inputs)
    stacked_targets = torch.stack(batch_targets)
    assert torch.not_equal(stacked_inputs, stacked_inputs[0]).any(), "inputs should not all be equal"
    assert torch.not_equal(stacked_targets, stacked_targets[0]).any(), "targets should not all be equal"


    # ensure that their concatenation in the right order is equivalent
    assert torch.cat(batch_inputs).equal(control_inputs), "concatenated inputs should be equal to full size tensor"
    assert torch.cat(batch_targets).equal(control_targets), "concatenated targets should be equal to full size target"

    # now lets perform the processing. First let's clear all gradients
    for model in models:
        for p in model.parameters():
            p.grad = None

    model_outputs = []

    for model, inputs, targets in zip(models, batch_inputs, batch_targets):
        logitsi, lossi = model(idx=inputs, targets=targets)
        # scale the loss by the sofmtax mean
        scaled_loss = lossi * Bn / B
        model_outputs.append((logitsi, scaled_loss))


    output_logits = [logitsi for logitsi, _ in model_outputs]
    stacked_logits = torch.stack(output_logits)
    assert len(output_logits) == N, "number of logits should be the same as the batch"
    assert torch.not_equal(stacked_logits, stacked_logits[0]).any(), "output logits should not be the same"


    stacked_losses = torch.stack([lossi for _, lossi in model_outputs])
    assert torch.not_equal(stacked_losses, stacked_losses[0]).any(), "output losses should not be the same"

    # calls .backward() on each respective model
    for _, lossi in model_outputs:
        lossi.backward()


    # but now by the principles of data parallelism, if we combine the gradients on both models,
    # they should now equal the same of a single sequential model
    loss_seq.backward()

    # now each model will have its own version of the scaled gradients. 
    # they should not be identical ot that of the sequential model
    for i, model in enumerate(models):
        for p, pseq in zip(model.parameters(), model_seq.parameters()):
            assert torch.not_equal(p.grad.data, pseq.grad.data).any(), "gradients should not yet equal sequential model"
        
        for j, modelj in enumerate(models):
            
            # no need to check self
            if i == j:
                continue

            for p, pj in zip(model.parameters(), modelj.parameters()):
                assert not torch.allclose(p.grad.data, pj.grad.data), "gradients should also not equal each other (unless the data is equal)"

    # now we just sum up all of the gradients onto the first model and we will naively copy them onto the other models later
    model0 = models[0]
    with torch.no_grad():
        for model in models[1:]:
            for p0, p in zip(model0.parameters(), model.parameters()):
                p0.grad.data += p.grad.data

    # now let's verify that the summed parameters are roughly equal to the other model

    # assert model 1 and model seq are now the same
    with torch.no_grad():
        max_diff = 0
        for p0, pseq in zip(model0.parameters(), model_seq.parameters()):
            diff = (p0.grad.data - pseq.grad.data).abs().max().item()
            max_diff = diff if diff > max_diff else max_diff
            assert p0.grad.data.allclose(pseq.grad.data, atol=1e-3), f"model gradients must be close, max_diff: {max_diff}"
        print(f"max diff: {max_diff}")

    # now lets do the update step for all of them!
    with torch.no_grad():
        lr = 1e-2
        # first the sequential model
        for p in model_seq.parameters():
            p.data -= lr * p.grad.data

        # next the data-parallel replicas
        model0 = models[0]
        for p in model0.parameters():
            p.data -= lr * p.grad.data
        for model in models[1:]:
            for p0, p in zip(model0.parameters(), model.parameters()):
                p.data.copy_(p0.data)
                assert torch.equal(p.data, p0.data)

        # assert equal
        for p, p_seq in zip(model0.parameters(), model_seq.parameters()):
            assert torch.allclose(p.data, p_seq.data, atol=1e-5), "model parameters should be roughly equal!"








def get_args():
    # parse command line args
    parser = argparse.ArgumentParser(description="Make More")
    # system/input/output
    parser.add_argument('--input-file', '-i', type=str, default='names.txt', help="input file with things one per line")
    parser.add_argument('--work-dir', '-o', type=str, default='out', help="output working directory")
    parser.add_argument('--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
    parser.add_argument('--sample-only', action='store_true', help="just sample from the model and quit, don't train")
    parser.add_argument('--num-workers', '-n', type=int, default=4, help="number of data workers for both train/test")
    parser.add_argument('--max-steps', type=int, default=-1, help="max number of optimization steps to run for, or -1 for infinite.")
    parser.add_argument('--device', type=str, default='cpu', help="device to use for compute, examples: cpu|cuda|cuda:2|mps")
    parser.add_argument('--seed', type=int, default=3407, help="seed")
    # sampling
    parser.add_argument('--top-k', type=int, default=-1, help="top-k for sampling, -1 means no top-k")
    # model
    parser.add_argument('--type', type=str, default='transformer', help="model class type to use, bigram|mlp|rnn|gru|bow|transformer")
    parser.add_argument('--n-layer', type=int, default=4, help="number of layers")
    parser.add_argument('--n-head', type=int, default=4, help="number of heads (in a transformer)")
    parser.add_argument('--n-embd', type=int, default=64, help="number of feature channels in the model")
    parser.add_argument('--n-embd2', type=int, default=64, help="number of feature channels elsewhere in the model")
    # optimization
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--use-torch-optim', action='store_true', default=False, help="Use the PyTorch optimizer implementations instead of our own.")
    parser.add_argument('--batch-size', '-b', type=int, default=32, help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4, help="learning rate")
    parser.add_argument('--momentum', '-m', type=float, default=0.0, help="momentum")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="weight decay")
    parser.add_argument('--nesterov', action='store_true', default=False)
    parser.add_argument('--alpha', type=float, default=0.99, help='The amount by which RMSProp decays its uncentered variance')
    parser.add_argument('--beta1', type=float, default=0.9, help="Exponential decay term for the first moment estimation in Adam & AdamW")
    parser.add_argument('--beta2', type=float, default=0.999, help="Exponential decay term for the second moment estimation in Adam & AdamW")
    args = parser.parse_args()
    print(vars(args))
    return args


def mp_forward_pass(i, queue: mp.Queue, model, batch, total_batch_size: int):
    torch.manual_seed(42)


    idx, targets = batch
    logits, loss = model(idx, targets)
    # we must scale the loss
    scaled_loss = loss * idx.shape[0] / total_batch_size
    scaled_loss.backward()
    # export gradients
    grads = [p.grad for p in model.parameters()]
    queue.put(grads)
    if i == 0:
        print(f'loss: {loss.item()}')
        


def run_better_dataparallel(args):
    print('inside dataparallel :)')
    assert dist.is_available(), "distributed should be available"
    torch.manual_seed(42)

    # XXX - we create datasets here not for consumption but to calculate values necessary to instantiate the model
    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    print(f"dataset determined that: {vocab_size=}, {block_size=}")

    # init model
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd, n_embd2=args.n_embd2)

    # setup model copy
    model = setup_model(args.type, config, args.device, args.work_dir, args.resume, args.sample_only)
    model = DataParallel(model, device_ids=[0, 1, 2, 3])
    optimizer = AdamW(model.parameters())
    # optimizer = SGD(model.parameters(), momentum=0.9)
    
    n_epochs = 1000
    for n in range(n_epochs):
        # zero gradients
        # for p in model.parameters():
        #     p.grad = None
        optimizer.zero_grad()

        inputs, targets = loader.next()
        logits, loss = model(inputs, targets)
        
        if n % 100 == 0:
            print(f"{n}: {loss.item():.4f}")
        loss.backward()
        optimizer.step()
        # with torch.no_grad():
        #     for p in model.parameters():
        #         p.data -= lr * p.grad.data






def run_dataparallel(args):
    print('inside dataparallel :)')
    assert dist.is_available(), "distributed should be available"
    torch.manual_seed(42)

    # XXX - we create datasets here not for consumption but to calculate values necessary to instantiate the model
    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    print(f"dataset determined that: {vocab_size=}, {block_size=}")

    # init model
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd, n_embd2=args.n_embd2)

    # setup model copy
    primary_model = setup_model(args.type, config, args.device, args.work_dir, args.resume, args.sample_only)
    primary_model.share_memory()

    # highly highly inefficient implementation
    epochs = 20
    for n in range(epochs):
        # extremely jank DataParallel training
        batch = loader.next()

        for p in primary_model.parameters():
            p.grad = None

        # now split up the batch into the split we t
        N = 4
        B = args.batch_size
        Bn = B // N

        # now we operate on inputs2
        inputs, targets = batch
        batch_inputs = torch.split(inputs, Bn, dim=0)
        batch_targets = torch.split(targets, Bn, dim=0)

        processes = []
        queue = mp.Queue()
        for i, batchi in enumerate(zip(batch_inputs, batch_targets)):
            proc = mp.Process(target=mp_forward_pass, args=(i, queue, primary_model, batchi, args.batch_size))
            proc.start()
            processes.append(proc)
    

        for i, process in enumerate(processes):
            print(f"awaiting process {i} to stop")
            process.join()

        # pull stuff from queue
        imported_grads = []
        while not queue.empty():
            grads = queue.get()
            imported_grads.append(grads)

        # now we have the imported gradients, we can successfully perform the update
        assert len(imported_grads) == N

        with torch.no_grad():
            # get the first gradient
            first_grad = imported_grads[0]
            for p, g in zip(primary_model.parameters(), first_grad):
                assert p.grad is None
                p.grad = g

            # sum up the remainder
            for grads in imported_grads[1:]:
                for p, g in zip(primary_model.parameters(), grads):
                    p.grad.data += g.data

            # update the weights
            lr = 1e-2
            for p in primary_model.parameters():
                p.data -= lr * p.grad.data









    # loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    # batch = loader.next()
    # idx, targets = batch
    # logits, loss = primary_model(idx, targets)
    # print(loss)



def mp_dev_main():
    args = get_args()
    torch.manual_seed(42)
    run_better_dataparallel(args)


    # init datasets
    # train_dataset, test_dataset = create_datasets(args.input_file)
    # vocab_size = train_dataset.get_vocab_size()
    # block_size = train_dataset.get_output_length()
    # print(f"dataset determined that: {vocab_size=}, {block_size=}")
    # # init model
    # config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
    #                    n_layer=args.n_layer, n_head=args.n_head,
    #                    n_embd=args.n_embd, n_embd2=args.n_embd2)

    # loader1 = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    # loader2 = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    # mock_dp(config, train_dataset)

def example_dp_train():
    args = get_args()

    # system inits
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    # init datasets
    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"dataset determined that: {vocab_size=}, {block_size=}")
    # init model
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd, n_embd2=args.n_embd2)

    model = setup_model(
        type=args.type,
        device=args.device,
        work_dir=args.work_dir,
        config=config,
        resume=args.resume,
        sample_only=args.sample_only,
    )

    if args.sample_only:
        print_samples(args=args, model=model, train_dataset=train_dataset, test_dataset=test_dataset, num=50)
        sys.exit()

    optimizer = setup_optimizer(args, model)

    # init dataloader
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    # batch_loader2 = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)


    train(
        args,
        model=model,
        optimizer=optimizer,
        batch_loader=batch_loader,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        writer=writer,
    )



def main():
    args = get_args()

    # system inits
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    # init datasets
    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"dataset determined that: {vocab_size=}, {block_size=}")
    # init model
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd, n_embd2=args.n_embd2)

    model = setup_model(
        type=args.type,
        device=args.device,
        work_dir=args.work_dir,
        config=config,
        resume=args.resume,
        sample_only=args.sample_only,
    )
    model = nn.DataParallel(model)

    if args.sample_only:
        print_samples(args=args, model=model, train_dataset=train_dataset, test_dataset=test_dataset, num=50)
        sys.exit()

    optimizer = setup_optimizer(args, model)

    # init dataloader
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    # batch_loader2 = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)


    train(
        args=args,
        model=model,
        optimizer=optimizer,
        batch_loader=batch_loader,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        writer=writer,
    )


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    mp_dev_main()
    # main()