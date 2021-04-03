import models
import sys
sys.path.append("..")
import vnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision
import os
import os.path

EVAL_ITER = 3
ADAM_LR = 3e-4
NUM_EPOCHS = 200
DEVICE = "cpu"

def load_cifar():
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10("../data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10("../data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
    return train_loader, test_loader

def load_mnist():
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=200, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=200, shuffle=False)
    return train_loader, test_loader

def eval_acc(model, loader, vectorized):
    num_correct = 0
    net_loss = 0.
    num_examples = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum") #note: sum, not mean here
    for batch_idx, (data, labels) in enumerate(loader):
        if vectorized and len(data.shape) == 4:
            input = vnn.expand_input_conv(data, 10)
        elif vectorized:
            input = vnn.expand_input(data, 10)
        else:
            input = data
        with torch.no_grad():
            out = model(input.to(DEVICE))
        if vectorized:
            out = out[:, :, 0]
        loss = loss_fn(out, labels.to(DEVICE)).item()
        net_loss += loss
        num_correct += (out.argmax(dim=1).cpu() == labels).int().sum().item()
        num_examples += len(data)
    acc = num_correct / num_examples
    loss = net_loss / num_examples
    return acc, loss

def save_snapshot(snapshot_dir, model, opt, epoch, train_loss, train_accuracy, test_loss, test_accuracy):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_accuracy,
        'test_loss': test_loss,
        'test_acc': test_accuracy
        }, "{}/epoch_{}.pt".format(snapshot_dir, epoch))
    print("saved snapshot at epoch {}".format(epoch))

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def files_in_dir(dir_name):
    filenames = sorted([os.path.join(dir_name, f) for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))])
    return filenames

def restart_from_snapshot(snapshot_dir, model, opt):
    make_dir(snapshot_dir)
    filenames = files_in_dir(snapshot_dir)
    if len(filenames) > 0:
        snapshot = torch.load(filenames[-1])
        model.load_state_dict(snapshot['model_state_dict'])
        opt.load_state_dict(snapshot['optimizer_state_dict'])
        epoch = snapshot['epoch']
        restarted = True
        print("loaded model from snapshot at epoch {}".format(epoch))
    else:
        epoch = 0
        restarted = False
    return epoch, restarted

def train_epoch(model, opt, train_loader, vectorized, learning_rule):
    epoch_loss = 0. #sum of batch-avg loss vals
    epoch_correct = 0 #sum of correct counts
    epoch_count = 0 #total # examples
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    for batch_idx, (data, labels) in enumerate(train_loader):
        if vectorized and len(data.shape) == 4:
            input = vnn.expand_input_conv(data, 10)
        elif vectorized:
            input = vnn.expand_input(data, 10)
        else:
            input = data
        opt.zero_grad()
        if learning_rule == "bp":
            #=====BP======
            out = model(input.to(DEVICE))
            if vectorized:
                out = out[:, :, 0]
            loss = loss_fn(out, labels.to(DEVICE))
            loss.backward()
        elif learning_rule == "df":
            #=====DF======
            with torch.no_grad():
                out = model(input.to(DEVICE))[:, :, 0]
            loss = loss_fn(out, labels.to(DEVICE))
        opt.step()
        if vectorized:
            vnn.post_step_callback(model)
        epoch_loss += loss.item()
        epoch_correct += (out.detach().argmax(dim=1).cpu() == labels).int().sum().item()
        epoch_count += len(data)
    current_loss = epoch_loss / (batch_idx + 1)
    current_acc = epoch_correct / epoch_count
    print("loss: {}, acc: {}".format(current_loss, current_acc))

def train_model(snapshot_dir, model, train_loader, test_loader, eval_iter, lr, num_epochs, vectorized, learning_rule):
    opt = optim.Adam(model.parameters(), lr=lr)
    snapshot_epoch, just_restarted = restart_from_snapshot(snapshot_dir, model, opt)
    for epoch in range(snapshot_epoch, num_epochs):
        if (epoch % eval_iter == 0) and not just_restarted:
            train_acc, train_loss = eval_acc(model, train_loader, vectorized=vectorized)
            test_acc, test_loss = eval_acc(model, test_loader, vectorized=vectorized)
            save_snapshot(snapshot_dir, model, opt, epoch, train_loss, train_acc, test_loss, test_acc)
        just_restarted = False
        train_epoch(model, opt, train_loader, vectorized, learning_rule)

        
def run_mnist_experiments(experiment_indices=None):
    train_loader, test_loader = load_mnist()
    common_params = {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "eval_iter": EVAL_ITER,
        "lr": ADAM_LR,
        "num_epochs": NUM_EPOCHS}
    if experiment_indices is None:
        experiment_indices = np.arange(15)
    for i in experiment_indices:
        #Fully connected
        if i == 0:
            model = models.make_mnist_nonvec_fc()
            train_model("models/mnist_nonvec_fc_bp", model, **common_params, vectorized=False, learning_rule="bp")
        elif i == 1:
            model = models.make_mnist_vec_fc(mono=False)
            train_model("models/mnist_vec_fc_bp_mixed", model, **common_params, vectorized=True, learning_rule="bp")
        elif i == 2:
            model = models.make_mnist_vec_fc(mono=True)
            train_model("models/mnist_vec_fc_bp_mono", model, **common_params, vectorized=True, learning_rule="bp")
        elif i == 3:
            model = models.make_mnist_vec_fc(mono=False)
            train_model("models/mnist_vec_fc_df_mixed", model, **common_params, vectorized=True, learning_rule="df")
        elif i == 4:
            model = models.make_mnist_vec_fc(mono=True)
            train_model("models/mnist_vec_fc_df_mono", model, **common_params, vectorized=True, learning_rule="df")
        #Convolutional
        elif i == 5:
            model = models.make_mnist_nonvec_conv()
            train_model("models/mnist_nonvec_conv_bp", model, **common_params, vectorized=False, learning_rule="bp")
        elif i == 6:
            model = models.make_mnist_vec_conv(mono=False)
            train_model("models/mnist_vec_conv_bp_mixed", model, **common_params, vectorized=True, learning_rule="bp")
        elif i == 7:
            model = models.make_mnist_vec_conv(mono=True)
            train_model("models/mnist_vec_conv_bp_mono", model, **common_params, vectorized=True, learning_rule="bp")
        elif i == 8:
            model = models.make_mnist_vec_conv(mono=False)
            train_model("models/mnist_vec_conv_df_mixed", model, **common_params, vectorized=True, learning_rule="df")
        elif i == 9:
            model = models.make_mnist_vec_conv(mono=True)
            train_model("models/mnist_vec_conv_df_mono", model, **common_params, vectorized=True, learning_rule="df")
        #Locally connected
        elif i == 10:
            model = models.make_mnist_nonvec_lc()
            train_model("models/mnist_nonvec_lc_bp", model, **common_params, vectorized=False, learning_rule="bp")
        elif i == 11:
            model = models.make_mnist_vec_lc(mono=False)
            train_model("models/mnist_vec_lc_bp_mixed", model, **common_params, vectorized=True, learning_rule="bp")
        elif i == 12:
            model = models.make_mnist_vec_lc(mono=True)
            train_model("models/mnist_vec_lc_bp_mono", model, **common_params, vectorized=True, learning_rule="bp")
        elif i == 13:
            model = models.make_mnist_vec_lc(mono=False)
            train_model("models/mnist_vec_lc_df_mixed", model, **common_params, vectorized=True, learning_rule="df")
        elif i == 14:
            model = models.make_mnist_vec_lc(mono=True)
            train_model("models/mnist_vec_lc_df_mono", model, **common_params, vectorized=True, learning_rule="df")

#if __name__ == "__main__":
#    run_mnist_experiments()
















