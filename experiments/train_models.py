import models
import sys
sys.path.append("..")
import vnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import os

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

def format_input(data, flatten, vectorized):
    if vectorized:
        if flatten:
            input = vnn.expand_input(torch.flatten(data, 1), 10)
        else:
            input = vnn.expand_input_conv(data, 10)
    else:
        if flatten:
            input = torch.flatten(data, 1)
        else:
            input = data
    return input

def eval_accuracy(model, loader, flatten, vectorized, device):
    loss_sum = 0.
    num_correct = 0
    num_examples = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum") #note: sum, not mean here
    model.eval()
    for batch_idx, (data, labels) in enumerate(loader):
        input = format_input(data, flatten, vectorized)
        with torch.no_grad():
            output = model(input.to(device))
        if vectorized:
            output = output[:, :, 0]
        loss = loss_fn(output, labels.to(device)).item()
        loss_sum += loss
        num_correct += (output.argmax(dim=1).cpu() == labels).int().sum().item()
        num_examples += len(data)
    accuracy = num_correct / num_examples
    loss = loss_sum / num_examples
    return accuracy, loss

def save_snapshot(snapshot_dir, model, opt, epoch, train_loss, train_accuracy, test_loss, test_accuracy,
    flatten, vectorized, learning_rule):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'flatten': flatten,
        'vectorized': vectorized,
        'learning_rule': learning_rule,
        'device': [p.device for p in model.parameters()][0]
        }, "{}/epoch_{}.pt".format(snapshot_dir, epoch))
    print("saved snapshot at epoch {}".format(epoch))
    print("train/test accuracy: {}/{}".format(train_accuracy, test_accuracy))

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def files_in_dir(dir_name):
    filenames = sorted([os.path.join(dir_name, f) for f in os.listdir(dir_name)
        if os.path.isfile(os.path.join(dir_name, f))])
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

def train_model(snapshot_dir, model, train_loader, test_loader, eval_iter, lr, num_epochs,
    flatten, vectorized, learning_rule, device):
    model = model.to(device)
    opt = optim.Adam([p for (name, p) in model.named_parameters() if name[-2:] != '.t'], lr=lr)
    #opt = optim.Adam(model.parameters(), lr=lr)
    snapshot_epoch, just_restarted = restart_from_snapshot(snapshot_dir, model, opt)
    #model = model.to(device)
    #opt = opt.to(device)
    for epoch in range(snapshot_epoch, num_epochs):
        if epoch % eval_iter == 0 and not just_restarted:
            train_accuracy, train_loss = eval_accuracy(model, train_loader, flatten, vectorized, device)
            test_accuracy, test_loss = eval_accuracy(model, test_loader, flatten, vectorized, device)
            save_snapshot(snapshot_dir, model, opt, epoch, train_loss, train_accuracy, test_loss, test_accuracy,
                flatten, vectorized, learning_rule)
        just_restarted = False
        train_epoch(model, opt, train_loader, flatten, vectorized, learning_rule, device)

def train_epoch(model, opt, train_loader, flatten, vectorized, learning_rule, device):
    avg_loss_sum = 0. #sum of batch-avg loss vals
    num_correct = 0 #sum of correct counts
    num_examples = 0 #total # examples
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        #print("Batch!")
        input = format_input(data, flatten, vectorized)
        opt.zero_grad()
        if learning_rule == "bp":
            #=====BP======
            output = model(input.to(device))
            if vectorized:
                output = output[:, :, 0]
            loss = loss_fn(output, labels.to(device))
            loss.backward()
        elif learning_rule == "df":
            #=====DF======
            with torch.no_grad():
                output = model(input.to(device))[:, :, 0]
            vnn.set_model_grads(model, output, labels)
            loss = loss_fn(output, labels.to(device))
        opt.step()
        if vectorized:
            vnn.post_step_callback(model)
        avg_loss_sum += loss.item()
        num_correct += (output.detach().argmax(dim=1).cpu() == labels).int().sum().item()
        num_examples += len(data)
    epoch_loss = avg_loss_sum / (batch_idx + 1)
    epoch_accuracy = num_correct / num_examples
    print("loss: {}, accuracy: {}".format(epoch_loss, epoch_accuracy))
        
def run_mnist_experiments(eval_iter=3, lr=3e-4, num_epochs=200, device="cpu", experiment_indices=None):
    train_loader, test_loader = load_mnist()
    common_params = {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "eval_iter": eval_iter,
        "lr": lr,
        "num_epochs": num_epochs,
        "device": device}
    if experiment_indices is None:
        experiment_indices = np.arange(15)
    for i in experiment_indices:
        #Fully connected
        if i == 0:
            model = models.make_mnist_nonvec_fc()
            train_model("models/mnist_nonvec_fc_bp", model, **common_params,
                flatten=True, vectorized=False, learning_rule="bp")
        elif i == 1:
            model = models.make_mnist_vec_fc(mono=False)
            train_model("models/mnist_vec_fc_bp_mixed", model, **common_params,
                flatten=True, vectorized=True, learning_rule="bp")
        elif i == 2:
            model = models.make_mnist_vec_fc(mono=True)
            train_model("models/mnist_vec_fc_bp_mono", model, **common_params,
                flatten=True, vectorized=True, learning_rule="bp")
        elif i == 3:
            model = models.make_mnist_vec_fc(mono=False)
            train_model("models/mnist_vec_fc_df_mixed", model, **common_params,
                flatten=True, vectorized=True, learning_rule="df")
        elif i == 4:
            model = models.make_mnist_vec_fc(mono=True)
            train_model("models/mnist_vec_fc_df_mono", model, **common_params,
                flatten=True, vectorized=True, learning_rule="df")
        #Convolutional
        elif i == 5:
            model = models.make_mnist_nonvec_conv()
            train_model("models/mnist_nonvec_conv_bp", model, **common_params,
                flatten=False, vectorized=False, learning_rule="bp")
        elif i == 6:
            model = models.make_mnist_vec_conv(mono=False)
            train_model("models/mnist_vec_conv_bp_mixed", model, **common_params,
                flatten=False, vectorized=True, learning_rule="bp")
        elif i == 7:
            model = models.make_mnist_vec_conv(mono=True)
            train_model("models/mnist_vec_conv_bp_mono", model, **common_params,
                flatten=False, vectorized=True, learning_rule="bp")
        elif i == 8:
            model = models.make_mnist_vec_conv(mono=False)
            train_model("models/mnist_vec_conv_df_mixed", model, **common_params,
                flatten=False, vectorized=True, learning_rule="df")
        elif i == 9:
            model = models.make_mnist_vec_conv(mono=True)
            train_model("models/mnist_vec_conv_df_mono", model, **common_params,
                flatten=False, vectorized=True, learning_rule="df")
        #Locally connected
        elif i == 10:
            model = models.make_mnist_nonvec_lc()
            train_model("models/mnist_nonvec_lc_bp", model, **common_params,
                flatten=False, vectorized=False, learning_rule="bp")
        elif i == 11:
            model = models.make_mnist_vec_lc(mono=False)
            train_model("models/mnist_vec_lc_bp_mixed", model, **common_params,
                flatten=False, vectorized=True, learning_rule="bp")
        elif i == 12:
            model = models.make_mnist_vec_lc(mono=True)
            train_model("models/mnist_vec_lc_bp_mono", model, **common_params,
                flatten=False, vectorized=True, learning_rule="bp")
        elif i == 13:
            model = models.make_mnist_vec_lc(mono=False)
            train_model("models/mnist_vec_lc_df_mixed", model, **common_params,
                flatten=False, vectorized=True, learning_rule="df")
        elif i == 14:
            model = models.make_mnist_vec_lc(mono=True)
            train_model("models/mnist_vec_lc_df_mono", model, **common_params,
                flatten=False, vectorized=True, learning_rule="df")

#if __name__ == "__main__":
#    run_mnist_experiments()
















