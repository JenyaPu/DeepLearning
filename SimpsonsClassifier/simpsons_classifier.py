# -*- coding: utf-8 -*-

# Лучший score на kaggle 0.98405
# Ник на kaggle Evgenii_Pustozerov_282335642
# Я работал с кодом в PyCharm и только в конце преобразовал его в ноутбук.
# Мне намного удобнее работать с инспектором и отладчиком, это было очень важно для первоначального редактирования кода.
# Поэтому мой итоговый файл очень сильно отличается от того, что был в ноутбуке, но зато сразу видно, что это
# самостоятельное решение.

import pickle
from abc import ABC

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
import torchvision
import torch
import random
# import warnings
# from skimage import io
# from multiprocessing.pool import ThreadPool
# from os.path import exists

# Здесь я настраиваю работу на видеокарту компьютера
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
torch.cuda.is_available()


# Важные переменные
DATA_MODES = ['train', 'val', 'test']
RESCALE_SIZE = 224
DEVICE = torch.device("cuda")
random.seed(1)


# Очень простая сеть, использовалась только первоначально
class SimpleCnn(nn.Module, ABC):

    def __init__(self, n_classes1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.out = nn.Linear(96 * 5 * 5, n_classes1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), -1)
        logits = self.out(x)
        return logits


def load_sample(file):
    image = Image.open(file)
    image.load()
    return image


def _prepare_sample(image):
    image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
    return image


# Загрузчик данных, доработан, чтобы выдавать на выходе тензоры
class SimpsonsDataset(Dataset):
    """
    Датасет с картинками, который паралельно подгружает их из папок
    производит скалирование и превращение в торчевые тензоры
    """

    def __init__(self, files, mode):
        super().__init__()
        # список файлов для загрузки
        self.files = sorted(files)
        # режим работы
        self.mode = mode

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        self.len_ = len(self.files)

        self.label_encoder = LabelEncoder()

        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            with open('label_encoder.pkl', 'wb') as le_dump_file:
                pickle.dump(self.label_encoder, le_dump_file)

    def __len__(self):
        return self.len_

    def __getitem__(self, index):
        # Я пробовал много разных методов аугментации, но лучший результат получился при использовании только
        # Random Horizontal Flip
        transform1 = transforms.Compose([
            transforms.ColorJitter(brightness=0.25),
            transforms.ColorJitter(contrast=0.25),
            transforms.ColorJitter(saturation=0.25),
            transforms.ColorJitter(hue=0.25),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        x = load_sample(self.files[index])
        x = _prepare_sample(x)
        x = transform1(x)
        x = x / 255

        if self.mode == 'test':
            return x
        else:
            label1 = self.labels[index]
            label_id = self.label_encoder.transform([label1])
            y = label_id.item()
            return x, y


def imshow(inp, plt_ax=plt):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    plt_ax.grid(False)


def fit_epoch(model1, train_loader, criterion, optimizer):
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0
    all_preds = []
    all_labels = []

    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model1(inputs)
        loss1 = criterion(outputs, labels)
        loss1.backward()
        optimizer.step()
        #
        preds1 = torch.argmax(outputs, 1)
        running_loss += loss1.item() * inputs.size(0)
        # noinspection PyTypeChecker
        running_corrects += torch.sum(preds1 == labels.data)
        processed_data += inputs.size(0)
        all_labels.extend(labels.data.cpu().numpy())
        all_preds.extend(preds1.cpu().numpy())

    train_loss1 = running_loss / processed_data
    train_acc1 = running_corrects.cpu().numpy() / processed_data
    # PyUnresolvedReferences
    train_f11 = f1_score(all_labels, all_preds, average="micro")
    return train_loss1, train_acc1, train_f11


def eval_epoch(model1, val_loader, criterion):
    model1.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0
    all_preds = []
    all_labels = []

    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(False):
            outputs = model1(inputs)
            loss1 = criterion(outputs, labels)
            preds1 = torch.argmax(outputs, 1)

        running_loss += loss1.item() * inputs.size(0)
        # noinspection PyTypeChecker
        running_corrects += torch.sum(preds1 == labels.data)
        processed_size += inputs.size(0)
        all_labels.extend(labels.data.cpu().numpy())
        all_preds.extend(preds1.cpu().numpy())
    val_loss1 = running_loss / processed_size
    # noinspection PyUnresolvedReferences
    val_acc1 = running_corrects.double() / processed_size
    val_f11 = f1_score(all_labels, all_preds, average="micro")
    return val_loss1, val_acc1, val_f11


# Функция доработана, чтобы выдавать F1-score
def train(train_dataset1, val_dataset1, model, epochs, batch_size):
    train_loader = DataLoader(train_dataset1, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset1, batch_size=batch_size, shuffle=False)

    history1 = []
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} val_loss: {v_loss:0.4f} train_acc: {t_acc:0.4f} " \
                   "val_acc: {v_acc:0.4f} train_f1: {t_f1:0.4f} val_f1: {v_f1:0.4f}"

    # Здесь я пробовал разные оптимизаторы и schedulers, лучший результат получился при Adam c lr=0.000005
    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        opt = torch.optim.Adam(model.parameters(), lr=0.000005)
        # opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            train_loss1, train_acc1, train_f11 = fit_epoch(model, train_loader, criterion, opt)

            val_loss1, val_acc1, val_f11 = eval_epoch(model, val_loader, criterion)
            history1.append((train_loss1, train_acc1, val_loss1, val_acc1, train_f11, val_f11))

            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch + 1, t_loss=train_loss1, v_loss=val_loss1, t_acc=train_acc1,
                                           v_acc=val_acc1, t_f1=train_f11, v_f1=val_f11))

    return history1


def predict(model1, test_loader1):
    with torch.no_grad():
        logits = []

        for inputs in test_loader1:
            inputs = inputs.to(DEVICE)
            model1.eval()
            outputs = model1(inputs).cpu()
            logits.append(outputs)

    probs1 = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    return probs1


def predict_one_sample(model1, inputs, device=DEVICE):
    with torch.no_grad():
        inputs = inputs.to(device)
        model1.eval()
        logit = model1(inputs).cpu()
        probs1 = torch.nn.functional.softmax(logit, dim=-1).numpy()
    return probs1


# MAIN WORKING CYCLE
# LOAD DATA
TRAIN_DIR = Path('journey-springfield/train/train')
TEST_DIR = Path('journey-springfield/test/test')

train_val_files = sorted(list(TRAIN_DIR.rglob('*.jpg')))
test_files = sorted(list(TEST_DIR.rglob('*.jpg')))
print(len(train_val_files))

train_val_labels = [path.parent.name for path in train_val_files]
train_files, val_files = train_test_split(train_val_files, test_size=0.25, stratify=train_val_labels)

val_dataset = SimpsonsDataset(val_files, mode='val')

# Первоначальное отображение персонажей
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(8, 8), sharey='all', sharex='all')
for fig_x in ax.flatten():
    random_characters = int(np.random.uniform(0, 1000))
    im_val, label = val_dataset[random_characters]
    img_label = " ".join(map(lambda x: x.capitalize(),
                             val_dataset.label_encoder.inverse_transform([label])[0].split('_')))
    imshow(im_val.data.cpu(), plt_ax=fig_x)
n_classes = len(np.unique(train_val_labels))
print("Amount of classes: {}".format(n_classes))

# CHOOSE MODEL
# Я пробовал разные модели разных конфигураций, но лучший результат у меня получился с предобученной resnet50
simple_cnn = SimpleCnn(n_classes).to(DEVICE)
resnet18 = torchvision.models.resnet18(pretrained=True).to(DEVICE)
resnet50 = torchvision.models.resnet50(pretrained=True).to(DEVICE)
vgg16 = torchvision.models.vgg16().to(DEVICE)
selected_model = resnet50

# Я пробовал замораживать разные слои, но лучший результат получился при заморозке только первого слоя
ct = 0
for child in selected_model.children():
    ct += 1
    if ct < 8:
        for param in child.parameters():
            param.requires_grad = False


# TRAIN MODEL
if val_dataset is None:
    val_dataset = SimpsonsDataset(val_files, mode='val')
train_dataset = SimpsonsDataset(train_files, mode='train')
history = train(train_dataset, val_dataset, model=selected_model, epochs=15, batch_size=64)
# Сохраняю сеть, чтобы можно было потом ей воспользоваться
torch.save(selected_model, "models/resnet50_7")

# Построение кривых обучения
train_loss, train_acc, train_f1, val_loss, val_acc, val_f1 = zip(*history)
plt.figure(figsize=(15, 9))
plt.plot(train_loss, label="train_loss")
plt.plot(val_loss, label="val_loss")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

# EVALUATE MODEL
selected_model = torch.load("models/resnet50_7")
selected_model.eval()
random_characters = int(np.random.uniform(0, 1000))
ex_img, true_label = val_dataset[random_characters]
# probs_im = predict_one_sample(selected_model, ex_img.unsqueeze(0))
idxs = list(map(int, np.random.uniform(0, 1000, 20)))
imgs = [val_dataset[id1][0].unsqueeze(0) for id1 in idxs]
# imgs = [val_dataset[id1][0] for id1 in idxs]
probs_ims = predict(selected_model, imgs)
label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))
y_pred = np.argmax(probs_ims, -1)
actual_labels = [val_dataset[id1][1] for id1 in idxs]
preds_class = [i for i in y_pred]
# preds_class_labels = [label_encoder.classes_[i] for i in y_pred]

# Расчет F1-score
f1 = f1_score(actual_labels, preds_class, average='micro')
print("F1 score: %s, calculated on %s objects" % (f1, len(actual_labels)))
all_idxs = np.arange(len(val_dataset))
all_imgs = [val_dataset[id1][0].unsqueeze(0) for id1 in all_idxs]
all_actual_labels = [val_dataset[id1][1] for id1 in all_idxs]
all_probs_ims = predict(selected_model, all_imgs)
y_pred_all = np.argmax(all_probs_ims, -1)
f1 = f1_score(all_actual_labels, y_pred_all, average='micro')
print("F1 score: %s, calculated on %s objects" % (f1, len(all_actual_labels)))

# Отображение рисунков с надписями
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 12), sharey='all', sharex='all')
for fig_x in ax.flatten():
    random_characters = int(np.random.uniform(0, 1000))
    im_val, label = val_dataset[random_characters]
    img_label = " ".join(map(lambda x: x.capitalize(),
                             val_dataset.label_encoder.inverse_transform([label])[0].split('_')))

    imshow(im_val.data.cpu(), plt_ax=fig_x)

    actual_text = "Actual : {}".format(img_label)

    fig_x.add_patch(patches.Rectangle((0, 53), 86, 35, color='white'))
    font0 = FontProperties()
    font = font0.copy()
    font.set_family("fantasy")
    prob_pred = predict_one_sample(selected_model, im_val.unsqueeze(0))
    predicted_proba = np.max(prob_pred) * 100
    y_pred = np.argmax(prob_pred)

    predicted_label = label_encoder.classes_[y_pred]
    predicted_label = predicted_label[:len(predicted_label) // 2] + '\n' + predicted_label[len(predicted_label) // 2:]
    predicted_text = "{} : {:.0f}%".format(predicted_label, predicted_proba)

    fig_x.text(1, 59, predicted_text, horizontalalignment='left', fontproperties=font,
               verticalalignment='top', fontsize=8, color='black', fontweight='bold')
plt.show()

# Подготовка решения для Kaggle
test_dataset = SimpsonsDataset(test_files, mode="test")
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)
probs = predict(selected_model, test_loader)
preds = label_encoder.inverse_transform(np.argmax(probs, axis=1))
test_filenames = [path.name for path in test_dataset.files]

my_submit = pd.DataFrame({'Id': test_filenames, 'Expected': preds})
my_submit.head()
my_submit.to_csv('journey-springfield/submission_final.csv', index=False)
