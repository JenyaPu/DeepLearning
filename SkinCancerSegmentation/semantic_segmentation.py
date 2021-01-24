# -*- coding: utf-8 -*-
# В данной секции импортируются все используемые в работе библиотеки
from abc import ABC
from skimage.io import imread
from skimage.transform import resize
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch
import torch.nn as nn
import random
from time import time
import pandas as pd

# Ниже все комментарии из ноутбука для справки, собраны в одном месте для упрощения навигации по коду
"""
1.Для начала мы скачаем датасет: [ADDI project](https://www.fc.up.pt/addi/ph2%20database.html).
! wget https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar
get_ipython().system_raw("unrar x PH2Dataset.rar")
Стуктура датасета у нас следующая:
    IMD_002/
        IMD002_Dermoscopic_Image/
            IMD002.bmp
        IMD002_lesion/
            IMD002_lesion.bmp
        IMD002_roi/
            ...
    IMD_003/
        ...
        ...
Для загрузки датасета я предлагаю использовать skimage: 
[`skimage.io.imread()`](https://scikit-image.org/docs/dev/api/skimage.io.html)
"""
"""
Изображения имеют разные размеры. Давайте изменим их размер на $256\times256 $ пикселей. 
[`skimage.transform.resize()`](https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize) 
можно использовать для изменения размера изображений. 
Эта функция также автоматически нормализует изображения в диапазоне $[0,1]$.
"""
"""
Чтобы убедиться, что все корректно, мы нарисуем несколько изображений
"""
"""
Разделим наши 200 картинок на 100/50/50 для валидации и теста
"""
"""# Реализация различных архитектур:
Ваше задание будет состоять в том, 
чтобы написать несколько нейросетевых архитектур для решения задачи семантической сегментации. 
Сравнить их по качеству на тесте и испробовать различные лосс функции для них.
-----------------------------------------------------------------------------------------
# SegNet [2 балла]
* Badrinarayanan, V., Kendall, A., & Cipolla, R. (2015). [SegNet: A deep convolutional
encoder-decoder architecture for image segmentation](https://arxiv.org/pdf/1511.00561.pdf)
Внимательно посмотрите из чего состоит модель и для чего выбраны те или иные блоки.
"""
"""## Метрика
В данном разделе предлагается использовать следующую метрику для оценки качества:
$I o U=\frac{\text {target } \cap \text { prediction }}{\text {target } \cup{prediction }}$
Пересечение (A ∩ B) состоит из пикселей, найденных как в маске предсказания, так и в основной маске истины, 
тогда как объединение (A ∪ B) просто состоит из всех пикселей, 
найденных либо в маске предсказания, либо в целевой маске.
To clarify this we can see on the segmentation:
![alt text](https://www.jeremyjordan.me/content/images/2018/05/target_prediction.png)
And the intersection will be the following:
![alt text](https://www.jeremyjordan.me/content/images/2018/05/intersection_union.png)
"""
"""## Функция потерь [1 балл]
Теперь не менее важным, чем построение архитектуры, является определение **оптимизатора** и **функции потерь.**
Функция потерь - это то, что мы пытаемся минимизировать. 
Многие из них могут быть использованы для задачи бинарной семантической сегментации. 
Популярным методом для бинарной сегментации является *бинарная кросс-энтропия*, которая задается следующим образом:
$$\mathcal L_{BCE}(y, \hat y) = -\sum_i \left[y_i\log\sigma(\hat y_i) + (1-y_i)\log(1-\sigma(\hat y_i))\right].$$
где $y$ это  таргет желаемого результата и $\hat y$ является выходом модели. 
$\sigma$ - это [*логистическая* функция](https://en.wikipedia.org/wiki/Sigmoid_function), 
который преобразует действительное число $\mathbb R$ в вероятность $[0,1]$.
Однако эта потеря страдает от проблем численной нестабильности. 
Самое главное, что $\lim_{x\rightarrow0}\log(x)=\infty$ приводит к неустойчивости в процессе оптимизации. 
Рекомендуется посмотреть следующее [упрощение]
(https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits) 
в Тарая функция эквивалентна и не так подвержена численной неустойчивости.
$$\mathcal L_{BCE} = \hat y - y\hat y + \log\left(1+\exp(-\hat y)\right).$$
"""
"""## Тренировка [1 балл]
Мы определим цикл обучения в функции, чтобы мы могли повторно использовать его.
"""
"""
## Основной момент: обучение
Обучите вашу модель. Обратите внимание, что обучать необходимо до сходимости. 
Если указанного количества эпох (20) не хватило, попробуйте изменять количество эпох до сходимости алгоритма. 
Сходимость определяйте по изменению функции потерь на валидационной выборке.
 С параметрами оптимизатора можно спокойно играть, пока вы не найдете лучший вариант для себя.
"""
"""## Инференс [1 балл]
После обучения модели эту функцию можно использовать для прогнозирования сегментации на новых данных:
"""
"""
## [BONUS] Мир сегментационных лоссов [5 баллов]
В данном блоке предлагаем вам написать одну функцию потерь самостоятельно. 
Для этого необходимо прочитать статью и имплементировать ее. 
Кроме тако провести численное сравнение с предыдущими функциями.
Какие варианты? 
1) Можно учесть Total Variation
2) Lova
3) BCE но с Soft Targets (что-то типа label-smoothing для многослассовой классификации)
4) Любой другой 
* [Physiological Inspired Deep Neural Networks for Emotion Recognition]
(https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8472816&tag=1)". IEEE Access, 6, 53930-53943.
* [Boundary loss for highly unbalanced segmentation]
(https://arxiv.org/abs/1812.07032)
* [Tversky loss function for image segmentation using 3D fully convolutional deep networks]
(https://arxiv.org/abs/1706.05721)
* [Correlation Maximized Structural Similarity Loss for Semantic Segmentation]
(https://arxiv.org/abs/1910.08711)
* [Topology-Preserving Deep Image Segmentation]
(https://papers.nips.cc/paper/8803-topology-preserving-deep-image-segmentation)
Так как Тверский лосс очень похож на данные выше, то за него будет проставлено только 3 балла 
(при условии, если в модели нет ошибок при обучении). Постарайтесь сделать что-то интереснее.
"""
"""-----------------------------------------------------------------------------------------
# U-Net [2 балла]
[**U-Net**](https://arxiv.org/abs/1505.04597) это архитектура нейронной сети, 
которая получает изображение и выводит его. 
Первоначально он был задуман для семантической сегментации (как мы ее будем использовать), 
но он настолько успешен, что с тех пор используется в других контекстах. 
Учитывая медицинское изображение, он выводит изображение в оттенках серого, представляющее вероятность того, 
что каждый пиксель является интересующей областью.
У нас в архитектуре все так же существует енкодер и декодер, как в **SegNet**, 
но отличительной особеностью данной модели являются skip-connections. 
Элементы соединяющие части декодера и енкодера. 
То есть для того чтобы передать на вход декодера тензор, мы конкатенируем симметричный выход с энкодера 
и выход предыдущего слоя декодера.
* Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. 
"[U-Net: Convolutional networks for biomedical image segmentation.](https://arxiv.org/pdf/1505.04597.pdf)" 
International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.
"""
"""Ответьте себе на вопрос: не переобучается ли моя модель?
# Нет, модель не переобучается, поскольку train loss и val loss примерно на одном уровне.

-----------------------------------------------------------------------------------------

## Дополнительные функции потерь [2 балла]
В данном разделе вам потребуется имплементировать две функции потерь:
DICE и Focal loss. 
Если у вас что-то не учится, велика вероятность, что вы ошиблись или учите слишком мало, 
прежде чем бить тревогу попробуйте оперебирать различные варианты, убедитесь, 
что во всех других сетапах сетть достигает желанного результата. 
СПОЙЛЕР: учиться она будет при всех лоссах предложенных в этом задании.

**1. Dice coefficient:** 
Учитывая две маски $X$ и $Y$, 
общая метрика для измерения расстояния между этими двумя масками задается следующим образом:
$$D(X,Y)=\frac{2|X\cap Y|}{|X|+|Y|}$$
Эта функция не является дифференцируемой, но это необходимое свойство для градиентного спуска. 
В данном случае мы можем приблизить его с помощью:
$$\mathcal L_D(X,Y) = 1-\frac{1}{256 \times 256} \times \sum_i\frac{2X_iY_i}{X_i+Y_i}.$$
Не забудьте подумать о численной нестабильности, возникаемой в математической формуле.
"""
"""[**2. Focal loss:**](https://arxiv.org/pdf/1708.02002.pdf) 
Окей, мы уже с вами умеем делать BCE loss:
$$\mathcal L_{BCE}(y, \hat y) = -\sum_i \left[y_i\log\sigma(\hat y_i) + (1-y_i)\log(1-\sigma(\hat y_i))\right].$$
Проблема с этой потерей заключается в том, что она имеет тенденцию приносить пользу классу **большинства** (фоновому)
по отношению к классу **меньшинства** ( переднему). Поэтому обычно применяются весовые коэффициенты к каждому классу:
$$\mathcal L_{wBCE}(y, \hat y) = 
-\sum_i \alpha_i\left[y_i\log\sigma(\hat y_i) + (1-y_i)\log(1-\sigma(\hat y_i))\right].$$
Традиционно вес $\alpha_i$ определяется как обратная частота класса этого пикселя $i$, 
так что наблюдения миноритарного класса весят больше по отношению к классу большинства.
Еще одним недавним дополнением является взвешенный пиксельный вариант, 
которая взвешивает каждый пиксель по степени уверенности, которую мы имеем в предсказании этого пикселя.
$$\mathcal L_{focal}(y, \hat y) = -\sum_i \left[\left(1-\sigma(\hat y_i)\right)^\gamma 
y_i\log\sigma(\hat y_i) + (1-y_i)\log(1-\sigma(\hat y_i))\right].$$
Зафиксируем значение $\gamma=2$.
--------------------------------------------------------------------------------
"""
"""
Новая модель путем изменения типа пулинга:
 **Max-Pooling** for the downsampling and **nearest-neighbor Upsampling** for the upsampling.
Down-sampling:
        conv = nn.Conv2d(3, 64, 3, padding=1)
        pool = nn.MaxPool2d(3, 2, padding=1)
Up-Sampling
        upsample = nn.Upsample(32)
        conv = nn.Conv2d(64, 64, 3, padding=1)
Замените max-pooling на convolutions с stride=2 и upsampling на transpose-convolutions с stride=2.
"""
"""Сделайте вывод какая из моделей лучше
# Отчет (6 баллов): 
Ниже предлагается написать отчет о проделанно работе и построить графики для лоссов, метрик на валидации и тесте. 
Если вы пропустили какую-то часть в задании выше, то вы все равно можете получить основную часть баллов в отчете, 
если правильно зададите проверяемые вами гипотезы.
Аккуратно сравните модели между собой и соберите наилучшую архитектуру. Проверьте каждую модель с различными лоссами. 
Мы не ограничиваем вас в формате отчета, но проверяющий должен отчетливо понять для чего построен каждый график, 
какие выводы вы из него сделали и какой общий вывод можно сделать на основании данных моделей. 
Если вы захотите добавить что-то еще, чтобы увеличить шансы получения максимального балла, 
то добавляйте отдельное сравнение.

Дополнительные комментарии: 
Пусть у вас есть N обученных моделей.
- Является ли отчетом N графиков с 1 линей? 
Да, но очень низкокачественным, потому что проверяющий не сможет сам сравнить их.
- Является ли отчетом 1 график с N линиями? 
Да, но скорее всего таким образом вы отразили лишь один эффект. 
Этого мало, чтобы сделать досточно суждений по поводу вашей работа.
- Я проверял метрики на трейне, и привел в результате таблицу с N числами, что не так? 
ключейвой момент тут, что вы измеряли на трейне ваши метрики, 
уверены ли вы, что заивисмости останутся такими же на отложенной выборке?
- Я сделал отчет содержащий график лоссов и метрик, и у меня нет ошибок в основной части, 
но за отчет не стоит максимум, почему? 
Естестественно максимум баллов за отчет можно получить не за 2 графика (даже при условии их полной правильности). 
Проверяющий хочет видеть больше сравнений моделей, чем метрики и лоссы (особенно, если они на трейне).
Советы: попробуйте правильно поставить вопрос на который вы себе отвечаете и продемонстрировать таблицу/график, 
помогающий проверяющему увидеть ответ на этот вопрос. 
Пример: Ваня хочет узнать, с каким из 4-х лоссов модель (например, U-Net) имеет наилучшее качество. 
Что нужно сделать Ване? Обучить 4 одинаковых модели с разными лосс функциями. И измерить итогововое качество. 
Продемонстрировать результаты своих измерений и итоговый вывод. 
(warning: конечно же, это не идеально ответит на наш вопрос, 
так как мы не учитываем в экспериментах возможные различные типы ошибок, 
но для первого приближения этого вполне достаточно).
Примерное время на подготовку отчета 1 час, он содержит сравнеение метрик, график лоссов, 
выбор лучших моделей из нескольких кластеров и выбор просто лучшей модели, небольшой вывод по всему дз, 
возможно сравнение результирующих сегментаций, времени или числа параметров модели, проявляйте креативность.
"""

# Настройка вычислений по GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
    DEVICE = torch.device("cpu")
else:
    print('CUDA is available!  Training on GPU ...')
    DEVICE = torch.device("cuda")


# Первичная загрузка и предобработка датасета
def plot_images():
    plt.figure(figsize=(18, 6))
    for i in range(6):
        plt.subplot(2, 6, i+1)
        plt.axis("off")
        plt.imshow(X[i])
        plt.subplot(2, 6, i+7)
        plt.axis("off")
        plt.imshow(Y[i])
    plt.show()


random.seed(1)
images = []
lesions = []
root = 'PH2Dataset'
for root, dirs, files in os.walk(os.path.join(root, 'PH2 Dataset images')):
    if root.endswith('_Dermoscopic_Image'):
        images.append(imread(os.path.join(root, files[0])))
    if root.endswith('_lesion'):
        lesions.append(imread(os.path.join(root, files[0])))

size = (256, 256)
X = [resize(x, size, mode='constant', anti_aliasing=True,) for x in images]
Y = [resize(y, size, mode='constant', anti_aliasing=False) > 0.5 for y in lesions]
X = np.array(X, np.float32)
Y = np.array(Y, np.float32)
print(f'Loaded {len(X)} images')
len(lesions)
ix = np.random.choice(len(X), len(X), False)
tr, val, ts = np.split(ix, [100, 150])
print(len(tr), len(val), len(ts))


# Настройка Data Loader
batch_size = 12
# noinspection PyTypeChecker
data_tr = DataLoader(list(zip(np.rollaxis(X[tr], 3, 1), Y[tr, np.newaxis])),
                     batch_size=batch_size, shuffle=True)
# noinspection PyTypeChecker
data_val = DataLoader(list(zip(np.rollaxis(X[val], 3, 1), Y[val, np.newaxis])),
                      batch_size=batch_size, shuffle=True)
# noinspection PyTypeChecker
data_ts = DataLoader(list(zip(np.rollaxis(X[ts], 3, 1), Y[ts, np.newaxis])),
                     batch_size=batch_size, shuffle=True)


# Ниже идут подготовленные мной архитектуры и используемые ими вспомогательные функции
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        # nn.BatchNorm2d(in_channels, momentum=batchnorm_momentum),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        # nn.BatchNorm2d(in_channels, momentum=batchnorm_momentum),
        nn.ReLU(inplace=True))


def crop_image(input_tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = input_tensor.size()[2]
    delta = (tensor_size - target_size) // 2
    return input_tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]


class SegNet(nn.Module, ABC):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )
        self.pool0 = nn.MaxPool2d(2, 2, return_indices=True)  # 256 -> 128
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)  # 128 -> 64
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)  # 64 -> 32
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)  # 32 -> 16

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # decoder (upsampling)
        self.upsample0 = nn.MaxUnpool2d(2, stride=2)  # 16 -> 32
        self.dec_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.upsample1 = nn.MaxUnpool2d(2, stride=2)  # 32 -> 64
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.upsample2 = nn.MaxUnpool2d(2, stride=2)  # 64 -> 128
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.upsample3 = nn.MaxUnpool2d(2, stride=2)  # 128 -> 256
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = x.to(DEVICE)

        # encoder
        e0, ind0 = self.pool0(self.enc_conv0(x))
        e1, ind1 = self.pool1(self.enc_conv1(e0))
        e2, ind2 = self.pool2(self.enc_conv2(e1))
        e3, ind3 = self.pool3(self.enc_conv3(e2))

        # bottleneck
        b = self.bottleneck_conv(e3)

        # decoder
        d0 = self.dec_conv0(self.upsample0(b, ind3))
        d1 = self.dec_conv1(self.upsample1(d0, ind2))
        d2 = self.dec_conv2(self.upsample2(d1, ind1))
        d3 = self.dec_conv3(self.upsample3(d2, ind0))  # no activation

        return d3


class UNet(nn.Module, ABC):

    def __init__(self):
        super().__init__()

        self.conv_down1 = double_conv(3, 64)
        self.conv_down2 = double_conv(64, 128)
        self.conv_down3 = double_conv(128, 256)
        self.conv_down4 = double_conv(256, 512)
        self.conv_down5 = double_conv(512, 1024)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.trans_up1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.conv_up1 = double_conv(1024, 512)
        self.trans_up2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv_up2 = double_conv(512, 256)
        self.trans_up3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv_up3 = double_conv(256, 128)
        self.trans_up4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv_up4 = double_conv(128, 64)
        self.last_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, image):
        # Encoder part
        x1 = self.conv_down1(image)
        x2 = self.maxpool(x1)
        x3 = self.conv_down2(x2)
        x4 = self.maxpool(x3)
        x5 = self.conv_down3(x4)
        x6 = self.maxpool(x5)
        x7 = self.conv_down4(x6)
        x8 = self.maxpool(x7)
        x9 = self.conv_down5(x8)

        # Decoder part
        x = self.trans_up1(x9)
        y = crop_image(x7, x)
        x = self.conv_up1(torch.cat([x, y], 1))
        x = self.trans_up2(x)
        y = crop_image(x5, x)
        x = self.conv_up2(torch.cat([x, y], 1))
        x = self.trans_up3(x)
        y = crop_image(x3, x)
        x = self.conv_up3(torch.cat([x, y], 1))
        x = self.trans_up4(x)
        y = crop_image(x1, x)
        x = self.conv_up4(torch.cat([x, y], 1))
        out = self.last_conv(x)
        return out


class UNet2(nn.Module, ABC):

    def __init__(self):
        super().__init__()

        self.conv_down1 = double_conv(3, 64)
        self.conv_down2 = double_conv(64, 128)
        self.conv_down3 = double_conv(128, 256)
        self.conv_down4 = double_conv(256, 512)
        self.conv_down5 = double_conv(512, 1024)

        self.maxpool1 = nn.Conv2d(3, 64, 3, padding=1)
        self.maxpool2 = nn.Conv2d(64, 128, 3, padding=1)
        self.maxpool3 = nn.Conv2d(128, 256, 3, padding=1)
        self.maxpool4 = nn.Conv2d(256, 512, 3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.trans_up1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.maxpool1 = double_conv(1024, 512)
        self.trans_up2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.maxpool2 = double_conv(512, 256)
        self.trans_up3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.maxpool3 = double_conv(256, 128)
        self.trans_up4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.maxpool4 = double_conv(128, 64)
        self.last_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, image):
        # Encoder part
        x1 = self.conv_down1(image)
        x2 = self.upsample(x1)
        x3 = self.conv_down2(x2)
        x4 = self.upsample(x3)
        x5 = self.conv_down3(x4)
        x6 = self.upsample(x5)
        x7 = self.conv_down4(x6)
        x8 = self.upsample(x7)
        x9 = self.conv_down5(x8)

        # Decoder part
        x = self.trans_up1(x9)
        y = crop_image(x7, x)
        x = self.conv_up1(torch.cat([x, y], 1))
        x = self.trans_up2(x)
        y = crop_image(x5, x)
        x = self.conv_up2(torch.cat([x, y], 1))
        x = self.trans_up3(x)
        y = crop_image(x3, x)
        x = self.conv_up3(torch.cat([x, y], 1))
        x = self.trans_up4(x)
        y = crop_image(x1, x)
        x = self.conv_up4(torch.cat([x, y], 1))
        out = self.last_conv(x)
        return out


# Используемая в работе метрика
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1).byte()  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1).byte()
    smooth = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    iou = (intersection + smooth) / (union + smooth)  # We smooth our division to avoid 0/0
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresholds
    return thresholded


# Ниже идут все подготовленные мною функции потерь
def bce_loss(y_pred, y_real):
    loss = torch.maximum(y_pred, torch.zeros_like(y_pred)) - torch.mul(y_real, y_pred) + \
           torch.log(1 + torch.exp(-torch.abs(y_pred)))
    return torch.mean(loss)


def bce_loss_standard(y_pred, y_real):
    loss = nn.BCEWithLogitsLoss()
    output = loss(y_pred, y_real)
    return output


def dice_loss(y_pred, y_real):
    smooth = 1.*1e-5
    y_pred = torch.clamp(y_pred, smooth, 1-smooth)
    intersection = (y_real * y_pred).sum(dim=(1, 2, 3))
    union = (y_real + y_pred).sum(dim=(1, 2, 3))
    loss = 1 - (2. * intersection)/union
    return torch.mean(loss)


def focal_loss(y_pred, y_real, gamma=2):
    smooth = 1.*1e-5
    y_pred = torch.clamp(y_pred, smooth, 1-smooth)
    bce_loss1 = torch.maximum(y_pred, torch.zeros_like(y_pred)) - torch.mul(y_real, y_pred) + \
        torch.log(1 + torch.exp(-torch.abs(y_pred)))
    loss = torch.pow((1-torch.exp(bce_loss1)), gamma) * bce_loss1
    return torch.mean(loss)
    # bce_loss1 = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_real, reduction='none')
    # alpha = 1
    # # targets = targets.type(torch.long)
    # # at = self.alpha.gather(0, targets.data.view(-1))
    # pt = torch.exp(-bce_loss1)
    # loss = alpha * (1 - pt) ** gamma * bce_loss1
    # return torch.mean(loss)


def tversky_loss(y_pred, y_real):
    smooth = 1.*1e-5
    alpha = 1
    betta = 1
    y_pred = torch.clamp(y_pred, smooth, 1-smooth)
    numerator = (y_pred * y_real).sum(dim=(1, 2, 3))
    denominator1 = alpha * (y_pred * y_real).sum(dim=(1, 2, 3))
    denominator2 = betta * (y_pred * (1-y_real)).sum(dim=(1, 2, 3))
    denominator3 = ((1-y_pred) * y_real).sum(dim=(1, 2, 3))
    loss = 1 - (numerator / (denominator1 + denominator2 + denominator3))
    return torch.mean(loss)


# Данный лосс более сложный, чем tversky, был найден в отдельной статье
def tversky_focal_loss(y_pred, y_real, gamma=2):
    # https://arxiv.org/pdf/1810.07842.pdf
    smooth = 1.*1e-5
    alpha = 1
    betta = 1
    y_pred = torch.clamp(y_pred, smooth, 1-smooth)
    numerator = (y_pred * y_real).sum(dim=(1, 2, 3))
    denominator1 = alpha * (y_pred * y_real).sum(dim=(1, 2, 3))
    denominator2 = betta * (y_pred * (1-y_real)).sum(dim=(1, 2, 3))
    denominator3 = ((1-y_pred) * y_real).sum(dim=(1, 2, 3))
    loss = 1 - (numerator / (denominator1 + denominator2 + denominator3))
    loss = torch.pow(loss, 1/gamma)
    return torch.mean(loss)


# Цикл тренировки модели
def train(model1, opt, loss_fn, epochs, data_tr1, data_val1):
    graph1 = []
    graph2 = []
    graph3 = []
    graph4 = []
    x_val, y_val = next(iter(data_val1))
    start_time = time()
    for epoch in range(epochs):
        avg_loss = 0
        avg_loss2 = 0
        model1.train()  # train mode
        for X_batch, Y_batch in data_tr1:
            inputs = X_batch.to(DEVICE)
            labels = Y_batch.to(DEVICE)
            opt.zero_grad()
            y_pred = model1(inputs)
            loss = loss_fn(y_pred, labels)
            loss.backward()
            opt.step()
            avg_loss += loss / len(data_tr1)
        model1.eval()
        for X_batch2, Y_batch2 in data_val1:
            inputs2 = X_batch2.to(DEVICE)
            y_pred2 = model(inputs2).cpu().detach()
            loss2 = loss_fn(y_pred2, Y_batch2)
            avg_loss2 += loss2 / len(data_val1)
        train_score = score_model(model1, iou_pytorch, data_tr)
        test_score = score_model(model1, iou_pytorch, data_val)
        if (epoch+1) % 1 == 0:
            print('Epoch: %d/%d, train loss: %f, test loss: %f, train score: %f, test score: %f'
                  % (epoch+1, epochs, avg_loss, avg_loss2, train_score, test_score))
        if epoch > 0:  # Evaluations on the first epoch are unstable
            graph1.append(avg_loss.cpu().detach().numpy().tolist())
            graph2.append(avg_loss2.cpu().detach().numpy().tolist())
            graph3.append(train_score)
            graph4.append(test_score)
        if epoch+1 == epochs:
            y_hat = model(x_val.to(DEVICE)).detach().to('cpu')
            show_images(x_val, y_val, y_hat, epoch+1, epochs, avg_loss)
    print('Elapsed time: %s' % np.round((time() - start_time), 1))
    elapsed_times.append(np.round((time() - start_time), 1))
    train_loss_graphs.append(graph1)
    val_loss_graphs.append(graph2)
    train_score_graphs.append(graph3)
    val_score_graphs.append(graph4)


# Функции для оценки моделей
def predict(model1, data):
    model1.eval()
    y_pred = [X_batch for X_batch, _ in data]
    return np.array(y_pred)


def score_model(model1, metric, data):
    model1.eval()
    scores = 0
    threshold = 0.5
    for X_batch, Y_label in data:
        y_pred = model(X_batch.to(DEVICE))
        y_pred = (y_pred > threshold).float()
        scores += metric(y_pred, Y_label.to(DEVICE)).mean().item()
    return scores/len(data)


# Это блок визуалиции, в котором я строю графики и диаграммы
def show_images(x_val, y_val, y_hat, epoch, epochs, avg_loss):
    clear_output(wait=True)
    for k in range(6):
        plt.subplot(3, 6, k + 1)
        plt.imshow(np.rollaxis(x_val[k].numpy(), 0, 3), cmap='gray')
        plt.title('Image')
        plt.axis('off')
        plt.subplot(3, 6, k + 7)
        plt.imshow(y_val[k, 0], cmap='gray')
        plt.title('Labels')
        plt.axis('off')
        plt.subplot(3, 6, k + 13)
        plt.imshow(np.round(y_hat[k, 0].numpy()), cmap='gray')
        plt.title('Predicted')
        plt.axis('off')
    plt.suptitle('%d / %d - loss: %f' % (epoch, epochs, avg_loss))
    plt.show()


def plot_learning_curves(lines, legend, name):
    for line in lines:
        plt.plot(line)
    plt.legend(legend)
    plt.title(name)
    plt.show()


def plot_histogram(data, labels):
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/2, data)
    ax.set_xticklabels(labels)
    plt.show()


def export_to_excel(df):
    writer = pd.ExcelWriter('Report/report_data.xlsx')
    df.to_excel(writer, 'data', index=False)
    writer.save()


# Главный цикл выполнения программы и вывод данных для отчета в виде графиков и XLSX таблиц
combination_names = ["SegNet+BCE", "SegNet+Dice", "SegNet+Focal", "SegNet+Tversky", "SegNet+TverskyFocal"
                     "UNet+BCE", "UNet+Dice", "UNet+Focal", "UNet+Tversky", "UNet+TverskyFocal"
                     "UNet2+BCE", "UNet2+Dice", "UNet2+Focal", "UNet2+Tversky", "UNet2+TverskyFocal"]
combination_names = ["SegNet+BCE", "SegNet+Dice", "SegNet+Focal", "SegNet+Tversky", "SegNet+TverskyFocal"]
models = [SegNet()]
losses = [bce_loss, dice_loss, focal_loss, tversky_loss, tversky_focal_loss]
max_epochs = 80
train_loss_graphs = []
val_loss_graphs = []
train_score_graphs = []
val_score_graphs = []
elapsed_times = []
final_scores = []
df_report = pd.DataFrame()
j = 0
for model in models:
    for loss_function in losses:
        print(combination_names[j])
        model.__init__()
        model.to(DEVICE)
        optim = torch.optim.AdamW(model.parameters(), lr=0.00001)
        j = j + 1
        train(model, optim, loss_function, max_epochs, data_tr, data_val)
        final_score = score_model(model, iou_pytorch, data_ts)
        final_scores.append(final_score)

plot_learning_curves(train_loss_graphs, combination_names, "Train Loss graphs")
plot_learning_curves(val_loss_graphs, combination_names, "Val Loss graphs")
plot_learning_curves(train_score_graphs, combination_names, "Train scores")
plot_learning_curves(val_score_graphs, combination_names, "Test scores")
plot_histogram(elapsed_times, combination_names)
df_report["train_loss_graphs"] = train_loss_graphs[0]
df_report["val_loss_graphs"] = val_loss_graphs[0]
df_report["train_score_graphs"] = train_score_graphs[0]
df_report["val_score_graphs"] = val_score_graphs[0]
print(final_scores)

export_to_excel(df_report)
