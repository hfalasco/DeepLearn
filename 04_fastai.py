"""
04 - Fast.ai: boas práticas prontas em cima do PyTorch
=======================================================

Fast.ai é uma camada de alto nível SOBRE o PyTorch (o modelo continua sendo
um nn.Module). O que ela acrescenta são as "boas práticas por padrão":

  - DataLoaders ...... junta treino + validação num objeto só
  - Learner .......... modelo + dados + perda + otimizador + métricas
  - lr_find() ........ testa várias taxas de aprendizado e sugere uma boa
  - fit_one_cycle() .. agenda de LR "one cycle" (sobe e depois desce), costuma
                        convergir mais rápido que LR fixo
  - vision_learner / fine_tune .. transfer learning com modelos pré-treinados
                        em 2 linhas (ver comentário no final)
  - get_preds / ClassificationInterpretation .. previsões, matriz de confusão, piores erros

Rodar:  python 04_fastai.py          (EPOCAS=1 para ser rápido)
"""
import os

from fastai.vision.all import (CrossEntropyLossFlat, DataLoader, DataLoaders, Learner, accuracy, nn, torch)
from torchvision import datasets, transforms

EPOCAS = int(os.environ.get("EPOCAS", 2))
DADOS = os.environ.get("DADOS", os.path.expanduser("~/estudo_dl_dados"))  # datasets e modelos salvos
os.makedirs(DADOS, exist_ok=True)
torch.manual_seed(42)

# ---------------------------------------------------------------- 1. Dados
# Reaproveitamos os datasets do torchvision. Para imagens em pastas
# (dados/gato/*.jpg, dados/cachorro/*.jpg) o idiomático seria
# ImageDataLoaders.from_folder(caminho), que já faz augmentação.
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
treino = datasets.FashionMNIST(DADOS, train=True, download=True, transform=tf)
teste = datasets.FashionMNIST(DADOS, train=False, download=True, transform=tf)
classes = ["Camiseta", "Calça", "Pulôver", "Vestido", "Casaco",
           "Sandália", "Camisa", "Tênis", "Bolsa", "Bota"]
# Atenção: é o DataLoader do fastai (aceita qualquer Dataset do PyTorch, mas tem
# recursos extras que o Learner usa, como `.new()` e decodificação).
dls = DataLoaders(DataLoader(treino, bs=128, shuffle=True),
                  DataLoader(teste, bs=512))

# --------------------------------------------------------------- 2. Modelo
# A mesma CNN dos outros scripts, escrita como nn.Sequential do PyTorch.
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 10),
)

# Learner = o "treinador". Ele substitui todo o loop que escrevemos no 01_pytorch.py.
# CrossEntropyLossFlat é a versão fastai da CrossEntropyLoss: além da perda, ela sabe
# "decodificar" logits em classe (argmax) e aplicar softmax no get_preds.
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)

# --------------------------------------------------------------- 3. Treino
# lr_find faz um mini-treino variando a LR e devolve uma sugestão.
sugestao = learn.lr_find(show_plot=False)
lr = sugestao.valley
print(f"Taxa de aprendizado sugerida pelo lr_find: {lr:.2e}")

learn.fit_one_cycle(EPOCAS, lr_max=lr)   # imprime uma tabela por época

# ------------------------------------------------- 4. Analisar e salvar
# get_preds roda o modelo no conjunto de validação e devolve probabilidades e rótulos.
probs, alvos = learn.get_preds()
previstos = probs.argmax(1)
cm = torch.zeros(10, 10, dtype=torch.long)       # linhas = real, colunas = previsto
for r, p in zip(alvos, previstos):
    cm[r, p] += 1
pares = [(int(cm[i, j]), classes[i], classes[j]) for i in range(10) for j in range(10) if i != j]
print("Pares de classes mais confundidos (real -> previsto: quantidade):")
for n, real, prev in sorted(pares, reverse=True)[:5]:
    print(f"  {real:9s} -> {prev:9s} {n}")
# Com dados carregados pelas APIs do próprio fastai (ImageDataLoaders etc.), isso vira:
#   interp = ClassificationInterpretation.from_learner(learn)
#   interp.plot_confusion_matrix(); interp.plot_top_losses(9); interp.most_confused()

torch.save(learn.model.state_dict(), os.path.join(DADOS, "cnn_fastai.pt"))

# Transfer learning, o ponto forte do fastai (precisa baixar pesos e imagens RGB):
#   from fastai.vision.all import *
#   dls = ImageDataLoaders.from_folder(caminho, valid_pct=0.2, item_tfms=Resize(224))
#   learn = vision_learner(dls, resnet18, metrics=accuracy)
#   learn.fine_tune(3)
