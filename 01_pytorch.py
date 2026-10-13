"""
01 - PyTorch: classificação do Fashion-MNIST "na mão"
=====================================================

Ideia central do PyTorch: você escreve Python comum. O modelo é uma classe,
o loop de treino é um `for` que VOCÊ escreve. Nada fica escondido, o que
torna o PyTorch ótimo para pesquisa, depuração e modelos fora do padrão.

Conceitos que aparecem aqui:
  - Tensor ........ array tipo NumPy que pode rodar na GPU e guardar gradientes
  - Dataset/DataLoader .. como os dados são lidos e agrupados em lotes (batches)
  - nn.Module ..... a classe base de qualquer modelo/camada
  - autograd ...... `loss.backward()` calcula os gradientes automaticamente
  - optimizer ..... `optimizer.step()` atualiza os pesos com esses gradientes

Rodar:  python 01_pytorch.py        (EPOCAS=1 python 01_pytorch.py para ser rápido)
"""
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

EPOCAS = int(os.environ.get("EPOCAS", 2))
BATCH = 128
DADOS = os.environ.get("DADOS", os.path.expanduser("~/estudo_dl_dados"))  # datasets e modelos salvos
os.makedirs(DADOS, exist_ok=True)

torch.manual_seed(42)
# Em PyTorch você escolhe explicitamente onde cada tensor vive (CPU ou GPU).
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- 1. Dados
# `transforms` convertem cada imagem PIL em tensor [1, 28, 28] com valores 0..1
# e depois normalizam (média/desvio do Fashion-MNIST).
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
treino = datasets.FashionMNIST(DADOS, train=True, download=True, transform=tf)
teste = datasets.FashionMNIST(DADOS, train=False, download=True, transform=tf)

# DataLoader embaralha, agrupa em lotes e pode usar vários processos.
dl_treino = DataLoader(treino, batch_size=BATCH, shuffle=True)
dl_teste = DataLoader(teste, batch_size=512)


# --------------------------------------------------------------- 2. Modelo
# Mesma CNN usada nos outros 3 scripts, para comparar de forma justa.
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 10)
        )

    # `forward` define o cálculo. Pode ter if/for/print à vontade: é Python puro.
    def forward(self, x):
        return self.classifier(self.features(x))


model = CNN().to(device)
print(model)
print("Parâmetros treináveis:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# A saída do modelo são "logits" (sem softmax); CrossEntropyLoss aplica o softmax internamente.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ------------------------------------------------------- 3. Loop de treino
def avaliar():
    model.eval()                     # desliga Dropout
    acertos = total = 0
    with torch.no_grad():            # não precisa de gradiente na avaliação
        for x, y in dl_teste:
            x, y = x.to(device), y.to(device)
            acertos += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return acertos / total


for epoca in range(1, EPOCAS + 1):
    model.train()                    # liga Dropout
    inicio, soma_loss = time.time(), 0.0
    for x, y in dl_treino:
        x, y = x.to(device), y.to(device)
        logits = model(x)            # 1) forward
        loss = loss_fn(logits, y)    # 2) erro
        optimizer.zero_grad()        # 3) zera gradientes do passo anterior
        loss.backward()              # 4) autograd calcula os gradientes
        optimizer.step()             # 5) atualiza os pesos
        soma_loss += loss.item()
    acc = avaliar()
    print(f"Época {epoca}/{EPOCAS}  loss={soma_loss / len(dl_treino):.4f}  "
          f"acurácia_teste={acc:.4f}  ({time.time() - inicio:.1f}s)")

# ----------------------------------------------------- 4. Salvar e prever
# O jeito idiomático é salvar só os pesos (state_dict).
torch.save(model.state_dict(), os.path.join(DADOS, "cnn_pytorch.pt"))

classes = ["Camiseta", "Calça", "Pulôver", "Vestido", "Casaco",
           "Sandália", "Camisa", "Tênis", "Bolsa", "Bota"]
x, y = teste[0]
model.eval()
with torch.no_grad():
    prob = torch.softmax(model(x.unsqueeze(0).to(device)), dim=1)[0]
print(f"Exemplo: real={classes[y]}  previsto={classes[prob.argmax()]}  (confiança {prob.max():.2%})")
