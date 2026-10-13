"""
03 - Keras: o mesmo problema em ~20 linhas úteis
=================================================

Keras é uma API de alto nível: você declara as camadas, chama `compile()`
(escolhe otimizador, perda, métricas) e `fit()` (treina). Loop, métricas,
validação, callbacks e logs já vêm prontos.

Desde o Keras 3, ele é MULTI-BACKEND: o mesmo código roda em cima de
TensorFlow, PyTorch ou JAX. Basta definir a variável de ambiente:

    KERAS_BACKEND=torch      python 03_keras.py
    KERAS_BACKEND=tensorflow python 03_keras.py   (padrão)

Conceitos que aparecem aqui:
  - Sequential / API funcional .. formas de montar o modelo
  - compile / fit / evaluate / predict .. o ciclo de vida padrão
  - callbacks ... ganchos no treino (EarlyStopping, ModelCheckpoint, TensorBoard...)

Rodar:  python 03_keras.py           (EPOCAS=1 para ser rápido)
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras
import numpy as np
from keras import layers

EPOCAS = int(os.environ.get("EPOCAS", 2))
DADOS = os.environ.get("DADOS", os.path.expanduser("~/estudo_dl_dados"))  # datasets e modelos salvos
os.makedirs(DADOS, exist_ok=True)
keras.utils.set_random_seed(42)
print("Keras", keras.__version__, "rodando sobre o backend:", keras.backend.backend())

# ---------------------------------------------------------------- 1. Dados
# Com Keras, NumPy puro já basta: fit() cuida dos lotes e do embaralhamento.
(x_tr, y_tr), (x_te, y_te) = keras.datasets.fashion_mnist.load_data()
x_tr = ((x_tr / 255.0 - 0.2860) / 0.3530)[..., None].astype("float32")
x_te = ((x_te / 255.0 - 0.2860) / 0.3530)[..., None].astype("float32")

# --------------------------------------------------------------- 2. Modelo
# A normalização poderia virar uma camada (layers.Rescaling/Normalization),
# assim o modelo salvo já faz o pré-processamento sozinho.
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
    layers.Flatten(), layers.Dense(128, activation="relu"), layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",    # rótulos inteiros (0..9)
    metrics=["accuracy"],
)

# --------------------------------------------------------------- 3. Treino
# validation_split separa 10% do treino para validação, automaticamente.
historico = model.fit(
    x_tr, y_tr, epochs=EPOCAS, batch_size=128, validation_split=0.1,
    callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
    verbose=2,
)
loss, acc = model.evaluate(x_te, y_te, verbose=0)
print(f"Teste: loss={loss:.4f}  acurácia_teste={acc:.4f}")
print("Histórico disponível em historico.history:", list(historico.history))

# ----------------------------------------------------- 4. Salvar e prever
model.save(os.path.join(DADOS, "cnn_keras.keras"))   # formato nativo do Keras 3

classes = ["Camiseta", "Calça", "Pulôver", "Vestido", "Casaco",
           "Sandália", "Camisa", "Tênis", "Bolsa", "Bota"]
prob = model.predict(x_te[:1], verbose=0)[0]
print(f"Exemplo: real={classes[y_te[0]]}  previsto={classes[int(np.argmax(prob))]}  "
      f"(confiança {prob.max():.2%})")
