"""
02 - TensorFlow "por baixo do Keras": tf.data + GradientTape
=============================================================

No dia a dia, quase todo mundo usa TensorFlow através do Keras (veja o 03).
Este script mostra a camada de baixo, que é o que o TensorFlow oferece de
próprio:

  - tf.data ........ pipeline de dados eficiente (shuffle, batch, prefetch, cache)
  - tf.GradientTape  "grava" as operações para calcular gradientes
                      (equivalente ao autograd do PyTorch)
  - @tf.function ... compila a função Python em um GRAFO otimizado (mais rápido,
                      exportável para TF Serving, TFLite (celular), TF.js (navegador))
  - SavedModel ..... formato de exportação para produção

Use esse estilo quando o `model.fit()` do Keras não for suficiente
(ex.: GANs, perdas exóticas, vários otimizadores).

Rodar:  python 02_tensorflow.py      (EPOCAS=1 para ser rápido)
"""
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # menos logs do TF
import tensorflow as tf

EPOCAS = int(os.environ.get("EPOCAS", 2))
BATCH = 128
DADOS = os.environ.get("DADOS", os.path.expanduser("~/estudo_dl_dados"))  # datasets e modelos salvos
os.makedirs(DADOS, exist_ok=True)
tf.random.set_seed(42)

# ---------------------------------------------------------------- 1. Dados
# O Keras (que vem dentro do TF) traz o Fashion-MNIST como arrays NumPy.
(x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()


def preparar(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = (x - 0.2860) / 0.3530
    return x[..., tf.newaxis], y            # TF usa formato "canais por último": [28, 28, 1]


# tf.data: cada etapa é lazy e roda em paralelo com o treino (prefetch).
ds_treino = (tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
             .shuffle(10_000, seed=42)
             .map(preparar, num_parallel_calls=tf.data.AUTOTUNE)
             .batch(BATCH)
             .prefetch(tf.data.AUTOTUNE))
ds_teste = (tf.data.Dataset.from_tensor_slices((x_te, y_te))
            .map(preparar).batch(512).prefetch(tf.data.AUTOTUNE))

# --------------------------------------------------------------- 2. Modelo
# As camadas vêm do Keras mesmo: no TF 2.x não existe outra API de camadas "oficial".
L = tf.keras.layers
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    L.Conv2D(32, 3, padding="same", activation="relu"), L.MaxPooling2D(),
    L.Conv2D(64, 3, padding="same", activation="relu"), L.MaxPooling2D(),
    L.Flatten(), L.Dense(128, activation="relu"), L.Dropout(0.3), L.Dense(10),
])
model.summary()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(1e-3)
acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()


# ------------------------------------------------------- 3. Loop de treino
# @tf.function transforma o passo em grafo: a 1ª chamada "traça" o código,
# as seguintes rodam o grafo compilado (bem mais rápido que Python puro).
@tf.function
def passo_treino(x, y):
    with tf.GradientTape() as tape:                    # grava as operações...
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)   # ...e deriva
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


@tf.function
def passo_teste(x, y):
    acc_metric.update_state(y, model(x, training=False))


for epoca in range(1, EPOCAS + 1):
    inicio, soma, n = time.time(), 0.0, 0
    for x, y in ds_treino:
        soma += float(passo_treino(x, y)); n += 1
    acc_metric.reset_state()
    for x, y in ds_teste:
        passo_teste(x, y)
    print(f"Época {epoca}/{EPOCAS}  loss={soma / n:.4f}  "
          f"acurácia_teste={float(acc_metric.result()):.4f}  ({time.time() - inicio:.1f}s)")

# ------------------------------------------------- 4. Exportar e prever
# SavedModel é o formato que TF Serving / TFLite / TF.js consomem.
model.export(os.path.join(DADOS, "cnn_tf_savedmodel"), verbose=False)

classes = ["Camiseta", "Calça", "Pulôver", "Vestido", "Casaco",
           "Sandália", "Camisa", "Tênis", "Bolsa", "Bota"]
x, y = preparar(x_te[:1], y_te[:1])
prob = tf.nn.softmax(model(x, training=False))[0]
print(f"Exemplo: real={classes[int(y[0])]}  previsto={classes[int(tf.argmax(prob))]}  "
      f"(confiança {float(tf.reduce_max(prob)):.2%})")
