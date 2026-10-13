# Estudo prático: PyTorch, TensorFlow, Keras e Fast.ai

O mesmo problema resolvido nas quatro bibliotecas, para comparar lado a lado:
classificar as 70 mil imagens 28x28 do **Fashion-MNIST** (10 tipos de roupa)
com **a mesma CNN** (2 convoluções + 2 camadas densas, ~422 mil parâmetros).

| Arquivo | O que mostra |
|---|---|
| [01_pytorch.py](01_pytorch.py) | Tudo explícito: `Dataset`/`DataLoader`, classe `nn.Module`, loop de treino escrito à mão |
| [02_tensorflow.py](02_tensorflow.py) | TensorFlow "por baixo": pipeline `tf.data`, `GradientTape`, `@tf.function`, exportação SavedModel |
| [03_keras.py](03_keras.py) | Alto nível: `Sequential` + `compile` + `fit`, callbacks, e o mesmo código rodando sobre TensorFlow **ou** PyTorch |
| [04_fastai.py](04_fastai.py) | Alto nível sobre PyTorch: `Learner`, `lr_find`, `fit_one_cycle`, análise dos erros |

## Como rodar

```bash
pip install -r requirements.txt
python 01_pytorch.py
python 02_tensorflow.py
python 03_keras.py
KERAS_BACKEND=torch python 03_keras.py   # mesmo script, outro motor
python 04_fastai.py
```

`EPOCAS=5 python 01_pytorch.py` muda o número de épocas (padrão 2).
Os dados (~30 MB) e os modelos treinados vão para `~/estudo_dl_dados` (mude com `DADOS=...`).
Funciona em CPU; se houver GPU, PyTorch/TF/fastai a usam sozinhos.

## Resultados (2 épocas, CPU de 4 núcleos)

| Script | Acurácia no teste | Tempo por época | Linhas de código "útil" |
|---|---|---|---|
| PyTorch | 89,1% | ~15 s | ~60 |
| TensorFlow (GradientTape) | 89,2% | ~9 s | ~55 |
| Keras (backend TensorFlow) | 89,8% | ~8 s | ~25 |
| Keras (backend PyTorch) | 88,9% | ~20 s | mesmas ~25 |
| Fast.ai | 86,9% | ~20 s | ~30 |

Como ler: com o mesmo modelo, **as bibliotecas chegam ao mesmo lugar** (~89%).
As diferenças vêm de detalhes, não de uma ser "mais inteligente": o Keras usa 10% do
treino para validação; o fastai usou a taxa de aprendizado sugerida pelo `lr_find`
(4e-4, menor que o 1e-3 dos outros) e com mais épocas alcançaria os demais.
O TF é mais rápido aqui porque `@tf.function` compila o passo de treino em grafo;
no PyTorch o equivalente seria `torch.compile(model)`.

## O que cada uma é, em uma frase

- **PyTorch**: biblioteca de tensores + gradientes automáticos, em Python "normal".
  Você controla tudo. É o padrão em pesquisa e na maioria dos modelos novos
  (Hugging Face, LLMs, difusão).
- **TensorFlow**: motor do Google com foco em **produção e implantação**:
  `tf.data` para pipelines grandes, SavedModel, TF Serving, **TFLite/LiteRT** (celular,
  microcontrolador), TF.js (navegador), TPUs. Hoje você quase sempre o usa via Keras.
- **Keras**: API de alto nível (`compile`/`fit`). Desde o Keras 3 é **multi-backend**:
  o mesmo modelo roda sobre TensorFlow, PyTorch ou JAX.
- **Fast.ai**: camada de alto nível sobre PyTorch com **boas práticas embutidas**
  (one-cycle, `lr_find`, transfer learning com `fine_tune`, augmentação, interpretação).
  Também é um curso gratuito excelente (course.fast.ai).

## Qual usar para quê

| Situação | Melhor escolha | Por quê |
|---|---|---|
| Aprender o que acontece "por dentro" | PyTorch | O loop explícito mostra forward, loss, backward e step |
| Pesquisa, arquiteturas novas, reproduzir papers | PyTorch | É onde o código dos papers está; depura com `print`/debugger |
| NLP/LLMs, modelos do Hugging Face | PyTorch | O ecossistema `transformers` é PyTorch em primeiro lugar |
| Protótipo rápido de rede "padrão" (tabular, imagem, texto) | Keras | Poucas linhas, callbacks prontos, fácil de ler |
| Colocar modelo no celular, navegador ou dispositivo embarcado | TensorFlow (via Keras) | TFLite/LiteRT e TF.js são o caminho mais maduro |
| Servir em produção no ecossistema Google Cloud / TPU | TensorFlow | TF Serving, Vertex AI, TPUs |
| Treino customizado sem largar o Keras (GANs, perdas especiais) | TensorFlow (`GradientTape`) | Controle total do passo, mantendo as camadas Keras |
| Classificar imagens/textos/tabelas próprios com pouco dado e rápido | Fast.ai | `vision_learner` + `fine_tune` dão resultado forte em poucas linhas |
| Equipe que não quer se prender a um motor | Keras 3 | Troca de backend com uma variável de ambiente |

**Regra prática**: se for escolher só uma para começar, **PyTorch** (é a mais usada hoje
e o fastai roda em cima dela). Some **Keras** quando quiser produtividade ou implantar em
mobile/web, e **fast.ai** quando quiser resultado rápido com transfer learning.

## Mapa de conceitos equivalentes

| Conceito | PyTorch | TensorFlow | Keras | Fast.ai |
|---|---|---|---|---|
| Dados em lotes | `DataLoader` | `tf.data.Dataset` | arrays NumPy direto no `fit` | `DataLoaders` (treino + validação) |
| Modelo | classe `nn.Module` | `tf.keras.Model` | `Sequential` / API funcional | qualquer `nn.Module` |
| Gradientes | `loss.backward()` | `tf.GradientTape` | escondido no `fit` | escondido no `Learner` |
| Atualizar pesos | `optimizer.step()` | `apply_gradients` | escondido no `fit` | escondido no `Learner` |
| Loop de treino | você escreve | você escreve | `model.fit()` | `learn.fit_one_cycle()` |
| Formato das imagens | `[lote, canal, alt, larg]` | `[lote, alt, larg, canal]` | igual ao TF | igual ao PyTorch |
| Salvar | `state_dict` (.pt) | SavedModel | `.keras` | `learn.export()` / `state_dict` |

## Próximos passos sugeridos

1. Rode cada script com `EPOCAS=10` e compare as curvas de perda.
2. No 01, troque `Adam` por `SGD(lr=0.1, momentum=0.9)` e veja a diferença.
3. No 03, acrescente `keras.callbacks.TensorBoard("logs")` e abra `tensorboard --logdir logs`.
4. No 04, faça transfer learning de verdade com uma pasta de fotos suas
   (o código está comentado no fim do arquivo).
