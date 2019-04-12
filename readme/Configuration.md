Configuration of NeuralClassifier uses JSON.

## Common

* **task\_info**
    * **label_type**:  Candidates: "single-label", "multi-label".
    * **hierarchical**: Boolean.
    * **hierar_taxonomy**: 
    * **hierar_penalty**:
* **device**: Candidates: "cuda", "cpu".
* **model\_name**: Candidates: "FastText", "TextCNN", "TextRNN", "TextRCNN", "DRNN", "VDCNN", "DPCNN", "Region embedding", "AttentiveConvNet", "Transformer".
* **checkpoint\_dir**: checkpoint directory
* **model\_dir**: model directory
* **data**
    * **train\_json\_files**: train input data.
    * **validate\_json\_files**: validation input data.
    * **test\_json\_files**: test input data.
    * **generate\_dict\_using\_json\_files**: generate dict using train data.
    * **generate\_dict\_using\_all\_json\_files**: generate dict using train, validate, test data.
    * **generate\_dict\_using\_pretrained\_embedding**: generate dict from pre-trained embedding.
    * **dict\_dir**: dict directory.
    * **num\_worker**: number of porcess to load data.


## Feature

* **feature\_names**: Candidates: "token", "char".
* **min\_token\_count**
* **min\_char\_count**
* **token\_ngram**
* **min\_token\_ngram\_count**
* **min\_keyword\_count**
* **min\_topic\_count**
* **max\_token\_dict\_size**
* **max\_char\_dict\_size**
* **max\_token\_ngram\_dict\_size**
* **max\_keyword\_dict\_size**
* **max\_topic\_dict\_size**
* **max\_token\_len**
* **max\_char\_len**
* **max\_char\_len\_per\_token**
* **token\_pretrained\_file**: token pre-trained embedding.
* **keyword\_pretrained\_file**: keyword pre-trained embedding.


## Train

* **batch\_size**
* **predict\_batch\_size**
* **eval\_train\_data**: whether evaluate training data when training.
* **start\_epoch**
* **num\_epochs**
* **num\_epochs\_static\_embedding**: number of epochs that input embedding does not update.
* **decay\_steps**
* **decay\_rate**
* **clip\_gradients**
* **l2\_lambda**
* **loss\_type**: Candidates: "SoftmaxCrossEntropy", "SoftmaxFocalCrossEntropy", "SigmodFocalCrossEntropy", "BCEWithLogitsLoss".
* **sampler**
* **num\_sampled**
* **hidden\_layer\_dropout**
* **visible\_device\_list**: GPU list to use


## Embedding

* **type**: Candidates: "embedding", "region_embedding"
* **dimension**
* **region\_embedding\_type**: Candidates: "word\_context", "context\_word"
* **region_size**
* **initializer**: Candidates: "uniform", "normal", "xavier\_uniform", "xavier\_normal", "kaiming\_uniform", "kaiming\_normal", "orthogonal"
* **fan\_mode**: Candidates: "FAN\_IN", "FAN\_OUT"
* **uniform\_bound**
* **random\_stddev**
* **dropout**: dropout of embedding layer


## Optimizer

* **optimizer\_type**: Candidates: "Adam", "Adadelta"
* **learning\_rate**
* **adadelta\_decay\_rate**: useful when optimizer\_type is Adadelta.
* **adadelta\_epsilon**: useful when optimizer\_type is Adadelta.


## Eval

* **text\_file**
* **threshold**: float trunc threshold for predict probabilities.
* **dir**: output dir of evaluation.
* **batch\_size**: batch size of evaluation.
* **is\_flat**: Boolean, flat evaluation or hierarchical evaluation.


## Log

* **logger\_file**: log file path.
* **log\_level**: Candidates: "debug", "info", "warn", "error".


## Encoder

### TextCNN

* **kernel\_sizes**
* **num\_kernels**
* **top\_k\_max\_pooling**

### TextRNN

* **hidden\_dimension**
* **rnn\_type**: Candidates: "RNN", "LSTM", "GRU".
* **num\_layers**
* **doc\_embedding\_type**: Candidates: "AVG", "Attention", "LastHidden".
* **attention\_dimension**
* **bidirectional**: Boolean, use Bi-RNNs.

### RCNN

see TextCNN and TextRNN

### DRNN

* **hidden\_dimension**
* **window\_size**
* **rnn\_type**: Candidates: "RNN", "LSTM", "GRU".
* **bidirectional**
* **cell\_hidden\_dropout**

### VDCNN

* **vdcnn\_depth**
* **top\_k\_max\_pooling**

### DPCNN

* **kernel\_size**
* **pooling\_stride**
* **num\_kernels**
* **blocks**

### AttentiveConvNet

* **attention\_type**: Candidates: "dot", "bilinear", "additive_projection".
* **margin\_size**
* **type**:  Candidates: "light", "advanced".
* **hidden\_size**

### Transformer

* **d\_inner**
* **d\_k**
* **d\_v**
* **n\_head**
* **n\_layers**
* **dropout**
* **use\_star**: whether use Star-Transformer, see [Star-Transformer](https://arxiv.org/pdf/1902.09113v2.pdf "Star-Transformer") 
