# Awesome Recurrent Neural Networks for natural language programming

copied from awesome rnn 


## Table of Contents

- [Codes](#codes)
- [Theory](#theory)
  - [Lectures](#lectures)
  - [Books / Thesis](#books--thesis)
  - [Architecture Variants](#architecture-variants)
    - [Structure](#structure)
    - [Memory](#memory)
  - [Surveys](#surveys)
- [Applications](#applications)
  - [Natural Language Processing](#natural-language-processing)
    - [Language Modeling](#language-modeling)
    - [Speech Recognition](#speech-recognition)
    - [Machine Translation](#machine-translation)
    - [Conversation Modeling](#conversation-modeling)
    - [Question Answering](#question-answering)
  - [Turing Machines](#turing-machines)
  - [Other](#other)
- [Datasets](#datasets)
- [Blogs](#blogs)
- [Online Demos](#online-demos)

## Tutorials
* Tensorflow [[RNN Tutorial] (https://www.tensorflow.org/versions/master/tutorials/recurrent/index.html)] [[seq2seq Tutorial](https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html)] [[char-rnn](https://github.com/sherjilozair/char-rnn-tensorflow)]
* Torch [[char rnn karpathy](https://github.com/karpathy/char-rnn)] [[faster torch-rnn library](https://github.com/jcjohnson/torch-rnn)] [[neuraltalk2](https://github.com/karpathy/neuraltalk2)][[language model on PTB](https://github.com/wojzaremba/lstm)] [[rnn library](https://github.com/Element-Research/rnn) ] 

## Theory
### Lectures
* Stanford NLP [[Lecture Note 4](http://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf)] : RNN language models, bi-directional RNN, GRU, LSTM
* Oxford [Machine Learning](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/) by Nando de Freitas
  * [Lecture 12](https://www.youtube.com/watch?v=56TYLaQN4N8) : Recurrent neural networks and LSTMs
  * [Lecture 13](https://www.youtube.com/watch?v=-yX1SYeDHbg) : (guest lecture) Alex Graves on Hallucination with RNNs

### Books / Thesis
* Alex Graves (2008)
  * [Supervised Sequence Labelling with Recurrent Neural Networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
* Tomas Mikolov (2012)
  * [Statistical Language Models based on Neural Networks](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)
* Ilya Sutskever (2013)
  * [Training Recurrent Neural Networks](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)
* Richard Socher (2014)
  * [Recursive Deep Learning for Natural Language Processing and Computer Vision](http://nlp.stanford.edu/~socherr/thesis.pdf)
* Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  * [The Deep Learning Book chapter 10](http://www.deeplearningbook.org/contents/rnn.html)


### Architecture Variants

#### Structure

* Bi-directional RNN [[Paper](http://www.di.ufpe.br/~fnj/RNA/bibliografia/BRNN.pdf)]
  * Mike Schuster and Kuldip K. Paliwal, *Bidirectional Recurrent Neural Networks*, Trans. on Signal Processing 1997
* Multi-dimensional RNN [[Paper](http://arxiv.org/pdf/0705.2011.pdf)]
  * Alex Graves, Santiago Fernandez, and Jurgen Schmidhuber, *Multi-Dimensional Recurrent Neural Networks*, ICANN 2007
* GFRNN [[Paper-arXiv](http://arxiv.org/pdf/1502.02367)] [[Paper-ICML](http://jmlr.org/proceedings/papers/v37/chung15.pdf)] [[Supplementary](http://jmlr.org/proceedings/papers/v37/chung15-supp.pdf)]
  * Junyoung Chung, Caglar Gulcehre, Kyunghyun Cho, Yoshua Bengio, *Gated Feedback Recurrent Neural Networks*, arXiv:1502.02367 / ICML 2015
* Tree-Structured RNNs
  * Kai Sheng Tai, Richard Socher, and Christopher D. Manning, *Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks*, arXiv:1503.00075 / ACL 2015 [[Paper](http://arxiv.org/pdf/1503.00075)]
  * Samuel R. Bowman, Christopher D. Manning, and Christopher Potts, *Tree-structured composition in neural networks without tree-structured architectures*, arXiv:1506.04834 [[Paper](http://arxiv.org/pdf/1506.04834)]
* Grid LSTM [[Paper](http://arxiv.org/pdf/1507.01526)] [[Code](https://github.com/coreylynch/grid-lstm)]
  * Nal Kalchbrenner, Ivo Danihelka, and Alex Graves, *Grid Long Short-Term Memory*, arXiv:1507.01526
* Segmental RNN [[Paper](http://arxiv.org/pdf/1511.06018v2.pdf)]
  * Lingpeng Kong, Chris Dyer, Noah Smith, "Segmental Recurrent Neural Networks", ICLR 2016.
* Seq2seq for Sets [[Paper](http://arxiv.org/pdf/1511.06391v4.pdf)]
  * Oriol Vinyals, Samy Bengio, Manjunath Kudlur, "Order Matters: Sequence to sequence for sets", ICLR 2016.
* Hierarchical Recurrent Neural Networks [[Paper](http://arxiv.org/abs/1609.01704)]
  * Junyoung Chung, Sungjin Ahn, Yoshua Bengio, "Hierarchical Multiscale Recurrent Neural Networks", arXiv:1609.01704

#### Memory

* LSTM [[Paper](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)]
  * Sepp Hochreiter and Jurgen Schmidhuber, *Long Short-Term Memory*, Neural Computation 1997
* GRU (Gated Recurrent Unit) [[Paper](http://arxiv.org/pdf/1406.1078.pdf)]
  * Kyunghyun Cho, Bart van Berrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio, *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*, arXiv:1406.1078 / EMNLP 2014
* NTM [[Paper](http://arxiv.org/pdf/1410.5401)]
  * A.Graves, G. Wayne, and I. Danihelka., *Neural Turing Machines,* arXiv preprint arXiv:1410.5401
* Neural GPU [[Paper](http://arxiv.org/pdf/1511.08228.pdf)]
  * Łukasz Kaiser, Ilya Sutskever, arXiv:1511.08228 / ICML 2016 (under review)
* Memory Network [[Paper](http://arxiv.org/pdf/1410.3916)]
  * Jason Weston, Sumit Chopra, Antoine Bordes, *Memory Networks,* arXiv:1410.3916
* Pointer Network [[Paper](http://arxiv.org/pdf/1506.03134)]
  * Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly, *Pointer Networks*, arXiv:1506.03134 / NIPS 2015
* Deep Attention Recurrent Q-Network [[Paper](http://arxiv.org/abs/1512.01693)]
  * Ivan Sorokin, Alexey Seleznev, Mikhail Pavlov, Aleksandr Fedorov, Anastasiia Ignateva, *Deep Attention Recurrent Q-Network* , arXiv:1512.01693
* Dynamic Memory Networks [[Paper](http://arxiv.org/abs/1506.07285)]
  * Ankit Kumar, Ozan Irsoy, Peter Ondruska, Mohit Iyyer, James Bradbury, Ishaan Gulrajani, Victor Zhong, Romain Paulus, Richard Socher, "Ask Me Anything: Dynamic Memory Networks for Natural Language Processing", arXiv:1506.07285

### Surveys
* Yann LeCun, Yoshua Bengio, and Geoffrey Hinton, [Deep Learning](http://www.nature.com/nature/journal/v521/n7553/pdf/nature14539.pdf), Nature 2015
* Klaus Greff, Rupesh Kumar Srivastava, Jan Koutnik, Bas R. Steunebrink, Jurgen Schmidhuber, [LSTM: A Search Space Odyssey](http://arxiv.org/pdf/1503.04069), arXiv:1503.04069
* Zachary C. Lipton, [A Critical Review of Recurrent Neural Networks for Sequence Learning](http://arxiv.org/pdf/1506.00019), arXiv:1506.00019
* Andrej Karpathy, Justin Johnson, Li Fei-Fei, [Visualizing and Understanding Recurrent Networks](http://arxiv.org/pdf/1506.02078), arXiv:1506.02078
* Rafal Jozefowicz, Wojciech Zaremba, Ilya Sutskever, [An Empirical Exploration of Recurrent Network Architectures](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf), ICML, 2015.

## Applications

### Natural Language Processing

#### Language Modeling
* Tomas Mikolov, Martin Karafiat, Lukas Burget, Jan "Honza" Cernocky, Sanjeev Khudanpur, *Recurrent Neural Network based Language Model*, Interspeech 2010 [[Paper](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)]
* Tomas Mikolov, Stefan Kombrink, Lukas Burget, Jan "Honza" Cernocky, Sanjeev Khudanpur, *Extensions of Recurrent Neural Network Language Model*, ICASSP 2011 [[Paper](http://www.fit.vutbr.cz/research/groups/speech/publi/2011/mikolov_icassp2011_5528.pdf)]
* Stefan Kombrink, Tomas Mikolov, Martin Karafiat, Lukas Burget, *Recurrent Neural Network based Language Modeling in Meeting Recognition*, Interspeech 2011 [[Paper](http://www.fit.vutbr.cz/~imikolov/rnnlm/ApplicationOfRNNinMeetingRecognition_IS2011.pdf)]
* Jiwei Li, Minh-Thang Luong, and Dan Jurafsky, *A Hierarchical Neural Autoencoder for Paragraphs and Documents*, ACL 2015 [[Paper](http://arxiv.org/pdf/1506.01057)], [[Code](https://github.com/jiweil/Hierarchical-Neural-Autoencoder)]
* Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, and Richard S. Zemel, *Skip-Thought Vectors*, arXiv:1506.06726 / NIPS 2015 [[Paper](http://arxiv.org/pdf/1506.06726.pdf)]
* Yoon Kim, Yacine Jernite, David Sontag, and Alexander M. Rush, *Character-Aware Neural Language Models*, arXiv:1508.06615 [[Paper](http://arxiv.org/pdf/1508.06615)]
* Xingxing Zhang, Liang Lu, and Mirella Lapata, *Tree Recurrent Neural Networks with Application to Language Modeling*, arXiv:1511.00060 [[Paper](http://arxiv.org/pdf/1511.00060.pdf)]
* Felix Hill, Antoine Bordes, Sumit Chopra, and Jason Weston, *The Goldilocks Principle: Reading children's books with explicit memory representations*, arXiv:1511.0230 [[Paper](http://arxiv.org/pdf/1511.02301.pdf)]


#### Speech Recognition
* Geoffrey Hinton, Li Deng, Dong Yu, George E. Dahl, Abdel-rahman Mohamed, Navdeep Jaitly, Andrew Senior, Vincent Vanhoucke, Patrick Nguyen, Tara N. Sainath, and Brian Kingsbury, *Deep Neural Networks for Acoustic Modeling in Speech Recognition*, IEEE Signam Processing Magazine 2012 [[Paper](http://cs224d.stanford.edu/papers/maas_paper.pdf)]
* Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton, *Speech Recognition with Deep Recurrent Neural Networks*, arXiv:1303.5778 / ICASSP 2013 [[Paper](http://www.cs.toronto.edu/~fritz/absps/RNN13.pdf)]
* Jan Chorowski, Dzmitry Bahdanau, Dmitriy Serdyuk, Kyunghyun Cho, and Yoshua Bengio, *Attention-Based Models for Speech Recognition*, arXiv:1506.07503 / NIPS 2015 [[Paper](http://arxiv.org/pdf/1506.07503)]
* Haşim Sak, Andrew Senior, Kanishka Rao, and Françoise Beaufays. *Fast and Accurate Recurrent Neural Network Acoustic Models for Speech Recognition*, arXiv:1507.06947 2015 [[Paper](http://arxiv.org/pdf/1507.06947v1.pdf)].

#### Machine Translation
* Oxford [[Paper](http://www.nal.ai/papers/kalchbrennerblunsom_emnlp13)]
  * Nal Kalchbrenner and Phil Blunsom, *Recurrent Continuous Translation Models*, EMNLP 2013
* Univ. Montreal
  * Kyunghyun Cho, Bart van Berrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio, *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*, arXiv:1406.1078 / EMNLP 2014 [[Paper](http://arxiv.org/pdf/1406.1078)]
  * Kyunghyun Cho, Bart van Merrienboer, Dzmitry Bahdanau, and Yoshua Bengio, *On the Properties of Neural Machine Translation: Encoder-Decoder Approaches*, SSST-8 2014 [[Paper](http://www.aclweb.org/anthology/W14-4012)]
  * Jean Pouget-Abadie, Dzmitry Bahdanau, Bart van Merrienboer, Kyunghyun Cho, and Yoshua Bengio, *Overcoming the Curse of Sentence Length for Neural Machine Translation using Automatic Segmentation*, SSST-8 2014
  * Dzmitry Bahdanau, KyungHyun Cho, and Yoshua Bengio, *Neural Machine Translation by Jointly Learning to Align and Translate*, arXiv:1409.0473 / ICLR 2015 [[Paper](http://arxiv.org/pdf/1409.0473)]
  * Sebastian Jean, Kyunghyun Cho, Roland Memisevic, and Yoshua Bengio, *On using very large target vocabulary for neural machine translation*, arXiv:1412.2007 / ACL 2015 [[Paper](http://arxiv.org/pdf/1412.2007.pdf)]
* Univ. Montreal + Middle East Tech. Univ. + Univ. Maine [[Paper](http://arxiv.org/pdf/1503.03535.pdf)]
  * Caglar Gulcehre, Orhan Firat, Kelvin Xu, Kyunghyun Cho, Loic Barrault, Huei-Chi Lin, Fethi Bougares, Holger Schwenk, and Yoshua Bengio, *On Using Monolingual Corpora in Neural Machine Translation*, arXiv:1503.03535
* Google [[Paper](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)]
  * Ilya Sutskever, Oriol Vinyals, and Quoc V. Le, *Sequence to Sequence Learning with Neural Networks*, arXiv:1409.3215 / NIPS 2014
* Google + NYU [[Paper](http://arxiv.org/pdf/1410.8206)]
  * Minh-Thang Luong, Ilya Sutskever, Quoc V. Le, Oriol Vinyals, and Wojciech Zaremba, *Addressing the Rare Word Problem in Neural Machine Transltaion*, arXiv:1410.8206 / ACL 2015
* ICT + Huawei [[Paper](http://arxiv.org/pdf/1506.06442.pdf)]
  * Fandong Meng, Zhengdong Lu, Zhaopeng Tu, Hang Li, and Qun Liu, *A Deep Memory-based Architecture for Sequence-to-Sequence Learning*, arXiv:1506.06442
* Stanford [[Paper](http://arxiv.org/pdf/1508.04025.pdf)]
  * Minh-Thang Luong, Hieu Pham, and Christopher D. Manning, *Effective Approaches to Attention-based Neural Machine Translation*, arXiv:1508.04025
* Middle East Tech. Univ. + NYU + Univ. Montreal [[Paper](http://arxiv.org/pdf/1601.01073.pdf)]
  * Orhan Firat, Kyunghyun Cho, and Yoshua Bengio, *Multi-Way, Multilingual Neural Machine Translation with a Shared Attention Mechanism*, arXiv:1601.01073

#### Conversation Modeling
* Lifeng Shang, Zhengdong Lu, and Hang Li, *Neural Responding Machine for Short-Text Conversation*, arXiv:1503.02364 / ACL 2015 [[Paper](http://arxiv.org/pdf/1503.02364)]
* Oriol Vinyals and Quoc V. Le, *A Neural Conversational Model*, arXiv:1506.05869 [[Paper](http://arxiv.org/pdf/1506.05869)]
* Ryan Lowe, Nissan Pow, Iulian V. Serban, and Joelle Pineau, *The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems*, arXiv:1506.08909 [[Paper](http://arxiv.org/pdf/1506.08909)]
* Jesse Dodge, Andreea Gane, Xiang Zhang, Antoine Bordes, Sumit Chopra, Alexander Miller, Arthur Szlam, and Jason Weston, *Evaluating Prerequisite Qualities for Learning End-to-End Dialog Systems*, arXiv:1511.06931 [[Paper](http://arxiv.org/pdf/1511.06931)]
* Jason Weston, *Dialog-based Language Learning*, arXiv:1604.06045, [[Paper](http://arxiv.org/pdf/1604.06045)]
* Antoine Bordes and Jason Weston, *Learning End-to-End Goal-Oriented Dialog*, arXiv:1605.07683 [[Paper](http://arxiv.org/pdf/1605.07683)]

#### Question Answering
* FAIR
  * Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, and Alexander M. Rush, *Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks*, arXiv:1502.05698 [[Web](https://research.facebook.com/researchers/1543934539189348)] [[Paper](http://arxiv.org/pdf/1502.05698.pdf)]
  * Antoine Bordes, Nicolas Usunier, Sumit Chopra, and Jason Weston, *Simple Question answering with Memory Networks*, arXiv:1506.02075 [[Paper](http://arxiv.org/abs/1506.02075)]
  * Felix Hill, Antoine Bordes, Sumit Chopra, Jason Weston, "The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations", ICLR 2016 [[Paper](http://arxiv.org/abs/1511.02301)]
* DeepMind + Oxford [[Paper](http://arxiv.org/pdf/1506.03340.pdf)]
  * Karl M. Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom, *Teaching Machines to Read and Comprehend*, arXiv:1506.03340 / NIPS 2015
* MetaMind [[Paper](http://arxiv.org/pdf/1506.07285.pdf)]
  * Ankit Kumar, Ozan Irsoy, Jonathan Su, James Bradbury, Robert English, Brian Pierce, Peter Ondruska, Mohit Iyyer, Ishaan Gulrajani, and Richard Socher, *Ask Me Anything: Dynamic Memory Networks for Natural Language Processing*, arXiv:1506.07285

#### Turing Machines
*  A.Graves, G. Wayne, and I. Danihelka., *Neural Turing Machines,* arXiv preprint arXiv:1410.5401 [[Paper](http://arxiv.org/pdf/1410.5401)]
* Jason Weston, Sumit Chopra, Antoine Bordes, *Memory Networks,* arXiv:1410.3916 [[Paper](http://arxiv.org/pdf/1410.3916)]
* Armand Joulin and Tomas Mikolov, *Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets*, arXiv:1503.01007 / NIPS 2015 [[Paper](http://arxiv.org/pdf/1503.01007)]
* Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus, *End-To-End Memory Networks*, arXiv:1503.08895 / NIPS 2015 [[Paper](http://arxiv.org/pdf/1503.08895)]
* Wojciech Zaremba and Ilya Sutskever, *Reinforcement Learning Neural Turing Machines,* arXiv:1505.00521 [[Paper](http://arxiv.org/pdf/1505.00521)]
* Baolin Peng and Kaisheng Yao, *Recurrent Neural Networks with External Memory for Language Understanding*, arXiv:1506.00195 [[Paper](http://arxiv.org/pdf/1506.00195.pdf)]
* Fandong Meng, Zhengdong Lu, Zhaopeng Tu, Hang Li, and Qun Liu, *A Deep Memory-based Architecture for Sequence-to-Sequence Learning*, arXiv:1506.06442 [[Paper](http://arxiv.org/pdf/1506.06442.pdf)]
* Arvind Neelakantan, Quoc V. Le, and Ilya Sutskever, *Neural Programmer: Inducing Latent Programs with Gradient Descent*, arXiv:1511.04834 [[Paper](http://arxiv.org/pdf/1511.04834.pdf)]
* Scott Reed and Nando de Freitas, *Neural Programmer-Interpreters*, arXiv:1511.06279 [[Paper](http://arxiv.org/pdf/1511.06279.pdf)]
* Karol Kurach, Marcin Andrychowicz, and Ilya Sutskever, *Neural Random-Access Machines*, arXiv:1511.06392 [[Paper](http://arxiv.org/pdf/1511.06392.pdf)]
* Łukasz Kaiser and Ilya Sutskever, *Neural GPUs Learn Algorithms*, arXiv:1511.08228 [[Paper](http://arxiv.org/pdf/1511.08228.pdf)]
* Ethan Caballero, *Skip-Thought Memory Networks*, arXiv:1511.6420 [[Paper](https://pdfs.semanticscholar.org/6b9f/0d695df0ce01d005eb5aa69386cb5fbac62a.pdf)]
* Wojciech Zaremba, Tomas Mikolov, Armand Joulin, and Rob Fergus, *Learning Simple Algorithms from Examples*, arXiv:1511.07275 [[Paper](http://arxiv.org/pdf/1511.07275.pdf)]

### Other
* Alex Graves, *Generating Sequences With Recurrent Neural Networks,* arXiv:1308.0850 [[Paper]](http://arxiv.org/abs/1308.0850)
* Wojciech Zaremba and Ilya Sutskever, *Learning to Execute*, arXiv:1410.4615 [[Paper](http://arxiv.org/pdf/1410.4615.pdf)] [[Code](https://github.com/wojciechz/learning_to_execute)]
* Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer, *Scheduled Sampling for Sequence Prediction with
Recurrent Neural Networks*, arXiv:1506.03099 / NIPS 2015 [[Paper](http://arxiv.org/pdf/1506.03099)]
* Cesar Laurent, Gabriel Pereyra, Philemon Brakel, Ying Zhang, and Yoshua Bengio, *Batch Normalized Recurrent Neural Networks*, arXiv:1510.01378 [[Paper](http://arxiv.org/pdf/1510.01378)]

## Datasets
* Speech Recognition
  * [OpenSLR](http://www.openslr.org/resources.php) (Open Speech and Language Resources)
    * [LibriSpeech ASR corpus](http://www.openslr.org/12/)
  * [VoxForge](http://voxforge.org/home)
* Question Answering
  * [The bAbI Project](http://fb.ai/babi) - Dataset for text understanding and reasoning, by Facebook AI Research. Contains:
    * The (20) QA bAbI tasks - [[Paper](http://arxiv.org/abs/1502.05698)]
    * The (6) dialog bAbI tasks - [[Paper](http://arxiv.org/abs/1605.07683)]
    * The Children's Book Test - [[Paper](http://arxiv.org/abs/1511.02301)]
    * The Movie Dialog dataset - [[Paper](http://arxiv.org/abs/1511.06931)]
    * The MovieQA dataset - [[Data](http://www.thespermwhale.com/jaseweston/babi/movie_dialog_dataset.tgz)]
    * The Dialog-based Language Learning dataset - [[Paper](http://arxiv.org/abs/1604.06045)]
    * The SimpleQuestions dataset - [[Paper](http://arxiv.org/abs/1506.02075)]
  * [SQuAD](https://stanford-qa.com/) - Stanford Question Answering Dataset :  [[Paper](http://arxiv.org/pdf/1606.05250)]

## Blogs
* [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/)
* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) in [Colah's blog](http://colah.github.io/)
* [WildML](http://www.wildml.com/) blog's RNN tutorial [[Part1](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)], [[Part2](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)], [[Part3](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)], [[Part4](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/)]
* [RNNs in Tensorflow, a Practical Guide and Undocumented Features](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/)
* [Optimizing RNN Performance](https://svail.github.io/) from Baidu's Silicon Valley AI Lab.
* [Character Level Language modelling using RNN](http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139) by Yoav Goldberg
* [Implement an RNN in Python](http://peterroelants.github.io/posts/rnn_implementation_part01/).
* [LSTM Backpropogation](http://arunmallya.github.io/writeups/nn/lstm/index.html#/)
* [Introduction to Recurrent Networks in TensorFlow](https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/) by Danijar Hafner
* [Variable Sequence Lengths in TensorFlow](https://danijar.com/variable-sequence-lengths-in-tensorflow/) by Danijar Hafner

## Online Demos
* LSTMVis: Visual Analysis for Recurrent Neural Networks [[link](http://lstm.seas.harvard.edu/)]
