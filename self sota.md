## Cifar 10

* Best model   6.29    0.13
  * th main.lua -dataset cifar10 -nGPU 1 -depth 26 -shareGradInput false -optnet true -nEpochs 30 -netType shakeshake -lrShape cosine -widenFactor 6 -LR 0.2 -forwardShake true -backwardShake true -shakeImage true -batchSize 128

