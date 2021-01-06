# text-adversarial-training

对抗训练在NLP中的应用。这里提供的实现是把扰动加到Embedding矩阵上，即论文[Adversarial Training Methods for Semi-Supervised Text Classification](https://arxiv.org/abs/1605.07725)上的思路。在Tensorflow2.x上实现很简单，具体见文件`adversarial_trainer.py`。

`AdversarialTrainer`类的使用，假设已经实现好模型的输入和输出，那么

```python
model = AdversarialTrainer(inputs, outputs)
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"],
              epsilon=0.8)
model.fit(X, y)
```

就可以实现对抗训练。


根据实验需要和数据集位置修改相关参数后，运行：

```bash
$ python3 adversarial_training.py
```

具体可参看源码。


通过我的实验，对抗训练在一些数据集上有1%+提升。需要说明，这里的对抗训练作为一种正则化方案，不能保证在任何数据集上都有提升，可能需要多跑几组实验调整下参数才有结果。


## 参考

[1] https://tensorflow.google.cn/tutorials/generative/adversarial_fgsm?hl=en
[2] [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
[3] [Adversarial Training Methods for Semi-Supervised Text Classification](https://arxiv.org/abs/1605.07725)
