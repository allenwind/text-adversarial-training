import tensorflow as tf

class AdversarialTrainer(tf.keras.Model):
    """对抗训练器，像tf.keras.Model一样使用，
    这里实现的是Fast Gradient Method。"""

    def compile(
        self,
        optimizer,
        loss,
        metrics,
        embedding_name="embedding",
        epsilon=0.5):
        super(AdversarialTrainer, self).compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        self.epsilon = epsilon
        # 需要注意Embedding的名称，以便搜索该层
        self.embedding_name = embedding_name

    def train_step(self, data):
        embedding_layer = self.get_layer(self.embedding_name)
        embeddings = embedding_layer.embeddings
        x, y = data
        with tf.GradientTape() as tape:
            tape.watch(embeddings)
            y_pred = self(x, training=False)
            loss = self.compiled_loss(y, y_pred)
        grads = tape.gradient(loss, embeddings) # 计算Embedding梯度
        grads = tf.zeros_like(grads) + grads
        delta = self.epsilon * grads / (tf.norm(grads) + 1e-8) # 计算扰动
        embeddings.assign_add(delta) # 添加扰动到Embedding矩阵
        results = super(AdversarialTrainer, self).train_step(data) # 执行普通的train_step
        embeddings.assign_sub(delta) # 删除Embedding矩阵上的扰动
        return results
