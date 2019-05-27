        with tf.name_scope("accuracy"):
            # 求准确率，首先把布尔类型转化为浮点类型
            accuracy = acc
            tf.summary.scalar("accuracy", accuracy)
        with tf.name_scope("loss"):
            # 二次代价函数
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_))
            tf.summary.scalar("loss",loss)

        init = tf.global_variables_initializer()
        merged=tf.summary.merge_all()

        sess=tf.Session()
        sess.run(init)
            # 保存Tensorboard文件
        writer = tf.summary.FileWriter("logs/", sess.graph)
        for epoch in range(65):
            for batch in range(1):
                # 使用函数获取一个批次图片
                summary,_ = sess.run([merged,train_op], feed_dict={x: x_train_a, y_: y_train_a})
                print(summary)
            writer.add_summary(summary, epoch)
            print("Iter " + str(epoch) )