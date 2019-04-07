# coding = utf-8
import time,collections,math,random,zipfile
import numpy as np
import tensorflow as tf


TRAINING_STEPS = 1000  # 最大训练步数
SAVE_CHECKPOINT_SECS = 600  # 每隔600秒保存一次模型
DATA_PATH = "./data/data/text8.zip"  # 数据位置


NUM_SKIPS = 2
SKIP_WINDOW = 1
VOCABULARY_SIZE = 50000
embedding_size = 128  # 单词转化为稠密词向量的维度
VALID_SIZE = 16  # 验证单词数
VALID_WINDOW = 100  # 验证单词数从频数最高的100个单词里面抽取
NUM_SAMPLED = 64
BATCH_SIZE = 128
LEARNING_RATE_BASE = 1.0  # 学习率


X = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
Y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])


def build_dataset(words):
    count = [['UNK', -1]]
    "统计单词列表中单词的频数，把前50000的放入字典"
    count.extend(collections.Counter(words).most_common(VOCABULARY_SIZE - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    """
    不在前50000里面 编码为0
    """
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        "将数据转化为单词列表"
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


words = read_data(DATA_PATH)
data, count, dictionary, reverse_dictionary = build_dataset(words)
del words
data_index = 0


def next_batch(step):
    """

    :param batch_size:
    :param num_skips:  对每个单词生成多少样本 不大于2*skip_window
    :param skip_window: 滑动窗口步长
    :return: batch
              labels
    """
    global data_index
    assert BATCH_SIZE % NUM_SKIPS == 0
    assert NUM_SKIPS <= 2 * SKIP_WINDOW
    batch = np.ndarray(shape=(BATCH_SIZE), dtype=np.int32)
    labels = np.ndarray(shape=(BATCH_SIZE, 1), dtype=np.int32)
    span = 2 * SKIP_WINDOW + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(BATCH_SIZE // NUM_SKIPS):  # 一块batch里面有包含的目标单词数
        target = SKIP_WINDOW
        target_to_avoid = [SKIP_WINDOW]  # 需要避免的单词列表
        for j in range(NUM_SKIPS):
            # 找到可以使用的语境词语
            while target in target_to_avoid:
                target = random.randint(0, span - 1)
            target_to_avoid.append(target)
            batch[i * NUM_SKIPS + j] = buffer[SKIP_WINDOW]  # 目标词汇
            labels[i * NUM_SKIPS + j, 0] = buffer[target]  # 语境词汇
        "buffer此时已经填满，后续的数据会覆盖掉前面的数据"
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


global global_step
global loss
global train_op
start_time = time.time()


# 实现创建模型接口，在这里，我们没有用到传入的参数is_chief（代表是否是分布式中的主节点）
def build_model(is_chief):
    global global_step
    global loss
    global train_op

    embeddings = tf.Variable(
        tf.random_uniform([VOCABULARY_SIZE, embedding_size], -1.0, 1.0)
    )
    embed = tf.nn.embedding_lookup(embeddings, X)  # 查找输入对应的向量
    nce_weights = tf.Variable(
        tf.truncated_normal([VOCABULARY_SIZE, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size))
    )
    nce_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE]))

    global_step = tf.contrib.framework.get_or_create_global_step()  # 创建全局global_step，必写
    loss = tf.reduce_mean(tf.nn.nce_loss(
        weights=nce_weights,
        biases=nce_biases,
        labels=Y_,
        inputs=embed,
        num_sampled=NUM_SAMPLED,
        num_classes=VOCABULARY_SIZE
    ))

    # 调用minimize函数时，必须传入global_step参数
    train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE_BASE).minimize(loss, global_step=global_step)


# 实现训练接口，step是本地的step数，每调用一次train_model函数，该step会+1
def train_model(session, step):
    xs, ys = next_batch(step)
    _, loss_value, global_step_value = session.run([train_op, loss, global_step], feed_dict={X: xs, Y_: ys})

    if step % 200 == 0:
        # 用yield传出需要打印的信息
        yield "After %s seconds training,loss is %s,global_step is %s" % ((time.time() - start_time), loss_value, global_step_value)