# -*- coding: utf-8 -*-
import os
import csv
import resource
import numpy as np
import tensorflow as tf
from input_data import InputData

tf.app.flags.DEFINE_boolean("eval", False, "評価モードフラグ")
tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint/', """チェックポイント保存先""")
tf.app.flags.DEFINE_string('log_dir', 'logs/', """学習ログ出力先""")
tf.app.flags.DEFINE_string('train_data', 'data/train_data.txt', """訓練データパス""")
tf.app.flags.DEFINE_string('validation_data', 'data/validation_data.txt', """バリデーションデータパス""")
tf.app.flags.DEFINE_string('test_data', 'data/test_data.txt', """テストデータパス""")
tf.app.flags.DEFINE_integer('max_step', 50000, """訓練回数""")
tf.app.flags.DEFINE_integer('batch_size', 256, """バッチサイズ""")
tf.app.flags.DEFINE_string('output_dir', './data', """評価機能の出力ファイルのディレクトリ""")


FLAGS = tf.app.flags.FLAGS


def train():
  '''訓練を実施する関数
  '''
  with tf.Graph().as_default():
    print('Loading data...')
    data = InputData(train_data_path=FLAGS.train_data, validation_data_path=FLAGS.validation_data, batch_size=FLAGS.batch_size)
    input_ph = tf.placeholder(tf.int32, [None, data.max_len])
    training_ph = tf.placeholder(tf.bool, [])
    label_ph = tf.placeholder(tf.float32, [None, data.num_category])
    # summary用のplace holder
    summary_ph = tf.placeholder(tf.float32, [3])
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

    with tf.Session(config=config) as sess:
      convolution_op = convolution(input_ph, training_ph, data.num_chars, data.num_category)
      loss_op = loss(convolution_op, label_ph)
      train_op = minimize(loss_op)
      accuracy_op = accuracy(convolution_op, label_ph)
      top_3_accuracy_op = top_3_accuracy(convolution_op, label_ph)
      summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
      write_validation_summary_op = write_validation_summary(summary_ph)

      for vals in tf.trainable_variables():
        tf.summary.histogram(vals.name, vals)

      summary_op = tf.summary.merge_all()
      saver = tf.train.Saver()
      load_checkpoint(sess, saver)
      for i in range(FLAGS.max_step):

        label_, text_ = data.next_batch()
        _ = sess.run(train_op, feed_dict={input_ph: text_, label_ph: label_, training_ph: True})
        if i % 100 == 0:
          loss_, accuracy_, top_3_accuracy_ = sess.run([loss_op, accuracy_op, top_3_accuracy_op], feed_dict={input_ph: text_, label_ph: label_, training_ph: False})
          print('global step: %04d, train loss: %01.7f, train accuracy_top_1 %01.5f train accuracy_top_3 %01.5f' % (i, loss_, accuracy_, top_3_accuracy_))

        if i % 1000 == 0:
          validation_loss = []
          validation_top_3_accuracy = []
          validation_top_1_accuracy = []
          for validation_label, validation_text in data.next_batch_evaluation_data():
            loss_, top_3_accuracy_, accuracy_ = sess.run([loss_op, top_3_accuracy_op, accuracy_op], 
                          feed_dict={input_ph: validation_text, label_ph: validation_label, training_ph: False})
            validation_loss.append(loss_)
            validation_top_1_accuracy.append(accuracy_)
            validation_top_3_accuracy.append(top_3_accuracy_)
          loss_ = sum(validation_loss) / len(validation_loss)
          accuracy_ = sum(validation_top_1_accuracy) / len(validation_top_1_accuracy)
          top_3_accuracy_ = sum(validation_top_3_accuracy) / len(validation_top_3_accuracy)
          print('Validation loss: %s validation accuracy_top_1: %01.5f validation accuracy_top_3: %01.5f' % (loss_, accuracy_, top_3_accuracy_))
          saver.save(sess, FLAGS.checkpoint_dir, global_step=i)
          # サマリー出力
          _, summary_str = sess.run([write_validation_summary_op, summary_op], feed_dict={input_ph: text_, label_ph: label_, training_ph: False, summary_ph: [loss_, accuracy_, top_3_accuracy_]})
          summary_writer.add_summary(summary_str, i)
          ru = resource.getrusage(resource.RUSAGE_SELF)
          print('Max memory usage(byte): ' + str(ru.ru_maxrss))


def write_validation_summary(summary):
  '''サマリを出力する関数

  Arguments:
    summary (Tensor): Shape[3]のTensor
  
  Returns:
    Tensor: StringTensor
  '''
  tf.summary.scalar('validation_loss', summary[0])
  tf.summary.scalar('validation_accuracy_top1', summary[1])
  return tf.summary.scalar('validation_accuracy_top3', summary[2])


def convolution(input_, training, num_chars, num_category):
  '''畳み込みニューラルネットワークを表現する関数

  Arguments:
    input_ (Tensor): Shape[batch_size, max_length]のTensor
    training (Tensor): bool値が格納されたスカラーテンソル 訓練時にTrue
    num_chars (int): 文字辞書に記載された文字種類数
    num_category (int): カテゴリーの総数
  
  Returns:
    Tensor: Shape[batch_size, num_ccategory]のTensor
  '''
  w = tf.get_variable("embedding", [num_chars, 128])
  output = tf.gather(w, input_)
  convs = []
  for filter_size in (2,3,4,5):
    conv = tf.layers.conv1d(output, 64, filter_size, 1, padding='SAME')
    # output = tf.layers.max_pooling1d(output, filter_size, 1, padding='SAME')
    conv = tf.nn.relu(conv)
    convs.append(conv)
  
  output = tf.concat(convs, axis=1)
  output = tf.contrib.layers.flatten(output)

  for _ in range(2):
    output = tf.layers.dense(output, 1024)
    output = tf.layers.batch_normalization(output, training=training)
    output = tf.nn.relu(output)

  output = tf.layers.dense(output, num_category)
  output = tf.nn.softmax(output)
  return output


def loss(logits, labels):
  '''損失の計算を実施
  損失関数は交差エントロピー

  Arguments:
    logits (Tensor): Shape[batch_size, num_category]のTensor 予測値を表現する
    labels (Tensor): Shape[batch_size, num_category]のTensor 正解ラベルを表現する
  
  Returns:
    Tensor: 損失
  '''
  loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits + 1e-10), reduction_indices=[1]))
  tf.summary.scalar('loss', loss)
  return loss


def minimize(loss):
  '''勾配を計算して、損失を最小化する

  Arguments:
    loss (Tensor): 損失
  
  Returns:
    Tensor: 最小化のオペレーション
  '''
  return tf.train.AdamOptimizer().minimize(loss)


def accuracy(logits, labels):
  '''正解率を計算する
  予測値の内、最大の値を示すカテゴリとラベルが示すカテゴリの一致率を計算する

  Arguments:
    logits (Tensor): 予測値
    labels (Tensor): 正解ラベル
  
  Returns:
    Tensor: 正解率が格納されたテンソル
  '''
  correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)
  return accuracy


def top_3_accuracy(logits, labels):
  '''TOP3正解率を計算する
  予測値の内、一番目、二番目、三番目に大きな値を示すカテゴリとラベルが示すカテゴリの一致率を計算する

  Arguments:
    logits (Tensor): 予測値
    labels (Tensor): 正解ラベル
  
  Returns:
    Tensor: 正解率が格納されたテンソル
  '''
  correct_prediction = tf.nn.in_top_k(logits, tf.argmax(labels,1), 3)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return accuracy


def evaluate():
  '''テストデータに対する評価を実施する関数
  '''
  data = InputData(test_data_path=FLAGS.test_data, train=False)
  input_ph = tf.placeholder(tf.int32, [None, data.max_len])
  training_ph = tf.placeholder(tf.bool, [])
  label_ph = tf.placeholder(tf.float32, [None, data.num_category])
  with tf.Session() as sess:
    output = convolution(input_ph, training_ph, data.num_chars, data.num_category)
    values, indices = tf.nn.top_k(output, k=10)
    saver = tf.train.Saver()
    load_checkpoint(sess, saver)
    with open(FLAGS.output_dir + '/evaluate.tsv', 'w') as f:
      writer = csv.writer(f, delimiter='\t')
      for test_labels, test_texts, unique_ids, item_names in data.next_batch_evaluation_data():
        values_, indices_ = sess.run([values, indices], feed_dict={input_ph: test_texts, training_ph: False})
        for (value, index, test_label, unique_id, item_name) in zip(values_, indices_, test_labels, unique_ids, item_names):
          row = [unique_id] + [data.category_dict[np.argmax(test_label)]] + [data.chars_to_unknown(item_name)]  + list(value) + list(map(lambda x: data.category_dict[x], index)) + [index[0] == np.argmax(test_label)] + [np.argmax(test_label) in index[0:3]]
          writer.writerow(row)
    
    num_records = len(open(FLAGS.output_dir + '/evaluate.tsv', 'r').readlines())
    with open(FLAGS.output_dir + '/evaluate.tsv', 'r') as f:
      reader = csv.reader(f, delimiter='\t')
      accuracy_count = [(line[23], line[24]) for line in reader]
      accuracy_top1 = len(list(filter(lambda x: x[0] == 'True', accuracy_count))) / num_records
      accuracy_top3 = len(list(filter(lambda x: x[1] == 'True', accuracy_count))) / num_records

    with open(FLAGS.output_dir + '/test_accuracy.tsv', 'w') as f:
      writer = csv.writer(f, delimiter='\t')
      writer.writerow([accuracy_top1, accuracy_top3])
        



def load_checkpoint(sess, saver):
  '''チェックポイントを読み込む関数
  チェックポイントファイルが存在しない場合は全てのパラメータが初期化されて実行される

  Arguments:
    sess (tf.Session): TensorFlowのSession
    saver (tf.train.Saver): ModelのSaver
  
  Returns:
    なし
  '''
  if os.path.exists(FLAGS.checkpoint_dir + 'checkpoint'):
    print('restore parameters...')
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
  else:
    print('initilize parameters...')
    init_op = tf.global_variables_initializer()
    sess.run(init_op)


def main():
  '''メイン関数
  '''
  if FLAGS.eval:
    evaluate()
  else:
    train()
  

if __name__ == '__main__':
  main()