# -*- coding: utf-8 -*-

"""
入力データクラスです。
"""

import csv
import random
import numpy as np

UNKNOWN = 'UNK_'

class InputData(object):


  def __init__(self
  , train_data_path='data/train_data.txt'
  , validation_data_path='data/validation_data.txt'
  , test_data_path='data/test_data.txt'
  , batch_size=64
  , max_length=20
  , train=True):
    '''初期化処理
    trainは訓練データとバリデーションデータの読み込み
    eval時はtestデータの読み込みを行う

    Arguments:
      train_data_path (str): トレーニングデータのパス
      validation_data_path (str): バリデーションデータのパス
      test_data_path (str): テストデータのパス
      batch_size (int): バッチサイズ
      max_length (int): 入力データのアイテム名の最大長
      train (bool): 訓練時True, 評価時False
    
    Returns:
      なし
    '''
    with open('./data/ids/category_ids.txt', 'r') as f:
      lines = f.readlines() 
      self.num_category = len(lines)
      self.category_dict = {int(line.split(',')[1]) : line.split(',')[0] for i, line in enumerate(lines)}
    self.batch_size = batch_size
    self.chars_dict = self._load_char_dict()
    self.num_chars = len(list(self.chars_dict.keys()))
    self.max_len = max_length
    self.train = train
    # 以下はtrainモードのみ
    if self.train:
      self.train_data = self._read_data(train_data_path)
      self.validation_data = self._read_data(validation_data_path)
      self.idx = 0
    else:
      self.test_data = self._read_data(test_data_path)
    self.evaluation_idx = 0


  def _load_char_dict(self):
    '''文字辞書の読み込みを行う
    外部からの呼び出し非推奨

    Arguments:
      なし
    Returns:
      なし
    '''
    with open('data/ids/char_ids.txt', 'r') as f:
      return {c.rstrip("\n"):i for i, c in enumerate(f)}


  def _read_data(self, data_path):
    '''ファイル読み込みを行う
    外部からの呼び出し非推奨

    Arguments:
      data_path (str): 読み込み対象ファイルのパス
    
    Returns
      list: 訓練時 [ラベルを表現する数値, ベクトル化された文字列]　評価時 [ラベルを表現する数値, ベクトル化された文字列, ユニークID, アイテム名]の配列
    '''
    data = []
    with open(data_path, 'r') as f:
      reader = csv.reader(f)
      for line in reader:
        ids = self.sentence_to_vector(line[1])
        if self.train:
          data.append([line[0], ids])
        else:
          # testの場合はアイテム名とユニークIDを追加
          data.append([line[0], ids, line[2], line[1]])
    return data


  def sentence_to_vector(self, sentence):
    '''文字列を文字IDのベクトルに変換する
    返却値は最大長まで0埋めされたベクトル
    ベクトルの各要素は文字IDを示すint

    Arguments:
      sentence (str): アイテム名
    
    Returns
      list: 文字列をベクトル化した配列
    '''
    sentence = list(sentence)
    sentence = [c if c in self.chars_dict else UNKNOWN for c in sentence]
    ids = [self.chars_dict[c] for c in sentence]
    ids = ids + [0 for _ in range(self.max_len - len(ids))]
    return ids


  def chars_to_unknown(self, sentence):
    '''文字列中のUNKNOWN文字を*に変換する
    前処理された文字列を可視化するために用いる
    アイテム名内に*が存在した場合、前処理段階で削除されているので、
    前処理後の文字列に*が実際に含まれることはないことに注意すること

    Arguments:
      sentence (str): アイテム名
    
    Returns
      str: アイテム名内のUNKNOWN文字を*に変換した文字列
    '''
    sentence = [c if c in self.chars_dict else '*' for c in sentence]
    return ''.join(sentence)


  def next_batch(self):
    '''データバッチを返却する

    Auguments:
      なし
    
    Returns:
      list: one-hot encodingされた正解ラベルが格納された配列 要素数はバッチサイズ
      list: id化されたアイテム名が格納された配列 要素数はバッチサイズ
    '''
    if self.idx + self.batch_size > len(self.train_data):
      self.idx = 0
      random.shuffle(self.train_data)
    batch = self.train_data[self.idx:self.idx+self.batch_size]
    labels = [self._one_hot_vector(e[0], self.num_category) for e in batch]
    texts = [e[1] for e in batch]
    self.idx += self.batch_size
    return labels, texts


  def next_batch_evaluation_data(self):
    '''評価用のデータバッチを返却する
    trainモードならvalidation_data, evalモードならtest_dataを使用する
    計算の煩雑さを解消するため、validation_dataはvalidation_data全体のレコード数に対して、
    バッチサイズに分割した際の端数は無視される

    Auguments:
      なし
    
    Returns:
      list: one-hot encodingされた正解ラベルが格納された配列 要素数はバッチサイズ
      list: id化されたアイテム名が格納された配列 要素数はバッチサイズ
    '''

    if self.train:
      data = self.validation_data
    else:
      data = self.test_data

    while self.evaluation_idx + self.batch_size < len(data):
      batch = data[self.evaluation_idx:self.evaluation_idx+self.batch_size]
      labels, texts, unique_ids, item_names = self._shape_data(batch)
      self.evaluation_idx += self.batch_size
      yield (labels, texts) if self.train else (labels, texts, unique_ids, item_names)
    
    # evalモード時は端数分も返却
    if not self.train:
      batch = data[self.evaluation_idx:]
      labels, texts, unique_ids, item_names = self._shape_data(batch)
      yield (labels, texts) if self.train else (labels, texts, unique_ids, item_names)
    self.evaluation_idx = 0
  

  def _shape_data(self, batch):
    '''バッチをメイン処理へ渡す形に整形する

    Auguments:
      batch (list): ラベル、テキスト、（テスト時のみ、ユニークID、アイテム名が追加される）のタプルが格納された配列 要素数はバッチサイズ
    
    Returns:
      list: one-hot encodingされた正解ラベルが格納された配列 要素数はバッチサイズ
      list: id化されたアイテム名が格納された配列 要素数はバッチサイズ
      list: ユニークIDが格納された配列 要素数はバッチサイズ ただし、訓練時はNoneが返却される
      list: アイテム名が格納された配列 要素数はバッチサイズ ただし、訓練時はNoneが返却される
    '''
    labels = [self._one_hot_vector(e[0], self.num_category) for e in batch]
    texts_vectors = [e[1] for e in batch]
    unique_ids = [e[2] for e in batch] if not self.train else None
    item_names = [e[3] for e in batch] if not self.train else None
    return labels, texts_vectors, unique_ids, item_names


  def _one_hot_vector(self, index, length):
    '''one-hotベクトルを生成する

    Auguments:
      index (int): one-hotのindex
      length (int): one-hot ベクトルの長さ
    
    Returns:
      Numpy array: one-hot encodingされたNumpy配列 
    '''
    one_hot = np.zeros(length)
    one_hot[int(index)] = 1
    return one_hot