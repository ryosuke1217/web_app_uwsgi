# -*- coding: utf-8 -*-

"""
データ分割、行の除外、文字列前処理を実行します。
"""

import csv
import argparse
import random
import resource
import mojimoji
import numpy as np
from collections import defaultdict

UNKNOWN = 'UNK_'

class Preprocessor(object):
  '''前処理実行クラス
  '''

  def __init__(self):
    '''初期化処理
    予め用意されたファイルから削除対象文字を読み込む
    '''
    self.delete_chars = self._load_delete_chars()


  def execute(self, data_path):
    '''前処理実行関数
    コマンドライン引数で指定されたデータを
    21文字以上のデータを除外、文字列に前処理を施した上で、
    訓練：バリデーション：テスト＝８：１：１にランダムに分割します
    分割後のデータは./data/に保存されます

    Arguments:
      data_path (str): Rawデータのパス
    Returns:
      なし
    '''
    with open(data_path, 'r', encoding='shift_jisx0213') as f:
      # ヘッダ行を除く
      f.readline()
      lines = list(filter(lambda x: len(x.split("\t")[33]) <= 20, f.readlines()))
      extracted = self._extract(lines)
      num_records = len(extracted)
      # train:validation:test = 8:1:1になるように分割
      num_validation_data = num_test_data = int(num_records / 10)
      data = random.shuffle(extracted)
      test_data = extracted[0:num_test_data]
      validation_data = extracted[num_test_data:num_validation_data+num_test_data]
      train_data = extracted[num_test_data+num_validation_data:]
      self._create_char_dict(train_data)
      self._write(validation_data, 'validation_data.txt')
      self._write(test_data, 'test_data.txt')
      self._write(train_data, 'train_data.txt')


  def _extract(self, lines):
    lines = list(map(lambda y: (y[38] if y[38] != '-' else y[36], self._convert_chars(y[33]), y[4]), filter(lambda x: len(x[33]) <= 20, map(lambda l: l.split('\t'), lines))))
    lines = self._convert_ids(lines)
    return lines


  def _convert_ids(self, data):
    '''小項目IDを変換します
    再利用できるようにファイルに保存します

    Arguments:
      data (list): rawデータの配列
    Returns:
      list: 変換されたIDの配列
    '''
    # 順序の一貫性を保つためにlistに変換
    ids = list(set([datum[0] for datum in data]))
    ids_dict = {id_:i for i, id_ in enumerate(ids)}
    # あとで使えるようにファイル保存
    self._write(ids_dict.items(), 'ids/category_ids.txt')
    return list(map(lambda x: (ids_dict[x[0]], x[1], x[2]), data))


  def _create_char_dict(self, train_data):
    '''文字の辞書を作成します
    再利用可能なようにファイルに保存します

    Arguments:
      train_data (list): 学習データの配列 
    Returns:
      なし
    '''
    counter = defaultdict(int)
    for datum in train_data:
      for c in datum[1]:
        counter[c] += 1
    sum_count = sum(counter.values())
    # 累積出現比率を算出
    frequency = np.array(sorted(list(map(lambda x: x / sum_count, counter.values())), reverse=True)).cumsum()
    # 出現度の高い順に並べたときに、累積出現比率95%になる文字数を算出
    num_chars = len(list(filter(lambda x: x < 0.95, frequency))) + 1
    chars = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    chars = [c[0] for c in chars]
    chars = chars[:num_chars]
    chars = [UNKNOWN] + chars
    self._write(chars, 'ids/char_ids.txt', is_csv=False)


  def _convert_chars(self, chars):
    '''文字列に対して前処理を実施します

    Arguments:
      chars (str): アイテム名
    Returns:
      変換後のアイテム名
    '''
    chars = mojimoji.han_to_zen(chars, digit=False, ascii=False)
    chars = mojimoji.zen_to_han(chars, kana=False).lower()
    chars = ''.join(list(filter(lambda c: c not in self.delete_chars, chars)))
    # 1-9までの数字を全て0に変換
    chars = ''.join(list(map(lambda c: '0' if c in [str(i) for i in range(1, 10)] else c, chars)))
    chars = chars.replace(' ', '')
    return chars


  def _write(self, lines, filename, is_csv=True):
    '''データをファイルに出力します
    
    Arguments:
      lines (list): データ行の配列
      filename (str): ファイル名
    
    Returns:
      なし
    '''
    with open('./data/' + filename, 'w', encoding='utf-8') as f:
      if is_csv:
        writer = csv.writer(f)
        writer.writerows(lines)
      else:
        for l in lines:
          f.write(l + '\n')
  

  def _load_delete_chars(self):
    '''削除対象文字を読み込みます
    
    Arguments:
      なし
    
    Returns:
      list: 削除対象文字のリスト
    '''
    with open('./data/delete_char.txt', 'r') as f:
      return list(map(lambda x: x.rstrip(), f.readlines()))


def main(args):
  '''メイン関数

  Arguments:
    args: コマンドライン引数
  Returns:
    なし
  '''
  print('Start preprocessing.')
  preprocessor = Preprocessor()
  preprocessor.execute(args.data_path)
  ru = resource.getrusage(resource.RUSAGE_SELF)
  print('Max memory usage(byte): ' + str(ru.ru_maxrss))
  print('Finish.')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='This script is ...')
  parser.add_argument('--data_path', default='./data/raw/test.txt', help='input data path.')
  args = parser.parse_args()
  main(args)