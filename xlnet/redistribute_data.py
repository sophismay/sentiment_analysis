from pathlib import Path
from typing import List, Tuple
import random

def sample_data(file_path: Path, n: int = 14650) -> List[str]:
  with file_path.open(mode= 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)
    return lines[:n]

def distribute_dev_test_data(file_path: Path) -> Tuple[List[str]]:
  with file_path.open(mode= 'r') as f:
    lines = f.readlines()
    result = []
    for line in lines:
      parts = line.split(',')
      parts = parts[1:]
      result.append(",".join(parts))

    return (result[:350], result[350:550], result[550:])

def convert_to_text(lines: List[str]) -> str:
  text = ""
  for line in lines:
    text += "{}".format(line)
  return text

def write_files(train_data: List[str], dev_test_data: Tuple[List[str]]) -> Tuple[List[str]]:
  """
    Generate a train(combined with some target data), dev and test training sets
    @param: result_file, type: Path
  """
  train_extra_data, dev_data, test_data = dev_test_data
  train_data.extend(train_extra_data)
  train_file = Path('data/train_redist.txt')
  dev_file = Path('data/dev_redist.txt')
  test_file = Path('data/test_redist.txt')
  train_file.write_text(convert_to_text(train_data))
  dev_file.write_text(convert_to_text(dev_data))
  test_file.write_text(convert_to_text(test_data))

  return

if __name__ == "__main__":
  dir = Path('data/')
  train_file = dir / 'train.txt'
  train_data = sample_data(train_file)
  print("{} data points shuffled from original training data ".format(len(train_data)))
  dev_test_data = distribute_dev_test_data(dir / 'dev_test_all.csv')
  write_files(train_data, dev_test_data)