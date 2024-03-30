# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2024/03/29 (C) antillia.com


import os
import sys
import numpy as np

import traceback
import csv
from ConfigParser import ConfigParser

import matplotlib.pyplot as plt

class LineGraphPlotter:
  # Constructor
  def __init__(self):
    print("=== LineGraphPlotter")
    self.line_style          = "solid"
    self.marker              = "o"
    self.y_label             = "score"
    self.output_image_format = ".png"

  def plot(self, eval_dir):
    train_metrics = os.path.join(eval_dir, "train_metrics.csv")
    train_losses  = os.path.join(eval_dir, "train_losses.csv")

    self.plot_csv_file(train_metrics, eval_dir)
    self.plot_csv_file(train_losses,  eval_dir)


  def plot_csv_file(self, csv_file, output_dir):
    print("=== plot_csv")

    rows = []
    basename = os.path.basename(csv_file)

    with open(csv_file) as f:   
      reader = csv.reader(f)
      rows = [row for row in reader]

    header = rows.pop(0)

    data = np.float_(np.array(rows).T)
    print(header)

    fig, ax = plt.subplots()

    ax.plot(data[0], data[1], linestyle=self.line_style, 
                     marker=self.marker, label= header[1])
    ax.plot(data[0], data[2], linestyle=self.line_style, 
                     marker=self.marker, label= header[2])

    ax.set_xlabel(header[0])

    ax.set_ylabel(self.y_label)

    ax.legend()

    plt.title(basename)
    output_file = os.path.join(output_dir, basename + self.output_image_format)

    plt.savefig(output_file)
    print("=== Saved {}".format(output_file))

    plt.close()
    #plt.show()

if __name__ == "__main__":
  try:
    eval_dir = "./eval"
    config_file = "./train_eval_infer.config"

    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    if not os.path.exists(config_file):
      raise Exception("Not found " + config_file)
    config   = ConfigParser(config_file)
    eval_dir = config.get("train", "eval_dir")
    if not os.path.exists(eval_dir):
      raise Exception("Not found "+ eval_dir) 
    plotter = LineGraphPlotter()
    plotter.plot(eval_dir)

  except:
    traceback.print_exc()
