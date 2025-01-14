import glob
import os
import shutil
import time
import sys
pjoin = os.path.join
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

class LogPrint():
  def __init__(self, file, ExpID):
    self.file = file
    self.ExpID = ExpID
  def __call__(self, some_str):
    print(self.ExpID[-6:] + "-%s-" % os.getpid() + time.strftime("%Y/%m/%d-%H:%M:%S] ") + str(some_str), file=self.file, flush=True)

def check_path(x):
  if x:
    complete_path = glob.glob(x)
    assert(len(complete_path) == 1)
    x = complete_path[0]
  return x

colors = ["gray", "blue", "black", "yellow", "green", "yellowgreen", "gold", "royalblue", "peru", "purple"]
markers = [".", "x"]
def feat_visualize(ax, feat, label, if_right):
  '''
    feat:  N x 2 # 2-d feature, N: number of examples
    label: N x 1
  '''
  index_1 = np.where(if_right == 1)[0]
  index_0 = np.where(if_right == 0)[0]
  for ix in index_1:
    x = feat[ix]; y = label[ix]
    ax.scatter(x[0], x[1], color=colors[y], marker=".")
  for ix in index_0:
    x = feat[ix]; y = label[ix]
    ax.scatter(x[0], x[1], color="red", marker="x")
  return ax

def get_previous_step(e2, resume):
  previous_epoch = previous_step = 0
  if e2 and resume:
    for clip in os.path.basename(e2).split("_"):
      if clip[0] == "E" and "S" in clip:
        num1 = clip.split("E")[1].split("S")[0]
        num2 = clip.split("S")[1]
        if num1.isdigit() and num2.isdigit():
          previous_epoch = int(num1)
          previous_step  = int(num2)
  return previous_epoch, previous_step
  
def set_up_dir(project_name, resume, debug):
  TimeID = time.strftime("%Y%m%d-%H%M%S")
  ExpID = "SERVER" + os.environ["SERVER"] + "-" + TimeID
  if not debug:
    assert(project_name != "") # For a formal exp, name it!
    project_path = pjoin("Experiments", ExpID + "_" + project_name)
    rec_img_path = pjoin(project_path, "reconstructed_images")
    weights_path = pjoin(project_path, "weights")
    os.makedirs(project_path)
    os.makedirs(rec_img_path)
    os.makedirs(weights_path)
    log_path = pjoin(weights_path, "log_" + ExpID + ".txt")
    log = open(log_path, "w+")
  else:
    rec_img_path = pjoin(os.environ["HOME"], "Trash")
    weights_path = pjoin(os.environ["HOME"], "Trash")
    if not os.path.exists(rec_img_path):
     os.makedirs(rec_img_path)
    log = sys.stdout # print to the screen
  print(" ".join(["CUDA_VISIBLE_DEVICES='0' python", *sys.argv]), file=log, flush=True) # save the script
  return TimeID, ExpID, rec_img_path, weights_path, log

# Exponential Moving Average
class EMA():
  def __init__(self, mu):
    self.mu = mu
    self.shadow = {}
  def register(self, name, value):
    self.shadow[name] = value.clone()
  def __call__(self, name, x):
    assert name in self.shadow
    new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
    self.shadow[name] = new_average.clone()
    return new_average

def register_ema(net, k=0.9):
  ema = EMA(k)
  for name, param in net.named_parameters():
    if param.requires_grad:
      ema.register(name, param.data)
  return ema
  
def apply_ema(net, ema):
  for name, param in net.named_parameters():
    if param.requires_grad:
      param.data = ema(name, param.data)
 
def get_CodeID():
  script = "git log --pretty=oneline >> wh_CodeID_file.tmp"
  os.system(script)
  x = open("wh_CodeID_file.tmp").readline()
  os.remove("wh_CodeID_file.tmp")
  return x[:8]

class LogHub(object):
  def __init__(self, momentum=0):
    self.losses = {}
    self.momentum = momentum
  def update(self, name, value):
    if name not in self.losses:
      self.losses[name] = value
    else:
      self.losses[name] = self.losses[name] * self.momentum + value * (1 - self.momentum)
  def format(self, num_digit=3):
    keys = self.losses.keys()
    keys = sorted(keys)
    keys_str = " | ".join(keys)
    values = []
    for k in keys:
      values.append(("%"+".%df" % num_digit) % self.losses[k])
    values_str = " | ".join(values)
    return keys_str, values_str