# File:
# -*- coding: utf-8 -*-
# @Time    : 11/14/2018 12:42 PM
# @Author  : Derek Hu

f1_r = open("train.en", encoding="utf-8")
f1_w = open("train_2.en", 'w', encoding="utf-8")
f2_r = open("train.vi", encoding="utf-8")
f2_w = open("train_2.vi", 'w', encoding="utf-8")

en = f1_r.readlines()
vi = f2_r.readlines()
assert len(en) == len(vi)

for i in range(len(en)):
    en_line = en[i]
    vi_line = vi[i]
    if en_line.strip() == "" or vi_line.strip() == "":
        continue
    else:
        f1_w.write(en_line)
        f2_w.write(vi_line)
