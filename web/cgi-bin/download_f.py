#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import param_f

print('Content-type: text/csv; charset=UTF-8')
print()
with open(param_f.SUBMIT_PATH,'r') as fin:
    while True:
        chunk = fin.read(100000)
        if not chunk:
            break
        print(chunk)