#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import param

import cgi, cgitb, os, sys, codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
import pandas as pd
import joblib 


PRED_NAME = 'target'


def load_file():
    form = cgi.FieldStorage()

    form_file = form['file']

    if not form_file.file:
        return pd.DataFrame(), False

    if not form_file.filename:
        return pd.DataFrame(), False

    file_name = os.path.basename(form_file.filename)
    uploaded_file_path = os.path.join(param.TESTS_DIR, file_name)
    with open(uploaded_file_path, 'wb') as fout:
        while True:
            chunk = form_file.file.read(100000)
            if not chunk:
                break
            fout.write(chunk)
    
    return pd.read_parquet(uploaded_file_path), True

def analyse(df):
    pca = joblib.load('model/pca_final.joblib')
    rf = joblib.load('model/final_model_final.joblib')

    # в случае Nan будет присвоено предыдущее значение в столбце

    df.fillna(method='ffill', inplace=True)
    X_valid = df.iloc[18::19, 2:]
    
    X_valid = pca.transform(X_valid)

    y_pred = rf.predict(X_valid)
    result = df.iloc[18::19, :2]
    result[PRED_NAME] = y_pred

    return result


if not os.path.exists(param.TESTS_DIR):
   os.makedirs(param.TESTS_DIR)

print('Content-Type: text/html; charset=UTF-8')
print()
print('''<html>
    <head><title>просмотр результата - data squad</title>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"></head>
    
<link rel="stylesheet" type="text/css" href="./style.css">
<style>
  table,
  th,
  td {
    border: 1px solid black;
  }
</style>
    <body>''')
cgitb.enable()
test, loaded = load_file()

if not loaded:   
    print('<div><label>Файл не был загружен</lable></div>')
else:
    if test.empty:
        print('<div><label>Файл пуст</lable></div>')

    else:
        submit = analyse(test)
        submit.to_csv(param.SUBMIT_PATH, index=False)

        headers = submit.columns
        print('''<div style="overflow-y: auto; max-height: 60%;">
                <table><tr>''')
        for header in headers:
            print('<th>',header,'</th>')
        print('</tr>')

        for i, row in submit.iterrows():
            print('<tr>')
            for header in headers:
                value = row[header]
                if header == PRED_NAME:
                    value = "{:.1f}".format(value)
                else:
                    value = str(int(value))
                print('<td>',value,'</td>')
            print('</tr>')
        print('</table></div>')


        print('''<p><a target="_blank" href="download.py"download="submit.csv">
                    <button>Скачать результат</button>
                </a></p>''')

print(        '''<p><a href="''',param.MAIN_PAGE,'''">
                    <button>Назад</button>
                 </a></p>
    </center></body></html>''')
