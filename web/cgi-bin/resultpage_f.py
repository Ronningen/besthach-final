#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import param_f
import cgi
import cgitb
import os
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


PRED_NAME = 'target'


def load_file():
    form = cgi.FieldStorage()

    form_file = form['file']

    if not form_file.file:
        return pd.DataFrame(), False

    if not form_file.filename:
        return pd.DataFrame(), False

    file_name = os.path.basename(form_file.filename)
    uploaded_file_path = os.path.join(param_f.TESTS_DIR, file_name)
    with open(uploaded_file_path, 'wb') as fout:
        while True:
            chunk = form_file.file.read(100000)
            if not chunk:
                break
            fout.write(chunk)

    return pd.read_parquet(uploaded_file_path), True


def analyse(df):
    clear_test = df.fillna(method='ffill')
    cols = set(['prev_fr_id',
                     'prev_is_load',
                     'snd_st_id',
                     'rsv_st_id',
                     'fr_id',
                     'is_load',
                     'common_ch',
                     'vidsobst',
                     'distance',
                     'prev_fr_group',
                     'fr_group'])
    df_cols = clear_test.columns
    for col in df_cols:
        if col not in cols:
            clear_test.drop(columns=[col], inplace=True)

    label_encoder = LabelEncoder()
    cat_columns = clear_test.select_dtypes('object').columns
    for col in cat_columns:
        clear_test[col] = label_encoder.fit_transform(clear_test[col])

    rf = joblib.load(param_f.MODEL_PATH)
    pca = joblib.load(param_f.PCA_PATH)
    return pd.DataFrame(rf.predict(pca.transform(clear_test)), columns=[PRED_NAME])


def label(str):
    return '<div><label><h3>' + str + '</h3></lable></div>'


if not os.path.exists(param_f.TESTS_DIR):
    os.makedirs(param_f.TESTS_DIR)
cgitb.enable()
test, loaded = load_file()

print('Content-Type: text/html; charset=UTF-8')
print()
print('''
<html>

<head>
  <title>просмотр результата - data squad</title>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
</head>
<link rel="stylesheet" type="text/css" href="/style.css">
<style>
  table,
  th,
  td {
    border: 1px solid black;
  }
</style>

<body>
  <center>
    <div style="font-size: 200%;">
      <h2 class="teamtext">ПРЕДСКАЗАНИЕ</h2>
    </div>
    <div class="text">''')

has_data = False
if not loaded:
    print(label('Файл не был загружен'))
else:
    if test.empty:
        print(label('Файл пуст'))

    else:
        # try:
        submit = analyse(test)
        submit.to_csv(param_f.SUBMIT_PATH, index=False)
        submit = pd.DataFrame({'wagnum':test['wagnum'], PRED_NAME:submit[PRED_NAME]})

        headers = submit.columns
        print('''
            <div style="overflow-y: auto; height:30vh;">
        <center>
                <table  BGCOLOR="#FEF9E9"><tr>''')
        for header in headers:
            print('<th >', header, '</th>')
        print('</tr>')

        for i, row in submit.iterrows():
            print('<tr>')

            for header in headers:
                value = row[header]
                if header == PRED_NAME:
                    value = "{:.1f}".format(value)
                elif submit.dtypes[header] != 'object':
                    value = str(int(value))
                print('<td>', value, '</td>')

            print('</tr>')

        print('''</table>
        </center>
            </div>''')
        has_data = True
        # except:
        #     print(label('Данные не корректны'))

print('''
    <center>
        <p>''')
if has_data:
    print('''<a target="_blank" href="download_f.py" download="submit.csv" style="text-decoration: none">
                    <div class="file-upload-container" id="fours">
                        <div class="file-upload-text">Скачать результат</div>
                    </div>
                </a>''')

print('''       <div class="file-upload-container" id="five">
                    <div class="file-upload-text">Назад</div>
                    <a class="file-upload" href=" ''', param_f.MAIN_PAGE, ''' "></a>
                </div>
        </p>
      </center>
</body></html>''')
