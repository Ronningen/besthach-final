#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

import param_f
import cgi, cgitb, os, sys, codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
import pandas as pd
import joblib 
from sklearn.preprocessing import StandardScaler, LabelEncoder


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
    df = df.fillna(method='ffill')
    clear_test = df.drop(columns=['prev_date_depart','prev_snd_org_id','prev_rsv_org_id',
                                    'snd_org_id','rsv_org_id','freight','prev_freight','rod',
                                    'prev_distance','wagnum','prev_date_arrival'])

    label_encoder = LabelEncoder()
    cat_columns = clear_test.select_dtypes('object').columns
    for col in cat_columns:
        clear_test[col] = label_encoder.fit_transform(clear_test[col])

    rf = joblib.load('models/final_model_41.joblib')
    pca = joblib.load('models/pca_41.joblib')
    return pd.Series(rf.predict(pca.transform(clear_test)), name=PRED_NAME)

def label(str):
    return '<div><label>'+ str +'</lable></div>'

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
<link rel="stylesheet" type="text/css" href="./style.css">
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
        try:
            submit = analyse(test)
            submit.to_csv(param_f.SUBMIT_PATH, index=False)

            headers = submit.columns
            print('''
        <center>
            <div style="overflow-y: auto; max-height: 60%;">
                <table  BGCOLOR="#358f81"><tr>''')
            for header in headers:
                print('<th>',header,'</th>')
            print('</tr>')

            for i, row in submit.iterrows():
                print('<tr>')

                for header in headers:
                    value = row[header]
                    if header == PRED_NAME:
                        value = "{:.1f}".format(value)
                    elif submit.dtypes[header] != 'object':
                        value = str(int(value))
                    print('<td>',value,'</td>')
                
                print('</tr>')

            print('''</table>
            </div>
        </center>''')
            has_data = True
        except:
            print(label('Данные не корректны'))

print('''
    <center>
        <p>''')
if has_data:
        print('''<a target="_blank" href="download.py" download="submit.csv" style="text-decoration: none">
                    <div class="file-upload-container" id="fours">
                        <div class="file-upload-text">Скачать результат</div>
                    </div>
                </a>''')

print('''       <div class="file-upload-container" id="five">
                    <div class="file-upload-text">Назад</div>
                    <a class="file-upload" href=" ''',param_f.MAIN_PAGE,''' "></a>
                </div>
        </p>
      </center>
</body></html>''')
