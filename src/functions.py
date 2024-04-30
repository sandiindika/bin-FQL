# LIBRARY / MODULE / PUSTAKA

import streamlit as st
import librosa
import itertools, os, pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score

from warnings import simplefilter

simplefilter(action= "ignore", category= FutureWarning)

# DEFAULT FUNCTIONS

"""Make Space

Fungsi-fungsi untuk membuat jarak pada webpage menggunakan margin space dengan
ukuran yang bervariatif.
"""

def ms_20():
    st.markdown("<div class= \"ms-20\"></div>", unsafe_allow_html= True)

def ms_40():
    st.markdown("<div class= \"ms-40\"></div>", unsafe_allow_html= True)

def ms_60():
    st.markdown("<div class= \"ms-60\"></div>", unsafe_allow_html= True)

def ms_80():
    st.markdown("<div class= \"ms-80\"></div>", unsafe_allow_html= True)

"""Make Layout

Fungsi-fungsi untuk layouting webpage menggunakan fungsi columns() dari
Streamlit.

Returns
-------
self : object containers
    Mengembalikan layout container.
"""

def ml_center():
    left, center, right = st.columns([.3, 2.5, .3])
    return center

def ml_split():
    left, center, right = st.columns([1, .1, 1])
    return left, right

def ml_left():
    left, center, right = st.columns([2, .1, 1])
    return left, right

def ml_right():
    left, center, right = st.columns([1, .1, 2])
    return left, right

"""Cetak text

Fungsi-fungsi untuk menampilkan teks dengan berbagai gaya menggunakan method
dari Streamlit seperti title(), write(), dan caption().

Parameters
----------
text : str
    Teks yang ingin ditampilkan dalam halaman.

size : int
    Ukuran Heading untuk teks yang akan ditampilkan.

division : bool
    Kondisi yang menyatakan penambahan garis divisi teks ditampilkan.
"""

def show_title(text, division= False):
    st.title(text)
    if division:
        st.markdown("---")

def show_text(text, size= 3, division= False):
    heading = "#" if size == 1 else (
        "##" if size == 2 else (
            "###" if size == 3 else (
                "####" if size == 4 else "#####"
            )
        )
    )

    st.write(f"{heading} {text}")
    if division:
        st.markdown("---")

def show_caption(text, size= 3, division= False):
    heading = "#" if size == 1 else (
        "##" if size == 2 else (
            "###" if size == 3 else (
                "####" if size == 4 else "#####"
            )
        )
    )

    st.caption(f"{heading} {text}")
    if division:
        st.markdown("---")

def show_paragraf(text):
    st.markdown(f"<div class= \"paragraph\">{text}</div>", unsafe_allow_html= True)

"""Load file

Fungsi-fungsi untuk membaca file dalam lokal direktori.

Parameters
----------
filepath : str
    Jalur tempat data tersedia di lokal direktori.

Returns
-------
self : object DataFrame or str
    Obyek dengan informasi yang berhasil diekstrak.
"""

def get_csv(filepath):
    return pd.read_csv(filepath)

def get_excel(filepath):
    return pd.read_excel(filepath)

# ----------

def mk_dir(dirpath):
    """Buat folder
    
    Fungsi ini akan memeriksa path folder yang diberikan. Jika tidak ada
    folder sesuai path yang dimaksud, maka folder akan dibuat.

    Parameters
    ----------
    dirpath : str
        Jalur tempat folder akan dibuat.
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

# CUSTOM FUNCTIONS

def get_list_musik(directory):
    """Baca file musik
    
    File musik yang telah dibagi per folder (genre) akan dibaca disini.
    Penjelajahan file dilakukan dengan module os yang nantinya akan didapatkan
    3 elemen dari data musik: filepath, filename, dan label genre musik.
    
    Parameters
    ----------
    directory : str
        Jalur utama tempat file musik akan diakses.

    Returns
    -------
    df : object DataFrame or TextFileReader
        File csv (comma-separated values) dikembalikan sebagai struktur data
        dua dimensi dengan sumbu yang diberi label.
    """
    temp_filepath, temp_genre, temp_filename = [], [], []
    for _dir in os.listdir(directory): # main directory
        folderpath = os.path.join(directory, _dir)
        if os.path.isdir(folderpath):
            for filename in os.listdir(folderpath): # genre directory
                filepath = os.path.join(folderpath, filename)

                temp_filepath.append(filepath)
                temp_filename.append(filename)
                temp_genre.append(_dir)
        else:
            temp_filepath.append(folderpath)
            temp_filename.append(_dir)
            temp_genre.append(directory)

    df = pd.DataFrame({
        "filepath": temp_filepath,
        "filename": temp_filename,
        "genre": temp_genre
    })
    return df

@st.cache_data(ttl= 3600, show_spinner= "MFCC Features Extraction...")
def feature_extraction_mfcc(df, duration= 30, coef= 13):
    """Ekstraksi Fitur MFCC

    Fitur audio MFCC didasarkan pada persepsi pendengaran manusia. Ekstraksi
    MFCC dilakukan dengan menggunakan module Librosa yang menyediakan
    pemrosesan audio.

    Parameters
    ----------
    df : object DataFrame
        Object DataFrame tempat semua file musik (path file) tersimpan.

    duration : int or float
        Durasi musik yang ingin di ekstraksi fiturnya.

    coef : int
        Jumlah koefisien dari MFCC yang akan dihitung.

    Returns
    -------
    res : object DataFrame
        DataFrame dari data musik dengan fitur yang telah di ekstraksi dan
        label genre musik. 
    """
    mfcc_features = []
    for _dir in df.iloc[:, 0]:
        y, sr = librosa.load(_dir, duration= duration)
        mfcc = librosa.feature.mfcc(y= y, sr= sr, n_mfcc= coef)

        feature = np.mean(mfcc, axis= 1)
        mfcc_features.append(feature)

    res = pd.DataFrame({
        "filename": df.iloc[:, 1],
        **{f"mfcc_{i + 1}": [x[i] for x in mfcc_features] for i in range(coef)},
        "genre": df.iloc[:, -1]
    })
    return res

def min_max_scaler(feature_names, df):
    """Transformasikan fitur dengan menskalakan setiap fitur ke rentang
    tertentu

    Estimator ini menskalakan dan menerjemahkan setiap fitur satu per satu
    sehingga berada dalam rentang tertentu pada set pelatihan, misalnya
    antara nol dan satu.
    
    Transformasi dilakukan dengan::
    
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    dimana min, max = feature_range.

    Transformasi ini sering digunakan sebagai alternatif terhadap mean nol,
    penskalaan unit varians.

    Parameters
    ----------
    feature_names : ndarray of shape
        Nama fitur dari kumpulan data (DataFrame). Didefinisikan hanya ketika
        `X` memiliki nama fitur yang semuanya berupa string.

    df : object DataFrame
        Object DataFrame yang menyimpan fitur musik beserta label genrenya.
    
    Returns
    -------
    self : object DataFrame
        DataFrame dari data musik dengan fitur yang telah di skalakan. 
    """
    for col in feature_names: # loop untuk setiap fitur dalam `X`
        min_ = df[col].min()
        max_ = df[col].max()
        df[col] = (df[col] - min_) / (max_ - min_)
    return df

def permutation_importance(model, X_test, y_test, metric= accuracy_score):
    """Calculate Permutation Importance

    Teknik yang digunakan untuk mengukur pentingnya fitur dalam model machine
    learning. Teknik ini bekerja dengan mengacak urutan nilai fitur untuk satu
    contoh pada satu waktu, dan kemudian mengamati perubahan performa model.

    Parameters
    ----------
    model : trained Naive Bayes model
        Trained Naive Bayes model ready for inference.

    X_test : ndarray or shape (n_samples, n_features)
        Sampel OOB (Out-of-Bag) dari data test yang dihasilkan KFold.

    y_test : ndarray or shape (n_samples, 1, n_outputs)
        Label sampel OOB dari data test yang dihasilkan KFold.

    metric : metric function, default= accuracy_score
        Perhitungan metric yang digunakan untuk mengevaluasi hasil permutation
        importance.

    Returns
    -------
    self : ndarray
        Nilai score dari setiap fitur yang dihitung menggunakan permutation
        importance.
    """
    baseline = metric(y_test, model.predict(X_test))
    scores = []

    for feature in range(X_test.shape[1]):
        permuted_X = X_test.copy()
        permuted_X[:, feature] = np.random.permutation(permuted_X[:, feature])
        permuted_score = metric(y_test, model.predict(permuted_X))
        scores.append(baseline - permuted_score)
    scores = np.array(scores)
    return scores, np.argsort(scores[::-1])

@st.cache_data(ttl= 3600, show_spinner= "Naive Bayes Classification...")
def nbc_model(features, labels, K= 5):
    """Naive Bayes Clasifier Model

    Pelatihan model menggunakan Naive Bayes dengan beberapa persiapan seperti
    setting parameter dan validasi KFold.

    Parameters
    ----------
    features : ndarray or shape (n_samples, n_features)
        Sampel OOB (Out-of-Bag).

    labels : ndarray or shape (n_samples, 1, n_outputs)
        Label sampel OOB.

    K : int
        Jumlah subset Fold.

    Returns
    -------
    dump_model : trained Naive Bayes model
        Trained Naive Bayes model ready for inference.

    avg_score : float
        Rata-rata dari score model selama pelatihan dengan KFold.

    dump_features_test : ndarray or shape (n_samples, n_features)
        Sampel OOB (Out-of-Bag) dari data test yang dihasilkan KFold.

    dump_labels_test : ndarray or shape (n_samples, 1, n_outputs)
        Label sampel OOB dari data test yang dihasilkan KFold.
    """
    kfold = KFold(n_splits= K, shuffle= True, random_state= 42)

    temp_score, avg_score = 0, 0
    list_labels_test, list_labels_predict = [], []

    for tr_index, ts_index in kfold.split(features):
        X_train, X_test = features[tr_index], features[ts_index]
        y_train, y_test = labels[tr_index], labels[ts_index]

        model = GaussianNB()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        avg_score += score

        if temp_score < score:
            temp_score = score

            dump_model = model
            dump_features_test = X_test
            dump_labels_test = y_test

        list_labels_test.append(y_test)
        list_labels_predict.append(y_pred)

    mk_dir("./data/pickle")
    with open("./data/pickle/labels_test.pickle", "wb") as file:
        pickle.dump(list_labels_test, file)
    with open("./data/pickle/labels_predict.pickle", "wb") as file:
        pickle.dump(list_labels_predict, file)
    return dump_model, avg_score / K, dump_features_test, dump_labels_test

def evaluation_metrics():
    """Perhitungan evaluasi

    Returns
    -------
    score_list : list
        Daftar akurasi dari keseluruhan model dalam fold.

    precision_list : list
        Daftar precision dari keseluruhan model dalam fold.

    recall_list : list
        Daftar recall dari keseluruhan model dalam fold.
    """
    with open("./data/pickle/labels_predict.pickle", "rb") as file:
        list_labels_predict = pickle.load(file)
    with open("./data/pickle/labels_test.pickle", "rb") as file:
        list_labels_test = pickle.load(file)

    score_list, precision_list, recall_list = [], [], []
    for y_test, y_pred in zip(list_labels_test, list_labels_predict):
        score = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average= "macro")
        recall = recall_score(y_test, y_pred, average= "macro")

        score_list.append(score)
        precision_list.append(precision)
        recall_list.append(recall)

    return score_list, precision_list, recall_list

def plot_confusion_matrix(normalize= False, title= "Confusion Matrix"
    ):
    """Plot Confusion Matrix
    
    Fungsi untuk menggambar confusion matrix

    Parameters
    ----------
    normalize : bool
        Jika True, maka confusion matrix akan dinormalisasi, dan False
        sebaliknya.

    title : str
        Menampilkan judul dari Plot.
    """

    with open("./data/pickle/labels_predict.pickle", "rb") as file:
        list_labels_predict = pickle.load(file)
    with open("./data/pickle/labels_test.pickle", "rb") as file:
        list_labels_test = pickle.load(file)

    for key, (y_test, y_pred) in enumerate(zip(list_labels_test, list_labels_predict)):
        show_caption(f"Fold-{key + 1}")
        cm = confusion_matrix(y_test, y_pred)

        classes = np.unique(y_test)

        fig = plt.figure(figsize= (10, 10))
        plt.imshow(cm, interpolation= "nearest", cmap= plt.cm.Blues)
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation= 45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis= 1)[:, np.newaxis]
            st.write("Normalized Confusion Matrix")
        else:
            st.write("Confusion Matrix, without Normalization")

        thresh = cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j, i, cm[i, j], horizontalalignment= "center", fontsize= 12,
                color= 'white' if cm[i, j] > thresh else 'black'
            )

        plt.tight_layout()
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        st.pyplot(fig)