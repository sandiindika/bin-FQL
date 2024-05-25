# LIBRARY / MODULE / PUSTAKA

import streamlit as st
from streamlit import session_state as ss

from functions import *
from warnings import simplefilter

simplefilter(action= "ignore", category= FutureWarning)

# PAGE CONFIG

st.set_page_config(
    page_title= "App", layout= "wide",
    page_icon= "globe", initial_sidebar_state= "expanded"
)

## hide menu, header, and footer
st.markdown(
    """<style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .st-emotion-cache-z5fcl4 {padding-top: 1rem;}
    </style>""",
    unsafe_allow_html= True
)

## CSS on style.css
with open("./css/style.css") as file:
    st.markdown(
        "<style>{}</style>".format(file.read()), unsafe_allow_html= True
    )

class MyApp():
    """Class dari MyApp

    Parameters
    ----------
    message : bool, default= False
        Jika False, maka pesan error tidak akan ditampilkan dalam Webpage
        Sistem. Jika True, maka akan menampilkan pesan dalam Webpage Sistem
        yang dapat dianalisis.
    
    Attributes
    ----------
    message : bool
        Tampilkan pesan error pada Webpage Sistem atau tidak.

    pathdata : str
        Path (jalur) data disimpan dalam lokal direktori.

    menu_ : list
        Daftar menu yang akan ditampilkan dalam Webpage Sistem.
    """

    def __init__(self, message= False):
        self.message = message
        self.pathdata = "./data/music"
        self.menu_ = [
            "Beranda", "Dataset", "Ekstraksi Fitur", "Permutation Importance",
            "Klasifikasi", "Evaluasi"
        ]

    def _exceptionMessage(self, e):
        """Tampilkan pesan galat
        
        Parameters
        ----------
        e : exception object
            Obyek exception yang tersimpan.
        """
        ms_20()
        with ml_center():
            st.error("Terjadi masalah...")
            if self.message:
                st.exception(e) # tampilkan keterangan galat

    def _pageBeranda(self):
        """Tab beranda
        
        Halaman ini akan menampilkan judul penelitian dan abstrak dari proyek.
        """
        try:
            ms_20()
            show_text(
                "Klasifikasi Musik Berdasarkan Genre Menggunakan Metode Naive\
                    Bayes",
                size= 2, division= True
            )

            ms_40()
            with ml_center():
                show_paragraf(
                    "Musik telah menjadi satu kesatuan dalam kehidupan\
                    mayoritas orang. Pembawaan musik pada era modern telah\
                    tergantikan dari yang semula album fisik kini menjadi\
                    musik digital. Label kategorikal (genre) diberikan pada\
                    setiap karya musik untuk mengidentifikasi jenis genre\
                    musiknya. Berdasarkan fitur yang di ekstrak menggunakan\
                    teknik machine learning, musik dapat diklasifikasikan\
                    secara otomatis sesuai genrenya sehingga dinilai lebih\
                    efisien jika dibandingkan dengan melakukan klasifikasi\
                    secara manual setiap trek dari database musik besar.\
                    Penelitian ini dilakukan untuk mengklasifikasikan genre\
                    musik menggunakan dataset GTZAN dengan fitur ekstraksi\
                    yang digunakan adalah Mel Frequency Cepstral Coefficients\
                    (MFCC) dan Naïve Bayes untuk klasifikasi. Keunggulan\
                    penggunaan MFCC sebagai fitur ekstraksi adalah kemampuan\
                    dalam mengenali karakteristik suara yang sangat penting\
                    bagi pengenalan suara dan menghasilkan data yang sangat\
                    penting bagi pengenalan musik dan menghasilkan data\
                    seminimal mungkin tanpa menghilangkan informasi penting.\
                    Berdasarkan hasil penelitian yang telah dilakukan,\
                    rata-rata akurasi dari model Naive Bayes menunjukkan\
                    performa yang baik dalam klasifikasi pada dataset dengan\
                    rata-rata akurasi sekitar 92,80% dengan menggunakan 5\
                    genre optimal yaitu: classical, disco, hiphop, pop, dan\
                    reggae. Hal ini berbanding terbalik jika menggunakan\
                    model yang dilatih dengan 10 genre, didapatkan akurasi\
                    rata-rata sebesar 68,90%. Dengan meningkatnya jumlah\
                    genre, kompleksitas data juga meningkat. Ini berarti\
                    terdapat lebih banyak variasi dalam fitur-fitur yang\
                    digunakan untuk membedakan antara genre-genre tersebut.\
                    Dengan menggunakan permutation importance, didapatkan\
                    model optimal dengan menggunakan 12 fitur dalam MFCC\
                    dengan rata-rata akurasi sebesar 95%."
                )
        except Exception as e:
            self._exceptionMessage(e)

    def _pageDataset(self):
        """Halaman Dataset
        
        Bagian ini akan menampilkan obyek DataFrame dengan detail data
        penelitian.
        """
        try:
            ms_20()
            show_text("Data Musik", division= True)

            ms_40()
            with ml_center():
                df = get_list_musik(self.pathdata)
                st.dataframe(df, use_container_width= True, hide_index= True)

                mk_dir("./data/dataframe")
                df.to_csv("./data/dataframe/list-music.csv", index= False)
        except Exception as e:
            self._exceptionMessage(e)
    
    def _pageFeatureExtraction(self):
        """Ekstraksi Fitur MFCC

        Halaman ini akan mengekstraksi fitur MFCC dari data dengan membaca
        filepath setiap file musik. Number input disediakan untuk optimasi
        pada durasi musik dan koefisien hitung MFCC.
        """
        try:
            ms_20()
            show_text("Ekstraksi Fitur")
            show_caption("Mel Frequency Cepstral Coefficients", division= True)

            left, right = ml_right()
            with left:
                ms_20()
                duration = st.number_input(
                    "Durasi Musik (detik)", min_value= 1, value= 30, step= 1,
                    key= "Number input untuk nilai durasi musik"
                )
                coef = st.number_input(
                    "Koefisien MFCC", min_value= 1, value= 13, step= 1, max_value= 20,
                    key= "Number input untuk nilai koefisien hitung MFCC"
                )

                ms_40()
                with ml_center():
                    btn_extract = st.button(
                        "Submit", key= "Button fit feature extraction", use_container_width= True
                    )
            with right:
                if btn_extract:
                    ss.fit_extract = True

                    with st.spinner("Feature extraction is running..."):
                        df = feature_extraction_mfcc(
                            get_csv("./data/dataframe/list-music.csv"),
                            duration= duration, coef= coef
                        )
                    
                    df.to_csv("./data/dataframe/mfcc_features.csv", index= False)
                if ss.fit_extract:
                    show_caption("Fitur MFCC", size= 2)
                    st.dataframe(
                        get_csv("./data/dataframe/mfcc_features.csv"),
                        use_container_width= True, hide_index= True
                    )
        except Exception as e:
            self._exceptionMessage(e)

    def _pagePermutationImportance(self):
        """Permutation Importance
        
        Teknik yang digunakan untuk mengukur pentingnya fitur dalam model
        machine learning. Teknik ini bekerja dengan mengacak urutan nilai
        fitur untuk satu contoh pada satu waktu, dan kemudian mengamati
        perubahan performa model.
        """
        try:
            ms_20()
            show_text("Permutation Importance", division= True)

            df = get_csv("./data/dataframe/mfcc_features.csv")
            features = df.iloc[:, 1:-1]
            feature_names = features.columns
            
            left, right = ml_right()
            with left:
                choice_genre = st.radio(
                    "Gunakan semua genre?", ["Yes", "No"], horizontal= True,
                    key= "Radio button pilihan jenis genre"
                )

                choice_genre = True if choice_genre == "Yes" else False
                if choice_genre:
                    labels_selected = st.multiselect(
                        "Pilih genre musik (multi)", df.iloc[:, -1].unique(),
                        placeholder= "Pilih opsi",
                        key= "Multiselect untuk pilihan genre"
                    )

                ms_40()
                with ml_center():
                    btn_perm = st.button(
                        "Submit", key= "Button fit permutation importance", use_container_width= True
                    )

            with right:
                features["genre"] = df.iloc[:, -1]
                if choice_genre:
                    unwanted_genres = [genre for genre in features["genre"].unique() if genre not in labels_selected]
                    features = features[~features["genre"].isin(unwanted_genres)]
                
                temp_title = st.empty()
                temp_data = st.empty()

                with temp_title:
                    show_caption("Final Data", size= 2)
                temp_data.dataframe(features, use_container_width= True, hide_index= True)

                labels = features.iloc[:, -1].values
                features = features.iloc[:, :-1].values

                if btn_perm:
                    temp_title.empty()
                    temp_data.empty()

                    with st.spinner("Permutation Importance is running..."):
                        model, score, X_test, y_test = nbc_model(
                            features, labels
                        )

                        permutation_scores, sorted_idx = permutation_importance(model, X_test, y_test)
                    show_caption("Permutation Importance Scores", size= 2)
                    st.dataframe(
                        pd.DataFrame({
                            "Feature": [f"mfcc_{i + 1}" for i in range(len(permutation_scores))],
                            "Score": permutation_scores
                        }),
                        hide_index= True,
                        use_container_width= True
                    )
        except Exception as e:
            self._exceptionMessage(e)

    def _pageKlasifikasi(self):
        """Klasifikasi Naive Bayes

        Halaman ini untuk setting dan training model klasifikasi.
        """
        try:
            ms_20()
            show_text("Klasifikasi", division= True)

            df = get_csv("./data/dataframe/mfcc_features.csv")
            features = df.iloc[:, 1:-1]
            feature_names = features.columns
            
            left, right = ml_right()
            with left:
                kfold = st.selectbox(
                    "Pilih jumlah subset Fold",
                    [4, 5, 10], index= 1,
                    key= "Selectbox untuk memilih jumlah subset Fold"
                )

                choice_feature = st.radio(
                    "Gunakan semua fitur?", ["Yes", "No"], horizontal= True,
                    key= "Radio button untuk pilihan fitur"
                )
                choice_feature = True if choice_feature == "No" else False
                if choice_feature:
                    features_selected = st.multiselect(
                        "Pilih fitur musik (multi)", feature_names,
                        placeholder= "Pilih opsi",
                        key= "Multiselect untuk memilih fitur koefisien MFCC"
                    )

                choice_genre = st.radio(
                    "Gunakan semua genre?", ["Yes", "No"], horizontal= True,
                    key= "Radio button untuk pilihan jenis genre"
                )

                choice_genre = True if choice_genre == "No" else False
                if choice_genre:
                    labels_selected = st.multiselect(
                        "Pilih genre musik (multi)", df.iloc[:, -1].unique(),
                        placeholder= "Pilih opsi",
                        key= "Multiselect untuk memilih genre"
                    )

                ms_40()
                with ml_center():
                    btn_classify = st.button(
                        "Submit", key= "Button fit classification", use_container_width= True
                    )

            with right:
                if choice_feature:
                    unwanted_columns = [col for col in features.columns if col not in features_selected]
                    features.drop(unwanted_columns, axis= 1, inplace= True)
                
                features["genre"] = df.iloc[:, -1]
                if choice_genre:
                    unwanted_genres = [genre for genre in features["genre"].unique() if genre not in labels_selected]
                    features = features[~features["genre"].isin(unwanted_genres)]
                
                temp_title = st.empty()
                temp_data = st.empty()

                with temp_title:
                    show_caption("Final Data", size= 2)
                temp_data.dataframe(features, use_container_width= True, hide_index= True)

                labels = features.iloc[:, -1].values
                features = features.iloc[:, :-1].values

                if btn_classify:
                    temp_title.empty()
                    temp_data.empty()

                    with st.spinner("Classification is running..."):
                        model, score, X_test, y_test = nbc_model(
                            features, labels, K= kfold
                        )

                    ms_20()
                    st.success("Pelatihan model berhasil!")
                    st.info(f"Rata-rata score model: {score:.4f}")
        except Exception as e:
            self._exceptionMessage(e)

    def _pageEvaluasi(self):
        """Evaluasi Model Klasifikasi

        Halaman ini untuk evaluasi model hasil klasifikasi.
        """
        try:
            ms_20()
            show_text("Evaluasi", division= True)

            scores, precision, recall = evaluation_metrics()

            left, right = ml_split()
            with left:
                ms_20()
                for key, val in enumerate(scores):
                    st.success(f"Akurasi Fold {key + 1}: {val * 100:.2f}%")

                ms_20()
                for key, val in enumerate(precision):
                    st.info(f"Precision Fold {key + 1}: {val * 100:.2f}%")

                ms_20()
                for key, val in enumerate(recall):
                    st.success(f"Recall Fold {key + 1}: {val * 100:.2f}%")
            with right:
                plot_confusion_matrix()
        except Exception as e:
            self._exceptionMessage(e)

    def main(self):
        """Main program
        
        Setting session page diatur disini dan konfigurasi setiap halaman
        dipanggil disini.
        """
        with st.container():
            tabs = st.tabs(self.menu_)

            if "fit_extract" not in ss:
                ss.fit_extract = False

            with tabs[0]:
                self._pageBeranda()
            with tabs[1]:
                self._pageDataset()
            with tabs[2]:
                self._pageFeatureExtraction()
            with tabs[3]:
                self._pagePermutationImportance()
            with tabs[4]:
                self._pageKlasifikasi()
            with tabs[5]:
                self._pageEvaluasi()

if __name__ == "__main__":
    app = MyApp(message= True)
    app.main()