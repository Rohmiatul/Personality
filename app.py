import streamlit as st
import pandas as pd
import joblib # Untuk memuat model yang disimpan

--- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Prediksi Kepribadian",
    page_icon="👤",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Memuat Model (Menggunakan st.cache_resource agar model hanya dimuat sekali) ---
@st.cache_resource
def load_personality_model():
    """
    Memuat model K-Nearest Neighbors Classifier yang telah dilatih.
    Pastikan file 'personality_model.pkl' berada di direktori yang sama.
    """
    try:
        model = joblib.load('personality_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Error: File model 'personality_model.pkl' tidak ditemukan. Pastikan Anda telah mengunggahnya.")
        return None

# Muat model saat aplikasi dimulai
knn_model = load_personality_model()

# --- Judul dan Deskripsi Aplikasi ---
st.title("Aplikasi Prediksi Kepribadian")
st.write("Aplikasi ini memprediksi apakah seseorang cenderung Introvert atau Ekstrovert berdasarkan beberapa aktivitas sosial.")
st.markdown("---")

# --- Input Pengguna ---
st.header("Masukkan Data Aktivitas:")

# Menggunakan st.number_input untuk input numerik
social_event_attendance = st.number_input(
    "Jumlah Kehadiran Acara Sosial per bulan:",
    min_value=0, max_value=30, value=5, step=1,
    help="Contoh: Pesta, pertemuan komunitas, dll."
)
going_outside = st.number_input(
    "Jumlah Kegiatan di Luar Rumah per minggu:",
    min_value=0, max_value=7, value=3, step=1,
    help="Contoh: Jalan-jalan, olahraga di luar, nongkrong di kafe."
)
friends_circle_size = st.number_input(
    "Ukuran Lingkaran Pertemanan (jumlah teman dekat):",
    min_value=0, max_value=100, value=10, step=1,
    help="Jumlah orang yang Anda anggap teman dekat."
)

# --- Tombol Prediksi ---
st.markdown("---")
if st.button("Prediksi Kepribadian"):
    if knn_model is not None:
        # Buat DataFrame dari input pengguna
        new_data = pd.DataFrame(
            [[social_event_attendance, going_outside, friends_circle_size]],
            columns=['Social_event_attendance', 'Going_outside', 'Friends_circle_size']
        )

        try:
            # Lakukan prediksi
            predicted_code = knn_model.predict(new_data)[0] # Hasilnya 0 atau 1

            # Konversi hasil prediksi ke label yang mudah dibaca
            label_mapping = {1: 'Ekstrovert', 0: 'Introvert'}
            predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')

            st.subheader("Hasil Prediksi:")
            if predicted_code == 1:
                st.success(f"Prediksi Kepribadian: **{predicted_label}** 🎉")
                st.write("Anda cenderung memiliki sifat yang lebih terbuka dan suka berinteraksi dengan orang lain.")
            else:
                st.info(f"Prediksi Kepribadian: **{predicted_label}** 🧘")
                st.write("Anda cenderung memiliki sifat yang lebih tertutup dan nyaman dengan diri sendiri atau lingkungan kecil.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
    else:
        st.warning("Model belum dimuat. Harap periksa file model Anda.")

st.markdown("---")
st.caption("Aplikasi ini hanya untuk tujuan demonstrasi dan tidak menggantikan analisis psikologis profesional.")
