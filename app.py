import streamlit as st
import pandas as pd
import joblib # Import joblib untuk memuat model

# --- Memuat Model KNN ---
try:
    # Pastikan nama file model kamu adalah 'knn_model.joblib' di direktori yang sama
    with open('knn_model.joblib', 'rb') as file:
        knn_model = joblib.load(file) # Ganti 'knn' dengan 'knn_model' untuk kejelasan
except FileNotFoundError:
    st.error("Oops! File 'knn_model.joblib' tidak ditemukan. Pastikan model kamu ada di direktori yang sama dengan aplikasi ini.")
    st.stop() # Hentikan aplikasi jika model tidak ditemukan
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}. Pastikan file model kamu tidak korup.")
    st.stop() # Hentikan aplikasi jika ada error lain saat memuat model

# --- Judul Aplikasi Streamlit ---
st.title('Aplikasi Prediksi Personality')
st.write('Aplikasi ini memprediksi apakah seseorang cenderung **Extrovert** atau **Introvert** berdasarkan aktivitas sosialnya.')

# --- Input dari Pengguna ---
st.sidebar.header('Masukkan Data Anda')
st.sidebar.markdown('Geser slider di bawah ini untuk memasukkan data yang diperlukan:')

new_Social_event_attendance = st.sidebar.slider(
    'Jumlah Kegiatan Sosial yang Dihadiri:',
    min_value=0, max_value=20, value=5, step=1,
    help='Berapa kali Anda menghadiri acara sosial (misal: pesta, komunitas, dll.)?'
)
new_Going_outside = st.sidebar.slider(
    'Frekuensi Kegiatan di Luar Ruangan:',
    min_value=0, max_value=20, value=5, step=1,
    help='Seberapa sering Anda melakukan kegiatan di luar rumah (misal: jalan-jalan, olahraga di luar, dll.)?'
)
new_Friends_circle_size = st.sidebar.slider(
    'Jumlah Lingkaran Pertemanan:',
    min_value=0, max_value=50, value=10, step=1,
    help='Berapa banyak teman dekat atau orang yang Anda ajak berinteraksi secara rutin?'
)

# --- Menampilkan Input yang Diterima ---
st.subheader('Data yang Anda Masukkan:')
st.markdown(f"- **Kegiatan Sosial**: **{new_Social_event_attendance}** kali")
st.markdown(f"- **Kegiatan di Luar**: **{new_Going_outside}** kali")
st.markdown(f"- **Jumlah Pertemanan**: **{new_Friends_circle_size}** orang")

# --- Tombol untuk Melakukan Prediksi ---
st.write("---") # Garis pemisah
if st.button('Prediksi Personality Saya!'):
    try:
        # Buat DataFrame dari input baru dengan nama kolom yang sesuai dengan saat training model
        new_data_df = pd.DataFrame(
            [[new_Social_event_attendance, new_Going_outside, new_Friends_circle_size]],
            columns=['Social_event_attendance', 'Going_outside', 'Friends_circle_size']
        )

        # Lakukan prediksi menggunakan model KNN
        predicted_code = knn_model.predict(new_data_df)[0] # Hasilnya 0 atau 1

        # Konversi hasil prediksi dari kode numerik ke label yang mudah dibaca
        label_mapping = {1: 'Extrovert', 0: 'Introvert'}
        predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')

        # Tampilkan hasil prediksi
        st.success(f"**Berdasarkan data yang Anda berikan, prediksi Personality Anda adalah: {predicted_label}**")
        st.balloons() # Efek balon jika prediksi berhasil!

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}. Mohon coba lagi.")

st.markdown("""
---
*Catatan: Pastikan file model `knn_model.joblib` kamu sudah disimpan dengan benar menggunakan `joblib.dump()` dan berada di direktori yang sama dengan file `app.py` ini di GitHub.*
""")