"""CACAONEX / field-first cacao intelligence. Run: python -m streamlit run app.py"""
from __future__ import annotations

import hashlib
import html
import io
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import streamlit as st
from PIL import Image

from database import all_records, delete_all, save
from inference import CLASS_NAMES, LABELS, gradcam, overlay, predict

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title='CACAONEX — Cocoa intelligence', page_icon='◈', layout='wide', initial_sidebar_state='collapsed')

STYLE = (ROOT / "assets" / "theme_forest.css").read_text(encoding="utf-8")

st.markdown(STYLE, unsafe_allow_html=True)

REFERENCES = {
    'busuk': [
        {'kind': 'PEDOMAN RESMI', 'year': 'Kementerian Pertanian', 'title': 'Penyakit Busuk Buah pada Tanaman Kakao', 'desc': 'Pengenalan gejala, penyebab, dan prinsip pengendalian busuk buah kakao.', 'url': 'https://sinta.ditjenbun.pertanian.go.id/penyakit-busuk-buah/'},
        {'kind': 'PANDUAN PRAKTIS', 'year': 'Kementerian Pertanian', 'title': 'Tips Kendalikan Busuk Kakao', 'desc': 'Langkah sanitasi kebun dan pengelolaan penyakit pada tingkat budidaya.', 'url': 'https://pustaka.bppsdmp.pertanian.go.id/info-literasi/tips-kendalikan-busuk-kakao-2'},
        {'kind': 'JURNAL ILMIAH', 'year': 'Journal of Fungi · 2020', 'title': 'Phytophthora palmivora–Cocoa Interaction', 'desc': 'Kajian tentang interaksi patogen dengan buah kakao dan mekanisme infeksi.', 'url': 'https://doi.org/10.3390/jof6030167'},
        {'kind': 'JURNAL ILMIAH', 'year': 'Phytopathology · 2019', 'title': 'Chocolate Under Threat from Old and New Cacao Diseases', 'desc': 'Tinjauan penyakit utama kakao serta strategi penelitian dan pengelolaannya.', 'url': 'https://doi.org/10.1094/PHYTO-12-18-0477-RVW'},
    ],
    'pbk': [
        {'kind': 'PEDOMAN RESMI', 'year': 'Kementerian Pertanian', 'title': 'Hama Penggerek Buah Kakao (PBK)', 'desc': 'Pengenalan PBK dan pendekatan pengendalian yang dapat dipertimbangkan.', 'url': 'https://sinta.ditjenbun.pertanian.go.id/hama-penggerek-buah-kakao-pbk/'},
        {'kind': 'PANDUAN TEKNIS', 'year': 'Kementerian Pertanian', 'title': 'Pedoman Teknis Pengendalian Hama Penggerek Buah Kakao', 'desc': 'Buku teknis tentang gejala, pemantauan, dan pengendalian PBK pada tanaman kakao.', 'url': 'https://repository.pertanian.go.id/handle/123456789/13167'},
        {'kind': 'JURNAL ILMIAH', 'year': 'Journal of Integrated Pest Management · 2026', 'title': 'Biological Control of Cocoa Pod Borer for Sustainable IPM', 'desc': 'Sintesis bukti pengendalian hayati untuk pengelolaan PBK yang berkelanjutan.', 'url': 'https://doi.org/10.1093/jipm/pmag042'},
        {'kind': 'JURNAL ILMIAH', 'year': 'Jurnal Hama dan Penyakit Tumbuhan Tropika · 2025', 'title': 'Beneficial Insects and Cocoa Pod Borer Damage', 'desc': 'Studi hubungan tumbuhan insektari, serangga bermanfaat, dan intensitas serangan PBK.', 'url': 'https://jhpttropika.fp.unila.ac.id/index.php/jhpttropika/article/view/810'},
    ],
    'umum': [
        {'kind': 'PANDUAN RESMI', 'year': 'Kementerian Pertanian', 'title': 'Pengendalian Terpadu Penyakit Busuk Buah Kakao', 'desc': 'Referensi metode pengendalian terpadu di tingkat perkebunan.', 'url': 'https://repository.pertanian.go.id/handle/123456789/16045'},
        {'kind': 'JURNAL ILMIAH', 'year': 'Pelita Perkebunan · 2005', 'title': 'Development of Monitoring Technique for Cocoa Pod Borer', 'desc': 'Kajian tentang metode pemantauan serangan penggerek buah kakao; bukan validasi grid CACAONEX.', 'url': 'https://doi.org/10.22302/iccri.jur.pelitaperkebunan.v21i3.24'},
        {'kind': 'PANDUAN RESMI', 'year': 'Kementerian Pertanian', 'title': 'Teknologi Budidaya Kakao', 'desc': 'Panduan budidaya dan pengendalian hama serta penyakit sebagai konteks pelaksanaan lapangan.', 'url': 'https://repository.pertanian.go.id/server/api/core/bitstreams/072bd6cc-b669-467f-95c5-244619391523/content'},
    ],
}

GUIDANCE = {
    'healthy': ('Pertahankan kondisi sehat', [
        'Pantau buah secara berkala dan catat perubahan gejala.',
        'Jaga kebersihan kebun, pemangkasan, dan sirkulasi udara.',
        'Tinjau ulang sampel bila ada perubahan kondisi atau cuaca ekstrem.',
    ], 'umum'),
    'black_pod_rot': ('Prioritaskan sanitasi dan kelembapan', [
        'Tandai pohon bergejala dan periksa buah di sekitarnya.',
        'Singkirkan buah terinfeksi sesuai panduan sanitasi perkebunan; hindari meninggalkan sumber inokulum di kebun.',
        'Perbaiki pemangkasan, sirkulasi udara, dan drainase kebun.',
        'Konsultasikan opsi pengendalian lanjutan dengan penyuluh sebelum memakai fungisida.',
    ], 'busuk'),
    'pod_borer': ('Terapkan pengendalian hama terpadu', [
        'Konfirmasi tanda serangan melalui pemeriksaan buah dan hasil panen.',
        'Lakukan panen sering dan sanitasi kebun sesuai rekomendasi penyuluh.',
        'Pertimbangkan penyarungan buah dan pengelolaan tajuk sesuai kondisi tanaman.',
        'Periksa pohon tetangga dan jadwalkan pengamatan ulang.',
    ], 'pbk'),
}

@st.cache_resource(show_spinner='Menyiapkan mesin diagnosis...')
def load_model(path: str, timestamp: float):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


def model_path():
    for name in ('lowest_val_loss.h5', 'lowest_val_loss.keras'):
        p = MODEL_DIR / name
        if p.is_file():
            return p
    return None


def fetch_model():
    p = model_path()
    if p is None:
        return None, None
    try:
        return load_model(str(p), p.stat().st_mtime), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def page_header(kicker, title, description):
    st.markdown(f'<div class="c-pagehead"><div class="c-eyebrow">{html.escape(kicker)}</div><h1>{html.escape(title)}</h1><p>{html.escape(description)}</p></div>', unsafe_allow_html=True)


def source_links(group):
    cards = []
    for ref in REFERENCES[group]:
        kind = html.escape(ref['kind'])
        year = html.escape(ref['year'])
        title = html.escape(ref['title'])
        desc = html.escape(ref['desc'])
        url = html.escape(ref['url'], quote=True)
        cards.append(f'''<a class="cn-reference-card" href="{url}" target="_blank" rel="noopener noreferrer" aria-label="Buka sumber: {title}">
            <div class="cn-reference-top"><span class="cn-reference-type">{kind}</span><span class="cn-reference-arrow">↗</span></div>
            <div class="cn-reference-body"><small>{year}</small><h3>{title}</h3><p>{desc}</p></div>
            <div class="cn-reference-bottom">BACA SUMBER <span>↗</span></div>
        </a>''')
    st.markdown('<div class="cn-reference-grid">'+''.join(cards)+'</div>', unsafe_allow_html=True)


def field_grid(area_ha: float, width_m: float, cell_m: float, row_space: float, tree_space: float, fruits: int):
    area = area_ha * 10000.0
    length = area / width_m
    nx = max(1, math.ceil(width_m / cell_m))
    ny = max(1, math.ceil(length / cell_m))
    if nx * ny > 1500:
        raise ValueError('Terlalu banyak petak. Perbesar ukuran petak atau periksa luas dan lebar kebun.')
    data = []
    idx = 0
    for row in range(ny):
        cols = range(nx) if row % 2 == 0 else range(nx-1, -1, -1)
        for col in cols:
            idx += 1
            x0 = col * cell_m
            y0 = row * cell_m
            x1 = min((col+1)*cell_m, width_m)
            y1 = min((row+1)*cell_m, length)
            # Choose nearest nominal tree to cell center (tree at half-spacing positions).
            target_x = (x0+x1)/2
            target_y = (y0+y1)/2
            xx = (round((target_x - tree_space/2)/tree_space) + .5) * tree_space
            yy = (round((target_y - row_space/2)/row_space) + .5) * row_space
            xx = min(max(xx, x0), x1)
            yy = min(max(yy, y0), y1)
            data.append({'Titik': f'S-{idx:03d}', 'Baris petak': row+1, 'Kolom petak': col+1,
                         'X dari sisi kiri (m)': round(xx, 1), 'Y dari sisi bawah (m)': round(yy, 1),
                         'Buah disarankan': fruits})
    return pd.DataFrame(data), length, nx, ny


def grid_figure(data, width, length, cell):
    fig, ax = plt.subplots(figsize=(min(11, max(6, width/12)), min(8, max(4, length/14))))
    fig.patch.set_facecolor('#F6F5EC')
    ax.set_facecolor('#E8EBD7')
    nx = math.ceil(width/cell)
    ny = math.ceil(length/cell)
    for j in range(ny):
        for i in range(nx):
            x=i*cell; y=j*cell
            ax.add_patch(Rectangle((x,y),min(cell,width-x),min(cell,length-y),
                         facecolor=('#D9E8B9' if (i+j)%2==0 else '#E8EBD5'),edgecolor='#91A67E',linewidth=.8))
    xs=data['X dari sisi kiri (m)'].tolist(); ys=data['Y dari sisi bawah (m)'].tolist()
    ax.scatter(xs, ys, s=35, c='#F06C4E',edgecolors='white',linewidths=.8,zorder=3)
    if len(data)<=70:
        for idx,(x,y) in enumerate(zip(xs,ys)):
            ax.annotate(str(idx+1),(x,y),xytext=(3,3),textcoords='offset points',fontsize=6,color='#1F3426')
    ax.set_xlim(0,width);ax.set_ylim(0,length);ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('Lebar kebun (meter)');ax.set_ylabel('Panjang kebun (meter)')
    ax.set_title('Rencana sampling · satu pohon contoh per petak',fontsize=11,fontweight='bold',color='#1F3426',pad=12)
    ax.grid(False)
    fig.tight_layout()
    return fig


# Native Streamlit controls remain functional; the custom skin is applied only to presentation.
def icon(name: str, size: int=23) -> str:
    paths = {
        'spark':'<path d="m12 2 1.7 6.3L20 10l-6.3 1.7L12 18l-1.7-6.3L4 10l6.3-1.7L12 2Z"/><path d="m19 17 .7 2.3L22 20l-2.3.7L19 23l-.7-2.3L16 20l2.3-.7L19 17Z"/>',
        'scan':'<rect x="5" y="5" width="14" height="14" rx="4"/><path d="M2 8V5a3 3 0 0 1 3-3h3M16 2h3a3 3 0 0 1 3 3v3M22 16v3a3 3 0 0 1-3 3h-3M8 22H5a3 3 0 0 1-3-3v-3"/><circle cx="12" cy="12" r="3"/>',
        'grid':'<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
        'leaf':'<path d="M20 4C9 4 4 9 4 15a5 5 0 0 0 5 5c6 0 11-5 11-16Z"/><path d="M4 20c4-6 7-8 12-11"/>',
        'arrow':'<path d="M5 12h14m-7-7 7 7-7 7"/>',
        'shield':'<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/><path d="m9 12 2 2 4-4"/>',
        'camera':'<path d="M14 4h-4l-2 3H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-4l-2-3Z"/><circle cx="12" cy="13" r="4"/>',
        'activity':'<path d="M2 12h5l3-8 4 16 3-8h5"/>',
    }
    return f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths.get(name,paths["spark"])}</svg>'

def section(kicker: str, title: str, sub: str=''):
    st.markdown(f'<div class="section-label">{html.escape(kicker)}</div><div class="section-header"><h2>{html.escape(title)}</h2><p>{html.escape(sub)}</p></div>',unsafe_allow_html=True)


def render_topbar(requested: str, model_ready: bool):
    items = []
    for slug, label in ROUTES.items():
        active = ' active' if slug == requested else ''
        items.append(f'<a class="cn-navlink{active}" href="?page={slug}">{html.escape(label)}</a>')
    status_label = 'MODEL ACTIVE' if model_ready else 'SETUP MODEL'
    status_class = 'ready' if model_ready else 'idle'
    st.markdown(
        f'''
        <nav class="cn-topbar">
            <a class="cn-brandwrap" href="?page=home" aria-label="CACAONEX home">
                <span class="cn-logo">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                        <rect x="1.5" y="1.5" width="21" height="21" rx="7" fill="#D8F16C"/>
                        <path d="M12 6.7 16.9 11.6 12 16.5 7.1 11.6 12 6.7Z" stroke="#173320" stroke-width="1.6"/>
                        <path d="M12 9.3 14.3 11.6 12 13.9 9.7 11.6 12 9.3Z" fill="#173320"/>
                    </svg>
                </span>
                <span class="cn-brandtext">CACAONEX</span>
            </a>
            <div class="cn-navglass">{''.join(items)}</div>
            <div class="cn-status {status_class}"><span class="cn-status-dot"></span><span>{status_label}</span></div>
        </nav>
        ''',
        unsafe_allow_html=True,
    )

# Query-string navigation makes the hero CTA and cards real links.
ROUTES = {'home': 'Beranda', 'scan': 'Scan Kakao', 'sampling': 'Rencana Sampling', 'guide': 'Panduan Penanganan', 'settings': 'Pengaturan'}
requested = st.query_params.get('page', 'home')
if requested not in ROUTES:
    requested = 'home'
page = ROUTES[requested]

model_ready = model_path() is not None
render_topbar(requested, model_ready)

rows = all_records()
counts = {key:sum(r[2] == key for r in rows) for key in CLASS_NAMES}

if page == 'Beranda':
    st.markdown('''<section class="cn-hero">
      <div class="cn-hero-copy">
        <div class="cn-kicker"><span></span> AI FOR HEALTHIER CACAO <b> / 2026</b></div>
        <div class="cn-mega">CACAONEX<span class="cn-swoosh">✳</span></div>
        <div class="cn-tagline"><em>Scan</em><i>·</i><em>Sample</em><i>·</i><em>Solve</em></div>
        <p>Kenali kondisi buah kakao, rancang titik pemeriksaan yang efisien, dan temukan langkah penanganan berdasarkan referensi ilmiah.</p>
        <div class="cn-actions"><a class="cn-action-primary" href="?page=scan">Mulai Pemeriksaan <span>↗</span></a><a class="cn-action-outline" href="?page=sampling">Rancang Sampling <span>→</span></a></div>
        <div class="cn-mini-meta"><span>01 / DETEKSI</span><span>02 / SAMPLING</span><span>03 / PENANGANAN</span></div>
      </div>
      <div class="cn-hero-visual" aria-label="Foto buah kakao pada pohon"><div class="cn-photo-glow"></div>
        <div class="cn-hero-sticker cn-sticker-a">FIELD INTELLIGENCE <strong>01 / SCAN</strong></div>
        <div class="cn-hero-sticker cn-sticker-b">THE COCOA PROJECT <strong>GROW SMARTER ↗</strong></div>
      </div>
    </section>
    <div class="cn-section-intro"><span>01 / EXPLORE THE PLATFORM</span><h2>Everything your field needs.<br><em>Nothing it doesn't.</em></h2></div>
    <div class="cn-feature-grid">
      <a href="?page=scan" class="cn-feature cn-feature-one"><div class="cn-feature-art cn-art-one"></div><div class="cn-feature-icon">◎</div><div class="cn-feature-content"><span>01 — AI DIAGNOSIS</span><h3>Scan Kakao</h3><p>Foto atau kamera langsung. Kenali tiga kondisi buah dan lihat Grad-CAM.</p></div><span class="cn-feature-arrow">↗</span></a>
      <a href="?page=sampling" class="cn-feature cn-feature-two"><div class="cn-feature-art cn-art-two"></div><div class="cn-feature-icon">▦</div><div class="cn-feature-content"><span>02 — FIELD PLANNING</span><h3>Rencana Sampling</h3><p>Ubah luas dan jarak tanam menjadi rekomendasi petak dan pohon contoh.</p></div><span class="cn-feature-arrow">↗</span></a>
      <a href="?page=guide" class="cn-feature cn-feature-three"><div class="cn-feature-art cn-art-three"></div><div class="cn-feature-icon">✳</div><div class="cn-feature-content"><span>03 — FIELD GUIDANCE</span><h3>Panduan Penanganan</h3><p>Langkah awal menghadapi penyakit, beserta rujukan teknis pertanian.</p></div><span class="cn-feature-arrow">↗</span></a>
    </div>''',unsafe_allow_html=True)
    st.markdown('<div class="cn-stat-heading">FIELD NOTES <span>MONITORING SNAPSHOT / LIVE LOCAL DATA</span></div>',unsafe_allow_html=True)
    n_all = len(rows)
    n_alert = counts['black_pod_rot'] + counts['pod_borer']
    n_healthy = counts['healthy']
    metric_cols=st.columns(3,gap='medium')
    for col,title,value,detail in [(metric_cols[0],'TOTAL SCANS',n_all,'Hasil yang disimpan'),(metric_cols[1],'NEED ATTENTION',n_alert,'Busuk buah dan penggerek'),(metric_cols[2],'HEALTHY SCANS',n_healthy,'Hasil skrining sehat')]:
        with col:
            st.markdown(f'<div class="cn-data-card"><span>{title}</span><strong>{value:02d}</strong><small>{detail}</small></div>',unsafe_allow_html=True)
    st.markdown('<div class="cn-disclaimer"><strong>FIELD FIRST, DATA AWARE.</strong> CACAONEX adalah alat bantu skrining. Rancangan sampling bukan standar epidemiologi tervalidasi, dan diagnosis akhir memerlukan konfirmasi lapangan.</div>',unsafe_allow_html=True)

elif page == 'Scan Kakao':
    page_header('01 / SMART SCAN', 'Cek buah kakao, semudah ambil foto.', 'Pilih gambar dari galeri atau gunakan kamera langsung. Hasil klasifikasi berasal dari model EfficientNet-B0 milikmu.')
    model,error=fetch_model()
    if model is None:
        st.warning('Model AI belum siap. Unggah lowest_val_loss.h5 pada halaman Pengaturan.')
        if error:st.error('Gagal memuat model: '+error)
    left,right=st.columns([1,1],gap='large')
    with left:
        st.markdown('<div class="c-section">Pilih cara mengambil gambar</div>',unsafe_allow_html=True)
        mode=st.segmented_control('Sumber foto', ['Unggah foto', 'Kamera langsung'], default='Unggah foto', label_visibility='collapsed', key='scan_source')
        if mode=='Kamera langsung':
            upload=st.camera_input('Arahkan kamera ke satu buah kakao')
        else:
            upload=st.file_uploader('Pilih gambar JPG atau PNG',type=['jpg','jpeg','png'])
        photo=None; raw=None
        if upload is not None:
            try:
                raw=upload.getvalue(); photo=Image.open(io.BytesIO(raw)).convert('RGB')
                st.image(photo,caption='Foto yang dipilih',use_container_width=True)
            except Exception as exc:
                st.error('Gambar tidak terbaca: '+str(exc))
        else:
            st.markdown('<div class="c-card c-yellow"><b>Tips foto yang efektif</b><p>Foto satu buah kakao dengan cahaya cukup, tidak buram, dan hindari objek yang menutupi permukaan buah.</p></div>',unsafe_allow_html=True)
        run=st.button('Analisis buah sekarang →',type='primary',use_container_width=True,disabled=photo is None or model is None)
    with right:
        st.markdown('<div class="c-section">Hasil analisis</div>',unsafe_allow_html=True)
        key=hashlib.sha256(raw).hexdigest() if raw else None
        if run and photo is not None and model is not None:
            try:
                with st.status('Menganalisis foto kakao...', expanded=False) as state:
                    shimmer=st.empty()
                    shimmer.markdown('<div class="shimmer-card"><div class="shimmer-line wide"></div><div class="shimmer-line"></div><div class="shimmer-line short"></div></div>', unsafe_allow_html=True)
                    lab,probs=predict(model,photo)
                    shimmer.empty()
                    state.update(label='Analisis selesai',state='complete')
                st.session_state['last_scan']={'key':key,'label':lab,'prob':probs}
                st.session_state.pop('last_cam',None)
            except Exception as exc:st.error('Prediksi gagal: '+str(exc))
        result=st.session_state.get('last_scan')
        if result and result['key']!=key:result=None
        if result:
            lab=result['label'];probs=result['prob']
            st.markdown(f'<div class="c-card"><div class="c-caption">KLASIFIKASI MODEL</div><div class="c-num" style="font-size:1.65rem">{html.escape(LABELS[lab])}</div><span class="c-pill">Skor output: {probs[lab]*100:.1f}%</span></div>',unsafe_allow_html=True)
            for cls in CLASS_NAMES:
                st.caption(f'{LABELS[cls]} · {probs[cls]*100:.1f}%')
                st.progress(max(0.0,min(1.0,probs[cls])))
            st.caption('Skor softmax bukan tingkat akurasi dan belum tentu merupakan probabilitas terkalibrasi.')
            title,actions,_ = GUIDANCE[lab]
            st.markdown(f'**Arah penanganan awal: {title}**')
            st.write('• '+actions[0])
            if st.button('Tampilkan Grad-CAM',use_container_width=True):
                try:
                    with st.status('Membangun visualisasi Grad-CAM...',expanded=False) as state:
                        shimmer=st.empty()
                        shimmer.markdown('<div class="shimmer-card"><div class="shimmer-line wide"></div><div class="shimmer-line"></div><div class="shimmer-line short"></div></div>',unsafe_allow_html=True)
                        st.session_state['last_cam']=(key,overlay(photo,gradcam(model,photo)))
                        shimmer.empty()
                        state.update(label='Grad-CAM siap',state='complete')
                except Exception as exc:st.warning('Grad-CAM belum kompatibel dengan model ini: '+str(exc))
            cam=st.session_state.get('last_cam')
            if cam and cam[0]==key:st.image(cam[1],caption='Grad-CAM — area yang memengaruhi keputusan AI, bukan bukti lokasi penyakit',use_container_width=True)
            note=st.text_input('Kode pohon / catatan (opsional)',placeholder='Contoh: Petak B-03, Pohon S-014')
            if st.button('Simpan hasil pemeriksaan',type='primary',use_container_width=True):
                save(lab,probs[lab],notes=note)
                st.success('Pemeriksaan tersimpan.');st.rerun()
        else:
            st.markdown('<div class="c-card"><b>Siap untuk diagnosis.</b><p>Unggah atau ambil satu foto, lalu klik Analisis. Hasil, rekomendasi awal, dan opsi Grad-CAM akan ditampilkan di sini.</p></div>',unsafe_allow_html=True)
    st.caption('Kamera pada Streamlit mengambil foto statis saat tombol ditekan; belum berupa deteksi video real-time frame demi frame.')

elif page == 'Rencana Sampling':
    page_header('02 / SMART SAMPLING', 'Satu kebun. Banyak petak. Lebih efisien.', 'Buat rancangan pemeriksaan berbasis luas kebun, bentuk lahan, jarak antarpohon, dan ukuran petak sampling. Tanpa memasukkan koordinat GPS.')
    st.markdown('<div class="c-card c-yellow"><b>Bagaimana sistem ini bekerja?</b><p>Satu titik rekomendasi menunjukkan satu pohon contoh pada setiap petak. Jumlah petak dapat kamu ubah dengan mengatur ukuran grid. Ini strategi sampling operasional awal, bukan perhitungan ukuran sampel statistik atau aturan resmi untuk semua kebun.</p></div>',unsafe_allow_html=True)
    col1,col2,col3=st.columns(3,gap='medium')
    with col1:
        area=st.number_input('Luas kebun (hektare)',min_value=.01,max_value=100.0,value=1.0,step=.25,format='%.2f')
        width=st.number_input('Perkiraan lebar kebun (meter)',min_value=5.0,max_value=10000.0,value=100.0,step=5.0)
    with col2:
        tree_space=st.number_input('Jarak antarpohon (meter)',min_value=1.0,max_value=20.0,value=3.0,step=.5)
        row_space=st.number_input('Jarak antarbaris (meter)',min_value=1.0,max_value=20.0,value=3.0,step=.5)
    with col3:
        cell=st.slider('Ukuran petak sampling (meter)',min_value=10,max_value=100,value=20,step=5,help='Misalnya 20 berarti petak 20 × 20 meter; satu pohon contoh tiap petak.')
        fruits=st.select_slider('Buah diamati pada tiap pohon contoh',options=[1,2,3,4,5],value=2)
    try:
        data,length,nx,ny=field_grid(area,width,cell,row_space,tree_space,fruits)
        n=len(data)
        approx_trees=math.floor(area*10000/(tree_space*row_space))
        metrics=st.columns(4)
        metrics[0].metric('Jumlah petak',f'{n:,}'.replace(',','.'))
        metrics[1].metric('Pohon contoh',f'{n:,}'.replace(',','.'))
        metrics[2].metric('Buah diperiksa',f'{n*fruits:,}'.replace(',','.'))
        metrics[3].metric('Estimasi pohon kebun',f'{approx_trees:,}'.replace(',','.'))
        st.caption(f'Perkiraan panjang kebun: {length:.1f} m · grid {nx} kolom × {ny} baris · estimasi jumlah pohon memakai jarak tanam teoritis, tidak memperhitungkan jalan dan ruang kosong.')
        left,right=st.columns([1.35,1],gap='large')
        with left:
            st.markdown('<div class="c-section">Denah sampling</div>',unsafe_allow_html=True)
            fig=grid_figure(data,width,length,cell)
            st.pyplot(fig,use_container_width=True)
            buf=io.BytesIO();fig.savefig(buf,format='png',dpi=165,bbox_inches='tight');plt.close(fig)
            st.download_button('Download denah PNG',buf.getvalue(),'cacaonex_denah_sampling.png','image/png',use_container_width=True)
        with right:
            st.markdown('<div class="c-section">Daftar titik sampel</div>',unsafe_allow_html=True)
            st.dataframe(data,height=410,use_container_width=True,hide_index=True)
            st.download_button('Download titik sampling CSV',data.to_csv(index=False).encode('utf-8-sig'),'cacaonex_titik_sampling.csv','text/csv',use_container_width=True)
        st.markdown('<div class="c-card c-pink"><b>Cara pakai di kebun</b><p>Mulai dari petak S-001, cari pohon terdekat dengan posisi rekomendasi, lalu pilih buah dari bagian tajuk yang berbeda. Catat pohon yang tidak berbuah, gejala terlihat, dan kondisi petak. Jika ditemukan gejala, periksa pohon tetangga dan perluas pengamatan. Koordinat X–Y hanya jarak lokal dari sudut kebun, bukan koordinat geografis.</p></div>',unsafe_allow_html=True)
        st.warning('Jangan gunakan hasil satu buah untuk menghitung prevalensi penyakit satu hektare. Untuk penelitian epidemiologi, jumlah pohon/buah, pemilihan acak, pengelompokan area, dan pengulangan waktu perlu protokol sampling tervalidasi.')
        st.markdown('**Bacaan mengenai sampling lapangan:**')
        source_links('umum')
    except ValueError as exc:st.error(str(exc))

elif page == 'Panduan Penanganan':
    page_header('03 / FIELD GUIDE', 'Tindakan yang tepat, bukan sekadar prediksi.', 'Panduan tindakan awal untuk tiga keluaran model CACAONEX. Setiap kategori dilengkapi sumber Kementerian Pertanian yang bisa diperiksa langsung.')
    option=st.segmented_control('Pilih kondisi', ['Buah sehat','Busuk buah','Penggerek buah'],default='Busuk buah', key='guide_condition')
    key={'Buah sehat':'healthy','Busuk buah':'black_pod_rot','Penggerek buah':'pod_borer'}[option]
    title,actions,refs=GUIDANCE[key]
    st.markdown(f'<div class="c-card"><div class="c-eyebrow">REKOMENDASI TERARAH</div><h2>{html.escape(title)}</h2><p>Gunakan tindakan berikut sebagai pedoman awal yang tetap perlu disesuaikan dengan pemeriksaan lapangan.</p></div>',unsafe_allow_html=True)
    for i,action in enumerate(actions,1):
        st.markdown(f'<div class="c-card"><span class="c-step">{i:02d}</span><b>{html.escape(action)}</b></div>',unsafe_allow_html=True)
    st.markdown('<div class="cn-references-heading"><div><span>LIBRARY / VERIFIED SOURCES</span><h2>Rujukan pilihan untuk penanganan yang lebih tepat.</h2></div><p>Dokumen teknis pemerintah dan artikel ilmiah yang dapat ditelusuri langsung.</p></div>',unsafe_allow_html=True)
    source_links(refs)
    st.markdown('<aside class="cn-safety-note"><div class="cn-safety-mark">i</div><p><strong>Catatan penggunaan.</strong> Hasil analisis foto merupakan skrining awal, bukan pengganti pemeriksaan lapangan. Penggunaan pestisida atau fungisida harus mengikuti label, ketentuan yang berlaku, dan arahan petugas berwenang.</p></aside>',unsafe_allow_html=True)

else:
    page_header('PENGATURAN', 'Bikin semuanya siap digunakan.', 'Pasang model hasil training Google Colab, lihat riwayat skrining, dan unduh hasil pemeriksaan.')
    st.markdown('<div class="c-section">Model AI</div>',unsafe_allow_html=True)
    uploaded=st.file_uploader('Masukkan model lowest_val_loss.h5 / .keras',type=['h5','keras'])
    if uploaded is not None and st.button('Pasang model sekarang',type='primary'):
        target=MODEL_DIR/('lowest_val_loss.keras' if uploaded.name.lower().endswith('.keras') else 'lowest_val_loss.h5')
        target.write_bytes(uploaded.getbuffer())
        other=MODEL_DIR/('lowest_val_loss.h5' if target.suffix=='.keras' else 'lowest_val_loss.keras')
        if other.exists():other.unlink()
        load_model.clear()
        st.success('File model terpasang. Buka Scan Kakao untuk mengujinya.')
    st.caption('Model harus memiliki tiga keluaran sesuai urutan: black_pod_rot, healthy, pod_borer. Input gambar 224 × 224 RGB tanpa normalisasi tambahan sebagaimana notebook referensi.')
    st.markdown('<div class="c-section">Riwayat pemeriksaan</div>',unsafe_allow_html=True)
    df=pd.DataFrame([(r[0],r[1],LABELS.get(r[2],r[2]),round(float(r[3])*100,1),r[6]) for r in rows],columns=['ID','Waktu UTC','Hasil','Skor model (%)','Catatan'])
    if len(df):
        st.dataframe(df,hide_index=True,use_container_width=True)
        st.download_button('Download riwayat CSV',df.to_csv(index=False).encode('utf-8-sig'),'cacaonex_riwayat.csv','text/csv')
        agreed=st.checkbox('Saya memahami penghapusan riwayat tidak dapat dibatalkan')
        if st.button('Hapus semua riwayat',disabled=not agreed):delete_all();st.rerun()
    else:st.info('Belum ada riwayat tersimpan.')
    st.caption('File gambar tidak disimpan ke database; informasi diagnosis dan catatan baru tersimpan setelah tombol Simpan ditekan.')

st.markdown('<div class="cn-footer"><span>CACAONEX / SCAN · SAMPLE · SOLVE</span><span>GROWN FOR THE NEXT GENERATION — 2026</span></div>',unsafe_allow_html=True)
