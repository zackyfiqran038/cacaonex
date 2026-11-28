"""CACAONEX / field-first cacao intelligence. Run: python -m streamlit run app.py"""
from __future__ import annotations

import hashlib
import html
import io
import csv
import math
import sqlite3, json, uuid
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from sampling import plan
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


@st.dialog('Metode dan rujukan', width='large')
def method_dialog():
    st.write('Grid sistematis dengan titik awal acak dalam polygon. Interval 20 m adalah default operasional, bukan standar ilmiah. Titik adalah lokasi rekomendasi, bukan pohon terverifikasi.')
    st.markdown('[Stehman & Foody (2019), Key issues in rigorous accuracy assessment of land cover products](https://doi.org/10.1016/j.rse.2019.05.018) — konteks desain sampling spasial, bukan validasi kakao.')
    st.markdown('[Selvaraju et al. (2017), Grad-CAM](https://doi.org/10.1109/ICCV.2017.74) — interpretasi model visual.')
    st.markdown('[Marelli et al. (2019), Chocolate Under Threat from Old and New Cacao Diseases](https://doi.org/10.1094/PHYTO-12-18-0477-RVW) — konteks penyakit kakao.')

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
ROUTES = {'home': 'Beranda', 'scan': 'Scan Kakao', 'sampling': 'Rencana Sampling', 'monitor': 'Monitoring', 'guide': 'Panduan Penanganan', 'settings': 'Pengaturan'}
requested = st.query_params.get('page', 'home')
if requested not in ROUTES:
    requested = 'home'
page = ROUTES[requested]

model_ready = model_path() is not None
render_topbar(requested, model_ready)

rows = all_records()
counts = {key:sum(r[2] == key for r in rows) for key in CLASS_NAMES}

(ROOT/'data').mkdir(exist_ok=True)
def db():
    c=sqlite3.connect(ROOT/'data'/'spatial.db');c.execute('CREATE TABLE IF NOT EXISTS plans(id TEXT PRIMARY KEY,name TEXT,data TEXT)');c.execute('CREATE TABLE IF NOT EXISTS scans(id TEXT PRIMARY KEY,plan_id TEXT,point TEXT,fruit TEXT,period TEXT,label TEXT,score REAL, UNIQUE(plan_id,point,fruit,period))');c.execute('CREATE TABLE IF NOT EXISTS visits(plan_id TEXT, point TEXT, period TEXT, note TEXT, PRIMARY KEY(plan_id,point,period))');c.commit();return c
with db() as c: plans={i:(n,json.loads(d)) for i,n,d in c.execute('SELECT id,name,data FROM plans')}
def basemap(center=(-2.5,118),zoom=5):
    m=folium.Map(location=center,zoom_start=zoom,tiles=None,control_scale=True)
    folium.TileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',attr='&copy; OpenStreetMap contributors',name='Peta jalan Indonesia',max_zoom=19,show=True).add_to(m)
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',attr='Esri, Maxar, Earthstar Geographics and GIS community',name='Satelit',overlay=False,show=False,max_zoom=19).add_to(m);folium.LayerControl().add_to(m);return m
def layers(m,p,colors=None):
    folium.Polygon(p['vertices'],color='#276749',fill=True,fill_opacity=.10).add_to(m)
    for x in p['points']:
        folium.CircleMarker([x['lat'],x['lon']],radius=6,popup=folium.Popup(f"<b>{x['code']}</b><br>{x['lat']:.6f}, {x['lon']:.6f}",max_width=240),color= (colors or {}).get(x['code'],'#d49b33'),fill=True,fill_opacity=.9,tooltip=x['code']).add_to(m)
    m.fit_bounds(p['vertices'])
def choose():
    if not plans: st.info('Simpan rancangan sampling terlebih dahulu.');return None
    return st.selectbox('Rencana kebun',list(plans),format_func=lambda k:plans[k][0]+' · '+k[:6])

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
      <a href="?page=sampling" class="cn-feature cn-feature-two"><div class="cn-feature-art cn-art-two"></div><div class="cn-feature-icon">▦</div><div class="cn-feature-content"><span>02 — FIELD PLANNING</span><h3>Rencana Sampling</h3><p>Gambar batas kebun dan buat titik pengamatan pada peta geografis.</p></div><span class="cn-feature-arrow">↗</span></a>
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
            linked=st.selectbox('Kebun (opsional)',[None]+list(plans),format_func=lambda k: plans[k][0] if k else 'Pemeriksaan bebas')
            if linked:
                point=st.selectbox('Titik sampling',[x['code'] for x in plans[linked][1]['points']]); period=st.date_input('Tanggal pengamatan'); fruit=st.text_input('ID buah unik')
            note=st.text_input('Kode pohon / catatan (opsional)',placeholder='Contoh: Petak B-03, Pohon S-014')
            if st.button('Simpan hasil pemeriksaan',type='primary',use_container_width=True):
                if linked:
                    if not fruit.strip(): st.error('Isi ID buah.'); st.stop()
                    try:
                        with db() as c: c.execute('INSERT INTO scans VALUES(?,?,?,?,?,?,?)',(uuid.uuid4().hex,linked,point,fruit.strip(),str(period),lab,probs[lab]))
                    except sqlite3.IntegrityError: st.error('Buah sudah dicatat pada titik dan tanggal ini.'); st.stop()
                    xy=next(x for x in plans[linked][1]['points'] if x['code']==point)
                    save(lab,probs[lab],xy['lat'],xy['lon'],note)
                else: save(lab,probs[lab],notes=note)
                st.success('Pemeriksaan tersimpan.');st.rerun()
        else:
            st.markdown('<div class="c-card"><b>Siap untuk diagnosis.</b><p>Unggah atau ambil satu foto, lalu klik Analisis. Hasil, rekomendasi awal, dan opsi Grad-CAM akan ditampilkan di sini.</p></div>',unsafe_allow_html=True)
    st.caption('Kamera pada Streamlit mengambil foto statis saat tombol ditekan; belum berupa deteksi video real-time frame demi frame.')

elif page == 'Rencana Sampling':
    page_header('02 / FIELD MAPPING','Rencana sampling kebun','Gambar batas kebun pada peta Indonesia, lalu buat titik pengamatan tetap.');st.caption('Peta Indonesia • pilih lapisan satelit melalui ikon layer di kanan atas. Gambar polygon batas kebun dengan alat di kiri peta.')
    if st.button('ⓘ Metode dan referensi'): method_dialog()
    left,right=st.columns([3,1]);
    with right:
        name=st.text_input('Nama kebun','Kebun kakao');spacing=st.number_input('Interval titik (meter)',2,1000,20);seed=st.number_input('Random seed',0,2147483647,42);st.caption('20 m hanyalah nilai awal yang dapat diubah, bukan standar ilmiah. Jumlah titik muncul dari luas dan interval.')
        if st.button('Kembali ke Indonesia'):
            st.session_state.map_center=(-2.5,118);st.session_state.map_zoom=5;st.session_state.pop('boundary',None);st.session_state.pop('draft',None);st.session_state['maprev']=st.session_state.get('maprev',0)+1;st.rerun()
        if st.button('Reset batas'): st.session_state.pop('boundary',None);st.session_state.pop('draft',None);st.session_state['maprev']=st.session_state.get('maprev',0)+1;st.rerun()
    with left:
        m=basemap(center=st.session_state.get('map_center',(-2.5,118)),zoom=st.session_state.get('map_zoom',5));Draw(export=False,draw_options={'polygon':True,'rectangle':False,'polyline':False,'circle':False,'circlemarker':False,'marker':False},edit_options={'edit':False,'remove':False}).add_to(m)
        if st.session_state.get('draft'): layers(m,st.session_state.draft)
        elif st.session_state.get('boundary'): folium.Polygon(st.session_state.boundary,color='green').add_to(m);m.fit_bounds(st.session_state.boundary)
        result=st_folium(m,height=650,use_container_width=True,key='draw'+str(st.session_state.get('maprev',0)),returned_objects=['last_active_drawing','center','zoom'])
        if result and result.get('center'):
            st.session_state.map_center=[result['center']['lat'],result['center']['lng']]
            st.session_state.map_zoom=result.get('zoom') or 5
        g=(result or {}).get('last_active_drawing') or {};g=g.get('geometry',{})
        if g.get('type')=='Polygon':
            v=[[lat,lon] for lon,lat in g['coordinates'][0][:-1]]
            if v!=st.session_state.get('boundary'): st.session_state.boundary=v;st.session_state.pop('draft',None)
        with st.expander('Alternatif jika menggambar tidak tersedia'):
            txt=st.text_area('Koordinat batas: satu latitude,longitude per baris')
            if st.button('Gunakan koordinat'):
                try: st.session_state.boundary=[list(map(float,t.split(','))) for t in txt.splitlines() if t.strip()];st.session_state.pop('draft',None);st.session_state['maprev']=st.session_state.get('maprev',0)+1;st.rerun()
                except ValueError: st.error('Format koordinat tidak valid.')
    if st.button('Buat titik sampling',type='primary'):
        try:
            p=plan(st.session_state.get('boundary',[]),spacing,seed)
            if not p['points']: raise ValueError('Tidak ada titik. Kurangi interval.')
            st.session_state.draft=p;st.session_state['maprev']=st.session_state.get('maprev',0)+1;st.rerun()
        except ValueError as e: st.error(str(e))
    if st.session_state.get('draft'):
        p=st.session_state.draft;a,b,c=st.columns(3);a.metric('Luas (ha)',f"{p['area']/10000:.3f}");b.metric('Titik sampling',len(p['points']));c.metric('Interval (m)',p['spacing']);st.dataframe(p['points'],use_container_width=True)
        buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=['code','lat','lon'],extrasaction='ignore');w.writeheader();w.writerows(p['points']);st.download_button('Unduh CSV',buf.getvalue(),'sampling.csv')
        if st.button('Simpan rencana tetap'):
            with db() as c: c.execute('INSERT INTO plans VALUES(?,?,?)',(uuid.uuid4().hex,name,json.dumps(p)))
            st.success('Rencana tersimpan. Gunakan titik ini untuk monitoring berikutnya.')

elif page == 'Monitoring':
    page_header('03 / FIELD ANALYSIS','Monitoring kebun','Pilih titik, catat kunjungan, dan periksa buah langsung dari monitoring.')
    k=choose()
    if k:
        p=plans[k][1]
        period=str(st.date_input('Tanggal pengamatan',key='monitor_date'))
        codes=[x['code'] for x in p['points']]
        with db() as c:
            df=pd.read_sql_query('SELECT point,fruit,period,label,score FROM scans WHERE plan_id=? AND period=?',c,params=(k,period))
            visits=dict(c.execute('SELECT point,note FROM visits WHERE plan_id=? AND period=?',(k,period)))
        scanned=set(df.point); done=set(visits)|scanned
        a,b,c=st.columns(3);a.metric('Total titik',len(codes));b.metric('Sudah dikunjungi',len(done));c.metric('Belum dikunjungi',len(set(codes)-done))
        st.progress(len(done)/max(1,len(codes)))
        colors={x:'#3979a8' for x in visits}
        colors.update({i:('#c35c43' if (g.label!='healthy').any() else '#39905d') for i,g in df.groupby('point')})
        m=basemap();layers(m,p,colors);st_folium(m,height=450,use_container_width=True,returned_objects=[],key='monitor_map_'+k+period+str(sorted(colors.items())))
        st.caption('Oranye: belum dikunjungi. Biru: kunjungan manual tanpa scan. Hijau: semua prediksi sehat. Merah: ada prediksi serangan.')
        table=pd.DataFrame([{'Kode':x,'Dikunjungi':x in done,'Jumlah scan':int((df.point==x).sum()),'Catatan':visits.get(x,'')} for x in codes])
        st.dataframe(table,hide_index=True,use_container_width=True)
        point=st.selectbox('Titik aktif',codes,key='monitor_point_'+k)
        tag=k+period+point
        with st.form('visit_'+tag):
            checked=st.checkbox('Tandai kunjungan manual',value=point in visits)
            note=st.text_area('Catatan lapangan',value=visits.get(point,''))
            if st.form_submit_button('Simpan checklist'):
                with db() as c:
                    if checked: c.execute('INSERT OR REPLACE INTO visits VALUES(?,?,?,?)',(k,point,period,note))
                    else: c.execute('DELETE FROM visits WHERE plan_id=? AND point=? AND period=?',(k,point,period))
                st.rerun()
        st.caption('Titik dengan hasil scan tetap tercatat dikunjungi meskipun checklist manual dihapus.')
        with st.expander('Deteksi AI dan Grad-CAM pada titik '+point,expanded=True):
            model,error=fetch_model()
            if model is None: st.warning('Pasang model pada Pengaturan. '+(error or ''))
            source=st.radio('Sumber foto',['Unggah','Kamera'],horizontal=True,key='src_'+tag)
            upload=st.file_uploader('Foto buah',type=['jpg','jpeg','png'],key='up_'+tag) if source=='Unggah' else st.camera_input('Foto buah',key='cam_'+tag)
            fruit=st.text_input('ID buah unik',key='fruit_'+tag)
            if upload is not None:
                try:
                    raw=upload.getvalue(); photo=Image.open(io.BytesIO(raw)).convert('RGB'); digest=hashlib.sha256(raw).hexdigest(); resultkey=tag+digest
                    st.image(photo,width=250)
                    if st.button('Analisis buah',disabled=model is None,key='an_'+tag):
                        lab,probs=predict(model,photo);st.session_state['monitor_result']=(resultkey,lab,probs);st.session_state.pop('monitor_cam',None)
                    r=st.session_state.get('monitor_result')
                    if r and r[0]==resultkey:
                        _,lab,probs=r;st.write(f'{LABELS[lab]} — skor {probs[lab]:.1%}')
                        if st.button('Grad-CAM',disabled=model is None,key='gc_'+tag):
                            st.session_state['monitor_cam']=(resultkey,overlay(photo,gradcam(model,photo)))
                        cam=st.session_state.get('monitor_cam')
                        if cam and cam[0]==resultkey: st.image(cam[1],caption='Grad-CAM: area yang memengaruhi prediksi')
                        if st.button('Simpan hasil ke titik '+point,key='sv_'+tag):
                            if not fruit.strip(): st.error('Isi ID buah unik.')
                            else:
                                try:
                                    with db() as c: c.execute('INSERT INTO scans VALUES(?,?,?,?,?,?,?)',(uuid.uuid4().hex,k,point,fruit.strip(),period,lab,float(probs[lab])))
                                except sqlite3.IntegrityError: st.error('ID buah sudah tercatat pada titik dan tanggal ini.')
                                else:
                                    xy=next(x for x in p['points'] if x['code']==point)
                                    save(lab,probs[lab],xy['lat'],xy['lon'],note)
                                    st.session_state.pop('monitor_result',None);st.rerun()
                except Exception as exc: st.error('Pemeriksaan gagal: '+str(exc))
        st.subheader('Analisis tanggal pengamatan')
        if not df.empty:
            a,b=st.columns(2);a.metric('Buah diperiksa',len(df));b.metric('Proporsi prediksi terserang',f"{(df.label!='healthy').mean():.1%}")
            st.bar_chart(df.label.value_counts().rename(index=LABELS));st.dataframe(df,use_container_width=True)
            st.download_button('Unduh hasil scan',df.to_csv(index=False),'monitoring.csv')
        else: st.info('Belum ada hasil scan pada tanggal ini.')
        st.download_button('Unduh checklist',table.to_csv(index=False),'checklist.csv')
        st.caption('Proporsi dihitung dari buah yang dipindai, bukan estimasi prevalensi seluruh kebun.')

elif page == 'Panduan Penanganan':
    page_header('03 / FIELD GUIDE', 'Tindakan yang tepat, bukan sekadar prediksi.', 'Panduan tindakan awal untuk tiga keluaran model CACAONEX. Setiap kategori dilengkapi sumber Kementerian Pertanian yang bisa diperiksa langsung.')
    option=st.segmented_control('Pilih kondisi', ['Buah sehat','Busuk buah','Penggerek buah'],default='Busuk buah', key='guide_condition')
    key={'Buah sehat':'healthy','Busuk buah':'black_pod_rot','Penggerek buah':'pod_borer'}[option]
    title,actions,refs=GUIDANCE[key]
    st.markdown(f'<div class="c-card"><div class="c-eyebrow">REKOMENDASI TERARAH</div><h2>{html.escape(title)}</h2><p>Gunakan tindakan berikut sebagai pedoman awal yang tetap perlu disesuaikan dengan pemeriksaan lapangan.</p></div>',unsafe_allow_html=True)
    for i,action in enumerate(actions,1):
        st.markdown(f'<div class="c-card"><span class="c-step">{i:02d}</span><b>{html.escape(action)}</b></div>',unsafe_allow_html=True)
    st.markdown('<div class="cn-references-heading"><div><span>LIBRARY / REFERENCES</span><h2>Rujukan pilihan untuk penanganan yang lebih tepat.</h2></div><p>Dokumen teknis pemerintah dan artikel ilmiah yang dapat ditelusuri langsung.</p></div>',unsafe_allow_html=True)
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
