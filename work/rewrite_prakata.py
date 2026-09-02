# -*- coding: utf-8 -*-
"""Tulis ulang isi PRAKATA: struktur poin 1-6 mengikuti skripsi Fadil
(Rektor, Dekan, Koorprodi, Dosen Wali, Pembimbing, Penguji) dengan penguji
Bapak Anggyi Trisnawan Putra; kalimat disusun ulang agar tidak sama persis
dengan dokumen acuan mana pun (Fadil maupun Thoriq).
"""
import os
import re
import shutil
import zipfile
from datetime import datetime

DOCX = r"C:/Users/Pratama/Downloads/smpro/sempro-hamza-1.docx"
WORK = os.path.dirname(os.path.abspath(__file__))

RPR = ('<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
       'w:cs="Times New Roman"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>')
RPR_I = ('<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
         'w:cs="Times New Roman"/><w:i/><w:iCs/><w:sz w:val="24"/>'
         '<w:szCs w:val="24"/></w:rPr>')
PPR_BODY = ('<w:pPr><w:spacing w:after="0" w:line="276" w:lineRule="auto"/>'
            '<w:ind w:firstLine="426"/><w:jc w:val="both"/>' + RPR + '</w:pPr>')
PPR_ITEM = ('<w:pPr><w:pStyle w:val="DaftarParagraf"/><w:numPr>'
            '<w:ilvl w:val="0"/><w:numId w:val="41"/></w:numPr>'
            '<w:spacing w:after="0" w:line="276" w:lineRule="auto"/>'
            '<w:jc w:val="both"/>' + RPR + '</w:pPr>')


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def run(text, italic=False):
    return ('<w:r>' + (RPR_I if italic else RPR) +
            '<w:t xml:space="preserve">' + esc(text) + '</w:t></w:r>')


def para(ppr, segments):
    """segments: list of (text, italic)"""
    return '<w:p>' + ppr + ''.join(run(t, i) for t, i in segments) + '</w:p>'


# --- isi baru -------------------------------------------------------------
INTRO_1 = [
    ('Segala puji dan syukur penulis panjatkan ke hadirat Allah Subhanahu wa '
     'Ta\u2019ala, yang telah melimpahkan rahmat, kesehatan, serta kemudahan '
     'sehingga penulis dapat merampungkan skripsi berjudul \u201c', False),
    ('Knowledge Distillation', True),
    (' Berbasis ', False),
    ('Teacher-Assistant', True),
    (' untuk Model Ringan WasteNet pada Klasifikasi Citra Sampah TrashNet'
     '\u201d. Skripsi ini disusun untuk memenuhi salah satu syarat memperoleh '
     'gelar Sarjana pada Program Studi Sarjana Teknik Informatika, Fakultas '
     'Matematika dan Ilmu Pengetahuan Alam, Universitas Negeri Semarang.',
     False),
]

INTRO_2 = [
    ('Penulis menyadari sepenuhnya bahwa perjalanan menyusun skripsi ini tidak '
     'ditempuh seorang diri. Banyak pihak yang turut menyumbangkan bantuan, '
     'bimbingan, dukungan, maupun doa hingga karya ini sampai pada bentuk '
     'akhirnya. Karena itu, dengan segala kerendahan hati penulis menyampaikan '
     'terima kasih yang sebesar-besarnya kepada:', False),
]

ITEMS = [
    'Bapak Prof. Dr. S. Martono, M.Si., selaku Rektor Universitas Negeri '
    'Semarang, atas arah kebijakan dan kepemimpinan beliau yang menjadikan '
    'kampus ini tempat yang baik untuk menempuh pendidikan.',

    'Bapak Prof. Dr. Edy Cahyono, M.Si., selaku Dekan Fakultas Matematika dan '
    'Ilmu Pengetahuan Alam Universitas Negeri Semarang, atas dukungan akademik '
    'serta sarana dan prasarana yang penulis terima selama masa studi.',

    'Bapak Dr. Alamsyah, S.Si., M.Kom., selaku Koordinator Program Studi '
    'Sarjana Teknik Informatika Fakultas Matematika dan Ilmu Pengetahuan Alam '
    'Universitas Negeri Semarang, atas kemudahan dan dukungan yang diberikan '
    'hingga skripsi ini dapat dirampungkan.',

    'Bapak Riza Arifudin, S.Pd., M.Cs., selaku dosen wali, yang telah '
    'mendampingi dan mengarahkan langkah akademik penulis sejak awal '
    'perkuliahan hingga akhir masa studi.',

    'Bapak Kholiq Budiman, S.Pd., M.Kom., selaku dosen pembimbing, yang dengan '
    'sabar meluangkan waktu, memberikan bimbingan, serta menyampaikan masukan '
    'yang menajamkan arah penelitian ini.',

    'Bapak Anggyi Trisnawan Putra, S.Si., M.Si., selaku dosen penguji, atas '
    'koreksi, pertanyaan kritis, dan saran perbaikan yang membuat skripsi ini '
    'tersusun jauh lebih baik.',

    'Seluruh dosen dan tenaga kependidikan di lingkungan Fakultas Matematika '
    'dan Ilmu Pengetahuan Alam, khususnya Program Studi Teknik Informatika, '
    'atas ilmu dan pelayanan yang penulis terima sepanjang masa pendidikan.',

    'Kedua orang tua beserta seluruh keluarga penulis, yang doanya tidak pernah '
    'putus dan kasih sayangnya menjadi alasan utama penulis bertahan sampai '
    'titik ini.',

    'Sahabat-sahabat terdekat penulis, tempat berbagi keluh, tawa, dan semangat '
    'sejak awal perkuliahan hingga skripsi ini selesai.',

    'Rekan-rekan mahasiswa Teknik Informatika serta seluruh pihak yang tidak '
    'dapat penulis sebutkan satu per satu, atas kebersamaan dan bantuan yang '
    'diberikan selama menempuh perjuangan di kampus.',

    'Terakhir, kepada diri sendiri, terima kasih karena telah memilih bertahan, '
    'menuntaskan apa yang sudah dimulai, dan tidak berhenti di tengah jalan.',
]

CLOSING = [
    ('Penulis menyadari skripsi ini belum sempurna dan masih menyimpan sejumlah '
     'keterbatasan. Oleh sebab itu, kritik dan saran yang membangun sangat '
     'penulis harapkan sebagai bahan perbaikan pada karya berikutnya. Semoga '
     'hasil penelitian ini tetap dapat memberikan manfaat bagi pengembangan '
     'ilmu pengetahuan dan bagi pembaca yang menekuni bidang serupa.', False),
]

# --- splice ---------------------------------------------------------------
ID_INTRO = '318259D8'   # paragraf pembuka PRAKATA
ID_CLOSE = '7466A19B'   # paragraf penutup PRAKATA


def para_span(d, para_id):
    i = d.find('w14:paraId="%s"' % para_id)
    assert i > 0, para_id
    start = d.rfind('<w:p ', 0, i)
    end = d.index('</w:p>', i) + len('</w:p>')
    return start, end


def main():
    with zipfile.ZipFile(DOCX) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}
    d = blobs['word/document.xml'].decode('utf-8')

    s_intro, _ = para_span(d, ID_INTRO)
    s_close, e_close = para_span(d, ID_CLOSE)

    # paragraf kosong pemisah tepat sebelum paragraf penutup: dipertahankan apa adanya
    s_spacer = d.rfind('<w:p ', 0, s_close)
    spacer_raw = d[s_spacer:s_close]
    assert '<w:t' not in spacer_raw, 'paragraf sebelum penutup ternyata berisi teks'

    frag = (para(PPR_BODY, INTRO_1) +
            para(PPR_BODY, INTRO_2) +
            ''.join(para(PPR_ITEM, [(t, False)]) for t in ITEMS) +
            spacer_raw +
            para(PPR_BODY, CLOSING))

    new_d = d[:s_intro] + frag + d[e_close:]

    # sanity: hanya bagian PRAKATA yang berubah
    assert len(new_d) > len(d) - 5000
    assert new_d.count('<w:body') == 1

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    shutil.copy2(DOCX, os.path.join(WORK, 'sempro-hamza-1_BACKUP_%s.docx' % ts))

    tmp = DOCX + '.tmp'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, new_d.encode('utf-8')
                       if n == 'word/document.xml' else blobs[n])
    os.replace(tmp, DOCX)
    print('OK, backup ts =', ts)


if __name__ == '__main__':
    main()
