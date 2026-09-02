# -*- coding: utf-8 -*-
"""Gabungkan judul bab ke dalam heading Judul1 supaya DAFTAR ISI menampilkan
"BAB 1 PENDAHULUAN", bukan "BAB 1" saja.

Sebelum : <w:p Judul1>BAB 1</w:p> + <w:p center bold>PENDAHULUAN</w:p>
Sesudah : <w:p Judul1>BAB 1<w:br/>PENDAHULUAN</w:p>

Tampilan di halaman tetap dua baris (Judul1 = TNR 12 tebal rata tengah, sama
persis dengan format paragraf judul yang digabungkan). Pola ini mengikuti
skripsi pasca-sidang Thoriq yang memakai satu heading + line break.

Pakai: python fix_judul_bab.py [target.docx]
"""
import os
import re
import shutil
import sys
import zipfile
from datetime import datetime

DEFAULT_DOCX = r"C:/Users/Pratama/Downloads/smpro/sempro-hamza-1.docx"
WORK = os.path.dirname(os.path.abspath(__file__))

JUDUL_BAB = {
    'BAB 1': 'PENDAHULUAN',
    'BAB 2': 'KAJIAN PUSTAKA',
    'BAB 3': 'METODE PENELITIAN',
    'BAB 4': 'HASIL DAN PEMBAHASAN',
    'BAB 5': 'PENUTUP',
}

P_RE = re.compile(r'<w:p\b[^>]*>.*?</w:p>', re.S)
T_RE = re.compile(r'<w:t(?: [^>]*)?>(.*?)</w:t>', re.S)


def text_of(p):
    return ''.join(T_RE.findall(p))


def main():
    docx = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DOCX
    with zipfile.ZipFile(docx) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}
    d = blobs['word/document.xml'].decode('utf-8')

    spans = [(m.start(), m.end(), m.group(0)) for m in P_RE.finditer(d)]
    edits = []          # (start_heading, end_title, xml_baru)
    for i, (s, e, p) in enumerate(spans):
        if 'w:val="Judul1"' not in p:
            continue
        bab = text_of(p).strip()
        if bab not in JUDUL_BAB:
            continue
        s2, e2, p2 = spans[i + 1]
        judul = text_of(p2).strip()
        assert judul == JUDUL_BAB[bab], (bab, judul)
        assert 'pStyle' not in p2, 'paragraf judul ternyata bergaya khusus: ' + bab
        assert s2 == e, 'ada isi lain di antara heading dan judul bab: ' + bab

        # Word mengubah <w:br/> menjadi SPASI saat menyusun entri DAFTAR ISI,
        # jadi run spasi di ekor heading ("BAB 1 ") harus dibuang dulu supaya
        # entri tidak menjadi "BAB 1  PENDAHULUAN" (spasi ganda).
        head = p[:-len('</w:p>')]
        head = re.sub(r'<w:r\b[^>]*>(?:(?!</w:r>).)*?<w:t[^>]*>\s+</w:t>'
                      r'</w:r>\s*$', '', head, flags=re.S)

        run = ('<w:r><w:br/><w:t xml:space="preserve">%s</w:t></w:r>' % judul)
        baru = head + run + '</w:p>'
        edits.append((s, e2, baru))

    assert len(edits) == 5, 'harus 5 bab, dapat %d' % len(edits)

    for s, e, baru in reversed(edits):
        d = d[:s] + baru + d[e:]

    for bab, judul in JUDUL_BAB.items():
        assert d.count('<w:t>%s</w:t>' % judul) + \
            d.count('<w:t xml:space="preserve">%s</w:t>' % judul) >= 1, judul

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    if os.path.abspath(docx) == os.path.abspath(DEFAULT_DOCX):
        shutil.copy2(docx, os.path.join(WORK,
                     'sempro-hamza-1_BACKUP_%s.docx' % ts))

    tmp = docx + '.tmp'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, d.encode('utf-8')
                       if n == 'word/document.xml' else blobs[n])
    os.replace(tmp, docx)
    print('OK 5 judul bab digabung ->', docx)


if __name__ == '__main__':
    main()
