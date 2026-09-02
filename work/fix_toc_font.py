# -*- coding: utf-8 -*-
"""Kunci font DAFTAR ISI / DAFTAR TABEL / DAFTAR GAMBAR ke Times New Roman 12pt.

Default dokumen adalah Calibri 11pt (tema minorHAnsi). Style TOC1/TOC2/TOC3 hanya
menetapkan Times New Roman pada w:cs (complex script), sedangkan TabelGambar tidak
punya rPr sama sekali. Akibatnya setiap kali field di-refresh (Ctrl+A, F9) Word
membuat ulang entri mengikuti style tersebut sehingga tampil Calibri.

Perbaikan dilakukan di level STYLE agar tahan terhadap refresh berikutnya.
"""
import zipfile, re, os, shutil, datetime

SRC = r"C:/Users/Pratama/Downloads/smpro/sempro-hamza-1.docx"
HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = ("TOC1", "TOC2", "TOC3", "TabelGambar")
FONTS = ('<w:rFonts w:ascii="Times New Roman" w:eastAsiaTheme="minorEastAsia" '
         'w:hAnsi="Times New Roman" w:cs="Times New Roman"/>')
SIZE = '<w:sz w:val="24"/><w:szCs w:val="24"/>'

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
bak = os.path.join(HERE, f"sempro-hamza-1_BACKUP_{ts}.docx")
shutil.copy2(SRC, bak)
print("backup ->", bak)

zin = zipfile.ZipFile(SRC, "r")
styles = zin.read("word/styles.xml").decode("utf-8")

for sid in TARGETS:
    m = re.search(r'<w:style [^>]*w:styleId="%s".*?</w:style>' % sid, styles, re.S)
    assert m, "style %s tidak ditemukan" % sid
    block = m.group(0)
    new = block

    if "<w:rPr>" in new:
        # ganti rFonts yang ada, lalu tambahkan ukuran bila belum ada
        new = re.sub(r'<w:rFonts[^>]*/>', FONTS, new, count=1)
        if "<w:sz " not in new:
            new = new.replace("</w:rPr>", SIZE + "</w:rPr>", 1)
    else:
        # belum punya rPr: sisipkan setelah pPr (urutan skema: pPr lalu rPr)
        assert "</w:pPr>" in new, "%s tidak punya pPr" % sid
        new = new.replace("</w:pPr>", "</w:pPr><w:rPr>" + FONTS + SIZE + "</w:rPr>", 1)

    assert new != block, "tidak ada perubahan pada %s" % sid
    styles = styles.replace(block, new, 1)
    print("OK ->", sid)

# verifikasi
for sid in TARGETS:
    b = re.search(r'<w:style [^>]*w:styleId="%s".*?</w:style>' % sid, styles, re.S).group(0)
    assert 'w:ascii="Times New Roman"' in b and '<w:sz w:val="24"/>' in b, sid

tmp = SRC + ".tmp"
zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    data = zin.read(item.filename)
    if item.filename == "word/styles.xml":
        data = styles.encode("utf-8")
    zout.writestr(item, data)
zin.close(); zout.close()
os.replace(tmp, SRC)
print("\nstyle daftar isi/tabel/gambar dikunci ke Times New Roman 12pt")
