# -*- coding: utf-8 -*-
"""Hilangkan cetak tebal pada caption (style Keterangan) dan pada entri
DAFTAR TABEL / DAFTAR GAMBAR (style TabelGambar), mengikuti konvensi skripsi
pembanding dari pembimbing yang sama.

Caption di badan tulisan adalah sumber format bagi field daftar, sehingga
menghilangkan tebal di caption membuat daftar tetap rapi walau field di-refresh.
"""
import zipfile, re, os, shutil, datetime

SRC = r"C:/Users/Pratama/Downloads/smpro/sempro-hamza-1.docx"
HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_STYLES = ("Keterangan", "TabelGambar")

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
bak = os.path.join(HERE, f"sempro-hamza-1_BACKUP_{ts}.docx")
shutil.copy2(SRC, bak)
print("backup ->", bak)

zin = zipfile.ZipFile(SRC, "r")
doc = zin.read("word/document.xml").decode("utf-8")

def strip_bold(block):
    """Buang <w:b/> dan <w:bCs/> (tanpa w:val, atau w:val bernilai benar)."""
    out = re.sub(r'<w:b(?:Cs)?(?:\s+w:val="(?:1|true|on)")?\s*/>', '', block)
    return out

paras = list(re.finditer(r'<w:p [^>]*>.*?</w:p>', doc, re.S))
changed = 0
pieces = []
last = 0
for m in paras:
    blk = m.group(0)
    st = re.search(r'<w:pStyle w:val="([^"]+)"', blk)
    if st and st.group(1) in TARGET_STYLES:
        new = strip_bold(blk)
        if new != blk:
            pieces.append(doc[last:m.start()])
            pieces.append(new)
            last = m.end()
            changed += 1
pieces.append(doc[last:])
doc = ''.join(pieces)
print("paragraf caption/daftar yang dibersihkan:", changed)

# verifikasi: tidak ada lagi <w:b/> di paragraf bergaya target
check = [m.group(0) for m in re.finditer(r'<w:p [^>]*>.*?</w:p>', doc, re.S)
         if (lambda s: s and s.group(1) in TARGET_STYLES)(re.search(r'<w:pStyle w:val="([^"]+)"', m.group(0)))]
sisa = sum(1 for b in check if re.search(r'<w:b(?:Cs)?(?:\s+w:val="(?:1|true|on)")?\s*/>', b))
assert sisa == 0, "masih ada %d paragraf tebal" % sisa
print("verifikasi: 0 caption/entri daftar yang masih tebal (dari %d paragraf)" % len(check))

tmp = SRC + ".tmp"
zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    data = zin.read(item.filename)
    if item.filename == "word/document.xml":
        data = doc.encode("utf-8")
    zout.writestr(item, data)
zin.close(); zout.close()
os.replace(tmp, SRC)
print("selesai")
