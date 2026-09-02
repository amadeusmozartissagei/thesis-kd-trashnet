# -*- coding: utf-8 -*-
"""Sisipkan gambar BAB 4 ke sempro-hamza-1.docx.

Pola mengikuti dokumen: paragraf gambar (inline drawing, rata tengah) lalu
caption di BAWAHnya memakai style Keterangan + field SEQ Gambar (\\s 1).
Penomoran Gambar 4.x mengikuti urutan kemunculan di dokumen.
"""
import zipfile, re, os, shutil, datetime
from PIL import Image

SRC = r"C:/Users/Pratama/Downloads/smpro/sempro-hamza-1.docx"
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
EMU_IN = 914400
MAX_W_IN = 5.2

RPR = ('<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
       'w:cs="Times New Roman"/><w:sz w:val="24"/><w:szCs w:val="24"/>')

_pid = [0x4C000000]
def pid():
    _pid[0] += 1
    return "%08X" % _pid[0]

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# urut sesuai posisi di dokumen -> menentukan nomor Gambar 4.x
PLAN = [
    ("Seluruh skenario yang diuji menggunakan partisi yang sama", "after",
     "gambar_4_1_distribusi_kelas.png",
     "Distribusi Jumlah Citra per Kelas dan per Subset Data"),
    ("Distilasi Langsung dari Guru Besar", "before",
     "gambar_4_3_perbandingan_skenario.png",
     "Perbandingan Akurasi Seluruh Skenario Pelatihan Student"),
    ("Perlu ditegaskan bahwa pembanding yang digunakan", "after",
     "gambar_4_2_kurva_pelatihan.png",
     "Kurva Loss dan Akurasi Pelatihan Skema TA-KD pada WasteNet-256K"),
    ("Perbaikan tidak tersebar merata", "after",
     "gambar_4_6_delta_f1_per_kelas.png",
     "Selisih F1 per Kelas pada Skema TA-KD terhadap Kontrolnya"),
    ("Sebaliknya, kelas trash pada skema K6", "after",
     "gambar_4_5_confusion_matrix.png",
     "Confusion Matrix Data Uji (kiri: kontrol cross-entropy, kanan: TA-KD)"),
    ("Temuan lain berkaitan dengan expand ratio", "after",
     "gambar_4_7_ablasi.png",
     "Hasil Ablasi Komponen Arsitektur pada Kedua Varian WasteNet"),
    ("Apabila ketiga jenis guru diurutkan berdasarkan kapasitasnya", "after",
     "gambar_4_4_kurva_capacity_gap.png",
     "Perubahan Akurasi Student menurut Kapasitas Guru"),
]


def drawing_par(rid, cx, cy, name):
    return (
        f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="004B5458">'
        f'<w:pPr><w:keepNext/><w:spacing w:before="120" w:after="60" w:line="276" w:lineRule="auto"/>'
        f'<w:jc w:val="center"/><w:rPr>{RPR}</w:rPr></w:pPr>'
        f'<w:r><w:rPr>{RPR}<w:noProof/></w:rPr><w:drawing>'
        f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{_pid[0] & 0x7FFFFFF}" name="{name}"/>'
        f'<wp:cNvGraphicFramePr><a:graphicFrameLocks '
        f'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/>'
        f'</wp:cNvGraphicFramePr>'
        f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        f'<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:nvPicPr><pic:cNvPr id="0" name="{name}"/>'
        f'<pic:cNvPicPr><a:picLocks noChangeAspect="1" noChangeArrowheads="1"/></pic:cNvPicPr>'
        f'</pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:srcRect/>'
        f'<a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr bwMode="auto"><a:xfrm><a:off x="0" y="0"/>'
        f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/>'
        f'<a:ln><a:noFill/></a:ln></pic:spPr></pic:pic></a:graphicData></a:graphic>'
        f'</wp:inline></w:drawing></w:r></w:p>')


def caption_par(text, n):
    return (
        f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="00000000">'
        f'<w:pPr><w:pStyle w:val="Keterangan"/>'
        f'<w:spacing w:after="160" w:line="276" w:lineRule="auto"/><w:jc w:val="center"/>'
        f'<w:rPr><w:b/><w:bCs/></w:rPr></w:pPr>'
        f'<w:r><w:rPr><w:b/><w:bCs/></w:rPr><w:t xml:space="preserve">Gambar 4.</w:t></w:r>'
        f'<w:fldSimple w:instr=" SEQ Gambar \\* ARABIC \\s 1 ">'
        f'<w:r><w:rPr><w:noProof/></w:rPr><w:t>{n}</w:t></w:r></w:fldSimple>'
        f'<w:r><w:rPr><w:b/><w:bCs/></w:rPr><w:t xml:space="preserve"> {esc(text)}</w:t></w:r></w:p>')


ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
bak = os.path.join(HERE, f"sempro-hamza-1_BACKUP_{ts}.docx")
shutil.copy2(SRC, bak)
print("backup ->", bak)

zin = zipfile.ZipFile(SRC, "r")
doc = zin.read("word/document.xml").decode("utf-8")
rels = zin.read("word/_rels/document.xml.rels").decode("utf-8")
existing = {n: zin.read(n) for n in zin.namelist()}

assert "SEQ Gambar" in doc
assert 'name="Gambar BAB4' not in doc, "gambar BAB 4 sudah disisipkan - dibatalkan"

used_rid = max(int(i) for i in re.findall(r'Id="rId(\d+)"', rels))
used_img = max([int(m) for m in re.findall(r'word/media/image(\d+)\.', " ".join(existing))] or [0])

new_media = {}
new_rels = []

for k, (anchor, where, fname, cap) in enumerate(PLAN, start=1):
    path = os.path.join(FIGS, fname)
    assert os.path.exists(path), path
    with Image.open(path) as im:
        pw, ph = im.size
    w_in = min(MAX_W_IN, pw / 200.0)
    cx = int(w_in * EMU_IN)
    cy = int(cx * ph / pw)

    used_rid += 1
    used_img += 1
    rid = "rId%d" % used_rid
    media_name = "word/media/image%d.png" % used_img
    new_media[media_name] = open(path, "rb").read()
    new_rels.append(
        f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
        f'officeDocument/2006/relationships/image" Target="media/image{used_img}.png"/>')

    block = drawing_par(rid, cx, cy, "Gambar BAB4 %d" % k) + caption_par(cap, k)

    ms = list(re.finditer(r'<w:p [^>]*>.*?</w:p>|<w:p/>', doc, re.S))
    def ptxt(p):
        return ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p))
    cands = [m for m in ms if ptxt(m.group(0)).strip().startswith(anchor)]
    assert len(cands) == 1, "anchor %r cocok %d paragraf" % (anchor, len(cands))
    pos = cands[0].end() if where == "after" else cands[0].start()
    doc = doc[:pos] + block + doc[pos:]
    print(f"Gambar 4.{k}: {fname}  ({w_in:.2f} in)  -> {where} {anchor[:38]!r}")

rels = rels.replace("</Relationships>", "".join(new_rels) + "</Relationships>")

tmp = SRC + ".tmp"
zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    data = existing[item.filename]
    if item.filename == "word/document.xml":
        data = doc.encode("utf-8")
    elif item.filename == "word/_rels/document.xml.rels":
        data = rels.encode("utf-8")
    zout.writestr(item, data)
for name, data in new_media.items():
    zout.writestr(name, data)
zin.close()
zout.close()
os.replace(tmp, SRC)
print("\n%d gambar disisipkan" % len(PLAN))
