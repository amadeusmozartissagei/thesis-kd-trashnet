# -*- coding: utf-8 -*-
"""Sisipkan subbab 'Perbandingan dengan Penelitian Terdahulu' sebelum 'Pembahasan'.

Angka penelitian terdahulu diambil dari Tabel 5 Zheng dkk. (2023),
Resource/Focus-RCNet ....pdf (terverifikasi langsung dari berkas paper).
"""
import zipfile, re, os, shutil, datetime

SRC = r"C:/Users/Pratama/Downloads/smpro/sempro-hamza-1.docx"
HERE = os.path.dirname(os.path.abspath(__file__))
NUMID_H2 = "40"

RPR = ('<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
       'w:cs="Times New Roman"/><w:sz w:val="24"/><w:szCs w:val="24"/>')
CRPR = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'

_pid = [0x4F000000]
def pid():
    _pid[0] += 1
    return "%08X" % _pid[0]

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def P(text):
    return (f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="00000000">'
            f'<w:pPr><w:spacing w:line="276" w:lineRule="auto"/><w:ind w:firstLine="426"/><w:jc w:val="both"/>'
            f'<w:rPr>{RPR}</w:rPr></w:pPr>'
            f'<w:r><w:rPr>{RPR}</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')

def H2(text):
    return (f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="00000000">'
            f'<w:pPr><w:pStyle w:val="Judul2"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="{NUMID_H2}"/></w:numPr>'
            f'<w:spacing w:after="0" w:line="276" w:lineRule="auto"/><w:ind w:left="426" w:hanging="426"/></w:pPr>'
            f'<w:r><w:t>{esc(text)}</w:t></w:r></w:p>')

def CAP(text, n):
    return (f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="00000000">'
            f'<w:pPr><w:pStyle w:val="Keterangan"/><w:keepNext/>'
            f'<w:spacing w:after="0" w:line="276" w:lineRule="auto"/></w:pPr>'
            f'<w:r><w:t xml:space="preserve">Tabel 4.</w:t></w:r>'
            f'<w:fldSimple w:instr=" SEQ Tabel \\* ARABIC \\s 1 ">'
            f'<w:r><w:rPr><w:noProof/></w:rPr><w:t>{n}</w:t></w:r></w:fldSimple>'
            f'<w:r><w:t xml:space="preserve"> {esc(text)}</w:t></w:r></w:p>')

def cell(text, w, header=False, align="center"):
    b = '<w:b/><w:bCs/>' if header else ''
    rpr = CRPR + b
    return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr>'
            f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="00000000">'
            f'<w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="{align}"/>'
            f'<w:rPr>{rpr}</w:rPr></w:pPr>'
            f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p></w:tc>')

def TBL(widths, rows):
    grid = ''.join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    out = ['<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/><w:jc w:val="center"/><w:tblBorders>'
           '<w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
           '<w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
           '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
           '<w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
           '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
           '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/></w:tblBorders>'
           '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" '
           'w:noHBand="0" w:noVBand="1"/></w:tblPr><w:tblGrid>' + grid + '</w:tblGrid>']
    for i, r in enumerate(rows):
        hdr = (i == 0)
        trpr = ('<w:trPr><w:tblHeader/><w:jc w:val="center"/></w:trPr>' if hdr
                else '<w:trPr><w:jc w:val="center"/></w:trPr>')
        cells = ''.join(cell(v, widths[j], header=hdr,
                             align=("left" if (j == 0 and not hdr) else "center"))
                        for j, v in enumerate(r))
        out.append(f'<w:tr w:rsidR="004B5458" w14:paraId="{pid()}" w14:textId="77777777">{trpr}{cells}</w:tr>')
    out.append('</w:tbl>')
    out.append(f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="00000000">'
               f'<w:pPr><w:spacing w:after="0" w:line="276" w:lineRule="auto"/><w:rPr>{RPR}</w:rPr></w:pPr></w:p>')
    return ''.join(out)

W = [2200, 1750, 1150, 1400, 1427]

body = []
A = body.append

A(H2("Perbandingan dengan Penelitian Terdahulu"))
A(P("Bagian ini menempatkan hasil penelitian terhadap penelitian terdahulu yang menggunakan dataset "
    "TrashNet. Angka penelitian terdahulu dikutip dari Zheng dkk. (2023) yang melaporkan perbandingan "
    "beberapa arsitektur pada dataset yang sama, sedangkan angka penelitian ini diambil dari skema enam "
    "kelas agar sebanding dengan jumlah kelas yang digunakan pada penelitian tersebut. Perbandingan "
    "selengkapnya disajikan pada Tabel 4.10."))
A(CAP("Perbandingan dengan penelitian terdahulu pada dataset TrashNet", 10))
A(TBL(W, [
    ["Model", "Sumber", "Akurasi", "Parameter", "FLOPs"],
    ["ResNet50", "Zheng dkk. (2023)", "0,87", "25,560 juta", "12,087 G"],
    ["DenseNet121", "Zheng dkk. (2023)", "0,88", "6,960 juta", "8,121 G"],
    ["MobileNetV1", "Zheng dkk. (2023)", "0,85", "2,232 juta", "0,952 G"],
    ["ShuffleNetV1", "Zheng dkk. (2023)", "0,86", "0,348 juta", "0,127 G"],
    ["EfficientNet-B4", "Zheng dkk. (2023)", "0,97", "17,559 juta", "4,490 G"],
    ["Focus-RCNet + KD", "Zheng dkk. (2023)", "0,92", "0,526 juta", "0,419 G"],
    ["EfficientNet-B4", "Penelitian ini", "0,9446", "17,56 juta", "—"],
    ["Focus-RCNet", "Penelitian ini", "0,8507", "0,521 juta", "—"],
    ["WasteNet-256K, TA-KD", "Penelitian ini", "0,7873", "0,256 juta", "0,084 G"],
    ["WasteNet-128K, CE", "Penelitian ini", "0,7652", "0,128 juta", "0,051 G"],
]))
A(P("Dari sisi efisiensi, model usulan penelitian ini berada pada tingkat yang lebih hemat daripada "
    "seluruh model pembanding. WasteNet-256K hanya memerlukan 0,256 juta parameter dan sekitar 0,084 G "
    "FLOPs, yaitu kurang dari separuh parameter dan sekitar lima kali lebih sedikit beban komputasi "
    "dibandingkan Focus-RCNet. Bahkan terhadap model pembanding yang paling ringan, yaitu ShuffleNetV1 "
    "dengan 0,348 juta parameter dan 0,127 G FLOPs, model usulan tetap lebih kecil. Dengan demikian "
    "penelitian ini bekerja pada rentang kapasitas yang lebih rendah daripada seluruh penelitian "
    "pembanding tersebut."))
A(P("Dari sisi akurasi, model usulan berada di bawah penelitian terdahulu. Focus-RCNet melaporkan "
    "akurasi 0,92, sedangkan WasteNet-256K hasil TA-KD mencapai 0,7873. Perbedaan ini perlu dibaca "
    "secara hati-hati karena dua hal. Pertama, terdapat perbedaan kelas kapasitas yang besar, sebab "
    "model usulan dirancang pada rentang 128 ribu hingga 256 ribu parameter yang tidak dijangkau oleh "
    "model pembanding mana pun. Kedua, protokol evaluasinya berbeda. Zheng dkk. (2023) memperbesar "
    "dataset melalui augmentasi dan melaporkan satu angka hasil pengujian, sedangkan penelitian ini "
    "mempertahankan ukuran dataset asli sebanyak 2.521 citra dan melaporkan rata-rata lima seed dengan "
    "pembagian data yang berbeda pada setiap seed. Protokol yang kedua cenderung menghasilkan angka "
    "yang lebih rendah namun lebih konservatif."))
A(P("Indikasi perbedaan protokol tersebut juga terlihat pada model pembanding yang direimplementasikan "
    "dalam penelitian ini. Focus-RCNet yang dilatih ulang menggunakan protokol penelitian ini mencapai "
    "0,8507, lebih rendah daripada 0,92 yang dilaporkan penulis aslinya, sedangkan EfficientNet-B4 "
    "mencapai 0,9446 dibandingkan 0,97. Selisih yang muncul pada kedua model tersebut menunjukkan bahwa "
    "perbedaan angka lebih banyak bersumber dari protokol evaluasi daripada dari implementasi modelnya, "
    "sehingga perbandingan langsung antarangka mentah tidak sepenuhnya setara."))
A(P("Perlu ditegaskan bahwa tujuan penelitian ini bukan mengejar akurasi tertinggi pada dataset "
    "TrashNet, melainkan menganalisis pengaruh kapasitas guru terhadap keberhasilan knowledge "
    "distillation pada student yang sangat kecil. Penelitian terdahulu umumnya berhenti pada kapasitas "
    "ratusan ribu hingga jutaan parameter dan tidak menguji bagaimana pemilihan guru memengaruhi "
    "student pada rentang di bawahnya. Pada rentang tersebut penelitian ini menunjukkan bahwa pemilihan "
    "guru berkapasitas menengah dapat meningkatkan akurasi secara signifikan tanpa menambah satu pun "
    "parameter, sehingga kontribusinya bersifat melengkapi dan bukan menggantikan penelitian terdahulu."))

BLOCK = ''.join(body)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
bak = os.path.join(HERE, f"sempro-hamza-1_BACKUP_{ts}.docx")
shutil.copy2(SRC, bak)
print("backup ->", bak)

zin = zipfile.ZipFile(SRC, "r")
doc = zin.read("word/document.xml").decode("utf-8")

assert "Perbandingan dengan Penelitian Terdahulu" not in doc, "subbab sudah ada - dibatalkan"

ms = list(re.finditer(r'<w:p [^>]*>.*?</w:p>', doc, re.S))
def ptxt(p):
    return ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p)).strip()
cands = [m for m in ms if ptxt(m.group(0)) == "Pembahasan" and '<w:pStyle w:val="Judul2"/>' in m.group(0)]
assert len(cands) == 1, "heading 'Pembahasan' cocok %d (harus 1)" % len(cands)

doc = doc[:cands[0].start()] + BLOCK + doc[cands[0].start():]

tmp = SRC + ".tmp"
zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    data = zin.read(item.filename)
    if item.filename == "word/document.xml":
        data = doc.encode("utf-8")
    zout.writestr(item, data)
zin.close(); zout.close()
os.replace(tmp, SRC)
print("subbab 'Perbandingan dengan Penelitian Terdahulu' disisipkan sebelum 'Pembahasan'")
