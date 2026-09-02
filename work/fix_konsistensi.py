# -*- coding: utf-8 -*-
"""Tiga perbaikan konsistensi metode-hasil:

A. BAB 3 Skenario Pengujian  : tambahkan metode ablasi komponen arsitektur + baris R10/R11.
B. BAB 4 Uji Signifikansi    : laporkan uji Wilcoxon (dijanjikan BAB 3 & abstrak, belum pernah dilaporkan).
C. BAB 3 Model Evaluation    : nyatakan analisis F1 per kelas sebagai bagian metode.

Angka Wilcoxon dihitung ulang dari data per-seed; paired t-test hasil hitung ulang
cocok persis dengan Tabel 4.4 sehingga sumber datanya terverifikasi.
"""
import zipfile, re, os, shutil, datetime

SRC = r"C:/Users/Pratama/Downloads/smpro/sempro-hamza-1.docx"
HERE = os.path.dirname(os.path.abspath(__file__))

RPR = ('<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
       'w:cs="Times New Roman"/><w:sz w:val="24"/><w:szCs w:val="24"/>')
CRPR = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'

_pid = [0x50000000]
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

def trow(vals, widths):
    cells = ''
    for j, v in enumerate(vals):
        al = "left" if j in (1, 3) else "center"
        cells += (f'<w:tc><w:tcPr><w:tcW w:w="{widths[j]}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr>'
                  f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="00000000">'
                  f'<w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="{al}"/>'
                  f'<w:rPr>{CRPR}</w:rPr></w:pPr>'
                  f'<w:r><w:rPr>{CRPR}</w:rPr><w:t xml:space="preserve">{esc(v)}</w:t></w:r></w:p></w:tc>')
    return (f'<w:tr w:rsidR="004B5458" w14:paraId="{pid()}" w14:textId="77777777">'
            f'<w:trPr><w:jc w:val="center"/></w:trPr>{cells}</w:tr>')

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
bak = os.path.join(HERE, f"sempro-hamza-1_BACKUP_{ts}.docx")
shutil.copy2(SRC, bak)
print("backup ->", bak)

zin = zipfile.ZipFile(SRC, "r")
doc = zin.read("word/document.xml").decode("utf-8")

# ---------------------------------------------------------------- A. baris R10/R11
tbl_m = None
for m in re.finditer(r'<w:tbl>.*?</w:tbl>', doc, re.S):
    first = re.search(r'<w:tr[ >].*?</w:tr>', m.group(0), re.S)
    if first and '<w:t>Kode</w:t>' in first.group(0):
        tbl_m = m
        break
assert tbl_m, "tabel Skenario Pengujian tidak ditemukan"
tbl = tbl_m.group(0)
assert 'R10' not in tbl, "R10 sudah ada - dibatalkan"
widths = [int(w) for w in re.findall(r'<w:gridCol w:w="(\d+)"/>', tbl)]
assert len(widths) == 4, widths

new_rows = (trow(["R10", "WasteNet-128K", "\u2013 (Cross-Entropy)", "Ablasi komponen arsitektur 128K"], widths) +
            trow(["R11", "WasteNet-256K", "\u2013 (Cross-Entropy)", "Ablasi komponen arsitektur 256K"], widths))
tbl_new = tbl.replace("</w:tbl>", new_rows + "</w:tbl>", 1)
doc = doc[:tbl_m.start()] + tbl_new + doc[tbl_m.end():]
print("A1 -> baris R10 & R11 ditambahkan ke tabel skenario")

# ---------------------------------------------------------------- A. prosa metode ablasi
anchor_a = ("Setiap skenario dilatih pada dua skema label, yaitu K6")
ms = list(re.finditer(r'<w:p [^>]*>.*?</w:p>', doc, re.S))
def ptxt(m):
    return ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', m.group(0))).strip()
cands = [m for m in ms if ptxt(m).startswith(anchor_a)]
assert len(cands) == 1, "jangkar prosa ablasi cocok %d" % len(cands)

abl_prose = (
    P("Selain kesembilan skenario distilasi tersebut, dilakukan pula ablasi komponen arsitektur pada "
      "kedua varian student, yaitu R10 untuk WasteNet-128K dan R11 untuk WasteNet-256K. Ablasi ini "
      "bertujuan mengetahui komponen rancangan mana yang paling menentukan akurasi. Komponen yang "
      "divariasikan meliputi fungsi aktivasi, expand ratio pada blok inverted residual, koneksi residual, "
      "dan lapisan tersembunyi pada pengklasifikasi. Koneksi residual hanya diablasi pada varian 256K, "
      "sebab pada varian 128K tidak terdapat blok dengan dimensi masukan dan keluaran yang sama sehingga "
      "koneksi tersebut tidak pernah aktif.") +
    P("Ablasi dijalankan pada skema K6 saja, sebab arsitektur WasteNet identik pada kedua skema label dan "
      "hanya berbeda pada jumlah neuron keluarannya sehingga kepentingan relatif antarkomponen tidak "
      "berubah. Setiap varian terdiri atas empat konfigurasi, yaitu baseline yang dilatih ulang beserta "
      "tiga variasi komponen, dan masing-masing dijalankan dengan lima seed menggunakan pengaturan "
      "pelatihan yang identik dengan baseline cross-entropy, sehingga seluruhnya berjumlah 40 kali "
      "pelatihan. Perlu dicatat bahwa variasi expand ratio dan lapisan tersembunyi mengubah jumlah "
      "parameter, sehingga jumlah parameter setiap konfigurasi dilaporkan bersama akurasinya agar "
      "pengaruh komponen tidak tertukar dengan pengaruh berkurangnya kapasitas."))
pos = cands[0].end()
doc = doc[:pos] + abl_prose + doc[pos:]
print("A2 -> prosa metode ablasi ditambahkan ke BAB 3")

# ---------------------------------------------------------------- C. analisis per kelas
# kalimat sumber terpecah antar-run, jadi tambahkan run baru di akhir paragrafnya
ms = list(re.finditer(r'<w:p [^>]*>.*?</w:p>', doc, re.S))
cands = [m for m in ms if 'macro F1' in ptxt(m) and ptxt(m).endswith('seperti kelas trash.')]
assert len(cands) == 1, "paragraf macro F1 cocok %d" % len(cands)
kal_c = (" Selain metrik agregat tersebut, nilai F1 setiap kelas juga ditelaah secara terpisah untuk "
         "mengetahui pada kelas mana perbaikan akibat distilasi terjadi, sebab perbaikan yang "
         "terkonsentrasi pada kelas-kelas yang saling menyerupai merupakan indikasi berpindahnya "
         "informasi kemiripan antarkelas dari guru ke student.")
run_c = f'<w:r><w:rPr>{RPR}</w:rPr><w:t xml:space="preserve">{esc(kal_c)}</w:t></w:r>'
blk = cands[0].group(0)
doc = doc[:cands[0].start()] + blk[:-len('</w:p>')] + run_c + '</w:p>' + doc[cands[0].end():]
print("C  -> analisis F1 per kelas dinyatakan di BAB 3")

# ---------------------------------------------------------------- B. Wilcoxon di BAB 4
anchor_b = "Seluruh skema lainnya tidak signifikan pada semua uji yang digunakan."
ms = list(re.finditer(r'<w:p [^>]*>.*?</w:p>', doc, re.S))
cands = [m for m in ms if ptxt(m).startswith(anchor_b)]
assert len(cands) == 1, "jangkar Wilcoxon cocok %d" % len(cands)

wil = (
    P("Uji Wilcoxon signed-rank memberikan hasil yang searah namun perlu dibaca dengan memperhatikan "
      "keterbatasannya. Pada skema TA-KD WasteNet-256K nilai p yang diperoleh adalah 0,062 untuk kedua "
      "skema label, sedangkan skema lainnya berkisar antara 0,273 hingga 1,000. Perlu dipahami bahwa "
      "dengan lima seed, nilai p dua sisi terkecil yang mungkin dihasilkan uji Wilcoxon adalah 0,0625, "
      "yaitu ketika seluruh pasangan bergerak searah. Dengan demikian nilai 0,062 pada skema TA-KD "
      "merupakan hasil terkuat yang dapat dicapai uji tersebut pada jumlah seed ini, dan angka itu muncul "
      "justru karena skema tersebut unggul pada kelima seed. Uji Wilcoxon di sini berfungsi memastikan "
      "arah dan konsistensi perbaikan, bukan menetapkan taraf signifikansi, sebab resolusinya memang "
      "terbatas pada ukuran sampel sekecil ini."))
pos = cands[0].start()
doc = doc[:pos] + wil + doc[pos:]
print("B  -> hasil uji Wilcoxon dilaporkan di BAB 4")

tmp = SRC + ".tmp"
zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    data = zin.read(item.filename)
    if item.filename == "word/document.xml":
        data = doc.encode("utf-8")
    zout.writestr(item, data)
zin.close(); zout.close()
os.replace(tmp, SRC)
print("\nselesai")
