# -*- coding: utf-8 -*-
"""Sisipkan BAB 5 PENUTUP (5.1 Simpulan, 5.2 Saran) ke sempro-hamza-1.docx.

numId baru: 44 = Judul2 "5.%1"; 42 = daftar Simpulan; 43 = daftar Saran.
"""
import zipfile, re, os, shutil, datetime, random

SRC = r"C:/Users/Pratama/Downloads/smpro/sempro-hamza-1.docx"
HERE = os.path.dirname(os.path.abspath(__file__))
NUM_H2, NUM_SIMP, NUM_SARAN = "44", "42", "43"

RPR = ('<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
       'w:cs="Times New Roman"/><w:sz w:val="24"/><w:szCs w:val="24"/>')

_pid = [0x4D000000]
def pid():
    _pid[0] += 1
    return "%08X" % _pid[0]

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def P(text, indent=True):
    ind = '<w:ind w:firstLine="426"/>' if indent else ''
    return (f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="00000000">'
            f'<w:pPr><w:spacing w:line="276" w:lineRule="auto"/>{ind}<w:jc w:val="both"/>'
            f'<w:rPr>{RPR}</w:rPr></w:pPr>'
            f'<w:r><w:rPr>{RPR}</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')

def H2(text):
    return (f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="00000000">'
            f'<w:pPr><w:pStyle w:val="Judul2"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="{NUM_H2}"/></w:numPr>'
            f'<w:spacing w:after="0" w:line="276" w:lineRule="auto"/><w:ind w:left="426" w:hanging="426"/></w:pPr>'
            f'<w:r><w:t>{esc(text)}</w:t></w:r></w:p>')

def LI(text, numid):
    return (f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="00000000">'
            f'<w:pPr><w:pStyle w:val="DaftarParagraf"/>'
            f'<w:numPr><w:ilvl w:val="0"/><w:numId w:val="{numid}"/></w:numPr>'
            f'<w:spacing w:after="0" w:line="276" w:lineRule="auto"/><w:jc w:val="both"/>'
            f'<w:rPr>{RPR}</w:rPr></w:pPr>'
            f'<w:r><w:rPr>{RPR}</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')

body = []
A = body.append

A(H2("Simpulan"))
A(P("Berdasarkan hasil penelitian dan pembahasan mengenai penerapan knowledge distillation "
    "berbasis teacher assistant pada model ringan WasteNet untuk klasifikasi citra sampah "
    "TrashNet, dapat disimpulkan sebagai berikut:"))
A(LI("Model ringan WasteNet berhasil dibangun dan dilatih menggunakan knowledge distillation "
     "dalam dua varian kapasitas, yaitu WasteNet-128K dengan 128.086 parameter dan beban "
     "komputasi sekitar 51 juta FLOPs, serta WasteNet-256K dengan 256.002 parameter dan sekitar "
     "84 juta FLOPs pada resolusi masukan 160 × 160 piksel. Model terbaik diperoleh melalui skema "
     "teacher assistant pada varian WasteNet-256K dengan akurasi 0,7873 pada skema enam kelas dan "
     "0,8235 pada skema lima kelas. Model tersebut memiliki jumlah parameter sekitar 69 kali lebih "
     "sedikit daripada guru besarnya, namun tetap mempertahankan sekitar 83 persen dari akurasi "
     "guru tersebut, sehingga layak diterapkan pada perangkat dengan sumber daya terbatas.", NUM_SIMP))
A(LI("Kapasitas guru terbukti menentukan keberhasilan distilasi. Distilasi langsung dari guru besar "
     "EfficientNet-B4 tidak memberikan perbaikan, yaitu −0,05 poin persentase pada WasteNet-128K dan "
     "−0,47 poin persentase pada WasteNet-256K. Guru berkapasitas sama melalui skema self-distillation "
     "hanya menghasilkan perbaikan 0,47 poin persentase pada skema enam kelas dan 1,40 poin persentase "
     "pada skema lima kelas, dan keduanya tidak signifikan. Sebaliknya, asisten berkapasitas menengah "
     "Focus-RCNet menghasilkan perbaikan 1,00 poin persentase dan 1,40 poin persentase pada kedua skema "
     "label secara signifikan. Dengan demikian jenis guru yang paling efektif bukanlah guru yang paling "
     "besar, melainkan guru berkapasitas menengah yang jarak kapasitasnya terhadap student tidak terlalu "
     "jauh dan tidak terlalu dekat. Selain kapasitas guru, kapasitas student juga menentukan, sebab skema "
     "teacher assistant hanya berhasil pada WasteNet-256K dan tidak pada WasteNet-128K.", NUM_SIMP))
A(LI("Evaluasi statistik terhadap kontrol berpasangan menunjukkan bahwa hanya skema teacher assistant "
     "pada WasteNet-256K yang mencapai taraf signifikansi, dengan nilai p sebesar 0,024 pada skema enam "
     "kelas dan 0,045 pada skema lima kelas, serta unggul pada kelima seed yang diuji. Konfirmasi melalui "
     "uji McNemar menghasilkan nilai khi-kuadrat 6,06 dengan p sebesar 0,014 pada skema lima kelas, "
     "sedangkan pada skema enam kelas nilainya 2,97 dengan p sebesar 0,085 sehingga baru mendekati taraf "
     "signifikansi. Seluruh skema distilasi lainnya tidak signifikan pada semua uji yang digunakan. "
     "Perbaikan yang dihasilkan terkonsentrasi pada kelas glass, metal, dan plastic yang secara visual "
     "saling menyerupai, sehingga mendukung penjelasan bahwa manfaat distilasi berasal dari informasi "
     "kemiripan antarkelas.", NUM_SIMP))
A(LI("Ablasi komponen arsitektur menunjukkan bahwa fungsi aktivasi merupakan komponen yang paling "
     "menentukan, namun pengaruhnya hanya signifikan pada varian berkapasitas kecil. Penggantian SiLU "
     "dengan ReLU menurunkan akurasi WasteNet-128K sebesar 5,33 poin persentase dengan p sebesar 0,004, "
     "sedangkan pada WasteNet-256K tidak ada satu pun komponen yang berpengaruh signifikan. Selain itu "
     "pemangkasan 64 hingga 75 persen parameter melalui penurunan expand ratio tidak menurunkan akurasi, "
     "yang menunjukkan bahwa dimensi tersebut masih berlebih.", NUM_SIMP))

A(H2("Saran"))
A(P("Berdasarkan hasil penelitian ini, terdapat beberapa saran yang dapat menjadi bahan "
    "pertimbangan bagi penelitian selanjutnya:"))
A(LI("Model yang dihasilkan perlu diuji secara langsung pada perangkat dengan sumber daya terbatas "
     "seperti Raspberry Pi, NVIDIA Jetson Nano, atau perangkat berbasis mikrokontroler. Pengujian "
     "tersebut diperlukan untuk mengukur latensi inferensi, penggunaan memori, dan konsumsi daya pada "
     "kondisi penerapan yang sebenarnya, sebab penelitian ini baru mengukur efisiensi secara analitis "
     "melalui jumlah parameter dan FLOPs.", NUM_SARAN))
A(LI("Penelitian selanjutnya disarankan menggunakan data yang lebih banyak dan lebih beragam serta "
     "menambah jumlah seed yang diuji. Jumlah data pada penelitian ini sebanyak 2.521 citra dan jumlah "
     "seed sebanyak lima masih membatasi daya uji statistik, sehingga perbaikan yang kecil berpotensi "
     "tidak terdeteksi meskipun sebenarnya nyata.", NUM_SARAN))
A(LI("Rentang kapasitas asisten perlu ditelusuri lebih lanjut. Penelitian ini baru menguji satu titik "
     "kapasitas asisten, yaitu Focus-RCNet dengan sekitar 520 ribu parameter, sehingga titik kapasitas "
     "yang benar-benar optimal pada kurva capacity gap belum diketahui. Pengujian beberapa asisten dengan "
     "kapasitas yang berbeda akan memperjelas bentuk kurva tersebut.", NUM_SARAN))
A(LI("Skema teacher assistant yang digunakan pada penelitian ini perlu diterapkan pada arsitektur ringan "
     "lain seperti MobileNetV3, ShuffleNetV2, atau GhostNet. Pengujian lintas arsitektur diperlukan untuk "
     "mengetahui apakah temuan mengenai kapasitas guru bersifat umum atau khusus pada arsitektur WasteNet.", NUM_SARAN))
A(LI("Temuan mengenai berlebihnya kapasitas pada dimensi expand dapat dimanfaatkan untuk merancang varian "
     "WasteNet yang lebih hemat parameter. Perancangan ulang tersebut berpotensi menghasilkan model dengan "
     "akurasi setara namun berukuran jauh lebih kecil.", NUM_SARAN))

BAB5 = ''.join(body)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
bak = os.path.join(HERE, f"sempro-hamza-1_BACKUP_{ts}.docx")
shutil.copy2(SRC, bak)
print("backup ->", bak)

zin = zipfile.ZipFile(SRC, "r")
doc = zin.read("word/document.xml").decode("utf-8")
num = zin.read("word/numbering.xml").decode("utf-8")

assert "BAB 5" not in doc, "BAB 5 sudah ada - dibatalkan"

def clone_abs(src_id, new_id, old_txt=None, new_txt=None):
    m = re.search(r'<w:abstractNum w:abstractNumId="%s"[^>]*>.*?</w:abstractNum>' % src_id, num, re.S)
    b = m.group(0)
    b = b.replace('w:abstractNumId="%s"' % src_id, 'w:abstractNumId="%s"' % new_id, 1)
    b = re.sub(r'<w:nsid w:val="[0-9A-Fa-f]+"/>',
               '<w:nsid w:val="%08X"/>' % random.randint(0x10000000, 0x7FFFFFFF), b, count=1)
    b = re.sub(r'w:tplc="[0-9A-Fa-f]+"',
               lambda _: 'w:tplc="%08X"' % random.randint(0x10000000, 0x7FFFFFFF), b)
    if old_txt:
        assert old_txt in b, "lvlText %r tidak ada di abstractNum %s" % (old_txt, src_id)
        b = b.replace(old_txt, new_txt, 1)
    return b

# numId 9 -> abstractNum 15 (daftar bernomor), numId 14 -> abstractNum 2 (Judul2 "3.%1")
abs44 = clone_abs("2", "44", '<w:lvlText w:val="3.%1"/>', '<w:lvlText w:val="5.%1"/>')
abs42 = clone_abs("15", "42")
abs43 = clone_abs("15", "43")

for nid in (NUM_H2, NUM_SIMP, NUM_SARAN):
    assert 'w:numId="%s"' % nid not in num, "numId %s sudah dipakai" % nid

first_num = num.index("<w:num ")
num = num[:first_num] + abs44 + abs42 + abs43 + num[first_num:]
num = num.replace("</w:numbering>",
                  f'<w:num w:numId="{NUM_H2}"><w:abstractNumId w:val="44"/></w:num>'
                  f'<w:num w:numId="{NUM_SIMP}"><w:abstractNumId w:val="42"/></w:num>'
                  f'<w:num w:numId="{NUM_SARAN}"><w:abstractNumId w:val="43"/></w:num>'
                  '</w:numbering>')

ms = list(re.finditer(r'<w:p [^>]*>.*?</w:p>|<w:p/>', doc, re.S))
def ptxt(p):
    return ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p))
idx_dp = next(i for i, m in enumerate(ms)
              if ptxt(m.group(0)).startswith("DAFTAR PUSTAK")
              and '<w:pStyle w:val="Judul1"/>' in m.group(0))
sect_para = ms[idx_dp - 1].group(0)
assert "sectPr" in sect_para, "paragraf sectPr sebelum DAFTAR PUSTAKA tidak ditemukan"
insert_at = ms[idx_dp].start()

head = (f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="00000000">'
        f'<w:pPr><w:pStyle w:val="Judul1"/><w:spacing w:after="0" w:line="276" w:lineRule="auto"/></w:pPr>'
        f'<w:r><w:t>BAB 5</w:t></w:r></w:p>'
        f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="00000000">'
        f'<w:pPr><w:jc w:val="center"/><w:rPr>{RPR}<w:b/></w:rPr></w:pPr>'
        f'<w:r><w:rPr>{RPR}<w:b/></w:rPr><w:t>PENUTUP</w:t></w:r></w:p>')

closing = re.sub(r'w14:paraId="[0-9A-Fa-f]+"', 'w14:paraId="%s"' % pid(), sect_para, count=1)
doc = doc[:insert_at] + head + BAB5 + closing + doc[insert_at:]

tmp = SRC + ".tmp"
zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    data = zin.read(item.filename)
    if item.filename == "word/document.xml":
        data = doc.encode("utf-8")
    elif item.filename == "word/numbering.xml":
        data = num.encode("utf-8")
    zout.writestr(item, data)
zin.close(); zout.close()
os.replace(tmp, SRC)
print("BAB 5 disisipkan (4 simpulan, 5 saran)")
