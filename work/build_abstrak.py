# -*- coding: utf-8 -*-
"""Sisipkan halaman ABSTRAK dan ABSTRACT sebelum DAFTAR ISI, plus perbaiki typo nama.

Format mengikuti skripsi pembanding: judul identitas, Kata Kunci, lalu satu paragraf isi.
"""
import zipfile, re, os, shutil, datetime

SRC = r"C:/Users/Pratama/Downloads/smpro/sempro-hamza-1.docx"
HERE = os.path.dirname(os.path.abspath(__file__))

RPR = ('<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
       'w:cs="Times New Roman"/><w:sz w:val="24"/><w:szCs w:val="24"/>')

_pid = [0x4E000000]
def pid():
    _pid[0] += 1
    return "%08X" % _pid[0]

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def P(text, indent=False, bold_prefix=None):
    ind = '<w:ind w:firstLine="426"/>' if indent else ''
    runs = ""
    if bold_prefix:
        runs += (f'<w:r><w:rPr>{RPR}<w:b/></w:rPr>'
                 f'<w:t xml:space="preserve">{esc(bold_prefix)}</w:t></w:r>')
    runs += f'<w:r><w:rPr>{RPR}</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r>'
    return (f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="00000000">'
            f'<w:pPr><w:spacing w:after="160" w:line="276" w:lineRule="auto"/>{ind}<w:jc w:val="both"/>'
            f'<w:rPr>{RPR}</w:rPr></w:pPr>{runs}</w:p>')

def H1(text):
    return (f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="00000000">'
            f'<w:pPr><w:pStyle w:val="Judul1"/><w:spacing w:after="240" w:line="276" w:lineRule="auto"/></w:pPr>'
            f'<w:r><w:t>{esc(text)}</w:t></w:r></w:p>')

def PAGEBREAK():
    return (f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="00000000">'
            f'<w:pPr><w:spacing w:after="0"/><w:rPr><w:lang w:eastAsia="id-ID"/></w:rPr></w:pPr>'
            f'<w:r><w:rPr><w:lang w:eastAsia="id-ID"/></w:rPr><w:br w:type="page"/></w:r></w:p>')

JUDUL_ID = ("Knowledge Distillation Berbasis Teacher-Assistant untuk Model Ringan "
            "WasteNet pada Klasifikasi Citra Sampah TrashNet")
JUDUL_EN = ("Teacher-Assistant Based Knowledge Distillation for a Lightweight WasteNet "
            "Model on TrashNet Waste Image Classification")

blocks = []
B = blocks.append

# ---------------------------------------------------------------- ABSTRAK
B(PAGEBREAK())
B(H1("ABSTRAK"))
B(P(f"Pratama, Hamza. 2026. {JUDUL_ID}. Skripsi, Program Studi Sarjana Teknik Informatika "
    "Fakultas Matematika dan Ilmu Pengetahuan Alam Universitas Negeri Semarang. "
    "Pembimbing Kholiq Budiman, S.Pd., M.Kom."))
B(P("Knowledge Distillation, Teacher Assistant, Capacity Gap, Klasifikasi Citra Sampah, "
    "TrashNet, Model Ringan.", bold_prefix="Kata Kunci: "))
B(P("Klasifikasi citra sampah pada perangkat dengan sumber daya terbatas memerlukan model yang "
    "sangat ringan, namun model berukuran kecil umumnya sulit mencapai akurasi yang memadai. "
    "Knowledge distillation lazim digunakan untuk mengatasi persoalan tersebut, tetapi "
    "efektivitasnya tidak terjamin ketika selisih kapasitas antara guru dan student terlalu besar "
    "atau dikenal sebagai capacity gap. Penelitian ini menguji pengaruh kapasitas guru terhadap "
    "keberhasilan distilasi pada model ringan WasteNet yang disusun dari blok konvolusi "
    "depthwise-separable dengan struktur inverted residual, dalam dua varian kapasitas yaitu "
    "WasteNet-128K dengan 128.086 parameter dan WasteNet-256K dengan 256.002 parameter. Tiga jenis "
    "guru dibandingkan, yaitu guru besar EfficientNet-B4, asisten berkapasitas menengah "
    "Focus-RCNet, dan guru berkapasitas sama melalui skema self-distillation Born-Again Networks. "
    "Data yang digunakan adalah TrashNet sebanyak 2.521 citra yang diuji pada dua skema label, "
    "yaitu enam kelas dan lima kelas. Setiap skenario dilatih menggunakan lima seed dengan "
    "pembagian data 70:15:15 yang berbeda pada setiap seed, kemudian dibandingkan terhadap kontrol "
    "berpasangan melalui uji t berpasangan, uji Wilcoxon, dan uji McNemar. Hasil penelitian "
    "menunjukkan bahwa distilasi langsung dari guru besar tidak memberikan perbaikan, sedangkan "
    "self-distillation hanya menghasilkan perbaikan yang tidak signifikan. Sebaliknya, distilasi "
    "melalui asisten berkapasitas menengah meningkatkan akurasi WasteNet-256K sebesar 1,00 poin "
    "persentase pada skema enam kelas dengan p sebesar 0,024 dan 1,40 poin persentase pada skema "
    "lima kelas dengan p sebesar 0,045, serta unggul pada seluruh seed yang diuji. Model terbaik "
    "mencapai akurasi 0,7873 dengan beban komputasi sekitar 84 juta FLOPs dan mempertahankan "
    "sekitar 83 persen akurasi guru besarnya menggunakan parameter sekitar 69 kali lebih sedikit. "
    "Penelitian ini menyimpulkan bahwa guru yang efektif bukanlah guru yang paling besar, "
    "melainkan guru berkapasitas menengah, dan keberhasilan distilasi juga menuntut kapasitas "
    "student yang memadai.", indent=True))

# ---------------------------------------------------------------- ABSTRACT
B(PAGEBREAK())
B(H1("ABSTRACT"))
B(P(f"Pratama, Hamza. 2026. {JUDUL_EN}. Undergraduate Thesis, Informatics Engineering Study "
    "Program, Faculty of Mathematics and Natural Sciences, Universitas Negeri Semarang. "
    "Supervisor: Kholiq Budiman, S.Pd., M.Kom."))
B(P("Knowledge Distillation, Teacher Assistant, Capacity Gap, Waste Image Classification, "
    "TrashNet, Lightweight Model.", bold_prefix="Keywords: "))
B(P("Waste image classification on resource-constrained devices requires extremely lightweight "
    "models, yet small models generally struggle to reach adequate accuracy. Knowledge "
    "distillation is commonly used to address this problem, but its effectiveness is not "
    "guaranteed when the capacity difference between the teacher and the student is too large, a "
    "condition known as the capacity gap. This study examines how teacher capacity affects "
    "distillation success on a lightweight WasteNet model built from depthwise-separable "
    "convolution blocks with an inverted residual structure, in two capacity variants: "
    "WasteNet-128K with 128,086 parameters and WasteNet-256K with 256,002 parameters. Three types "
    "of teachers are compared, namely the large teacher EfficientNet-B4, the mid-capacity "
    "assistant Focus-RCNet, and a same-capacity teacher through Born-Again Networks "
    "self-distillation. The dataset is TrashNet, consisting of 2,521 images evaluated under two "
    "label schemes, six classes and five classes. Every scenario is trained with five seeds using "
    "a 70:15:15 split that differs per seed, then compared against a paired control using a paired "
    "t-test, the Wilcoxon test, and McNemar's test. The results show that direct distillation from "
    "the large teacher yields no improvement, while self-distillation produces only a "
    "non-significant improvement. In contrast, distillation through the mid-capacity assistant "
    "improves WasteNet-256K accuracy by 1.00 percentage point on the six-class scheme with p = "
    "0.024 and by 1.40 percentage points on the five-class scheme with p = 0.045, winning on all "
    "tested seeds. The best model reaches 0.7873 accuracy with approximately 84 million FLOPs and "
    "retains about 83 percent of its large teacher's accuracy using roughly 69 times fewer "
    "parameters. This study concludes that an effective teacher is not the largest one but a "
    "mid-capacity one, and that successful distillation also requires sufficient student capacity.",
    indent=True))

BLOCK = ''.join(blocks)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
bak = os.path.join(HERE, f"sempro-hamza-1_BACKUP_{ts}.docx")
shutil.copy2(SRC, bak)
print("backup ->", bak)

zin = zipfile.ZipFile(SRC, "r")
doc = zin.read("word/document.xml").decode("utf-8")

assert "ABSTRAK" not in doc, "ABSTRAK sudah ada - dibatalkan"

# perbaiki typo nama pada halaman PERNYATAAN
if "Praama" in doc:
    doc = doc.replace(">Praama<", ">Pratama<", 1)
    print("typo diperbaiki: 'Hamza Praama' -> 'Hamza Pratama'")

ms = list(re.finditer(r'<w:p [^>]*>.*?</w:p>|<w:p/>', doc, re.S))
def ptxt(p):
    return ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p))
idx = next(i for i, m in enumerate(ms)
           if ptxt(m.group(0)).strip() == "DAFTAR ISI"
           and '<w:pStyle w:val="Judul1"/>' in m.group(0))
# sisipkan sebelum paragraf page-break yang mendahului DAFTAR ISI
prev = ms[idx - 1].group(0)
assert '<w:br w:type="page"/>' in prev, "paragraf page-break sebelum DAFTAR ISI tidak ditemukan"
insert_at = ms[idx - 1].start()

doc = doc[:insert_at] + BLOCK + doc[insert_at:]

tmp = SRC + ".tmp"
zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    data = zin.read(item.filename)
    if item.filename == "word/document.xml":
        data = doc.encode("utf-8")
    zout.writestr(item, data)
zin.close(); zout.close()
os.replace(tmp, SRC)
print("ABSTRAK + ABSTRACT disisipkan sebelum DAFTAR ISI")
