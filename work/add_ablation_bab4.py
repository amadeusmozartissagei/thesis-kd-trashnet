# -*- coding: utf-8 -*-
"""Sisipkan subbab 'Ablasi Komponen Arsitektur' ke BAB 4, tepat sebelum subbab 'Pembahasan'.

Penomoran Judul2 memakai numId 40 (lvlText 4.%1) sehingga subbab baru otomatis
menjadi 4.8 dan 'Pembahasan' bergeser menjadi 4.9 tanpa renumber manual.
Caption memakai field SEQ sehingga tabel baru otomatis menjadi Tabel 4.8 dan 4.9.
"""
import zipfile, re, os, shutil, datetime

SRC = r"C:/Users/Pratama/Downloads/smpro/sempro-hamza-1.docx"
BAKDIR = os.path.dirname(os.path.abspath(__file__))
NUMID_H2 = "40"

RPR = ('<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
       'w:cs="Times New Roman"/><w:sz w:val="24"/><w:szCs w:val="24"/>')
CRPR = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'

_pid = [0x4B000000]
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
            f'<w:spacing w:after="0" w:line="276" w:lineRule="auto"/><w:rPr><w:b/><w:bCs/></w:rPr></w:pPr>'
            f'<w:r><w:rPr><w:b/><w:bCs/></w:rPr><w:t xml:space="preserve">Tabel 4.</w:t></w:r>'
            f'<w:fldSimple w:instr=" SEQ Tabel \\* ARABIC \\s 1 ">'
            f'<w:r><w:rPr><w:noProof/></w:rPr><w:t>{n}</w:t></w:r></w:fldSimple>'
            f'<w:r><w:rPr><w:b/><w:bCs/></w:rPr><w:t xml:space="preserve"> {esc(text)}</w:t></w:r></w:p>')

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

WA = [2300, 1700, 1400, 1000, 1527]

body = []
A = body.append

A(H2("Ablasi Komponen Arsitektur"))
A(P("Bagian ini menelaah komponen arsitektur mana yang paling menentukan akurasi WasteNet. Pengujian "
    "dilakukan pada skema K6 saja, sebab arsitektur WasteNet identik pada kedua skema label dan hanya "
    "berbeda pada jumlah neuron keluarannya, sehingga kepentingan relatif antarkomponen tidak berubah. "
    "Setiap konfigurasi dilatih ulang menggunakan lima seed dengan pengaturan pelatihan yang identik "
    "dengan baseline cross-entropy, sehingga seluruh pengujian pada bagian ini mencakup 40 kali pelatihan."))
A(P("Komponen yang divariasikan meliputi fungsi aktivasi, expand ratio pada blok inverted residual, "
    "koneksi residual, dan lapisan tersembunyi pada pengklasifikasi. Koneksi residual hanya diablasi pada "
    "varian 256K, sebab pada varian 128K tidak terdapat blok dengan dimensi masukan dan keluaran yang "
    "sama sehingga koneksi residual memang tidak pernah aktif dan ablasinya tidak akan berpengaruh. "
    "Perlu ditegaskan pula bahwa variasi expand ratio dan lapisan tersembunyi mengubah jumlah parameter, "
    "sehingga efek yang terukur bercampur antara hilangnya komponen dan berkurangnya kapasitas. Oleh "
    "karena itu jumlah parameter dicantumkan pada setiap baris, sedangkan variasi fungsi aktivasi dan "
    "koneksi residual tidak mengubah jumlah parameter sehingga efeknya dapat dibaca secara langsung. "
    "Hasil ablasi pada varian 128K disajikan pada Tabel 4.8 dan pada varian 256K pada Tabel 4.9."))
A(CAP("Hasil ablasi komponen arsitektur pada WasteNet-128K", 8))
A(TBL(WA, [
    ["Konfigurasi", "Akurasi", "Δ vs baseline", "p", "Parameter"],
    ["Baseline", "0,7689 ± 0,0172", "—", "—", "128.086"],
    ["Aktivasi SiLU menjadi ReLU", "0,7156 ± 0,0193", "−5,33 pp", "0,004", "128.086"],
    ["Expand ratio 2,0 menjadi 1,0", "0,7689 ± 0,0198", "+0,00 pp", "1,000", "45.638"],
    ["Tanpa lapisan tersembunyi", "0,7646 ± 0,0192", "−0,42 pp", "0,634", "122.870"],
]))
A(CAP("Hasil ablasi komponen arsitektur pada WasteNet-256K", 9))
A(TBL(WA, [
    ["Konfigurasi", "Akurasi", "Δ vs baseline", "p", "Parameter"],
    ["Baseline", "0,7694 ± 0,0164", "—", "—", "256.002"],
    ["Aktivasi SiLU menjadi ReLU", "0,7873 ± 0,0223", "+1,79 pp", "0,268", "256.002"],
    ["Tanpa koneksi residual", "0,7662 ± 0,0130", "−0,32 pp", "0,702", "256.002"],
    ["Expand ratio 2,5 menjadi 1,0", "0,7689 ± 0,0091", "−0,05 pp", "0,918", "64.206"],
]))
A(P("Pada varian 128K, fungsi aktivasi merupakan satu-satunya komponen yang berpengaruh secara "
    "signifikan. Penggantian SiLU dengan ReLU menurunkan akurasi sebesar 5,33 poin persentase dengan "
    "p sebesar 0,004, dan penurunan tersebut terjadi secara konsisten pada kelima seed. Karena "
    "penggantian ini tidak mengubah jumlah parameter sama sekali, penurunan tersebut murni berasal dari "
    "perubahan fungsi aktivasi dan bukan dari berkurangnya kapasitas model."))
A(P("Keadaan yang berbeda ditemukan pada varian 256K. Pada varian tersebut tidak ada satu pun komponen "
    "yang menghasilkan perbedaan signifikan, sebab seluruh nilai p berada di atas 0,26. Penggantian ke "
    "ReLU memang menaikkan akurasi sebesar 1,79 poin persentase, tetapi kenaikan tersebut tidak "
    "signifikan sehingga belum dapat dinyatakan sebagai perbaikan yang meyakinkan. Dengan demikian "
    "arsitektur WasteNet-256K tergolong tahan terhadap perubahan komponen penyusunnya."))
A(P("Perbedaan antara kedua varian menunjukkan bahwa pengaruh fungsi aktivasi bergantung pada kapasitas "
    "model. Ketika kapasitas terbatas seperti pada varian 128K, pemilihan fungsi aktivasi menjadi "
    "menentukan, sedangkan pada varian 256K perbedaannya menghilang. Hal ini diduga karena kapasitas yang "
    "lebih besar menyediakan cukup keleluasaan sehingga model tidak lagi bergantung pada satu fungsi "
    "aktivasi tertentu. Pola ini sejalan dengan temuan pada subbab sebelumnya bahwa kapasitas student "
    "turut menentukan bagaimana model merespons perlakuan yang diberikan kepadanya."))
A(P("Temuan lain berkaitan dengan expand ratio. Penurunan expand ratio memangkas 64 hingga 75 persen "
    "jumlah parameter, tetapi akurasinya hampir tidak berubah dengan nilai p paling rendah sebesar 0,92. "
    "Hal ini menunjukkan bahwa kapasitas pada dimensi expand cenderung berlebih, sehingga masih tersedia "
    "ruang untuk penghematan parameter tanpa mengorbankan akurasi. Temuan ini bermanfaat sebagai arah "
    "pengembangan lanjutan, meskipun perlu dibaca dengan hati-hati karena variasi tersebut termasuk "
    "variasi yang mengubah jumlah parameter."))
A(P("Hasil ablasi ini memberikan dukungan tidak langsung terhadap kesimpulan utama penelitian. Karena "
    "arsitektur WasteNet-256K terbukti tahan terhadap perubahan komponen penyusunnya, peningkatan akurasi "
    "sebesar 1,00 poin persentase yang diperoleh melalui skema TA-KD tidak dapat dijelaskan oleh "
    "kekhususan rancangan arsitektur, melainkan berasal dari cara model tersebut dilatih. Perlu dicatat "
    "pula bahwa penggunaan lima seed membatasi daya uji, sehingga sebagian selisih yang kecil pada varian "
    "256K berpotensi tidak terdeteksi meskipun sebenarnya nyata."))

BLOCK = ''.join(body)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
bak = os.path.join(BAKDIR, f"sempro-hamza-1_BACKUP_{ts}.docx")
shutil.copy2(SRC, bak)
print("backup ->", bak)

zin = zipfile.ZipFile(SRC, "r")
doc = zin.read("word/document.xml").decode("utf-8")

assert "Ablasi Komponen Arsitektur" not in doc, "subbab ablasi sudah ada - dibatalkan"

ms = list(re.finditer(r'<w:p [^>]*>.*?</w:p>|<w:p/>', doc, re.S))
def ptxt(p):
    return ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p))

cands = [m for m in ms
         if ptxt(m.group(0)).strip() == "Pembahasan"
         and '<w:pStyle w:val="Judul2"/>' in m.group(0)]
assert len(cands) == 1, "ditemukan %d heading 'Pembahasan' (harus tepat 1)" % len(cands)

insert_at = cands[0].start()
doc = doc[:insert_at] + BLOCK + doc[insert_at:]

tmp = SRC + ".tmp"
zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    data = zin.read(item.filename)
    if item.filename == "word/document.xml":
        data = doc.encode("utf-8")
    zout.writestr(item, data)
zin.close()
zout.close()
os.replace(tmp, SRC)
print("subbab ablasi disisipkan sebelum 'Pembahasan'")
