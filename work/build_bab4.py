# -*- coding: utf-8 -*-
"""Sisipkan BAB 4 (Hasil dan Pembahasan) ke sempro-hamza-1.docx.

Mengikuti machinery dokumen yang sudah ada:
  Judul2/Judul3 dengan numId per-BAB (baru: 4.%1 dan 4.3.%1)
  paragraf isi: TNR 12pt, rata kiri-kanan, first-line 426, line 276
  tabel: lebar auto, rata tengah, border hitam tunggal, header berulang
  caption: style Keterangan + field SEQ Tabel (\\s 1 -> reset tiap BAB)
"""
import zipfile, re, os, shutil, datetime, random

SRC = r"C:/Users/Pratama/Downloads/smpro/sempro-hamza-1.docx"
BAKDIR = os.path.dirname(os.path.abspath(__file__))
NUMID_H2 = "40"
NUMID_H3 = "41"

RPR = ('<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
       'w:cs="Times New Roman"/><w:sz w:val="24"/><w:szCs w:val="24"/>')
CRPR = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'

_pid = [0x4A000000]
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
            f'<w:pPr><w:pStyle w:val="Judul2"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="{NUMID_H2}"/></w:numPr>'
            f'<w:spacing w:after="0" w:line="276" w:lineRule="auto"/><w:ind w:left="426" w:hanging="426"/></w:pPr>'
            f'<w:r><w:t>{esc(text)}</w:t></w:r></w:p>')

def H3(text):
    return (f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="00000000">'
            f'<w:pPr><w:pStyle w:val="Judul3"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="{NUMID_H3}"/></w:numPr>'
            f'<w:spacing w:after="0" w:line="276" w:lineRule="auto"/><w:ind w:left="993" w:hanging="567"/></w:pPr>'
            f'<w:r><w:t>{esc(text)}</w:t></w:r></w:p>')

_seq = [0]
def CAP(text):
    _seq[0] += 1
    return (f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="00000000">'
            f'<w:pPr><w:pStyle w:val="Keterangan"/><w:keepNext/>'
            f'<w:spacing w:after="0" w:line="276" w:lineRule="auto"/><w:rPr><w:b/><w:bCs/></w:rPr></w:pPr>'
            f'<w:r><w:rPr><w:b/><w:bCs/></w:rPr><w:t xml:space="preserve">Tabel 4.</w:t></w:r>'
            f'<w:fldSimple w:instr=" SEQ Tabel \\* ARABIC \\s 1 ">'
            f'<w:r><w:rPr><w:noProof/></w:rPr><w:t>{_seq[0]}</w:t></w:r></w:fldSimple>'
            f'<w:r><w:rPr><w:b/><w:bCs/></w:rPr><w:t xml:space="preserve"> {esc(text)}</w:t></w:r></w:p>')

def cell(text, w, header=False, align="center"):
    b = '<w:b/><w:bCs/>' if header else ''
    rpr = CRPR + b
    return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr>'
            f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="00000000">'
            f'<w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="{align}"/>'
            f'<w:rPr>{rpr}</w:rPr></w:pPr>'
            f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p></w:tc>')

def TBL(widths, rows, left_align_first=True):
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
        cells = ''
        for j, v in enumerate(r):
            al = "left" if (j == 0 and left_align_first and not hdr) else "center"
            cells += cell(v, widths[j], header=hdr, align=al)
        out.append(f'<w:tr w:rsidR="004B5458" w14:paraId="{pid()}" w14:textId="77777777">{trpr}{cells}</w:tr>')
    out.append('</w:tbl>')
    out.append(f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="00000000">'
               f'<w:pPr><w:spacing w:after="0" w:line="276" w:lineRule="auto"/><w:rPr>{RPR}</w:rPr></w:pPr></w:p>')
    return ''.join(out)

W1 = [2000, 1700, 1300, 1500, 1427]
W2 = [1900, 1500, 1300, 1500, 1727]
W3 = [2100, 1600, 1200, 1500, 1527]
W4 = [2000, 2100, 1900, 1927]
W5 = [1400, 1500, 1600, 1700, 1727]
W7 = [2600, 1700, 1700, 1927]

body = []
A = body.append

A(H2("Hasil Penerapan Protokol Data"))
A(P("Penerapan protokol data menghasilkan pembagian 2.521 citra TrashNet menjadi 1.764 citra latih, "
    "378 citra validasi, dan 379 citra uji, atau setara dengan proporsi 70:15:15. Pembagian dilakukan "
    "secara terstratifikasi sehingga komposisi kelas pada ketiga subset tetap sebanding dengan komposisi "
    "keseluruhan data. Setiap seed menghasilkan partisi yang berbeda, sehingga lima seed yang digunakan "
    "pada penelitian ini menghasilkan lima partisi yang saling independen."))
A(P("Seluruh skenario yang diuji menggunakan partisi yang sama pada setiap seed. Dengan demikian "
    "perbandingan antarskenario dapat dilakukan secara berpasangan, yaitu membandingkan hasil pada seed "
    "yang sama, sehingga keragaman yang bersumber dari perbedaan pembagian data tidak tercampur dengan "
    "perbedaan yang berasal dari metode. Hal ini menjadi dasar penggunaan uji berpasangan pada subbab 4.4. "
    "Skema label K6 menggunakan enam kelas asli TrashNet, sedangkan skema K5 menghilangkan kelas trash "
    "dengan protokol yang identik pada aspek lainnya."))

A(H2("Kinerja Model Guru, Asisten, dan Baseline Student"))
A(P("Sebelum membahas pengaruh distilasi, terlebih dahulu disajikan kinerja masing-masing peran model "
    "yang menjadi acuan pada seluruh perbandingan berikutnya. Hasil selengkapnya disajikan pada Tabel 4.1."))
A(CAP("Kinerja model guru, asisten, dan baseline student"))
A(TBL(W1, [
    ["Model", "Peran", "Parameter", "Akurasi K6", "Akurasi K5"],
    ["EfficientNet-B4", "Guru besar", "17,56 juta", "0,9446 ± 0,0070", "0,9581 ± 0,0048"],
    ["Focus-RCNet", "Asisten menengah", "520,6 ribu", "0,8507 ± 0,0164", "0,8737 ± 0,0152"],
    ["WasteNet-256K", "Student, CE", "256,0 ribu", "0,7784 ± 0,0128", "0,7994 ± 0,0259"],
    ["WasteNet-128K", "Student, CE", "128,1 ribu", "0,7652 ± 0,0204", "0,8067 ± 0,0200"],
]))
A(P("Guru besar EfficientNet-B4 mencapai akurasi 0,9446 pada skema K6 dan menjadi batas atas kinerja pada "
    "penelitian ini, sedangkan asisten Focus-RCNet mencapai 0,8507 dengan jumlah parameter sekitar 34 kali "
    "lebih kecil. Adapun student WasteNet yang dilatih dengan cross-entropy memperoleh 0,7652 pada varian "
    "128K dan 0,7784 pada varian 256K. Selisih antara guru besar dan student mencapai sekitar 17,9 poin "
    "persentase, yang menunjukkan lebarnya jarak kapasitas antara keduanya."))
A(P("Hal yang perlu dicatat adalah urutan kedua varian student tidak konsisten pada kedua skema label. "
    "Pada K6 varian 256K lebih unggul 1,3 poin persentase dari varian 128K, tetapi pada K5 justru varian "
    "128K yang lebih tinggi, yaitu 0,8067 berbanding 0,7994, dengan simpangan baku varian 256K yang "
    "relatif besar. Dengan demikian penambahan kapasitas student dari 128K ke 256K belum tentu "
    "meningkatkan akurasi apabila pelatihan hanya menggunakan cross-entropy biasa."))
A(P("Karena asisten berperan sebagai sumber pengetahuan pada skema TA-KD, cara pelatihan asisten juga "
    "diuji terlebih dahulu agar pemilihannya dapat dipertanggungjawabkan. Hasilnya disajikan pada Tabel 4.2."))
A(CAP("Pengaruh cara pelatihan asisten Focus-RCNet terhadap akurasinya"))
A(TBL(W2, [
    ["Cara pelatihan asisten", "Akurasi K6", "Δ K6 (p)", "Akurasi K5", "Δ K5 (p)"],
    ["Cross-entropy", "0,8507 ± 0,0164", "—", "0,8737 ± 0,0152", "—"],
    ["Distilasi langsung", "0,8554 ± 0,0231", "+0,47 (0,62)", "0,8732 ± 0,0135", "−0,06 (0,97)"],
    ["Distilasi dua tahap", "0,8528 ± 0,0078", "+0,21 (0,77)", "0,8765 ± 0,0096", "+0,28 (0,77)"],
]))
A(P("Tidak satu pun varian pelatihan asisten menghasilkan perbedaan yang signifikan terhadap pelatihan "
    "cross-entropy, baik pada K6 maupun K5. Oleh karena itu asisten yang dilatih dengan cross-entropy "
    "digunakan sebagai guru pada seluruh skenario TA-KD. Temuan ini sekaligus memberikan petunjuk awal "
    "yang penting, yaitu bahwa jadwal pelatihan dua tahap tidak dengan sendirinya memperbaiki hasil "
    "apabila guru yang digunakan tetap EfficientNet-B4."))

A(H2("Pengaruh Strategi Distilasi terhadap Kinerja Student"))
A(P("Bagian ini membandingkan seluruh strategi distilasi yang diterapkan pada student WasteNet. "
    "Rangkuman akurasi setiap skenario disajikan pada Tabel 4.3, sedangkan pengujian signifikansinya "
    "dibahas pada subbab 4.4."))
A(CAP("Perbandingan akurasi seluruh skenario pelatihan student"))
A(TBL(W3, [
    ["Skenario", "Guru", "Jadwal", "Akurasi K6", "Akurasi K5"],
    ["WasteNet-128K, CE", "—", "Satu tahap", "0,7652 ± 0,0204", "0,8067 ± 0,0200"],
    ["WasteNet-128K, distilasi langsung", "EfficientNet-B4", "Satu tahap", "0,7646 ± 0,0127", "0,8061 ± 0,0181"],
    ["WasteNet-128K, TA-KD", "Focus-RCNet", "Dua tahap", "0,7578 ± 0,0322", "0,8117 ± 0,0200"],
    ["WasteNet-256K, CE", "—", "Satu tahap", "0,7784 ± 0,0128", "0,7994 ± 0,0259"],
    ["WasteNet-256K, distilasi langsung", "EfficientNet-B4", "Satu tahap", "0,7736 ± 0,0178", "0,8056 ± 0,0156"],
    ["WasteNet-256K, TA-KD", "Focus-RCNet", "Dua tahap", "0,7873 ± 0,0089", "0,8235 ± 0,0107"],
    ["WasteNet-256K, self-distillation", "WasteNet-256K", "Satu tahap", "0,7831 ± 0,0105", "0,8134 ± 0,0109"],
]))

A(H3("Distilasi Langsung dari Guru Besar"))
A(P("Distilasi langsung dari EfficientNet-B4 tidak memberikan perbaikan pada kedua varian student. "
    "Pada WasteNet-128K akurasi justru bergerak dari 0,7652 menjadi 0,7646, sedangkan pada WasteNet-256K "
    "dari 0,7784 menjadi 0,7736. Kedua selisih tersebut sangat kecil dan tidak signifikan secara statistik. "
    "Pola yang sama juga terlihat pada skema K5, sehingga kegagalan ini bukan merupakan akibat dari "
    "pemilihan skema label tertentu."))
A(P("Hasil tersebut sejalan dengan temuan Cho dan Hariharan (2019) mengenai capacity gap, yaitu bahwa "
    "guru yang terlalu besar tidak selalu menghasilkan student yang lebih baik. Dengan selisih kapasitas "
    "sekitar 69 kali antara EfficientNet-B4 dan WasteNet-256K, distribusi keluaran guru diduga terlalu "
    "tajam dan terlalu kompleks untuk dapat ditiru oleh student yang sangat kecil."))

A(H3("Distilasi melalui Teacher Assistant"))
A(P("Berbeda dengan distilasi langsung, penggunaan asisten berkapasitas menengah memberikan hasil yang "
    "bergantung pada kapasitas student. Pada WasteNet-128K, skema TA-KD menghasilkan 0,7578 pada K6 dan "
    "tidak lebih baik dari kontrolnya. Sebaliknya, pada WasteNet-256K skema TA-KD menghasilkan 0,7873 pada "
    "K6 dan 0,8235 pada K5, yang merupakan akurasi tertinggi di antara seluruh skenario student pada kedua "
    "skema label."))
A(P("Perlu ditegaskan bahwa pembanding yang digunakan untuk menilai efek distilasi bukanlah baseline "
    "cross-entropy satu tahap, melainkan kontrol berpasangan yang menggunakan jadwal dua tahap yang sama "
    "dengan distilasi dimatikan. Dengan pembanding tersebut, satu-satunya faktor yang berbeda adalah ada "
    "atau tidaknya transfer pengetahuan dari asisten, sehingga perbaikan yang terukur tidak dapat "
    "dikaitkan dengan perbedaan jadwal pelatihan."))

A(H3("Self-Distillation dengan Born-Again Networks"))
A(P("Skema self-distillation menggunakan guru berupa WasteNet-256K generasi sebelumnya, yaitu model "
    "dengan arsitektur dan kapasitas yang persis sama dengan student. Skema ini menghasilkan 0,7831 pada "
    "K6 dan 0,8134 pada K5. Kedua nilai tersebut berada di atas baseline cross-entropy, namun tetap lebih "
    "rendah daripada skema TA-KD dan tidak mencapai taraf signifikansi pada kedua skema label."))
A(P("Meskipun demikian, terdapat gejala yang layak dicatat, yaitu menurunnya keragaman antarseed pada "
    "skema K5, dari simpangan baku 0,0259 pada baseline menjadi 0,0109. Hal ini sejalan dengan penjelasan "
    "Furlanello dkk. (2018) bahwa sebagian manfaat self-distillation berasal dari efek regularisasi soft "
    "label yang menghaluskan target pelatihan, bukan semata-mata dari transfer pengetahuan baru."))

A(H2("Hasil Uji Signifikansi Statistik"))
A(P("Setiap skema distilasi diuji terhadap kontrol berpasangannya, yaitu skenario yang identik kecuali "
    "pada satu faktor yang diuji. Pengujian dilakukan menggunakan uji t berpasangan antarseed dan "
    "dikonfirmasi dengan uji McNemar pada tingkat sampel. Hasil selengkapnya disajikan pada Tabel 4.4."))
A(CAP("Hasil uji berpasangan setiap skema distilasi terhadap kontrolnya"))
A(TBL(W4, [
    ["Perbandingan", "Faktor yang diuji", "K6: Δ, p, menang", "K5: Δ, p, menang"],
    ["Distilasi langsung, 128K", "Distilasi vs tanpa distilasi", "−0,05 pp; 0,94; 2/5", "−0,06 pp; 0,96; 3/5"],
    ["TA-KD, 128K", "Asisten vs kontrol CE", "−0,74 pp; 0,20; 1/5", "−0,50 pp; 0,23; 1/5"],
    ["Distilasi langsung, 256K", "Distilasi vs tanpa distilasi", "−0,47 pp; 0,68; 3/5", "+0,61 pp; 0,68; 4/5"],
    ["TA-KD, 256K", "Asisten vs kontrol CE", "+1,00 pp; 0,024; 5/5", "+1,40 pp; 0,045; 5/5"],
    ["Self-distillation, 256K", "Guru diri sendiri vs tanpa distilasi", "+0,47 pp; 0,51; 3/5", "+1,40 pp; 0,38; 4/5"],
]))
A(P("Dari seluruh perbandingan, hanya skema TA-KD pada WasteNet-256K yang mencapai taraf signifikansi. "
    "Skema tersebut unggul pada kelima seed di kedua skema label, dengan peningkatan 1,00 poin persentase "
    "pada K6 dengan p sebesar 0,024 dan 1,40 poin persentase pada K5 dengan p sebesar 0,045. Konsistensi "
    "kemenangan pada seluruh seed memperkuat kesimpulan bahwa perbaikan tersebut bukan kebetulan akibat "
    "satu partisi data tertentu."))
A(P("Konfirmasi melalui uji McNemar memberikan gambaran yang perlu disampaikan secara berimbang. Pada "
    "skema K5 hasilnya signifikan dengan nilai khi-kuadrat 6,06 dan p sebesar 0,014, sedangkan pada skema "
    "K6 hasilnya mendekati taraf signifikansi dengan nilai khi-kuadrat 2,97 dan p sebesar 0,085. Perbedaan "
    "ini wajar karena kedua uji memandang data dari sudut yang berbeda, yaitu uji t berpasangan bekerja "
    "pada tingkat seed sedangkan uji McNemar bekerja pada tingkat sampel. Dengan demikian bukti terkuat "
    "berada pada skema K5, sementara pada skema K6 arah perbaikannya konsisten namun dukungan "
    "statistiknya lebih lemah."))
A(P("Seluruh skema lainnya tidak signifikan pada semua uji yang digunakan. Skema self-distillation "
    "menunjukkan arah perbaikan yang positif pada kedua skema label, tetapi besarnya perbaikan tidak "
    "konsisten antarseed sehingga tidak pernah mencapai taraf signifikansi."))

A(H2("Analisis Kinerja per Kelas"))
A(P("Untuk mengetahui letak perbaikan yang dihasilkan distilasi, dilakukan penelaahan terhadap nilai F1 "
    "setiap kelas. Selisih F1 antara skema TA-KD dan kontrolnya disajikan pada Tabel 4.5."))
A(CAP("Selisih F1 per kelas pada skema TA-KD terhadap kontrolnya"))
A(TBL(W5, [
    ["Kelas", "F1 TA-KD (K6)", "F1 kontrol (K6)", "ΔF1 K6 (pp)", "ΔF1 K5 (pp)"],
    ["glass", "0,731", "0,708", "+2,23", "+1,98"],
    ["metal", "0,742", "0,725", "+1,68", "+0,68"],
    ["paper", "0,863", "0,855", "+0,82", "+1,23"],
    ["cardboard", "0,901", "0,895", "+0,58", "+0,69"],
    ["plastic", "0,754", "0,751", "+0,33", "+2,34"],
    ["trash", "0,561", "0,572", "−1,03", "—"],
]))
A(P("Perbaikan tidak tersebar merata, melainkan terkonsentrasi pada kelas-kelas yang paling sulit "
    "dibedakan, yaitu glass dan metal pada skema K6 serta plastic dan glass pada skema K5. Ketiga kelas "
    "tersebut merupakan kelompok sampah daur ulang yang secara visual saling menyerupai. Pola ini "
    "konsisten dengan gagasan dark knowledge, yaitu bahwa manfaat distilasi berasal dari informasi "
    "kemiripan antarkelas yang terkandung pada distribusi keluaran guru, bukan sekadar dari efek "
    "penghalusan target."))
A(P("Sebaliknya, kelas trash pada skema K6 justru mengalami penurunan sebesar 1,03 poin persentase. "
    "Kelas ini bersifat heterogen dan berjumlah paling sedikit, sehingga struktur kemiripan yang "
    "ditawarkan guru kurang bermakna. Hal tersebut sekaligus menjelaskan mengapa perbaikan pada skema K5 "
    "yang tidak memuat kelas trash terlihat lebih besar dan lebih bersih."))
A(P("Penelaahan yang sama dilakukan pada skema self-distillation dan hasilnya disajikan pada Tabel 4.6."))
A(CAP("Selisih F1 per kelas pada skema self-distillation terhadap baseline"))
A(TBL(W5, [
    ["Kelas", "F1 self-KD (K6)", "F1 baseline (K6)", "ΔF1 K6 (pp)", "ΔF1 K5 (pp)"],
    ["glass", "0,750", "0,712", "+3,77", "+1,87"],
    ["trash", "0,603", "0,589", "+1,45", "—"],
    ["metal", "0,727", "0,717", "+1,05", "+2,09"],
    ["cardboard", "0,887", "0,893", "−0,58", "+2,60"],
    ["paper", "0,854", "0,860", "−0,60", "+1,95"],
    ["plastic", "0,744", "0,756", "−1,22", "−1,44"],
]))
A(P("Skema self-distillation sebagian meniru pola yang sama, terlihat dari naiknya kelas glass dan metal "
    "pada kedua skema label. Akan tetapi perbaikannya kurang terarah. Kelas plastic yang menjadi penerima "
    "manfaat terbesar pada skema TA-KD justru menurun pada kedua skema label, sedangkan kelas trash pada "
    "skema K6 bergerak berlawanan arah dibandingkan skema TA-KD. Temuan ini menunjukkan bahwa guru "
    "berkapasitas sama membawa informasi kemiripan antarkelas yang lebih lemah dan kurang terfokus "
    "dibandingkan asisten berkapasitas menengah, dan hal tersebut menjelaskan mengapa perbaikannya tidak "
    "mencapai taraf signifikansi meskipun arahnya positif."))

A(H2("Perbandingan Skema Lima Kelas dan Enam Kelas"))
A(P("Pada seluruh skenario, skema K5 menghasilkan akurasi yang lebih tinggi daripada skema K6, dengan "
    "selisih berkisar antara 1,3 hingga 5,4 poin persentase. Selisih pada nilai macro F1 bahkan lebih "
    "besar karena kelas trash yang sulit dikenali ikut menekan rata-rata F1 pada skema K6. Perbandingan "
    "ini bersifat deskriptif dan tidak diuji secara berpasangan, sebab kedua skema bekerja pada ruang "
    "label yang berbeda sehingga jumlah kelas dan tingkat kesulitan tugasnya tidak setara."))
A(P("Nilai penting dari perbandingan ini terletak pada konsistensi temuan. Skema TA-KD pada WasteNet-256K "
    "unggul pada kelima seed di kedua skema label, sehingga keunggulannya tidak bergantung pada ada atau "
    "tidaknya kelas trash. Sebaliknya, seluruh skema yang tidak signifikan pada skema K6 juga tidak "
    "signifikan pada skema K5. Kesesuaian arah temuan pada dua ruang label yang berbeda memperkuat "
    "keandalan kesimpulan penelitian ini."))

A(H2("Efisiensi Komputasi Model Student"))
A(P("Selain akurasi, kelayakan penerapan model juga dinilai dari kebutuhan komputasinya. Perbandingan "
    "jumlah parameter, FLOPs, dan akurasi setiap varian student disajikan pada Tabel 4.7."))
A(CAP("Efisiensi komputasi model student pada resolusi masukan 160 × 160 piksel"))
A(TBL(W7, [
    ["Model student", "Parameter", "FLOPs", "Akurasi K6"],
    ["WasteNet-128K, CE", "128.086", "≈ 51 juta", "0,7652 ± 0,0204"],
    ["WasteNet-256K, CE", "256.002", "≈ 84 juta", "0,7784 ± 0,0128"],
    ["WasteNet-256K, TA-KD", "256.002", "≈ 84 juta", "0,7873 ± 0,0089"],
]))
A(P("Perlu diperhatikan bahwa skema TA-KD meningkatkan akurasi tanpa menambah satu pun parameter maupun "
    "operasi komputasi, sebab distilasi hanya mengubah cara model dilatih dan tidak mengubah "
    "arsitekturnya. Perbaikan sebesar 1,00 poin persentase pada skema K6 dan 1,40 poin persentase pada "
    "skema K5 diperoleh dengan biaya inferensi yang sama persis, sehingga tergolong perbaikan yang murah "
    "dari sudut pandang penerapan."))
A(P("Apabila dibandingkan dengan guru besarnya, model WasteNet-256K hasil TA-KD memiliki parameter "
    "sekitar 69 kali lebih sedikit daripada EfficientNet-B4, namun tetap mempertahankan sekitar 83 persen "
    "dari akurasi guru tersebut. Dengan beban komputasi sekitar 84 juta FLOPs, model ini sesuai untuk "
    "diterapkan pada perangkat dengan sumber daya terbatas."))

A(H2("Pembahasan"))
A(P("Rangkaian hasil di atas menunjukkan bahwa keberhasilan distilasi pada penelitian ini ditentukan oleh "
    "kapasitas guru, bukan oleh jadwal pelatihan. Ketika guru yang digunakan adalah EfficientNet-B4, "
    "distilasi tidak memberikan perbaikan baik pada jadwal satu tahap maupun dua tahap. Sebaliknya, ketika "
    "guru diganti menjadi asisten berkapasitas menengah dengan jadwal yang sama, perbaikan yang signifikan "
    "baru muncul. Temuan ini mendukung skema teacher assistant yang diusulkan Mirzadeh dkk. (2020) sebagai "
    "cara mengatasi capacity gap."))
A(P("Apabila ketiga jenis guru diurutkan berdasarkan kapasitasnya, terbentuk pola yang runtut. Guru yang "
    "jauh lebih besar tidak memberikan perbaikan, guru yang berkapasitas sama dengan student memberikan "
    "arah perbaikan yang positif tetapi tidak signifikan, sedangkan guru berkapasitas menengah memberikan "
    "perbaikan yang signifikan dan konsisten. Dengan demikian guru yang efektif bukanlah guru yang paling "
    "kuat, melainkan guru yang jarak kapasitasnya terhadap student tidak terlalu jauh dan tidak terlalu "
    "dekat."))
A(P("Selain kapasitas guru, kapasitas student juga turut menentukan. Skema TA-KD tidak memberikan "
    "perbaikan pada WasteNet-128K, tetapi berhasil pada WasteNet-256K. Hal ini menunjukkan bahwa student "
    "memerlukan kapasitas yang memadai untuk dapat menampung pengetahuan yang ditransfer. Pada model yang "
    "terlalu kecil, kapasitas diduga telah habis terpakai untuk mempelajari tugas dasarnya sehingga tidak "
    "tersisa ruang untuk memanfaatkan informasi tambahan dari guru. Dengan demikian keberhasilan "
    "distilasi ditentukan oleh kesesuaian kapasitas pada kedua sisi, yaitu sisi guru dan sisi student."))
A(P("Analisis per kelas memberikan dukungan mekanistik terhadap kesimpulan tersebut. Perbaikan yang "
    "dihasilkan skema TA-KD terkonsentrasi pada kelompok kelas daur ulang yang saling menyerupai, "
    "sedangkan skema self-distillation menghasilkan perbaikan yang lebih menyebar dan kurang terarah. "
    "Perbedaan pola ini memperlihatkan bahwa yang berpindah dari asisten ke student bukan sekadar efek "
    "penghalusan target, melainkan informasi mengenai kemiripan antarkelas."))
A(P("Penelitian ini juga memiliki sejumlah keterbatasan yang perlu dikemukakan. Jumlah data yang "
    "digunakan tergolong kecil, yaitu 2.521 citra, sehingga keragaman antarpartisi cukup besar. Jumlah "
    "seed sebanyak lima juga membatasi daya uji statistik, sehingga perbaikan yang kecil berpotensi tidak "
    "terdeteksi meskipun sebenarnya nyata. Selain itu besarnya perbaikan yang diperoleh berkisar pada satu "
    "poin persentase, sehingga penerapannya perlu mempertimbangkan kebutuhan ketelitian pada kasus "
    "penggunaan yang dituju."))

BAB4 = ''.join(body)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
bak = os.path.join(BAKDIR, f"sempro-hamza-1_BACKUP_{ts}.docx")
shutil.copy2(SRC, bak)
print("backup ->", bak)

zin = zipfile.ZipFile(SRC, "r")
doc = zin.read("word/document.xml").decode("utf-8")
num = zin.read("word/numbering.xml").decode("utf-8")

def clone_abs(src_id, new_id, old_txt, new_txt):
    m = re.search(r'<w:abstractNum w:abstractNumId="%s"[^>]*>.*?</w:abstractNum>' % src_id, num, re.S)
    b = m.group(0)
    b = b.replace('w:abstractNumId="%s"' % src_id, 'w:abstractNumId="%s"' % new_id, 1)
    b = re.sub(r'<w:nsid w:val="[0-9A-Fa-f]+"/>',
               '<w:nsid w:val="%08X"/>' % random.randint(0x10000000, 0x7FFFFFFF), b, count=1)
    b = re.sub(r'w:tplc="[0-9A-Fa-f]+"',
               lambda _: 'w:tplc="%08X"' % random.randint(0x10000000, 0x7FFFFFFF), b)
    assert old_txt in b, "lvlText not found in abstractNum %s" % src_id
    return b.replace(old_txt, new_txt, 1)

abs40 = clone_abs("2", "40", '<w:lvlText w:val="3.%1"/>', '<w:lvlText w:val="4.%1"/>')
abs41 = clone_abs("8", "41", '<w:lvlText w:val="3.1.%1"/>', '<w:lvlText w:val="4.3.%1"/>')

assert 'w:abstractNumId="40"' not in num and 'w:numId="40"' not in num
first_num = num.index("<w:num ")
num = num[:first_num] + abs40 + abs41 + num[first_num:]
num = num.replace("</w:numbering>",
                  '<w:num w:numId="40"><w:abstractNumId w:val="40"/></w:num>'
                  '<w:num w:numId="41"><w:abstractNumId w:val="41"/></w:num></w:numbering>')

ms = list(re.finditer(r'<w:p [^>]*>.*?</w:p>|<w:p/>', doc, re.S))
def ptxt(p):
    return ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p))
idx_dp = next(i for i, m in enumerate(ms)
              if ptxt(m.group(0)).startswith("DAFTAR PUSTAK")
              and '<w:pStyle w:val="Judul1"/>' in m.group(0))
sect_para = ms[idx_dp - 1].group(0)
assert "sectPr" in sect_para, "expected sectPr paragraph before DAFTAR PUSTAKA"
assert "BAB 4" not in doc, "BAB 4 already present - aborting to avoid duplicate"
insert_at = ms[idx_dp].start()

head = (f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="00000000">'
        f'<w:pPr><w:pStyle w:val="Judul1"/><w:spacing w:after="0" w:line="276" w:lineRule="auto"/></w:pPr>'
        f'<w:r><w:t>BAB 4</w:t></w:r></w:p>'
        f'<w:p w14:paraId="{pid()}" w14:textId="77777777" w:rsidR="004B5458" w:rsidRDefault="00000000">'
        f'<w:pPr><w:jc w:val="center"/><w:rPr>{RPR}<w:b/></w:rPr></w:pPr>'
        f'<w:r><w:rPr>{RPR}<w:b/></w:rPr><w:t>HASIL DAN PEMBAHASAN</w:t></w:r></w:p>')

closing = re.sub(r'w14:paraId="[0-9A-Fa-f]+"', 'w14:paraId="%s"' % pid(), sect_para, count=1)

doc = doc[:insert_at] + head + BAB4 + closing + doc[insert_at:]

tmp = SRC + ".tmp"
zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    data = zin.read(item.filename)
    if item.filename == "word/document.xml":
        data = doc.encode("utf-8")
    elif item.filename == "word/numbering.xml":
        data = num.encode("utf-8")
    zout.writestr(item, data)
zin.close()
zout.close()
os.replace(tmp, SRC)
print("BAB 4 spliced into", SRC)
print("tabel disisipkan:", _seq[0])
