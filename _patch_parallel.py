import json

p = r'final_research_kd_k5/notebook2_r2_k5_focus_rcnet_ce_final.ipynb'
nb = json.load(open(p, encoding='utf-8'))
cells = nb['cells']


def patch(i, old, new):
    s = ''.join(cells[i]['source'])
    assert old in s, f"NOT FOUND in cell {i}: {old!r}"
    cells[i]['source'] = s.replace(old, new).splitlines(keepends=True)


# Cell 3: make SEED env-overridable (default stays explicit 42)
patch(3,
      "    # Change SEED to each of {42, 123, 777, 2026, 3407} and rerun the notebook.\n    SEED = 42",
      "    # Change SEED to each of {42, 123, 777, 2026, 3407} and rerun (or commit) the notebook.\n"
      "    # For parallel Kaggle commits, override per run with env var R2_K5_SEED.\n"
      "    SEED = int(os.environ.get(\"R2_K5_SEED\", \"42\"))")

# Cell 1: document the Kaggle parallel workflow
patch(1,
      "- Budget epoch final = 100, seragam dengan seluruh final di studi ini (R1 teacher dan R3-R8 student semua 100 epoch). Pilot CE memakai 60 epoch hanya untuk tuning LR; LR yang terpilih tetap valid dipakai di final 100 epoch (pola yang sama dengan R1: pilot 30 epoch, final 100).\n- Jalankan final seed `42`, `123`, `777`, `2026`, `3407` secara bergiliran tanpa mengubah hyperparameter (ganti `Config.SEED` lalu jalankan ulang notebook).",
      "- Budget epoch final = 100, seragam dengan seluruh final di studi ini (R1 teacher dan R3-R8 student semua 100 epoch). Pilot CE memakai 60 epoch hanya untuk tuning LR; LR yang terpilih tetap valid dipakai di final 100 epoch (pola yang sama dengan R1: pilot 30 epoch, final 100).\n"
      "- Jalankan final seed `42`, `123`, `777`, `2026`, `3407` tanpa mengubah hyperparameter: set `Config.SEED` (atau env `R2_K5_SEED`) lalu jalankan ulang/commit notebook.\n"
      "- Seed bersifat independen (manifest dan folder output terpisah per seed, tidak ada tabrakan file), sehingga aman dijalankan paralel. Di Kaggle gunakan **Save & Run All (Commit)** per seed; jumlah yang benar-benar paralel dibatasi kuota/sesi GPU Kaggle, sisanya mengantre.")

json.dump(nb, open(p, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print("patched: SEED env override + Kaggle parallel notes")

# verify + compile
nb2 = json.load(open(p, encoding='utf-8'))
for line in ''.join(nb2['cells'][3]['source']).splitlines():
    if 'SEED = ' in line or 'R2_K5_SEED' in line:
        print('  ', line.strip())
ok = True
for i, c in enumerate(nb2['cells']):
    if c['cell_type'] != 'code':
        continue
    try:
        compile(''.join(c['source']), f'<cell {i}>', 'exec')
    except SyntaxError as e:
        ok = False
        print('SYNTAX ERROR', i, e)
print('compile OK' if ok else 'COMPILE FAILED')
