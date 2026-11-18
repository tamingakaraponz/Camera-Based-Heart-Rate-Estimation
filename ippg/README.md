# โปรแกรม iPPG ปัจจุบัน

**ขอบเขตเอกสาร:** ตัวรันหลักสำหรับไฟล์วิดีโอคือ `run_ippg.py` ซึ่งพัก DWT และใช้ POS → bandpass → FFT; ตัวอย่าง DATA/cPPG และ DWT ด้านล่างเป็น workflow เดิมหรือการทดลองแยกและต้องมีข้อมูลในเครื่อง ค่าเริ่มต้นของโมดูลอื่นอาจต่างกัน

**ช่วงปัจจุบัน:** core iPPG และ `analyze_ground_truth.py` ใช้ bandpass/ค้นหา BPM ที่ 0.6–3.0 Hz (36–180 BPM) เป็นค่าเริ่มต้น รายงานและสคริปต์ทดลองเดิมบางตัวระบุ 0.6–2.0 Hz ไว้เองและอาจยังใช้ช่วงเก่า


## ทดลองข้อมูลทยอยเข้า 5/10/15 วินาที

รัน `replay_realtime.py` ด้วย project venv เพื่อทดลองจากคลิปทีละเฟรม คำนวณใหม่จาก RGB ที่มาถึงแล้วเท่านั้น
ดู [วิธีรัน ผลทดลอง และข้อจำกัด](REALTIME_REPLAY.md); ยังไม่ใช่เว็บหรือ benchmark กล้องสด

**กลับมาพัฒนาต่อ:** อ่าน [สถานะการพัฒนาและคะแนนความพร้อม Real time](../DEVELOPMENT_STATUS.md)
ไฟล์นี้บันทึกสิ่งที่ทำแล้ว งานถัดไป และผลตรวจที่ต้องอัปเดตหลังพัฒนาแต่ละรอบ

## บันทึกกราฟและสัญญาณสำหรับเทียบภายหลัง

`run_ippg.py` และ `analyze_ground_truth.py` บันทึกผลอัตโนมัติใต้ `outputs/signal_runs`
แต่ละรอบสร้างโฟลเดอร์ใหม่แยกกัน ไม่เขียนทับรอบเก่า และบันทึกก่อนเปิดหน้าต่างกราฟ
มี `graph.png`, `signal.csv`, `spectrum.csv`, `signals.npz`, `manifest.json` และ `RESULT.md`
ฝั่ง PPG มี `original_signal.csv` เพิ่มเพื่อเก็บสัญญาณ/BPM/timestamp เดิมก่อน interpolate
manifest บันทึก config, source/code/file hashes, ช่วงเวลาและวิธีนิยามเวลา; ใช้เฉพาะ run ที่ status เป็น complete

```powershell
& "..\.venv\Scripts\python.exe" "ippg\run_ippg.py" --no-show
& "..\.venv\Scripts\python.exe" "ippg\analyze_ground_truth.py" --no-show
```

ไม่ใส่ `--no-show` หากต้องการเปิดหน้าต่างกราฟด้วย
เปลี่ยนปลายทางได้ด้วย `--output-root "outputs/my-runs"`
โค้ดคำนวณสัญญาณไม่เปลี่ยนเพื่อการ export และไม่มีการเลื่อนเวลา กลับขั้ว หรือคูณ BPM ให้กราฟตรงกัน
แนวทางตรวจความถูกต้องเพิ่มเติมอยู่ใน `SIGNAL_COMPARISON_PLAN.md`

## อ่าน UBFC ground_truth.txt: band-pass และ FFT

ใช้ `analyze_ground_truth.py` สำหรับไฟล์ข้อความ 3 แถว:
แถว 1 = PPG, แถว 2 = BPM ที่มากับข้อมูล, แถว 3 = เวลาเป็นวินาที
โปรแกรมคำนวณใหม่จากแถว 1 และ 3 โดยไม่ใช้แถว BPM เลือก FFT peak

```powershell
& "..\.venv\Scripts\python.exe" "ippg\analyze_ground_truth.py" "C:\Users\Akkarapon\Downloads\subject1-20261104T091024Z-1-001\subject1\ground_truth.txt"
```

เมื่อไม่ส่ง path จะใช้ `FILE_PATH` ที่ตั้งไว้ด้านบนของสคริปต์ จึงกด Run ใน editor ได้
กราฟมี PPG เดิม, หลัง band-pass และ FFT พร้อมค่า BPM
ค่าเริ่มต้น Butterworth order 3, 0.6–3.0 Hz (36–180 BPM)
เปลี่ยนช่วงเองได้ด้วย `--low` และ `--high`
ใช้ `--output outputs/result-new.png --no-show` เพื่อบันทึกกราฟโดยไม่เปิดหน้าต่าง; ไม่เขียนทับ output ที่มีอยู่

Dependency เฉพาะสคริปต์นี้: `numpy scipy matplotlib` ไม่ต้องใช้กล้องหรือ MediaPipe
ช่วงเวลาที่ไม่สม่ำเสมอจะถูก interpolate; timestamp ซ้ำจะรวม amplitude ด้วยค่าเฉลี่ย
sampling rate เริ่มต้นคำนวณจากจำนวน timestamp ไม่ซ้ำและช่วงเวลารวม หรือกำหนดเองด้วย `--fs`
ไม่แก้ข้อมูลต้นฉบับและไม่ใช้ตัวคูณ BPM ผล FFT เป็นค่าประมาณรวมทั้งคลิป ไม่ใช่ผลรายจังหวะ

ทดสอบด้วยไฟล์ subject1 ข้างต้นได้ 106.97 BPM; แถว BPM ที่ให้มามีค่าเฉลี่ย 106.70 BPM
สองค่านี้เป็นคนละวิธีสรุปผล จึงไม่ใช้ส่วนต่างเพียงไฟล์เดียวเป็นข้อสรุปความแม่นยำ

ไฟล์ด้านบนของโฟลเดอร์นี้เป็น source code ที่ใช้งานปัจจุบัน ส่วน tests, assets และโค้ดเก่าแยกอยู่ในโฟลเดอร์ย่อยแล้ว

โปรแกรมหลักคือ `ippg_v2_refactored.py` และโปรแกรมเปรียบเทียบที่แนะนำคือ `compare_gr_pos_cppg.py`

ถ้าต้องการเปรียบเทียบเฉพาะ POS+DWT iPPG กับ cPPG ทั้ง waveform และ FFT ให้ใช้ `compare_ippg_cppg.py`

## เปรียบเทียบ pulse จาก iPPG และ cPPG

## แสดง G-R, POS เดิม และ POS+DWT เป็น 3 กราฟ

ใช้ `compare_gr_pos_cppg.py` สำหรับกราฟ G-R, POS เดิม และ POS+DWT เทียบ cPPG ตามลำดับ
ดึง RGB จากวิดีโอครั้งเดียว ใช้ cPPG ชุดเดียวกันและสเกลแกนเดียวกันในทั้งสามกราฟ
เก็บไฟล์ใหม่นี้ไว้ข้าง `compare_ppg.py` และโมดูลเดิมทั้งหมด

พิมพ์แค่เลข DATA ได้เลย เช่น `21`:

```powershell
& "..\.venv\Scripts\python.exe" "ippg\compare_gr_pos_cppg.py" 21
```

โปรแกรมจะหา `DATA\temp\21.avi` และ `DATAcppg\21.xlsx` ให้อัตโนมัติ
เลข `62` จะจับคู่ `DATA\temp\62.avi` กับ `DATAcppg\62.xlsx`
โดยอ้างอิงโฟลเดอร์โครงการจากตำแหน่งสคริปต์ ไม่ใช่โฟลเดอร์ที่เปิด terminal
ถ้าไฟล์ขาดจะแสดง path ที่หาไม่พบ โดยไม่เลือกไฟล์อื่นมาแทน

หรือรันโดยไม่ระบุอะไร แล้วตอบแค่เลขเมื่อโปรแกรมถาม:

```powershell
& "..\.venv\Scripts\python.exe" "ippg\compare_gr_pos_cppg.py"
```

ยังระบุ filepath เองแบบเดิมได้:

```powershell
& "..\.venv\Scripts\python.exe" "ippg\compare_gr_pos_cppg.py" "DATA\temp\11.avi" "DATAcppg\11.xlsx"
```

เพิ่ม `--zoom 5 10` เพื่อขยายทั้งสามกราฟไปที่เวลา 5–10 วินาที
การ zoom เปลี่ยนเฉพาะภาพ ไม่เปลี่ยนการ normalize หรือค่า r/MAE/RMSE ของช่วงข้อมูลร่วมทั้งหมด
หากตอบเป็น filepath ของวิดีโอแทนเลข DATA โปรแกรมจะถาม filepath ของ cPPG ต่อ

```powershell
& "..\.venv\Scripts\python.exe" "ippg\compare_gr_pos_cppg.py" "DATA\temp\11.avi" "DATAcppg\11.xlsx" --output "outputs\gr_pos.png" --metrics-json "outputs\gr_pos.json" --no-show
```

รองรับ `--duration`, `--preview`, `--no-mediapipe`, `--landmark-model`, `--cascade`
และ `--overwrite` เช่นเดียวกับไฟล์เปรียบเทียบแบบห้าจุด
ทั้งสามวิธีใช้ band-pass order 6 เทียบ cPPG order 3 ตาม pipeline เดิม
ค่าบนกราฟเป็น Z-score ไม่ใช่ค่า BPM และยังไม่ชดเชย lag หรือกลับ polarity
รายงาน Pearson r, MAE/RMSE หน่วย Z-score และ FFT BPM/error หน่วย BPM แยกกัน
FFT ทุกวิธีรวม cPPG ใช้ช่วงข้อมูลร่วมและ video grid เดียวกัน ไม่ขึ้นกับ zoom

## Wavelet หลัง POS

ติดตั้ง dependency ใน environment ที่ใช้รัน:

```powershell
& "..\.venv\Scripts\python.exe" -m pip install -r requirements-ippg.txt
```

Pipeline: RGB → resample → raw POS → DWT → soft threshold → IDWT → reconstructed POS → band-pass → FFT → BPM

- ค่าเริ่มต้น `db4`, level `min(4, dwt_max_level)`, boundary `symmetric`, strength `1.0`
- ทำ DWT กับ raw POS ทั้งช่วง ไม่ใช่แยกใน POS window 1.6 วินาที
- ประมาณ sigma จาก MAD ของ cD1 / 0.6745 และ threshold = strength × sigma × sqrt(2 ln N)
- ค่าเริ่มต้น `protect-pulse-band`: เก็บ detail ที่ทับช่วง 0.6–2.0 Hz และ threshold เฉพาะ detail ที่อยู่นอกช่วงนั้น; approximation ไม่ถูกแก้ และ IDWT ตัดปลายให้ยาวเท่าเดิม
- ใช้ `--wavelet-strategy all-details` เพื่อทำ soft threshold ทุก detail แบบเดิมสำหรับเปรียบเทียบ
- ไม่มี wavelet ใน G-R หรือ cPPG; ไม่ใช้ cPPG ปรับ threshold
- ปรับได้ด้วย `--wavelet db4 --wavelet-level auto --wavelet-strength 1.0 --wavelet-strategy protect-pulse-band`
- ตัวรันหลัก `run_ippg.py` พัก DWT แล้ว; core CLI และตัวเปรียบเทียบที่อธิบายในส่วนนี้ยังอาจใช้ DWT ตามค่า config ของตน
- โปรแกรมสามกราฟแสดง baseline และ DWT เสมอ จึงไม่มี `--denoise`
- หาก PyWavelets ขาด จะแจ้งคำสั่งติดตั้ง ไม่ข้ามขั้นตอนเงียบ ๆ
- level ระบุเองเกินขีดจำกัดจะ error; auto ลดระดับได้; ข้อมูลสั้นเกิน DWT จะเตือนและคืนสำเนาเดิม
- strength/threshold เป็นศูนย์คืนสำเนาเดิม; band-pass ยังต้องมีจำนวนจุดขั้นต่ำ; สัญญาณแบนไม่รายงาน BPM

`process_pulse()` คืน `raw`, `reconstructed`, `filtered` และ `denoise_metadata` แยกกัน
ผลโปรแกรมหลักเก็บ raw ใน `result.signals.pulse_signal`, reconstructed ใน
`result.reconstructed_pulse_signal` และ filtered ใน `result.filtered_pulse_signal`
JSON กราฟสามวิธีเก็บ wavelet/level/threshold/status และ BPM; benchmark บันทึก stages ใน NPZ เพิ่มด้วย

ทดสอบทุก DATA ที่กำหนดไว้โดยไม่เลือกเฉพาะผลดี (output ต้องเป็นโฟลเดอร์ใหม่):

```powershell
& "..\.venv\Scripts\python.exe" "ippg\benchmark_wavelet.py" --output "outputs\wavelet-run-new"
```

ค่าเริ่มต้น DATA `11 12 13 21 62` อ่านทั้งคลิป ส่งออก PNG ภาพรวม/zoom, JSON รายคลิป,
NPZ สัญญาณ และ summary.json พร้อม MAE/RMSE ของ BPM และรายงาน clip ที่ล้มเหลวด้วย
ผลทดสอบ protected strategy ล่าสุดอยู่ใน `outputs\\wavelet-protected-20260830\\VALIDATION.md`:
ช่วยป้องกัน BPM ผิดพลาดจากการ threshold ทุก detail แต่ยังไม่ดีขึ้นกว่า POS + band-pass
จึงเป็น baseline ที่ปลอดภัยกว่า ไม่ใช่การรับประกันความแม่นยำที่สูงขึ้น

## แสดงหนึ่งวิธีพร้อมตารางห้าจุด

รันคำสั่งจากโฟลเดอร์ `SCiUS-Project` ด้วย environment ของโครงการ
เก็บ `compare_ppg.py`, `ippg_v2_refactored.py` และ
`benchmark_background_methods.py` ไว้ในโฟลเดอร์เดียวกัน และติดตั้ง requirements ด้านบน

## เปิดกราฟ

```powershell
& "..\.venv\Scripts\python.exe" "ippg\compare_ppg.py" "DATA\temp\11.avi" "DATAcppg\11.xlsx"
```

หรือไม่ระบุ path เพื่อให้โปรแกรมถามทีละไฟล์:

```powershell
& "..\.venv\Scripts\python.exe" "ippg\compare_ppg.py"
```

ใช้ path แบบเต็มได้ โดยใส่เครื่องหมายคำพูดถ้า path มีช่องว่าง
cPPG ต้องเป็น `.xlsx` ชีตแรก คอลัมน์ A = amplitude และ B = timestamp **หน่วยวินาที**
(Unix timestamp หรือ relative time ก็ได้) ค่าที่ไม่ใช่ตัวเลขจะถูกข้าม

## บันทึกภาพโดยไม่เปิดหน้าต่าง

```powershell
& "..\.venv\Scripts\python.exe" "ippg\compare_ppg.py" "DATA\temp\11.avi" "DATAcppg\11.xlsx" --output "outputs\comparison_11.png" --metrics-json "outputs\comparison_11.json" --no-show
```

หากชื่อ output มีอยู่แล้ว โปรแกรมจะหยุด ให้ใช้ชื่อใหม่หรือเพิ่ม `--overwrite`
ไฟล์วิดีโอและ cPPG ต้นฉบับจะไม่ถูกแก้ไข

## เปลี่ยนเวลาที่เทียบ

```powershell
& "..\.venv\Scripts\python.exe" "ippg\compare_ppg.py" "DATA\temp\11.avi" "DATAcppg\11.xlsx" --sample-times 10 10.2 10.4 10.6 10.8 --zoom 9.8 11.2
```

- ค่าเริ่มต้น: 5.0, 5.2, 5.4, 5.6, 5.8 วินาที ต้องระบุให้ครบห้าค่า เรียงจากน้อยไปมาก
- `--zoom START END`: ต้องอยู่ในช่วงข้อมูลร่วมและครอบคลุมห้าจุดทั้งหมด
- `--duration 15`: จำกัดวิดีโอไว้ที่ 15 วินาที ค่าเริ่มต้นอ่านทั้งวิดีโอ
- `--preview`: แสดงกรอบใบหน้า/skin mask ระหว่างประมวลผล
- `--no-mediapipe`: ใช้ Haar fallback
- `--landmark-model PATH` / `--cascade PATH`: ระบุโมเดลเอง
- `--pulse-method g-r`: เปลี่ยนจาก POS เป็น G-R
- `--help`: ดูตัวเลือกทั้งหมด

## กราฟและการตีความ

1. กราฟภาพรวมของช่วงเวลาที่ข้อมูลทั้งสองมีร่วมกัน พร้อม Pearson r ของช่วงนี้
2. กราฟขยาย พร้อมจุดห้าคู่ที่เลือกไว้ล่วงหน้า
3. ตาราง iPPG, cPPG และส่วนต่างสัมบูรณ์ พร้อม MAE ของห้าจุด

ใช้ band-pass 0.6–2.0 Hz ตาม pipeline เดิม: iPPG order 6 และ cPPG order 3
นำ cPPG มา interpolate บน time grid ของวิดีโอ แล้วปรับแต่ละ pulse เป็น Z-score
ด้วยค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานจากช่วงข้อมูลร่วมทั้งหมด ก่อนเลือกห้าจุด
Interpolation ไม่ได้เพิ่มจำนวนการวัดจริงของ cPPG

ผลต่างมีหน่วย Z-score ไม่ใช่ BPM หรือเปอร์เซ็นต์ โปรแกรมนี้เทียบรูปคลื่น
ไม่ใช่ HR benchmark และไม่ได้ใช้ background เพื่อแก้รูปคลื่น

โปรแกรมสมมติว่าไฟล์เริ่มพร้อมกัน **ยังไม่ยืนยัน synchronization** และไม่หา lag
หรือกลับ polarity อัตโนมัติ ดังนั้น correlation ต่ำยังไม่พิสูจน์ว่า POS ไม่แม่น
ห้าจุดเป็นเพียงตัวอย่าง ไม่ใช่ผลสรุปทางสถิติของระบบทั้งหมด

## ทดสอบ

```powershell
& "..\.venv\Scripts\python.exe" -m unittest discover -s "ippg\tests" -p "test_*.py" -v
```

## Interface

| ไฟล์และ symbol | หน้าที่ |
|---|---|
| [run_ippg.py](run_ippg.py): `main` | CLI รับ path วิดีโอ, `--duration`, `--preview`, `--no-show`, `--output-root`; ใช้ POS/legacy/ไม่ใช้ DWT/no background |
| [ippg_v2_refactored.py](ippg_v2_refactored.py): `IPPGConfig`, `analyze_video`, `extract_video_signals`, `analyze_signals` | ตั้งค่า ดึง RGB และวิเคราะห์คลื่น คืน `IPPGResult` พร้อม stages/metadata |
| [replay_realtime.py](replay_realtime.py): `TimeBuffer`, `ReplayEngine`, `evaluate` | replay แบบหน้าต่าง trailing 5/10/15 วินาที; CLI ต้องมีวิดีโอและ reference |
| [analyze_ground_truth.py](analyze_ground_truth.py): `analyze_file`, `main` | วิเคราะห์ PPG text 3 แถว; รับ `--fs`, `--low`, `--high`, `--order` |
| [signal_recording.py](signal_recording.py): `save_bundle`, `save_ippg_run`, `save_reference_run` | บันทึก NPZ/CSV/กราฟ/manifest/report |
| [compare_dev_methods.py](compare_dev_methods.py): `chrom` | ตัวสกัด CHROM สำหรับการทดลอง แยกจาก core POS |

กลุ่มงานทดลองที่แยกจากตัวรันหลัก:

| กลุ่ม | จุดเริ่มอ่านและหน้าที่ |
|---|---|
| เปรียบเทียบคลื่นกับ cPPG | [compare_ppg.py](compare_ppg.py): `compare_filtered_signals`; [compare_gr_pos_cppg.py](compare_gr_pos_cppg.py): `prepare_comparison`; [compare_pos_ica_cppg.py](compare_pos_ica_cppg.py): `prepare_comparison` |
| อ่าน cPPG จาก Excel / ทดลอง wavelet | [benchmark_background_methods.py](benchmark_background_methods.py): `load_contact_ppg`; [benchmark_mother_wavelets.py](benchmark_mother_wavelets.py): `validate_records`, `aggregate_records` |
| ชุดข้อมูลใหม่และ POS/CHROM | [inspect_new_dataset.py](inspect_new_dataset.py): `inspect`; [benchmark_new_dataset.py](benchmark_new_dataset.py): `development_inputs`; [evaluate_waveform_dev.py](evaluate_waveform_dev.py): `measure` |
| เปรียบเทียบผลที่บันทึกแล้ว | [compare_saved_signals.py](compare_saved_signals.py): `read_run`, `compare`, `estimate_lag` |
| FFT ละเอียด / คลื่นแม่แบบ / attention | [compare_dense_fft.py](compare_dense_fft.py), [trial_short_templates.py](trial_short_templates.py), [trial_previous_bpm_real.py](trial_previous_bpm_real.py), [attention_real_video.py](attention_real_video.py) เป็นสคริปต์ทดลองที่ทำงานระดับ module |

โมดูลทดลองมี input/output เฉพาะงาน ต้องอ่าน source/config ก่อนใช้กับชุดข้อมูลใหม่; attention pilots ต้องมี Torch และไฟล์โมเดลจาก `third_party/tscan` เพิ่มเติม

## Invariants

- `ippg_v2_refactored.py:mean_skin_rgb` และ `resample_to_fps` ใช้ RGB รูป N×3 ตามลำดับ R/G/B; เวลา finite และเพิ่มขึ้นอย่างเคร่งครัด พร้อม FPS บวก
- `ippg_v2_refactored.py:process_pulse` เก็บ raw, reconstructed, filtered แยกกัน; ค่าเริ่มต้น DWT และ bandpass อยู่ใน `IPPGConfig` ส่วน G-R ไม่ผ่าน DWT
- `ippg_v2_refactored.py:estimate_heart_rate` เลือกยอด FFT ที่ใช้ได้ภายในช่วงค้นหา ค่าเริ่มต้น config 0.6–3 Hz การเปลี่ยน grid ไม่เพิ่มข้อมูลการวัด
- `replay_realtime.py:TimeBuffer` รับ observation ตามเวลาที่เพิ่มขึ้น; `ReplayEngine.push` คำนวณรอบก่อนรับเฟรมอนาคต และ `reference_window` ใช้เฉพาะ reference ที่มาถึงรอบนั้นพร้อมตรวจช่วงข้อมูลขาด
- `analyze_ground_truth.py:analyze_file` รวม timestamp ซ้ำและปฏิเสธเวลาย้อนกลับ แถว BPM ที่ให้มาถูกเก็บไว้แต่ไม่ใช้เลือกยอด FFT
- `signal_recording.py:save_ippg_run` รองรับ single POS ที่ background เป็น none; ค่า ROI quality ของ legacy เป็น placeholder และไม่ใช่ความเชื่อมั่น
- `benchmark_background_methods.py:load_contact_ppg` อ่าน XLSX โดยใช้สองคอลัมน์ตัวเลขแรกเป็น amplitude/time แล้วจัดเรียงและรวมเวลาซ้ำ; เป็น input คนละรูปแบบกับ PPG text 3 แถว
- `benchmark_new_dataset.py:development_inputs` ตรวจให้ development/held-out ไม่ซ้ำกัน พร้อมตรวจตำแหน่งและ size/mtime ของไฟล์; `compare_dev_methods.py:chrom` ใช้หน้าต่าง 1.6 วินาที overlap 50% และกรองภายใน 0.7–2.5 Hz
- `compare_saved_signals.py:read_run` ตรวจ manifest สถานะ complete, hashes และแกนเวลา; `estimate_lag` ทดลองเลื่อนเวลาภายในขอบเขตโดยไม่กลับขั้ว ควรแยกผลนี้จากการเทียบบนเวลาเดิม
- `check_cppg_five_new_people.py:FACTOR` และ `check_cppg_time_hypothesis.py` เป็นการทดลองความไวต่อสมมติฐาน 34/32 ไม่ใช่การแก้ reference ที่ยืนยันแล้ว

## Side Effects

- `ippg_v2_refactored.py:extract_video_signals` เปิดวิดีโอและ tracker มี calibration และอาจเปิด preview; `plot_result` อาจเปิดหน้าต่างกราฟ
- `run_ippg.py:main`, `analyze_ground_truth.py:main` ส่งออก successful runs; `signal_recording.py:save_bundle` สร้างโฟลเดอร์เฉพาะรอบและ manifest สถานะ writing/complete/failed พร้อม hashes
- `replay_realtime.py:main` บันทึก logs, CSV, NPZ ของแต่ละช่วงและกราฟในโฟลเดอร์เฉพาะรอบ แม้ buffer วิเคราะห์จำกัดความยาว แต่ logs/stages สะสมได้ตามความยาวคลิป
- `run_ippg.py:VIDEO_FILE`, `analyze_ground_truth.py:FILE_PATH`, `replay_realtime.py:DEFAULT_VIDEO` มี path เฉพาะเครื่องเดิม ควรส่ง path ต้นฉบับที่ต้องการเป็น argument
- `run_ippg.py:PROJECT_PYTHON` และ `replay_realtime.py:PROJECT_PYTHON` อาจเปลี่ยนไปใช้ `.venv/Scripts/python.exe` ใน parent ของ checkout โดยอัตโนมัติเมื่อรันเป็น script
- **อย่า import สคริปต์ทดลองโดยถือว่าไม่มีผลข้างเคียง:** เช่น `attention_*.py`, `compare_dense_fft.py`, `trial_short_templates.py`, `trial_previous_bpm_real.py`, `audit_waveform_morphology.py` และ `benchmark_five_ppg_peaks.py` มีคำสั่งอ่านข้อมูล/ประมวลผล/เขียนผลระดับ module บางไฟล์ใช้ output เดิมซ้ำและเขียนทับได้
- `compare_ppg.py:validate_output` ตรวจนามสกุล ไม่ให้ทับ input และต้องมี `--overwrite` หาก output มีอยู่แล้ว ข้อคุ้มครองนี้ไม่ได้ครอบคลุมสคริปต์ทดลองทั้งหมด

## Verification

รันจากโฟลเดอร์หลักด้วย environment ที่มี dependencies:

```powershell
python -m unittest discover -s ippg/tests -p "test_*.py"
```

คำสั่งนี้เป็นวิธีตรวจโค้ด ไม่ใช่หลักฐานว่ารอบ bootstrap รันทดสอบผ่านแล้ว งานทดลองที่ต้องมีวิดีโอ/PPG ต้องเตรียมไฟล์และโมเดลตาม source ก่อนรัน
