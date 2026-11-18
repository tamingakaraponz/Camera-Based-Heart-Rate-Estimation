# คัดเลือกตำแหน่ง ROI ก่อนทดสอบวิดีโอ

วันที่ 2026-11-16 | ภาพสร้างด้วย Imagegen (built-in) | สถานะ: ผู้ใช้เลือกเฉพาะ A/B/C ตรวจ overlay เท่านั้น

![ตำแหน่ง ROI บนหน้าคนตัวอย่าง](outputs/roi_selection/facial_roi_candidates_v1.png)

ภาพบุคคลสมมติใช้สื่อสารตำแหน่ง ไม่ใช่ใบหน้าของผู้เข้าร่วม ไม่ใช่แผนที่ perfusion และขอบ ROI ไม่ใช่ landmark specification ที่พร้อมใช้กับโค้ด ขนาดและตำแหน่งจริงต้องตรวจบนเฟรมวิดีโอ ซ้าย/ขวานับจากตัวบุคคล ไม่ใช่ผู้ดูภาพ

**ขอบเขตปัจจุบัน: A/B/C เท่านั้น D/E เป็นประวัติข้อเสนอ ไม่อยู่ในรอบนี้** ตรวจ 8,932 เฟรมของหกคน ไม่หลุดตรวจ แต่ A ของ Subject11 ยังทับผม B/C อยู่บนแก้มในภาพตัวอย่าง ยังไม่ยืนยันคุณภาพสัญญาณ ดู [ผลตรวจและภาพเดิม–ใหม่](outputs/roi_overlay_abc_20261116_171220/RESULT.md)

## คะแนนรอบคัดเลือก

คะแนนเป็นการประเมินเชิงออกแบบของผู้ช่วย ไม่ใช่คะแนนจาก paper โดยตรง ไม่ใช่ค่าที่วัดจากวิดีโอ ไม่ใช่เปอร์เซ็นต์ความแม่นยำ ทุกเกณฑ์ 0–5 ยิ่งสูงยิ่งน่าทดลองและให้น้ำหนักเท่ากัน:

- **หลักฐาน:** ความชัดของหลักฐานที่อ่านว่าบริเวณนั้นน่าสนใจสำหรับ rPPG
- **พื้นที่:** โอกาสได้พิกเซลผิวเพียงพอในภาพหน้าตรง ไม่ใช่ให้บริเวณใหญ่มีน้ำหนักสัญญาณมากโดยอัตโนมัติ
- **หลบสิ่งบัง:** ความสะดวกในการวาง ROI ให้หลบผม คิ้ว แว่น เครา ภายใต้การปรับตำแหน่งอย่างเหมาะสม
- **ความนิ่ง:** ความเสี่ยงเชิงออกแบบต่อการเปลี่ยนรูปจากการแสดงสีหน้าในตำแหน่งที่วาด ไม่ใช่ค่าการเคลื่อนไหวที่วัดแล้ว

| ROI | หลักฐาน | พื้นที่ | หลบสิ่งบัง | ความนิ่ง | รวม/20 | การตัดสินใจ |
|---|---:|---:|---:|---:|---:|---|
| A หน้าผากกลาง หลบแนวผม | 5 | 5 | 3 | 4 | 17 | ชุดหลักสำหรับทดลอง |
| B แก้มบนซ้าย | 5 | 4 | 4 | 3 | 16 | ชุดหลักสำหรับทดลอง |
| C แก้มบนขวา | 5 | 4 | 4 | 3 | 16 | ชุดหลักสำหรับทดลอง |
| D ผิวระหว่างคิ้ว | 4 | 2 | 3 | 3 | 12 | ทดลองเพิ่มแยกจาก A–C |
| E สันจมูกด้านบน | 3 | 2 | 3 | 3 | 11 | สำรองสำหรับการทดลองภายหลัง |

คะแนน B/C เท่ากันเพราะยังไม่มีข้อมูลแสงหรือการขยับที่รองรับให้ข้างใดชนะ ไม่ใช้คะแนนต่าง 1 จุดตัดสินว่าสัญญาณดีขึ้นจริง ตัวคะแนนนี้ต้องปรับหลังวัดข้อมูล ไม่ใช้คะแนนรวมเป็นน้ำหนัก RGB

## เหตุผลและข้อจำกัด

- **A:** สนับสนุนโดย Kwon 2015 และ Kim 2021 มีพื้นที่กว้าง แต่ในข้อมูลเรา mask เคยเลือกเส้นผม จึงต้องกำหนดขอบที่หลบผมจริง ไม่ใช้ ROI เดิมโดยถือว่าคะแนนนี้รับรองแล้ว
- **B/C:** สนับสนุนโดยงาน ROI เช่นเดียวกัน โดยเฉพาะ malar ใน Kim 2021 มีประโยชน์ให้เปรียบเทียบสองข้าง แต่เงาและมุมหน้าอาจทำให้ผลไม่เท่ากัน หลีกเลี่ยงขอบตา ริมฝีปากและรอยพับข้างจมูก
- **D:** มีหลักฐานจาก Kim 2021 แต่พื้นที่เล็กและอยู่ใกล้ขนคิ้ว ต้องตรวจว่าเหลือพิกเซลจริงพอและไม่รับ motion จากคิ้วมากเกินไป
- **E:** แยก upper nasal dorsum ออกจากปลายและด้านข้างจมูก มีพื้นที่เล็กและอาจมีแว่นหรือแสงสะท้อน ยังไม่ใช่ตัวเลือกแรก

การเลือก A–C เป็นการเลือก candidate สำหรับทดสอบ ไม่ใช่การยืนยันว่า geometry/mask/น้ำหนักของโค้ดปัจจุบันดีแล้ว ROI ทางกายวิภาคใกล้เคียงกันแต่ขอบต่างกันอาจให้ผลต่างกัน และการปรับ mask กับสูตรรวมเป็นคนละการทดลอง

## ขั้นยืนยันว่าใช้ได้จริง

1. กำหนด polygon ตาม landmarks และตรวจภาพ overlay ของทุกคนก่อน: เลือกผิวจริง ไม่กินผม ตา คิ้ว ริมฝีปาก เก็บภาพตัวอย่างและจำนวนพิกเซลตามเวลา
2. ขั้นอนาคตหลังผ่านการตรวจภาพและผู้ใช้ให้ดำเนินการ (ยังไม่รัน): สกัด RGB ของ A/B/C แยกกัน แล้วใช้ POS ไม่มี DWT, bandpass/search 0.6–3 Hz และ FFT วิธีเดียวกับ baseline เปรียบเทียบ PPG FFT บนช่วงข้อมูลร่วม; ข้าม PPG ที่ขาดตามเกณฑ์เดิม
3. รายงานแต่ละ ROI แยกคนและ 5/10/15 วินาที ทั้ง MAE/RMSE, waveform correlation แบบเวลาเดิม, spectral overlap, จำนวนช่วงที่คำนวณได้, การเคลื่อน ROI และเวลาประมวลผล
4. เปรียบเทียบ geometry/mask ใหม่กับเดิมก่อนเปลี่ยนการรวมข้อมูล จากนั้นจึงทดสอบการรวม A–C ทีละเรื่อง โดยไม่เพิ่ม D/E ในขอบเขตปัจจุบัน โดยกฎเลือกต้องใช้เฉพาะวิดีโอ ไม่ใช้ PPG เลือก ROI ย้อนหลัง
5. กำหนดเกณฑ์ยอมรับก่อนวัดผล และทดสอบคนใหม่หลังพัฒนากับข้อมูลหกคน ไม่ถือว่าผลดีในกลุ่มพัฒนายืนยันใช้ได้ทุกคน สีผิวและแสง

ตัวอย่างเกณฑ์ที่ต้องตกลงก่อนทดสอบ: mask ดีขึ้นจากภาพที่กำกับจริง, ไม่ลดสัดส่วนช่วงที่มีข้อมูลโดยไม่รายงาน, ผลแยกคนไม่แย่ลงจนถูกค่าเฉลี่ยกลบ และภาระประมวลผลยังเหมาะกับเป้าหมายกล้องสด ยังไม่มี threshold ตัวเลขที่ผ่านการยืนยันสำหรับเกณฑ์เหล่านี้ในโปรเจกต์

## แหล่งหลักฐาน

- [Kwon et al. 2015, ROI analysis for remote photoplethysmography on facial video](https://pubmed.ncbi.nlm.nih.gov/26737399/): หน้าผากและแก้มเป็น candidates; หลักฐานรอบนี้จาก abstract
- [Kim et al. 2021, Assessment of ROI Selection for Facial Video-Based rPPG](https://pmc.ncbi.nlm.nih.gov/articles/PMC8659899/): malar, glabella และ medial forehead ให้ผลดีในเงื่อนไขที่ศึกษา มีข้อจำกัดด้านชุดข้อมูลและสีผิว
- [Wong et al. 2022, Facial Surface Orientation](https://openaccess.thecvf.com/content/CVPR2022W/CVPM/html/Wong_Optimising_rPPG_Signal_Extraction_by_Exploiting_Facial_Surface_Orientation_CVPRW_2022_paper.html): มุมผิวเกี่ยวข้องกับคุณภาพ จึงไม่ควรจัดอันดับบริเวณโดยละเลยท่าหน้า
- [งานวิจัยประกอบทั้งหมด](FACIAL_SKIN_ROI_RESEARCH.md)
- [ผลตรวจ pipeline เดิม](outputs/pipeline_audit_20261116_150954/RESULT.md)
- [Baseline PPG FFT](outputs/ppg_fft_baseline_20261116_164353_032826/RESULT.md)

## Prompt ของภาพ

Use case: scientific-educational. Create an accurate, clear educational infographic to explain candidate facial skin regions of interest for camera-based photoplethysmography research, not a diagnostic or heatmap. Landscape 3:2 white/off-white background. Center a large anatomically plausible front-facing adult human face, neutral expression, medium brown skin, hair swept fully off forehead, no makeup, no glasses, no facial hair, both ears visible, even diffuse illumination. Semi-realistic medical illustration with realistic facial proportions, front view, not a photograph of a known person. Overlay five NON-OVERLAPPING translucent bounded patches on actual skin, and thin leader lines to English labels outside the face: A = 'Central forehead': compact horizontal patch on central forehead well below hairline, above and separate from eyebrows; B = 'Left upper cheek': subject's LEFT (viewer RIGHT), malar skin below lower eyelid with a clear gap avoiding the eyelid, lateral to nose, above mouth and nasolabial fold; C = 'Right upper cheek': subject's RIGHT (viewer LEFT), mirror of B; D = 'Glabella': small skin patch between eyebrows above nasal root, do not overlap eyebrow hair or A/E; E = 'Upper nasal bridge': small skin patch on upper nasal dorsum below glabella, avoiding nasal tip/sidewalls and eyes. A,B,C translucent teal/blue with solid outlines signify priority candidates; D translucent amber with dashed outline signifies optional test; E pale gray with dashed outline signifies reserve. Clearly label left/right from SUBJECT perspective in a small note 'Left/right = subject perspective'. Small footer legend exact text: 'A–C: priority candidates   D: optional test   E: reserve'. Header exact text 'FACIAL SKIN ROI CANDIDATES'. Subheader exact text 'Illustrative placement — requires video validation'. Bottom small excluded-regions note exact text 'Exclude hair, eyebrows, eyes, lips and facial hair'. Do not cover the eyes or eyebrows with candidate masks; no medical scores or estimated accuracy printed; do not draw vascular maps or suggest red means blood perfusion. High legibility, colored patch letters centered, uncluttered layout, sufficient spacing.
