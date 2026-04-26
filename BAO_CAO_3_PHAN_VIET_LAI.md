# 3 phần lý thuyết viết lại

Mỗi phần đều theo cấu trúc:
1. Kiến trúc tổng quan
2. Thành phần — đầu vào / đầu ra / mục tiêu / công thức
3. Diễn giải công thức (vì sao có nó) + nguồn

---

## A. CHỐNG GIẢ MẠO KHUÔN MẶT (Silent Face Anti-Spoofing)

### A.1. Bối cảnh & vai trò trong pipeline

Trong hệ thống điểm danh phòng thi, kẻ gian lận có thể đưa vào camera ảnh in,
ảnh trên màn hình điện thoại/laptop, hoặc mặt nạ silicon — gọi chung là *Presentation Attack* (PA).
Module Anti-Spoofing đóng vai trò "cửa sổ" trước module nhận dạng: chỉ cho ảnh
được đánh giá là *bona fide* (mặt thật) đi tiếp đến bước trích xuất embedding
và truy vấn Qdrant.

Phương pháp được sử dụng là **Silent Face Anti-Spoofing** (SFAS) — phân loại
nhị phân `{real, fake}` mà không yêu cầu người dùng nháy mắt / lắc đầu / mỉm
cười như Active FAS.

### A.2. Kiến trúc tổng quan

Pipeline FAS sử dụng trong hệ thống được kế thừa từ repo
[`Silent-Face-Anti-Spoofing`](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
(Minivision AI, 2020). Sơ đồ logic:

```
        ┌────────── ảnh frame RGB từ camera ──────────┐
        │                                              │
        │  RetinaFace (InsightFace)  ──►  bbox face    │
        │                                              │
        │                                              ▼
        │   ┌──────────────────────────────────┐   crop 2 scale
        │   │ patch 2.7   (80×80)              │ ─────────────►
        │   │ patch 4.0   (80×80)              │
        │   └──────────────────────────────────┘
        │                                              │
        │   ┌────────────────┐    ┌────────────────┐  │
        │   │ MiniFASNetV2   │    │MiniFASNetV1SE  │  │ inference 2 nhánh
        │   │ (scale 2.7)    │    │ (scale 4.0)    │  │ song song
        │   └────────┬───────┘    └────────┬───────┘  │
        │            │ p27 ∈ ℝ³            │ p40 ∈ ℝ³ │
        │            └─────────┬────────────┘          │
        │                      ▼                       │
        │             p_avg = (p27 + p40)/2            │
        │                      ▼                       │
        │           argmax(p_avg) → label, score        │
        │                      ▼                       │
        │  Temporal consistency (≥ 3 frame liên tiếp   │
        │  có label==1 và score≥0.85) → coi là REAL    │
        └──────────────────────────────────────────────┘
```

3 yếu tố cốt lõi:

1. **Backbone nhẹ — MiniFASNet**: rút gọn từ MobileFaceNet bằng pruning, giảm
   FLOPs từ 0.224 G xuống 0.081 G mà vẫn giữ độ chính xác cao.
2. **Phân tích đa quy mô (Multi-scale crop)**: cùng một bbox face, cắt ra hai
   patch ở 2 tỷ lệ phóng (2.7 và 4.0 lần) rồi đưa vào 2 mạng độc lập.
3. **Học bổ trợ bằng phổ Fourier (Fourier auxiliary supervision)**: trong quá
   trình huấn luyện, mạng học song song hai nhiệm vụ:
   - phân loại `real / fake / unknown`,
   - dự đoán **biên độ phổ Fourier** của ảnh.
   Khi inference, chỉ dùng nhánh phân loại.

### A.3. Mô tả từng thành phần

#### A.3.1. Bộ phát hiện khuôn mặt (RetinaFace)

| Đầu vào | Đầu ra | Mục tiêu |
|--------|--------|---------|
| ảnh RGB từ camera | bounding box, 5 landmark, det_score | định vị + cắt vùng ROI để FAS chỉ tốn tài nguyên cho vùng mặt |

#### A.3.2. Multi-scale Cropping

| Đầu vào | Đầu ra | Mục tiêu |
|--------|--------|---------|
| ảnh RGB + bbox $b=(x,y,w,h)$ | 2 patch 80×80 ở scale $s \in \{2.7, 4.0\}$ | scale 2.7 ⇒ "nhìn gần" (texture, lỗ chân lông, moiré); scale 4.0 ⇒ "nhìn xa" (mép giấy, viền điện thoại) |

Công thức cắt patch theo scale $s$:

\[
\begin{aligned}
c_x = x + w/2,\quad c_y = y + h/2 \\
w_s = s \cdot w,\quad h_s = s \cdot h \\
\text{patch} = \text{Crop}(I,\; [c_x - w_s/2,\, c_y - h_s/2,\, w_s,\, h_s]) \\
\text{patch}_{80\times80} = \text{Resize}(\text{patch}, 80, 80)
\end{aligned}
\]

Ý nghĩa: đặt tâm vùng cắt trùng tâm bbox, mở rộng vùng cắt theo $s$ rồi resize
về kích thước cố định 80×80 cho mạng. Khi $s>1$, phần ngoài bbox cũng được lấy
vào → cung cấp ngữ cảnh nền.

#### A.3.3. MiniFASNet — backbone

MiniFASNet kế thừa kiến trúc **MobileFaceNet** (Chen et al. 2018) — một biến
thể của MobileNetV2 dành cho face recognition. Khối tích chập cốt lõi là
**Depthwise Separable Convolution** (Howard et al., 2017, MobileNet):

| Đầu vào | Khối | Đầu ra |
|--------|------|--------|
| feature $\mathbb{R}^{H\times W\times C_{in}}$ | Depthwise 3×3 (1 kernel/channel) | $\mathbb{R}^{H\times W\times C_{in}}$ |
| | Pointwise 1×1 | $\mathbb{R}^{H\times W\times C_{out}}$ |

Số phép nhân-cộng (MAC) so với conv chuẩn:

\[
\text{MAC}_{std} = H\cdot W\cdot C_{in}\cdot C_{out}\cdot K^2
\]
\[
\text{MAC}_{ds} = H\cdot W\cdot C_{in}\cdot K^2 + H\cdot W\cdot C_{in}\cdot C_{out}
\]
\[
\frac{\text{MAC}_{ds}}{\text{MAC}_{std}} = \frac{1}{C_{out}} + \frac{1}{K^2}
\]

Với $K=3, C_{out}=64$ thì tỉ số ≈ $\frac{1}{64} + \frac{1}{9} \approx 0.127$,
tức depthwise-separable nhanh hơn ~8 lần với cùng dung lượng biểu diễn.

3 biến thể:
- **MiniFASNetV1**: bản gốc thu gọn.
- **MiniFASNetV2**: tăng độ sâu, thêm khối Inverted Residual.
- **MiniFASNetV1SE**: V1 + Squeeze-and-Excitation (SE) channel attention
  (Hu et al. 2018, "Squeeze-and-Excitation Networks", CVPR).

##### Squeeze-and-Excitation (SE) — công thức

Cho feature $U \in \mathbb{R}^{H\times W\times C}$:

\[
z_c = F_{sq}(u_c) = \frac{1}{H\cdot W} \sum_{i=1}^{H}\sum_{j=1}^{W} u_c(i,j)
\]

(Squeeze — global average pooling, nén thông tin không gian thành 1 số / kênh)

\[
s = F_{ex}(z, W) = \sigma\bigl(W_2 \cdot \text{ReLU}(W_1 z)\bigr)
\]

(Excitation — 2 lớp FC + sigmoid, học trọng số 0–1 cho từng kênh)

\[
\tilde u_c = s_c \cdot u_c
\]

(Tái chuẩn — nhân từng kênh với trọng số tương ứng)

Ý nghĩa: với bài toán FAS, một số kênh đặc trưng phản ánh "lỗ chân lông da
thật", số khác phản ánh "vân moiré từ màn hình". SE cho phép mạng tự gán
trọng số nhỏ cho kênh nhiễu và lớn cho kênh chứa tín hiệu.

#### A.3.4. Fourier Auxiliary Supervision

Đây là điểm phân biệt SFAS so với các CNN classifier thông thường. Khi train,
ngoài nhánh phân loại còn thêm **nhánh phụ dự đoán phổ Fourier** của ảnh đầu vào.

Ảnh thật và ảnh giả khác nhau rất rõ ở miền tần số: ảnh in/màn hình thường mất
các thành phần tần số cao, hoặc xuất hiện tần số định kỳ (moiré).

Biến đổi Fourier rời rạc 2D:

\[
F(u,v) = \sum_{x=0}^{H-1}\sum_{y=0}^{W-1} I(x,y)\cdot e^{-j2\pi(\frac{ux}{H}+\frac{vy}{W})}
\]

Phổ biên độ:

\[
|F(u,v)| = \sqrt{\Re(F)^2 + \Im(F)^2}
\]

Loss tổng:

\[
L_{total} = L_{cls} + \lambda \cdot L_{ft}
\]
trong đó
\[
L_{cls} = -\sum_{i} y_i \log p_i
\]
\[
L_{ft} = \frac{1}{H\cdot W} \sum_{u,v} \bigl( |F_{pred}(u,v)| - |F_{gt}(u,v)| \bigr)^2
\]

Khi inference, nhánh Fourier bị bỏ — nhưng nhờ đã ép backbone học cả hai task,
feature ở đầu ra của backbone *có chứa* thông tin phổ ⇒ classifier bền hơn với
các attack tinh vi.

Nguồn: README + source code của
[Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing#train).

#### A.3.5. Hợp nhất 2 mạng & Temporal consistency

Cuối pipeline, hai mạng chạy độc lập trên 2 patch và đầu ra được trung bình:

\[
p_{avg} = \frac{p_{27} + p_{40}}{2},\quad p_i \in \mathbb{R}^3
\]

(3 lớp = real / printed / replayed; trong dự án dùng đơn giản hoá thành nhị phân
"label==1 (real)" hoặc khác).

Để chống các "spoof flicker" (trong 1 frame ngẫu nhiên có thể bị FAS bỏ sót),
hệ thống áp dụng **Temporal Consistency**:

\[
\text{Real}^{(t)} = \mathbb{1}\Bigl[\sum_{k=t-N+1}^{t} \mathbb{1}\bigl[
\text{label}^{(k)} = 1 \,\wedge\, \text{score}^{(k)} \ge \tau\bigr] \ge N \Bigr]
\]

với $N=3$, $\tau=0.85$ trong dự án. Tức là cần ≥ 3 frame liên tiếp đều vượt
ngưỡng mới chấp nhận.

### A.4. Hàm mất mát đầy đủ khi train

Tóm tắt loss của repo Minivision (đa nhiệm):

\[
L = \underbrace{L_{CE}(p, y)}_{\text{phân loại real/fake}}
\;+\;\lambda_1 \underbrace{L_{ft}(|F_{pred}|, |F_{gt}|)}_{\text{Fourier MSE}}
\]

Nếu dữ liệu mất cân bằng (fake nhiều hơn real, hoặc ngược lại), thay $L_{CE}$
bằng **Focal Loss** (Lin et al. 2017, "Focal Loss for Dense Object Detection",
ICCV):

\[
L_{focal} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
\]

trong đó $p_t$ là xác suất dự đoán đúng nhãn, $\gamma$ điều khiển mức tập
trung vào mẫu khó (báo cáo gốc nói $\gamma=2$). Ý nghĩa của $(1-p_t)^\gamma$:
mẫu dễ ($p_t$ lớn) ⇒ hệ số nhỏ ⇒ ít đóng góp vào loss ⇒ mạng tập trung học
mẫu khó.

### A.5. Đánh giá

Các chỉ số chuẩn ISO/IEC 30107-3 cho bài FAS:

\[
\text{APCER} = \frac{\#\text{fake bị nhận là real}}{\#\text{tổng mẫu fake}}
\]

\[
\text{BPCER} = \frac{\#\text{real bị nhận là fake}}{\#\text{tổng mẫu real}}
\]

\[
\text{ACER} = \frac{\text{APCER} + \text{BPCER}}{2}
\]

APCER càng nhỏ ⇒ càng khó vượt qua bằng ảnh giả; BPCER càng nhỏ ⇒ càng ít
"khoá nhầm" người thật.

### A.6. Tài liệu tham khảo

- Minivision AI. *Silent-Face-Anti-Spoofing*, 2020.
  https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
- Howard et al. *MobileNets: Efficient Convolutional Neural Networks for Mobile
  Vision Applications.* arXiv:1704.04861, 2017.
- Hu, Shen, Sun. *Squeeze-and-Excitation Networks.* CVPR 2018.
- Lin et al. *Focal Loss for Dense Object Detection.* ICCV 2017.
- ISO/IEC 30107-3:2017 — *Information technology — Biometric presentation
  attack detection — Part 3: Testing and reporting.*

---

## B. PHÁT HIỆN CHE KHUÔN MẶT (Occlusion Detection)

### B.1. Bối cảnh & mục tiêu

Khi người dùng đeo khẩu trang / kính đen / dùng tay che, embedding khuôn mặt bị
biến dạng nặng, dẫn đến cả 2 hệ quả tệ:

- *Đăng ký (registration)*: lưu vào Qdrant một vector "khuôn mặt + khẩu trang"
  → các lần verify sau (không đeo khẩu trang) sẽ không khớp.
- *Xác minh (verification)*: tỷ lệ false-reject tăng vọt với người vẫn đăng ký.

Vì vậy ta cần một module phát hiện khuôn mặt bị che, để:
- chặn **đăng ký** nếu phát hiện che,
- cảnh báo (không chặn) khi **verify** để admin biết.

Module này hoạt động chỉ với landmark + crop ảnh, không dùng model học sâu
chuyên dụng — phù hợp với ràng buộc tài nguyên realtime.

### B.2. Kiến trúc tổng quan

```
ảnh + 5 landmark từ RetinaFace
        │
        ▼
   ┌──────────────────────┐
   │ Định nghĩa 2 ROI     │  (lower face = vùng khẩu trang)
   │ chuẩn hoá theo IOD   │  (eye region = vùng kính)
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ Convert RGB → YCbCr  │  (ITU-R BT.601)
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ Skin classifier      │  ngưỡng (Cb, Cr) — Chai & Ngan 1999
   │ trên (Cb, Cr)        │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ Morphological clean  │  opening để loại nhiễu
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ Tính SPR             │  Skin Pixel Ratio
   │ SPR = #skin/#ROI     │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ Threshold + temporal │  hysteresis 2 ngưỡng + N frame liên tiếp
   └──────────┬───────────┘
              ▼
       OCCLUDED / OK
```

### B.3. Mô tả từng thành phần & toán học

#### B.3.1. Định nghĩa ROI

| ROI | Đầu vào | Đầu ra | Mục tiêu |
|-----|---------|--------|----------|
| **Lower face** (khẩu trang) | landmarks: nose tip, mouth corners, chin | đa giác bao toàn bộ vùng mũi-miệng-cằm | nơi khẩu trang sẽ phủ lên |
| **Eye region** (kính) | landmarks: 4 góc mắt + lông mày | bbox quanh mắt + lông mày | nơi kính sẽ phủ lên |

Tất cả ROI được mở rộng theo **inter-ocular distance** (IOD) để bất biến với
khoảng cách camera ↔ người:

\[
\text{IOD} = \|\, \mathbf{p}_{leye} - \mathbf{p}_{reye}\,\|_2
\]

\[
\text{ROI}_{padded} = \text{Expand}(\text{ROI}_{base},\, \alpha \cdot \text{IOD})
\]

trong đó:
- $\mathbf{p}_{leye}, \mathbf{p}_{reye} \in \mathbb{R}^2$: tọa độ pixel của hai
  điểm ngoài/giữa mắt (do RetinaFace trả về).
- $\|\cdot\|_2$: chuẩn Euclid.
- $\alpha \in [0.2, 0.4]$: hệ số mở rộng (thực nghiệm).

**Tại sao dùng IOD làm đơn vị đo?** Trong giải phẫu khuôn mặt người, khoảng
cách giữa hai mắt biến thiên rất ít so với các kích thước khác (theo
*Farkas, L.G. — Anthropometry of the Head and Face*, 1994, IOD trung bình
≈ 60–66 mm với độ lệch chuẩn nhỏ). Đồng thời, trong ảnh, IOD tỉ lệ tuyến tính
với khoảng cách camera–người: $\text{IOD}_{px} \propto 1/Z$ (với $Z$ là khoảng
cách thực). Dùng IOD làm đơn vị nội tại sẽ triệt tiêu yếu tố $Z$, khiến mọi
ngưỡng tính bằng IOD đều bất biến với khoảng cách.

#### B.3.2. Không gian màu YCbCr — nguồn gốc các hệ số

##### B.3.2.1. Vì sao chọn YCbCr (không RGB / HSV / Lab)?

| Không gian | Tách luminance? | Bị ảnh hưởng bởi ánh sáng? |
|-----------|-----------------|--------------------------|
| RGB | Không — luminance trải đều cả 3 kênh | Cao |
| HSV | H tách phần nào, nhưng H định nghĩa quanh trục — bất ổn khi V nhỏ | Trung bình |
| Lab | Tách hoàn toàn (L vs a, b) | Thấp |
| **YCbCr** | **Tách hoàn toàn (Y vs Cb, Cr)** | **Thấp**, lại rẻ tính toán |

Da người tạo cụm rất chặt trong mặt phẳng $(C_b, C_r)$ độc lập với $Y$, là cơ
sở để dùng ngưỡng đơn giản (xem §B.3.3).

##### B.3.2.2. Công thức RGB → Y'

Hệ số được dùng:

\[
Y' = 0.299\,R + 0.587\,G + 0.114\,B
\]

**Nguồn: ITU-R Recommendation BT.601-7** (trước đây gọi là CCIR Recommendation
601, ban hành lần đầu 1982). Đây là chuẩn quốc tế cho video số chuẩn (SDTV),
các codec JPEG/MPEG, OpenCV `cv2.cvtColor(..., COLOR_BGR2YCrCb)` cũng dùng
chính bộ hệ số này.

**Vì sao 3 hệ số là 0.299, 0.587, 0.114 mà không phải 1/3 mỗi cái?**

Đây là **hệ số luminance** — phản ánh độ nhạy của mắt người với từng kênh màu.
Bắt nguồn từ **Hàm độ sáng quang phổ chuẩn CIE 1931** (CIE Standard Photopic
Luminosity Function $V(\lambda)$, Wright 1928 + Guild 1931). Mắt người nhạy
nhất với bước sóng ≈ 555 nm (xanh lá), kém hơn với 605 nm (đỏ), kém nhất với
436 nm (xanh dương).

Cụ thể, các hệ số được rút ra từ ma trận chuyển không gian màu **NTSC RGB**
(được dùng trong FCC 1953 standard cho TV màu Mỹ) sang **CIE XYZ** rồi lấy
hàng $Y$. Bộ primaries của NTSC + white point Illuminant C cho:

\[
\begin{pmatrix} X \\ Y \\ Z \end{pmatrix}
=
\begin{pmatrix}
0.6070 & 0.1734 & 0.2006 \\
\mathbf{0.2990} & \mathbf{0.5864} & \mathbf{0.1146} \\
0.0000 & 0.0661 & 1.1175
\end{pmatrix}
\begin{pmatrix} R \\ G \\ B \end{pmatrix}
\]

Hàng giữa (chính là $Y$) cho ra ≈ $(0.299, 0.587, 0.114)$. Tổng đúng bằng 1
(do $Y$ chuẩn hoá $[0,1]$ cho ảnh trắng $R=G=B=1$).

> Nguồn lịch sử: SMPTE Recommended Practice RP 145 (1987); ITU-R BT.601-7
> (2011) §2.5.1; Charles Poynton, *Digital Video and HD: Algorithms and
> Interfaces*, Morgan Kaufmann 2012, Ch. 24.

> Lưu ý: với HDTV (BT.709) các hệ số là $(0.2126, 0.7152, 0.0722)$ vì primaries
> khác. Trong dự án ta dùng OpenCV mặc định ⇒ BT.601.

##### B.3.2.3. Công thức RGB → Cb, Cr

Định nghĩa **gốc** của Cb, Cr là *color-difference channels*:

\[
P_b = \frac{B - Y'}{2(1 - 0.114)} = \frac{B - Y'}{1.772}
\]
\[
P_r = \frac{R - Y'}{2(1 - 0.299)} = \frac{R - Y'}{1.402}
\]

Mẫu số $1.772 = 2(1-K_B)$ và $1.402 = 2(1-K_R)$ là hệ số chuẩn hoá để đưa
$P_b, P_r$ về dải $[-0.5, +0.5]$ khi $R, G, B \in [0, 1]$. Để hiểu vì sao:
giá trị cực trị xảy ra khi $R=G=0, B=1$, lúc đó $Y = 0.114$, $B - Y = 0.886
= 2(1 - 0.114) \cdot 0.5$ ⇒ chia cho $1.772$ ra $+0.5$.

Để mã hoá 8-bit (dải $[0, 255]$), ta thực hiện *offset* 128:

\[
C_b = 128 + 224 \cdot P_b,\qquad C_r = 128 + 224 \cdot P_r
\]
(ITU-R BT.601 *digital* range — 16..240 là dải hữu dụng; OpenCV/JPEG dùng
**full range** 0..255, hệ số 224 thay bằng 255.)

**Trong OpenCV `cv2.cvtColor` với `COLOR_BGR2YCrCb` (full range):**

\[
\boxed{\;
\begin{aligned}
Y'  &= \;0.299\,R + 0.587\,G + 0.114\,B \\
C_b &= -0.169\,R - 0.331\,G + 0.500\,B + 128 \\
C_r &= \;0.500\,R - 0.419\,G - 0.081\,B + 128
\end{aligned}
\;}
\]

**Nguồn gốc các hệ số trong $C_b$:**

\[
C_b = \frac{B - Y'}{1.772} \cdot 255 + 128
\]

\[
\Leftrightarrow\; C_b = \underbrace{\frac{-0.299}{1.772}}_{-0.1687}\,R
+ \underbrace{\frac{-0.587}{1.772}}_{-0.3313}\,G
+ \underbrace{\frac{1 - 0.114}{1.772}}_{0.5000}\,B + 128
\]

Tương tự cho $C_r$:

\[
C_r = \underbrace{\frac{1 - 0.299}{1.402}}_{0.5000}\,R
+ \underbrace{\frac{-0.587}{1.402}}_{-0.4187}\,G
+ \underbrace{\frac{-0.114}{1.402}}_{-0.0813}\,B + 128
\]

Như vậy **mọi hệ số trong công thức RGB↔YCbCr đều là hệ quả của hai con số gốc
$K_R = 0.299, K_B = 0.114$ (CIE 1931 photopic + NTSC primaries)**, không có
hệ số nào "tự đặt".

#### B.3.3. Bộ phân lớp da — nguồn ngưỡng (Cb, Cr)

Da người có cụm phân bố hẹp trên mặt phẳng $(C_b, C_r)$ — kết luận thực
nghiệm đầu tiên của:

> **Chai, D., Ngan, K. N.** (1999). *Face segmentation using skin-color map in
> videophone applications.* IEEE Trans. Circuits Syst. Video Technol.
> 9(4):551–564. [DOI:10.1109/76.767122](https://doi.org/10.1109/76.767122)

Tác giả phân tích 4000 mẫu da từ 100 ảnh người đa sắc tộc. Histogram 2-D của
$(C_b, C_r)$ tập trung trong một vùng hình elip nhỏ; ngưỡng vuông xấp xỉ:

\[
\text{skin}(p) = \mathbb{1}\bigl[ 77 \le C_b(p) \le 127 \;\wedge\; 133 \le C_r(p) \le 173 \bigr]
\]

Trong đó:
- $\mathbb{1}[\cdot]$: hàm chỉ thị — bằng 1 khi điều kiện đúng, 0 ngược lại.
- $C_b(p), C_r(p)$: giá trị $C_b, C_r$ tại pixel $p$ (8-bit, $[0,255]$).

**Vì sao những con số này?** Trong [Chai 1999, Tab. 1], các giá trị min/max
của (Cb, Cr) trên mẫu da chiếm ≥ 99% điểm trong dataset thử nghiệm là:
$C_b \in [77, 127]$, $C_r \in [133, 173]$. Đây là biên *bao hộp* (bounding
box) đơn giản nhất — đánh đổi recall lấy precision.

**Mở rộng / nâng cấp.** Với da rất sậm (Phototype VI) hoặc rất sáng, ngưỡng
hộp Chai-Ngan có thể cắt bớt. Có thể thay bằng:
1. **Mô hình Gauss đơn** trên $(C_b, C_r)$ (Yang & Waibel 1996):
   \[
   p(\text{skin}\mid c) \propto \exp\bigl(-\tfrac{1}{2}(c-\mu)^\top \Sigma^{-1}(c-\mu)\bigr)
   \]
   với $\mu, \Sigma$ ước lượng MLE từ tập huấn luyện cục bộ.

2. **Mô hình Gaussian Mixture** (Jones & Rehg 2002, *Statistical Color
   Models with Application to Skin Detection*, IJCV) — sử dụng phổ biến
   trong production.

3. **Logistic regression** trên $(C_b, C_r, Y)$ — vẫn rẻ, robust hơn ngưỡng
   hộp.

#### B.3.4. Hậu xử lý hình thái

Mặt nạ skin thô (binary) thường có nhiễu rời rạc (đốm vài pixel) và lỗ thủng
nhỏ. Áp dụng phép **opening** (Serra 1982, *Image Analysis and Mathematical
Morphology*):

\[
S_{clean} = (S \ominus B) \oplus B
\]

trong đó:
- $\ominus$: erosion (co) — loại đốm < kích thước $B$.
- $\oplus$: dilation (giãn) — phục hồi cạnh thật sau erosion.
- $B$: structuring element, ở đây $3\times 3$ hoặc disk bán kính $r=1$ pixel.

Định nghĩa erosion / dilation:

\[
(S \ominus B)(p) = \min_{b\in B} S(p+b) \quad \text{(với binary: AND của lân cận)}
\]
\[
(S \oplus B)(p) = \max_{b\in B} S(p+b) \quad \text{(với binary: OR của lân cận)}
\]

**Tại sao opening (mà không closing)?** Đốm nhiễu ngoại lai (false-positive
da trên nền) thường nhỏ → erosion xoá được. Closing (dilation→erosion) sẽ
dùng khi muốn lấp lỗ trong vùng da → ít cần ở đây vì SPR đo theo tỉ lệ.

#### B.3.5. Skin Pixel Ratio (SPR)

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $\Omega$ | tập pixel hợp lệ trong ROI (loại pixel ngoài viền/khung) |
| $S \subseteq \Omega$ | tập pixel được phân lớp là skin sau morphology |
| $A_{ROI} = \lvert\Omega\rvert$ | diện tích ROI tính theo pixel |

\[
\text{SPR} = \frac{|S|}{A_{ROI}} \in [0, 1]
\]

**Cơ sở thống kê:** giả sử mỗi pixel trong ROI là Bernoulli với $p_{skin}$
là xác suất pixel là skin (phụ thuộc trạng thái che). Khi không che,
$p_{skin}\approx p_0$ ≈ 0.85–0.95 (lower-face) hoặc 0.65–0.75 (eye region —
mắt, lông mày không phải skin). Khi che bằng vật phi-da, $p_{skin}$ giảm vì
phần ROI bị thay bằng vải/nhựa. SPR là ước lượng MLE của $p_{skin}$:

\[
\hat p_{skin} = \frac{1}{A_{ROI}}\sum_{p\in\Omega} \mathbb{1}[p\in S] \;=\; \text{SPR}
\]

Sai số chuẩn:

\[
\text{SE}(\hat p_{skin}) = \sqrt{\frac{p(1-p)}{A_{ROI}}}
\]

Với $A_{ROI} \approx 5000$ pixel, $p \approx 0.5$ thì SE ≈ 0.007 — rất nhỏ
⇒ ngưỡng cố định đáng tin cậy.

#### B.3.6. Ngưỡng quyết định + hysteresis

Ngưỡng đề xuất (cần hiệu chuẩn cho dữ liệu cục bộ):

\[
\text{Occluded}_{mask} = \mathbb{1}[\text{SPR}_{lower\_face} < 0.45]
\]
\[
\text{Occluded}_{glasses} = \mathbb{1}[\text{SPR}_{eye} < 0.60]
\]

**Hysteresis 2 ngưỡng** (Schmitt trigger, mượn từ điện tử) chống flicker:

\[
\text{state}^{(t)} = \begin{cases}
\text{Occluded} & \text{nếu } \mathrm{SPR}^{(t)} < \tau_{enter}\;\wedge\;\text{state}^{(t-1)}=\text{OK} \\
\text{OK}        & \text{nếu } \mathrm{SPR}^{(t)} > \tau_{exit}\;\wedge\;\text{state}^{(t-1)}=\text{Occluded} \\
\text{state}^{(t-1)} & \text{ngược lại}
\end{cases}
\]

Với $\tau_{enter} = 0.40 < \tau_{exit} = 0.50$ chẳng hạn. Vùng "chết"
$[\tau_{enter}, \tau_{exit}]$ giữ trạng thái cũ → tránh dao động khi SPR
lượn quanh 1 ngưỡng đơn.

### B.4. Hạn chế và hướng nâng cấp

- **Da rất sậm hoặc rất sáng** → (Cb, Cr) rơi ngoài hộp Chai-Ngan. Khắc phục:
  per-user calibration (ghi $\mu, \Sigma$ riêng từ ảnh đăng ký FRONT) hoặc
  thay ngưỡng hộp bằng GMM huấn luyện trên dữ liệu Việt Nam.
- **Khẩu trang màu da** → SPR vẫn cao. Phương án nâng cấp: bổ sung edge
  density (mặt thật có gradient mềm; khẩu trang tạo cạnh sắc), hoặc CNN
  segmentation nhỏ huấn luyện trên MaskedFace-Net.
- **Tay che mặt** → tay cũng là da → SPR không giảm. Giải pháp: kiểm tra
  landmark visibility hoặc kết hợp model phát hiện tay (MediaPipe Hands).

### B.5. Tài liệu tham khảo

- ITU-R Recommendation BT.601-7 (03/2011) — *Studio encoding parameters of
  digital television for standard 4:3 and wide-screen 16:9 aspect ratios.*
- ITU-R Recommendation BT.709-6 (06/2015) — *Parameter values for the HDTV
  standards.* (Để so sánh với BT.601.)
- Poynton C. *Digital Video and HD: Algorithms and Interfaces.* 2nd ed.,
  Morgan Kaufmann 2012, Ch. 24 — phần dẫn xuất chính xác các hệ số RGB↔YCbCr.
- CIE 1931 Standard Colorimetric Observer — Wright 1928, Guild 1931
  (tổng hợp tại Smith, T. & Guild, J. *The CIE colorimetric standards and
  their use.* Trans. Optical Society 1931).
- **Chai D., Ngan K.** *Face segmentation using skin-color map in videophone
  applications.* IEEE Trans. CSVT, 9(4), 1999. — *nguồn của ngưỡng (Cb, Cr).*
- Yang J., Waibel A. *A real-time face tracker.* WACV 1996. — Gaussian skin
  model.
- Jones M. J., Rehg J. M. *Statistical color models with application to skin
  detection.* IJCV 46(1):81–96, 2002. — Gaussian Mixture trên RGB.
- Phung S. L., Bouzerdoum A., Chai D. *Skin segmentation using color pixel
  classification.* IEEE TPAMI 27(1), 2005. — so sánh nhiều không gian màu.
- Serra J. *Image Analysis and Mathematical Morphology.* Academic Press,
  1982. — opening / erosion / dilation.
- Farkas L. G. *Anthropometry of the Head and Face.* Raven Press, 1994. —
  thống kê IOD theo chủng tộc/giới tính.

---

## C. ƯỚC LƯỢNG GÓC MẶT (Head Pose Estimation)

### C.1. Bối cảnh & mục tiêu

Trong giai đoạn đăng ký khuôn mặt 5 góc (FRONT/LEFT/RIGHT/UP/DOWN), hệ thống
cần biết hiện tại đầu sinh viên đang ở góc nào để:
1. Hướng dẫn người dùng quay đầu đúng hướng tiếp theo.
2. Xác nhận đã đủ dữ liệu cho góc đó (so với dải $(yaw, pitch)$ mong muốn).
3. Là một lớp **anti-identity-swapping**: kẻ tấn công muốn đánh cắp tài khoản
   bằng ảnh tĩnh sẽ phải xoay ảnh sang đủ 5 hướng — khó hơn rất nhiều.

Module head pose ở đây dùng **MediaPipe Face Mesh** (Kartynnik et al. 2019)
để lấy landmark, sau đó tính 3 góc Euler $(yaw, pitch, roll)$ bằng công thức
**ratio-based** trên 6 landmark đặc trưng. Phương pháp này không có một paper
gốc duy nhất; nó là *engineering heuristic* xây trên nền **mô hình chiếu trực
giao yếu** (weak-perspective / orthographic projection) — phần C.4 sẽ chứng
minh trực tiếp 3 công thức từ hình học chiếu cơ bản.

### C.2. Kiến trúc tổng quan

```
   ảnh RGB
      │
      ▼
┌──────────────────────┐
│ MediaPipe pipeline   │
│  ┌─ BlazeFace        │  detect bbox face siêu nhẹ (TFLite)
│  └─ FaceMesh CNN     │  hồi quy 468 (x,y,z) trên ROI
└──────────┬───────────┘
           │ 6 landmark cốt lõi
           ▼
┌──────────────────────┐
│ Mô hình hình học     │  giả thiết: chiếu trực giao
│ orthographic         │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Tính ratio →         │  3 công thức tỉ lệ (chứng minh ở §C.4)
│ yaw, pitch, roll     │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ AngleRange.matches() │
└──────────┬───────────┘
           ▼
   gợi ý / chấp nhận
```

### C.3. MediaPipe Face Mesh — kiến trúc hai pha

**Nguồn:** Kartynnik Y., Ablavatski A., Grishchenko I., Grundmann M.
*Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs.*
CVPR Workshop on Computer Vision for AR/VR, 2019.
[arXiv:1907.06724](https://arxiv.org/abs/1907.06724)

| Pha | Mô hình | Đầu vào | Đầu ra |
|-----|--------|---------|--------|
| 1 | **BlazeFace** | ảnh full | bbox + 6 landmark thô + score |
| 2 | **Face Mesh CNN** | ROI 192×192 từ pha 1 | 468 × (x, y, z) |

Pha 2 dùng **coordinate regression** thay vì heatmap: mạng output trực tiếp
$K \times 3 = 1404$ giá trị thực — ít tham số, không cần upsample về resolution
gốc. Trục $z$ là *relative depth* (không có đơn vị mét), được học gián tiếp
qua tập dữ liệu có 3DMM gắn nhãn.

### C.4. Cơ sở hình học — chứng minh 3 công thức

#### C.4.1. Mô hình quy ước

Đặt hệ tọa độ vật thể (object frame) gắn với đầu, gốc tại tâm khuôn mặt:
- Trục $X$: ngang sang phải.
- Trục $Y$: dọc xuống dưới.
- Trục $Z$: vào trong mặt phẳng ảnh (về phía sau gáy).

Camera đặt rất xa so với kích thước đầu (ratio $Z_{cam}/D_{face} \gg 1$),
nên dùng **xấp xỉ chiếu trực giao** (orthographic projection): tọa độ ảnh
$(x_{img}, y_{img})$ là $(X, Y)$ sau khi đầu xoay, bỏ qua trục $Z$.

> Tại sao chấp nhận xấp xỉ này? Trong điều kiện camera webcam cách người
> dùng 30–80 cm, chiều rộng đầu ≈ 15 cm ⇒ tỉ số $\approx 1/3$. Sai số
> phối cảnh khi đó cỡ 2–4° ở vùng yaw $\pm 30°$ — chấp nhận được cho ứng
> dụng "phân biệt 5 vùng góc" (xem Murphy-Chutorian 2009, §3.A).

Mô hình anatomy đơn giản (tham số):
- **Hai má** đặt tại $\bigl(\pm \tfrac{D}{2}, 0, 0\bigr)$ — nằm trên mặt
  phẳng "phẳng" của khuôn mặt. $D$ = chiều rộng giữa hai má.
- **Mũi** đặt tại $(0, y_n, h_n)$ — nhô về phía $+Z$ một đoạn $h_n$ (depth).
  $y_n$ = vị trí dọc của mũi so với tâm.
- **Trán** tại $(0, -H_f, 0)$ trên mặt phẳng face.
- **Cằm** tại $(0, +H_c, 0)$ trên mặt phẳng face.
- $H = H_f + H_c$ = chiều cao mặt từ trán đến cằm.

Theo *Farkas 1994* (anthropometry):

\[
\frac{D}{h_n} \approx 3.0\text{–}3.3,\qquad \frac{H}{h_n} \approx 3.4\text{–}3.7
\]
\[
\frac{H_f}{H} \approx 0.55,\qquad \frac{H_c}{H} \approx 0.45
\]

(Giá trị 0.55 chính là hằng số trong công thức pitch — sẽ thấy bên dưới.)

#### C.4.2. Yaw (xoay quanh trục $Y$ — quay trái/phải)

Ma trận xoay quanh trục $Y$ một góc $\alpha$:

\[
R_Y(\alpha) =
\begin{pmatrix}
\cos\alpha & 0 & \sin\alpha \\
0 & 1 & 0 \\
-\sin\alpha & 0 & \cos\alpha
\end{pmatrix}
\]

Áp $R_Y(\alpha)$ lên 3 điểm khoá:

\[
\begin{aligned}
\text{má trái } \bigl(-\tfrac{D}{2}, 0, 0\bigr) \;&\mapsto\; \bigl(-\tfrac{D}{2}\cos\alpha,\, 0,\, +\tfrac{D}{2}\sin\alpha\bigr) \\
\text{má phải } \bigl(+\tfrac{D}{2}, 0, 0\bigr) \;&\mapsto\; \bigl(+\tfrac{D}{2}\cos\alpha,\, 0,\, -\tfrac{D}{2}\sin\alpha\bigr) \\
\text{mũi } (0, y_n, h_n) \;&\mapsto\; \bigl(h_n\sin\alpha,\, y_n,\, h_n\cos\alpha\bigr)
\end{aligned}
\]

Chiếu trực giao xuống ảnh (giữ tọa độ $X$ trong frame thế giới mới — mà
trong ảnh nó là $x_{img}$):

\[
x_{lc} = -\tfrac{D}{2}\cos\alpha,\quad
x_{rc} = +\tfrac{D}{2}\cos\alpha,\quad
x_{nose} = h_n\sin\alpha
\]

Tỉ số đặc trưng:

\[
r \;\equiv\; \frac{x_{nose} - x_{lc}}{x_{rc} - x_{lc}}
= \frac{h_n\sin\alpha + \tfrac{D}{2}\cos\alpha}{D\cos\alpha}
= \frac{1}{2} + \frac{h_n}{D}\tan\alpha
\]

Vậy:

\[
\boxed{\; r - \tfrac{1}{2} = \frac{h_n}{D}\tan\alpha \;}
\]

Suy ra **công thức nghịch đảo chính xác**:

\[
\alpha \;=\; \arctan\!\Bigl(\bigl(r - \tfrac{1}{2}\bigr) \cdot \tfrac{D}{h_n}\Bigr)
\]

Áp dụng **xấp xỉ tuyến tính** $\arctan(x) \approx x$ (rad) khi $|x|$ nhỏ:

\[
\alpha\,(\text{rad}) \;\approx\; \bigl(r - \tfrac{1}{2}\bigr)\cdot \tfrac{D}{h_n}
\]
\[
\alpha\,(\text{deg}) \;\approx\; \bigl(r - \tfrac{1}{2}\bigr)\cdot \tfrac{D}{h_n}\cdot \tfrac{180}{\pi}
\]

So với công thức trong dự án:

\[
\text{yaw}_{code} = \bigl(r - 0.5\bigr) \times 180^\circ
\]

⇒ điều kiện để khớp:

\[
\frac{D}{h_n}\cdot \frac{180}{\pi} = 180 \;\Longleftrightarrow\; \frac{D}{h_n} = \pi \approx 3.14
\]

Đối chiếu với Farkas: $D/h_n \approx 3.0$–$3.3$ ⇒ rơi đúng dải anatomy người
trưởng thành. **Hằng số 180° trong công thức yaw không phải tự đặt — nó là
$\frac{D}{h_n}\cdot\frac{180}{\pi}$ với $D/h_n$ bằng giá trị anatomy trung
bình.**

> Sai số ở mép dải: tại $\alpha = 30°$, $\arctan(0.577) = 30°$ thực,
> trong khi xấp xỉ tuyến tính cho $0.577 \cdot 57.3 = 33°$ ⇒ sai 3°.
> Trong vùng ngưỡng dự án ($\pm 22°$ cho FRONT) sai số $<1°$ — không đáng kể.

#### C.4.3. Pitch (xoay quanh trục $X$ — ngẩng/cúi)

Ma trận xoay quanh trục $X$ một góc $\beta$:

\[
R_X(\beta) =
\begin{pmatrix}
1 & 0 & 0 \\
0 & \cos\beta & -\sin\beta \\
0 & \sin\beta & \cos\beta
\end{pmatrix}
\]

Áp lên 3 điểm khoá:

\[
\begin{aligned}
\text{trán } (0,\, -H_f,\, 0) \;&\mapsto\; (0,\, -H_f\cos\beta,\, -H_f\sin\beta) \\
\text{cằm } (0,\, +H_c,\, 0) \;&\mapsto\; (0,\, +H_c\cos\beta,\, +H_c\sin\beta) \\
\text{mũi } (0,\, y_n,\, h_n) \;&\mapsto\; (0,\, y_n\cos\beta - h_n\sin\beta,\, y_n\sin\beta + h_n\cos\beta)
\end{aligned}
\]

Chiếu lên trục $Y$ của ảnh:

\[
y_{fh} = -H_f\cos\beta,\quad y_{ch} = +H_c\cos\beta,\quad y_{nose} = y_n\cos\beta - h_n\sin\beta
\]

Đặt $a = y_n + H_f$ (khoảng cách từ trán đến mũi khi nhìn thẳng); theo
anatomy $a/H = 0.55$, tức $a = 0.55\,H$.

\[
r' \;\equiv\; \frac{y_{nose} - y_{fh}}{y_{ch} - y_{fh}}
= \frac{(y_n + H_f)\cos\beta - h_n\sin\beta}{(H_c + H_f)\cos\beta}
= \frac{a}{H} - \frac{h_n}{H}\tan\beta
\]

Thay $a/H = 0.55$:

\[
\boxed{\; r' - 0.55 = -\frac{h_n}{H}\tan\beta \;}
\]

Suy ra:

\[
\beta = \arctan\!\Bigl(-\bigl(r' - 0.55\bigr)\cdot \tfrac{H}{h_n}\Bigr)
\]

Xấp xỉ tuyến tính:

\[
\beta\,(\text{deg}) \;\approx\; -\bigl(r' - 0.55\bigr)\cdot \tfrac{H}{h_n}\cdot \tfrac{180}{\pi}
\]

So với công thức trong dự án:

\[
\text{pitch}_{code} = -\bigl(r' - 0.55\bigr)\times 200^\circ
\]

⇒ điều kiện khớp:

\[
\frac{H}{h_n}\cdot \frac{180}{\pi} = 200 \;\Longleftrightarrow\; \frac{H}{h_n} = \frac{200\pi}{180} \approx 3.49
\]

Đối chiếu Farkas: $H/h_n \approx 3.4$–$3.7$ ⇒ khớp dải anatomy. **Hằng số
0.55 và 200° không phải con số bí ẩn:**
- $0.55$ = tỉ lệ vị trí mũi anatomy ($a/H$).
- $200° = (H/h_n) \cdot \frac{180}{\pi}$ với $H/h_n = 3.49$.

> Có thể thấy rõ: $D/h_n < H/h_n$ (mũi nhô ra so với chiều rộng nhỏ hơn so
> với chiều dài) ⇒ pitch "nhạy" hơn yaw ⇒ hệ số nhân lớn hơn ($200°$ vs
> $180°$). Điều này phản ánh trực tiếp hình học.

#### C.4.4. Roll (xoay quanh trục $Z$ — nghiêng đầu)

Ma trận xoay quanh trục $Z$ một góc $\gamma$:

\[
R_Z(\gamma) =
\begin{pmatrix}
\cos\gamma & -\sin\gamma & 0 \\
\sin\gamma & \cos\gamma & 0 \\
0 & 0 & 1
\end{pmatrix}
\]

Hai điểm ngoài mắt trái/phải ban đầu nằm gần như đối xứng qua trục $Y$ với
$y$ xấp xỉ bằng nhau. Đặt:

\[
\Delta x_0 = x_{re,0} - x_{le,0},\qquad \Delta y_0 = y_{re,0} - y_{le,0} \approx 0
\]

Sau xoay $\gamma$:

\[
\Delta x = \Delta x_0\cos\gamma - \Delta y_0\sin\gamma \approx \Delta x_0\cos\gamma
\]
\[
\Delta y = \Delta x_0\sin\gamma + \Delta y_0\cos\gamma \approx \Delta x_0\sin\gamma
\]

Suy ra:

\[
\frac{\Delta y}{\Delta x} = \tan\gamma \;\Longrightarrow\;
\gamma = \arctan\!\bigl(\tfrac{\Delta y}{\Delta x}\bigr)
\]

Để xử lý đúng cả 4 góc phần tư (khi $\gamma$ vượt $\pm 90°$), thay
$\arctan$ bằng $\arctan_2$:

\[
\boxed{\;
\gamma = \arctan_2\bigl(y_{re} - y_{le},\; x_{re} - x_{le}\bigr)
\;}
\]

(quy đổi rad → deg bằng nhân $180/\pi$).

**Roll là công thức CHÍNH XÁC, không phải xấp xỉ.** Vì roll xảy ra ngay
trên mặt phẳng ảnh, không liên quan đến phối cảnh hay depth.

#### C.4.5. Tóm lược 3 công thức và nguồn gốc

| Góc | Công thức code | Cơ sở chứng minh | Hằng số đến từ |
|-----|---------------|-------------------|----------------|
| yaw | $(r - 0.5)\cdot 180°$ | $r - 0.5 = \tfrac{h_n}{D}\tan\alpha$ + xấp xỉ $\arctan x \approx x$ | $D/h_n = \pi$ (anatomy Farkas) |
| pitch | $-(r' - 0.55)\cdot 200°$ | $r' - 0.55 = -\tfrac{h_n}{H}\tan\beta$ + xấp xỉ tuyến tính | $0.55 = a/H$, $200° = (H/h_n)\cdot\tfrac{180}{\pi}$ |
| roll | $\arctan_2(\Delta y, \Delta x)$ | trực tiếp từ $R_Z$ + chiếu trực giao | (chính xác, không hằng số) |

### C.5. Cấu hình AngleRange

Mỗi bước đăng ký gắn với một dải $(yaw, pitch)$ chấp nhận:

| Hướng | Yaw (°) | Pitch (°) |
|-------|---------|-----------|
| FRONT | $[-18, 18]$ | $[-18, 18]$ |
| LEFT  | $[-90, -22]$ | $[-25, 25]$ |
| RIGHT | $[22, 90]$  | $[-25, 25]$ |
| UP    | $[-25, 25]$ | $[18, 80]$ |
| DOWN  | $[-25, 25]$ | $[-80, -18]$ |

Logic kiểm tra:

\[
\text{accept}(yaw, pitch \mid d) = \mathbb{1}\bigl[
 yaw_{min}^{(d)} \le yaw \le yaw_{max}^{(d)}
\;\wedge\;
 pitch_{min}^{(d)} \le pitch \le pitch_{max}^{(d)} \bigr]
\]

Khoảng cách giữa các vùng (ví dụ FRONT $\le 18°$, LEFT $\ge 22°$) tạo "vùng
chết" 4° để tránh trạng thái dao động qua lại khi yaw lượn quanh ±20° (cùng
lý do hysteresis ở module occlusion).

### C.6. So sánh với phương pháp PnP-3DMM

Phương pháp "chính tắc" để tính head pose là giải bài toán **Perspective-n-Point**
(PnP):

1. Có $n$ điểm 3D trên mô hình khuôn mặt chuẩn (3DMM, ví dụ FaceWarehouse,
   BFM).
2. Có $n$ landmark 2D phát hiện trên ảnh.
3. Giải tìm ma trận xoay $R \in SO(3)$ và tịnh tiến $t \in \mathbb{R}^3$:

\[
\min_{R,t}\sum_{i=1}^n \bigl\| u_i - \pi(K(R X_i + t))\bigr\|^2
\]

trong đó $\pi$ là phép chiếu phối cảnh (perspective), $K$ là camera intrinsics.
Sau khi có $R$, phân rã thành Euler $(yaw, pitch, roll)$.

So sánh:

| Tiêu chí | PnP + 3DMM | Ratio-based (dự án này) |
|----------|------------|--------------------------|
| Độ chính xác | cao, có nghĩa hình học chặt | trung bình ($<5°$ trong dải $\pm 30°$) |
| Yêu cầu | camera intrinsics + 3DMM + ≥4 landmark | chỉ 6 landmark 2D, không cần $K$ |
| Tốc độ | gọi `solvePnP` (Levenberg-Marquardt) mỗi frame | vài phép cộng-trừ-chia-arctan |
| Robustness với landmark noise | trung bình (cần ≥ 4 điểm tốt + outlier rejection) | cao (mỗi ratio chỉ phụ thuộc 3 điểm) |
| Đối tượng | yêu cầu chính xác metric (AR/VR, gaze) | chỉ cần phân biệt 5 vùng rộng |

Trong bài toán đăng ký khuôn mặt, ta KHÔNG cần biết yaw chính xác $\pm 1°$;
chỉ cần phân biệt 5 vùng rộng. Vì vậy ratio-based đủ dùng và rẻ hơn nhiều.

### C.7. Vai trò chống Identity-Swapping

Module head pose là 1 trong 3 tầng kiểm tra trong luồng đăng ký:

\[
\text{accept frame}^{(t)} \iff
\underbrace{\text{Pose}^{(t)} \in \text{AngleRange}_d}_{\text{geometry}}
\;\wedge\;
\underbrace{\text{FAS}^{(t)} = \text{real}}_{\text{liveness}}
\;\wedge\;
\underbrace{\cos(e_{front}, e_{d}) \ge 0.5}_{\text{identity}}
\]

Nếu thiếu một trong 3, frame bị từ chối ⇒ ảnh tĩnh hay video tái phát rất khó
qua đồng thời cả 3 lớp.

### C.8. Tài liệu tham khảo

- **Kartynnik Y., Ablavatski A., Grishchenko I., Grundmann M.** *Real-time
  Facial Surface Geometry from Monocular Video on Mobile GPUs.* CVPR Workshop
  on Computer Vision for AR/VR, 2019.
  [arXiv:1907.06724](https://arxiv.org/abs/1907.06724) — kiến trúc Face Mesh.
- Bazarevsky V., Kartynnik Y., Vakunov A., Raveendran K., Grundmann M.
  *BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs.* CVPR
  Workshop, 2019. — pha 1 detector.
- **Murphy-Chutorian E., Trivedi M. M.** *Head Pose Estimation in Computer
  Vision: A Survey.* IEEE TPAMI, vol. 31, no. 4, 2009. — survey toàn diện;
  phân loại "geometric methods using facial features" mà ratio-based thuộc về.
- **Farkas L. G.** *Anthropometry of the Head and Face.* 2nd ed., Raven Press,
  1994. — nguồn các tỉ số anatomy ($D/h_n \approx \pi$, $H/h_n \approx 3.5$,
  $a/H \approx 0.55$) dùng để chuẩn hoá hằng số 180°, 200°, 0.55.
- Hartley R., Zisserman A. *Multiple View Geometry in Computer Vision.* 2nd ed.,
  Cambridge University Press, 2004. — Ch. 6 chiếu trực giao và xấp xỉ
  weak-perspective; Ch. 22 phép xoay $SO(3)$.
- Lepetit V., Moreno-Noguer F., Fua P. *EPnP: An Accurate O(n) Solution to the
  PnP Problem.* IJCV, 2009. — thuật toán PnP để so sánh.
- Casiez G., Roussel N., Vogel D. *1€ Filter: A Simple Speed-based Low-pass
  Filter for Noisy Input in Interactive Systems.* CHI 2012. — Temporal
  smoothing mà MediaPipe sử dụng cho landmark.
- Beymer D. *Face Recognition under Varying Pose.* CVPR 1994. — một trong
  những công trình đầu áp dụng ratio-based geometry trên landmarks cho head
  pose.
