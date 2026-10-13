# Write-up — Lab 25: GPU FinOps (NimbusAI)

> Số liệu lấy từ `python missions/run_all.py` (seed=25, snapshot giá tháng 6/2026) và khớp với `outputs/report.md`.

## 1. Baseline vs. Optimized

| Chỉ số | Baseline | Optimized |
|---|---|---|
| Chi phí GPU hằng tháng (tổng) | $27,133 | $14,626 |
| **Tiết kiệm tổng** | | **$12,507 (46%)** |
| Inference `$/1M-token` | **$6.488** | **$1.126** (giảm 82.6%) |
| Inference `$/ngày` | $48.87 | $8.48 |
| Mua GPU (on-demand → spot/reserved) | $25,667/tháng | $15,627/tháng (39.1%) |

Baseline inference là mô hình "ngây thơ": mọi request chạy trên model lớn, không cache, không batch. Chi phí mua GPU baseline là on-demand 100%.

## 2. Phân tích từng đòn bẩy

| Hạng | Lever | Tiết kiệm/tháng | % tổng tiết kiệm |
|---|---|---|---|
| 1 | Purchasing (spot/reserved) | $10,040 | 80% |
| 2 | Inference (cascade/cache/batch) | $1,212 | 10% |
| 3 | Right-size util-lies | $655 | 5% |
| 4 | Kill idle GPUs | $600 | 5% |

- **Purchasing lớn nhất** vì 5 job có thể gián đoạn (training, dev, batch-eval) chuyển sang spot và 3 dịch vụ inference chạy 18–24h/ngày chuyển sang reserved 3 năm. Reserved chỉ có lợi khi duty cycle ≥ 55% (13,2 giờ/ngày).
- **Inference giảm 82.6% về `$/1M-token`** nhưng chỉ là 10% tổng tiền vì tiền GPU thuê cố định lớn hơn nhiều chi phí token trong bộ dữ liệu này. Cascade đưa ~79% request sang model nhỏ (rẻ hơn ~15×); cache đóng góp $1.17/ngày; batch giảm 50% cho ~17% request.
- Hai lever còn lại nhỏ nhưng gần như không rủi ro (tắt GPU idle, hạ cấp GPU).

## 3. GPU-Util "nói dối"

| GPU | GPU-Util | MFU | Lãng phí/tháng |
|---|---|---|---|
| gpu-h100-4 | 98.2% | 0.19 | $1,451 |
| gpu-a10g-1 | 96.9% | 0.27 | $527 |

`nvidia-smi` chỉ báo "có kernel nào đang chạy trong khoảng lấy mẫu", không đo số SM bận hay tensor core có được cấp dữ liệu không. GPU chờ HBM (memory stall), chờ all-reduce/dataloader, hoặc chạy kernel nhỏ với launch overhead vẫn hiện ~100%. Ta trả tiền cả giờ H100 ($2.50) nhưng chỉ nhận ~19% FLOPs, tức bỏ phí ~$2.02/giờ. Hai GPU này đang rò rỉ khoảng **$1,978/tháng**. Hành động: sửa nút thắt (batch size, dataloader, overlap giao tiếp, fused kernel) trước khi mua thêm GPU; cảnh báo theo MFU/MBU thay vì GPU-Util.

## 4. Các phần mở rộng đã làm (5/5)

**Ext 1 — `recommend_tier_v2` (`finops/pricing.py`).** Thêm interruption rate theo loại GPU (H100 4%/giờ … A10G 12%/giờ), tính reserved theo cả 24 giờ cam kết (không phải chỉ giờ dùng), và so sánh 1yr/3yr theo `horizon_months`; job < 30 ngày không đủ điều kiện reserved. Kết quả M3 (so với on-demand): chính sách cũ báo **39.1%**, nhưng khi tính đúng chi phí cam kết chỉ còn **37.9%**; chính sách mới **38.0%**; nếu chỉ dám cam kết 1 năm là **29.5%**. Insight: hầu hết job giữ nguyên tier, giá trị chính là con số trung thực hơn và ma trận quyết định (GPU × giờ/ngày × interruptible), ví dụ A10G/L4 chạy 24/7 mà chịu được gián đoạn thì spot rẻ hơn reserved.

**Ext 2 — Right-sizing theo MBU (`missions/m1_efficiency_audit.py`).** Bảng `$/GB-VRAM`: MI300X rẻ nhất ($0.0102), A10G đắt nhất ($0.0417). Với GPU không phải training, chọn SKU rẻ nhất vẫn chứa vừa VRAM (+15%) và đáp ứng băng thông ở ≤60% MBU. Kết quả: gpu-a10g-0 A10G → L4, tiết kiệm **$144/tháng**; các GPU khác đã đúng cỡ. Không chọn "rẻ nhất theo `$/GPU-giờ`" vì L4 chỉ có 0.3 TB/s, sẽ nghẽn băng thông với decode.

**Ext 3 — `cache_is_worth_it` (`finops/pricing.py`, M2).** Break-even = `write_cost / (1 − read_discount)` = 0.28 lần đọc/prefix (ghi bằng 1.25× giá). Dữ liệu có cache-hit 31.9% ⇒ ~0.47 lần đọc/prefix (ước lượng `h/(1−h)`), nên cache **có lợi** (tiết kiệm $1.17/ngày). Nếu tính phí lưu trữ kiểu Gemini (giả định minh họa), ngưỡng tăng lên 5.8 (small) và 1.9 (large) lần đọc, khi đó cache **lỗ** ở mức hit này. Hàm được dùng làm cổng điều kiện trong M2.

**Ext 4 — Ngân sách reasoning (M2).** Reasoning chiếm **8.4% traffic**, **16.5% chi phí**, **94% năng lượng**; mỗi query reasoning tốn ~173× Wh (hệ số 80× nhân với ~2.2× số token). Quy tắc: chỉ cho reasoning với prompt dài/khó nhất hoặc khi câu trả lời model rẻ thất bại validation. Giới hạn 5% traffic: tiết kiệm $14/tháng và 31% năng lượng; 2%: $23/tháng và 68% năng lượng. Cap 10% không đổi gì vì hiện tại đã dưới 10%. Giá trị chính của lever này là năng lượng/carbon chứ không phải tiền.

**Ext 5 — Lập lịch theo carbon (M3).** Các job interruptible dùng 1,789 kWh. Chuyển từ us-east-1 sang europe-north1 tránh **626 kgCO2e (92%)**. Rẻ nhất về điện: us-east-wa; sạch nhất: europe-north1; cân bằng nhất (carbon $100/tấn): us-east-wa. Vùng "tối ưu" phụ thuộc ưu tiên công ty (tiền hay carbon). Đánh đổi: vùng xa người dùng chỉ phù hợp training/batch, inference nhạy latency nên ở gần người dùng.

## 5. Khuyến nghị cho NimbusAI (3 hành động đầu tiên)

1. **Chuyển training/dev/batch-eval sang spot có checkpoint và inference ổn định sang reserved 3 năm** (+$10,040/tháng, chưa tính rủi ro; nếu chưa chắc về nhu cầu 3 năm, bắt đầu bằng 1 năm).
2. **Tự động tắt GPU idle và thay cảnh báo GPU-Util bằng MFU/MBU**, sau đó sửa hoặc hạ cấp 2 GPU nói dối.
3. **Triển khai cascade + prompt caching + batch API và giới hạn reasoning ≤5% traffic**, đồng thời đặt job chịu được gián đoạn ở vùng sạch/rẻ; tag đầy đủ (coverage hiện 92%) để chuyển sang chargeback.

## Bonus (không chấm điểm)

- **LiteLLM tracker:** `team-chat` bị chặn sau 10 request khi vượt hạn mức $0.05 (chi $0.0507); `team-eval` (model nhỏ + batch) chỉ tốn $0.0003.
- **Model local (CPU):** `sshleifer/tiny-gpt2` đạt 242 tok/s (~$0.11/1M-token, nhưng model đồ chơi nên không có ý nghĩa chất lượng); `Qwen2.5-0.5B` chỉ 5.3 tok/s, quy đổi ~$5.22/1M-token so với ~$0.30 mô phỏng cho model nhỏ. Tức tok/s và mức sử dụng quyết định `$/token` nhiều hơn giá niêm yết `$/giờ`.
- **Prometheus + Grafana:** exporter xuất `gpu_wasted_cost_usd_per_hr = (1 − MFU) × $/giờ`; xếp hạng lãng phí cao nhất: gpu-h100-4 ($2.01/giờ), gpu-h100-5 ($1.85/giờ), gpu-h100-2 ($1.50/giờ).
