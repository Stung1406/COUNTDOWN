# Countdown với GRPO và LoRA

Dự án thử nghiệm fine-tune mô hình ngôn ngữ để giải bài toán Countdown bằng
reward có thể kiểm chứng tự động. Mỗi đề gồm một danh sách số và một số đích;
model phải tạo biểu thức dùng mỗi số đúng một lần, chỉ với `+`, `-`, `*`, `/`
và dấu ngoặc.

Đây là pipeline GRPO tối giản để minh họa. Script cập nhật LoRA một lần cho
mỗi batch on-policy; không triển khai importance sampling, PPO clipping hay
KL penalty như một GRPO đầy đủ.

## Action, State, Reward nằm ở đâu?

| Thành phần | Ý nghĩa trong dự án | Vị trí trong code |
|---|---|---|
| **State (trạng thái/đầu vào)** | Đề bài gồm `numbers` và `target`, được đưa vào prompt. Trong lúc sinh, ngữ cảnh còn bao gồm các token model đã sinh trước đó. Không có lớp environment hay hàm `step()` quản lý state riêng; đây là bài toán sinh câu trả lời theo prompt, chấm reward cuối cùng. | [`make_task()` và `build_messages()` trong `countdown.py`](countdown.py) tạo đề và prompt. [`encode_prompt()` và `generate()` trong `train_grpo.py`](train_grpo.py) chuyển prompt thành đầu vào model và sinh câu trả lời. |
| **Action (hành động)** | Completion do policy/model sinh ra: một chuỗi token, theo định dạng yêu cầu `FINAL: <expression>`. Biểu thức sau `FINAL:` là hành động ở mức bài toán Countdown. Trong train, mỗi prompt sinh `G` completion để so sánh trong cùng nhóm. | [`generate()` trong `train_grpo.py`](train_grpo.py) lấy mẫu completion; [`build_messages()` trong `countdown.py`](countdown.py) yêu cầu định dạng đầu ra. |
| **Reward (phần thưởng)** | Reward nhị phân ở cuối completion: `1.0` nếu biểu thức hợp lệ, dùng đúng mỗi số một lần và bằng target; `0.0` nếu không. Không có reward trung gian cho từng token hay từng bước giải. | [`verify()` trong `countdown.py`](countdown.py) kiểm tra và trả về `(reward, lý_do)`. Trong train, kết quả được lấy tại bước `VERIFY -> REWARD` trong [`train_grpo.py`](train_grpo.py). |

### Ví dụ

Với state/đề bài `numbers = [2, 3, 7, 10]`, `target = 47`, action:

```text
FINAL: (10 - 3) * 7 - 2
```

được chấm `reward = 1.0`, vì biểu thức dùng đủ bốn số mỗi số một lần và có giá
trị 47. Nếu thiếu/thừa số, sai cú pháp, dùng toán tử bị cấm hoặc kết quả không
bằng 47 thì reward là `0.0`.

### Quy tắc chấm reward

`verify(completion, numbers, target)` thực hiện các bước:

1. Lấy biểu thức sau `FINAL:` bằng `extract_final()`. Không có nhãn này nhận
   `0.0` (`no_final`).
2. Giới hạn biểu thức tối đa 100 ký tự, rồi phân tích cú pháp bằng `ast`.
3. Chỉ chấp nhận số nguyên, phép cộng/trừ/nhân/chia và ngoặc. Biểu thức như
   `**`, lời gọi hàm, tên biến, số thực và số âm đơn bị loại.
4. Từ chối phép chia cho 0; so sánh đa tập số đã dùng với `numbers`, nên số
   trùng lặp cũng được kiểm tra chính xác.
5. So sánh giá trị chính xác bằng `Fraction` với `target`.

Các nhánh lỗi trả reward `0.0` cùng mã lý do như `parse_error`,
`wrong_numbers`, `wrong_value` hoặc `div_zero`. Verifier không thực thi mã
Python tùy ý từ câu trả lời.

## Luồng huấn luyện

Code chính nằm trong [`train_grpo.py`](train_grpo.py); logic tạo đề, prompt,
kiểm chứng và advantage nằm trong [`countdown.py`](countdown.py).

1. **Tạo state:** `make_dataset()` tạo dữ liệu train và test. Target được sinh
   ngược từ một biểu thức hợp lệ, để đề có lời giải.
2. **Lấy action:** `generate()` sinh `G` completion từ cùng prompt.
3. **Chấm reward:** `verify()` chấm từng completion độc lập bằng 0 hoặc 1.
4. **Tính advantage theo nhóm:** `group_advantages()` tính
   `A_i = (r_i - mean(r)) / (std(r) + 1e-4)` theo mặc định. Nhóm có reward
   giống nhau có advantage bằng 0 và được bỏ qua vì không tạo tín hiệu phân biệt.
5. **Cập nhật policy:** `completion_logprobs()` lấy log-probability của các
   token completion; loss dùng advantage để tăng xác suất câu trả lời tốt hơn
   mức trung bình nhóm. Chỉ các tham số LoRA được tối ưu.

Vì reward chỉ được tính sau khi có toàn bộ câu trả lời, đây là **outcome-based
reward** chứ không phải chấm điểm từng bước suy luận.

## Cấu trúc dự án

- [`countdown.py`](countdown.py): sinh đề, tạo prompt, trích xuất biểu thức,
  verifier và group-relative advantage.
- [`train_grpo.py`](train_grpo.py): tải model, gắn LoRA, sinh completion, train
  và đánh giá pass@1.
- [`test_countdown.py`](test_countdown.py): kiểm tra verifier, bộ sinh dữ liệu
  và advantage.
- [`train_log.csv`](train_log.csv): log train; nội dung bị ghi mới khi chạy
  script.
- [`lora_countdown/`](lora_countdown/): adapter LoRA đã lưu kèm theo dự án;
  lần train mới cũng lưu vào thư mục này.

## Cài đặt và chạy

Cần Python cùng các thư viện `torch`, `transformers` và `peft`:

```powershell
pip install torch transformers peft
```

Chạy kiểm tra logic không cần tải model:

```powershell
python test_countdown.py
```

Đo baseline greedy trước khi cập nhật, chạy thử ngắn hoặc train lâu hơn:

```powershell
python train_grpo.py --steps 0
python train_grpo.py --steps 30
python train_grpo.py --steps 300
```

Các tham số thường dùng:

- `--model`: model nền, mặc định `Qwen/Qwen2.5-0.5B-Instruct`.
- `--G`: số completion trong mỗi nhóm, mặc định `4`.
- `--prompts_per_step`: số prompt được lấy ở mỗi bước, mặc định `4`.
- `--n_numbers`: số toán hạng, mặc định `3`.
- `--eval_n`: số đề đánh giá, mặc định `50`.
- `--eval_every`: chu kỳ đánh giá, mặc định `25` bước.
- `--seed`: seed cho Python random và PyTorch.

## Đọc kết quả

- `mean_reward` là reward trung bình của các completion train trong bước đó,
  không phải điểm eval.
- `groups_used` là số prompt có advantage khác 0 và được dùng để tạo gradient.
- `pass@1` là tỷ lệ bài test giải đúng khi model sinh greedy một completion cho
  mỗi bài. `why` trong log đếm các lý do verifier trả về.
- `train_log.csv` ghi `step`, `mean_reward`, `groups_used` và `loss`.

Khi chạy `--steps 0`, script vẫn tải model, đánh giá baseline, tạo/ghi log và
lưu adapter vào `lora_countdown/`. Sao lưu adapter hoặc log trước nếu cần giữ
kết quả cũ.

## Kiểm thử

Các ví dụ và ca biên của verifier, tính tái lập của dữ liệu và hành vi của
advantage được kiểm tra trong [`test_countdown.py`](test_countdown.py). Chạy:

```powershell
python test_countdown.py
```
