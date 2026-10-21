from collections import Counter

from countdown import group_advantages, make_dataset, verify

NUMS, TARGET = [2, 3, 7, 10], 47


def check(completion, want_reward, want_reason, nums=NUMS, target=TARGET):
    r, why = verify(completion, nums, target)
    assert (r, why) == (want_reward, want_reason), (completion, r, why)
    print(f"  ok  {completion!r:42} -> {r} ({why})")


print("[1] Verifier - các ví dụ trong slide")
check("FINAL: (10 - 3) * 7 - 2", 1.0, "ok")
check("FINAL: (10 - 3) * 7 - 2 = 47", 1.0, "ok")        # model hay viết kèm '= 47'
check("Let me think...\nFINAL: (10 - 3) * 7 - 2", 1.0, "ok")  # có chữ thừa phía trên
check("FINAL: 10 * 7 - 3 + 2", 0.0, "wrong_value")      # đủ số nhưng = 69
check("FINAL: (10 + 7) * 3 - 2", 0.0, "wrong_value")    # đủ số nhưng = 49
check("FINAL: 10 * 7 - 3", 0.0, "wrong_numbers")        # thiếu số 2 (slide: = 67)
check("FINAL: 47", 0.0, "wrong_numbers")                # không dùng numbers

print("[2] Verifier - các ca biên")
check("(10 - 3) * 7 - 2", 0.0, "no_final")              # thiếu 'FINAL:'
check("FINAL: (10 - 3 * 7", 0.0, "parse_error")         # ngoặc lệch
check("FINAL: 2 ** 3 + 7 * 10", 0.0, "bad_token")       # toán tử ** bị cấm
check("FINAL: -2 + 3 + 7 * 7", 0.0, "bad_token")        # số âm đơn bị cấm
check("FINAL: __import__('os')", 0.0, "bad_token")      # không thể chạy code lạ
check("FINAL: 10 / (3 - 3)", 0.0, "div_zero", nums=[10, 3, 3], target=5)
check("FINAL: 6 / 4 * 10 + 2", 0.0, "wrong_numbers", nums=[6, 4, 10, 4], target=17)
check("FINAL: 6 / 4 * 4", 1.0, "ok", nums=[6, 4, 4], target=6)  # phân số trung gian OK

print("[3] Data generator")
data = make_dataset(2000, seed=0, n_numbers=3)
assert len(data) == 2000
assert all(1 <= t <= 100 for _, t in data)
assert all(len(n) == 3 and all(1 <= x <= 10 for x in n) for n, _ in data)
assert make_dataset(5, seed=1) == make_dataset(5, seed=1)   # cùng seed -> cùng data
print("  ok  2000 bài, target trong [1,100], tái lập được theo seed")
print("  ví dụ:", data[:3])
print("  phân bố target (top 5):", Counter(t for _, t in data).most_common(5))

print("[4] Group advantage - group ở slide 15 (r = 1,0,0,0)")
raw = group_advantages([1, 0, 0, 0], normalize=False)
assert raw == [0.75, -0.25, -0.25, -0.25]
norm = group_advantages([1, 0, 0, 0], normalize=True)
print("  raw  :", raw)
print("  norm :", [round(a, 3) for a in norm])
assert norm[0] > 0 > norm[1]
assert group_advantages([0, 0, 0, 0]) == [0, 0, 0, 0]
assert group_advantages([1, 1, 1, 1]) == [0, 0, 0, 0]
print("  ok  cả group cùng reward -> advantage = 0 (không có tín hiệu học)")

print("\nTẤT CẢ TEST ĐỀU PASS")
