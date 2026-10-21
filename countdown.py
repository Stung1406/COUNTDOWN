"""Lõi của pipeline: Data -> Prompt -> Parser -> Verifier -> Reward -> Advantage.
Không phụ thuộc torch, chạy test được trên mọi máy."""
import ast
import random
import re
from collections import Counter
from fractions import Fraction

# ---------------------------------------------------------------- 1. DATA
def make_task(rng: random.Random, n_numbers: int = 3, max_num: int = 10):
    """Sinh (numbers, target). Target được tính NGƯỢC từ một biểu thức dùng đủ
    mọi số => bài nào cũng chắc chắn giải được, không cần gán nhãn."""
    while True:
        nums = [rng.randint(1, max_num) for _ in range(n_numbers)]
        pool = [Fraction(x) for x in nums]
        ok = True
        while len(pool) > 1:
            i, j = rng.sample(range(len(pool)), 2)
            a, b = pool[i], pool[j]
            op = rng.choice("+-*/")
            if op == "/" and b == 0:
                ok = False
                break
            val = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else 0}[op]
            pool = [p for k, p in enumerate(pool) if k not in (i, j)] + [val]
        if ok and pool[0].denominator == 1 and 1 <= pool[0] <= 100:
            return nums, int(pool[0])


def make_dataset(n: int, seed: int, n_numbers: int = 3, max_num: int = 10):
    rng = random.Random(seed)
    return [make_task(rng, n_numbers, max_num) for _ in range(n)]


# ---------------------------------------------------------------- 2. PROMPT
def build_messages(numbers, target):
    user = (
        f"Numbers: {numbers}\nTarget: {target}\n"
        "Use each number exactly once with + - * / and parentheses so the "
        "expression equals the target.\n"
        "Reply with ONE line only, in this exact format:\n"
        "FINAL: <expression>"
    )
    return [{"role": "user", "content": user}]


# ---------------------------------------------------------------- 3. PARSER
def extract_final(text: str):
    """Lấy biểu thức sau 'FINAL:' (lấy lần xuất hiện cuối), bỏ phần '= 47' nếu có."""
    m = re.findall(r"FINAL:\s*(.+)", text)
    if not m:
        return None
    expr = m[-1].strip().strip("`").rstrip(".")
    expr = expr.replace("×", "*").replace("÷", "/").split("=")[0].strip()
    return expr


# ---------------------------------------------------------------- 4. VERIFIER
_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}


class _Bad(Exception):
    def __init__(self, reason):
        self.reason = reason


def _eval(node, used):
    if isinstance(node, ast.Expression):
        return _eval(node.body, used)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        used.append(node.value)
        return Fraction(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _eval(node.left, used)
        right = _eval(node.right, used)
        if isinstance(node.op, ast.Div) and right == 0:
            raise _Bad("div_zero")
        return _OPS[type(node.op)](left, right)
    raise _Bad("bad_token")  # **, số âm đơn, hàm, tên biến, số thực... đều bị loại


def verify(completion: str, numbers, target):
    """Trả về (reward, lý do). Reward = 1.0 chỉ khi hợp lệ VÀ đạt target."""
    expr = extract_final(completion)
    if expr is None:
        return 0.0, "no_final"
    if len(expr) > 100:
        return 0.0, "too_long"
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return 0.0, "parse_error"
    used = []
    try:
        value = _eval(tree, used)
    except _Bad as e:
        return 0.0, e.reason
    if Counter(used) != Counter(numbers):
        return 0.0, "wrong_numbers"
    if value != target:
        return 0.0, "wrong_value"
    return 1.0, "ok"


# ---------------------------------------------------------------- 5. ADVANTAGE
def group_advantages(rewards, normalize: bool = True, eps: float = 1e-4):
    """A_i = r_i - mean(group), chia thêm std nếu normalize (đúng GRPO gốc).
    Cả group cùng reward => mọi A = 0 => không có tín hiệu học."""
    n = len(rewards)
    mean = sum(rewards) / n
    adv = [r - mean for r in rewards]
    if normalize:
        std = (sum(a * a for a in adv) / n) ** 0.5
        adv = [a / (std + eps) for a in adv]
    return adv
