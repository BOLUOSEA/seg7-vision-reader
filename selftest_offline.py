#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
selftest_offline.py —— 无相机离线自检

合成三位数码管图像（含光晕、漏光、噪声），跑完整判读链路，
校验输出读数是否与预期一致。用来在拿到相机之前先验算法，或改动后回归。

用法：
    python selftest_offline.py                # 打印报告，并写 selftest_report.txt
    python selftest_offline.py -o report.txt
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seg7_reader as R  # noqa: E402

W, H = 1280, 720
BOXES = [(180, 250, 130, 220), (400, 250, 130, 220), (620, 250, 130, 220)]
BOX_NAMES = ["左位(半位)", "中位(十位)", "右位(个位)"]


def build_template() -> dict:
    """按 calib 的逻辑生成模板：左位半位，中/右位全 7 段。"""
    digits = []
    for i, (nm, box) in enumerate(zip(BOX_NAMES, BOXES)):
        segs = R.segments_from_box(box, 0.14, 0.18)
        if i == 0:
            segs = {k: v for k, v in segs.items() if k in ("b", "c")}
            code = dict(R.HALF_CODE)
        else:
            code = dict(R.FULL_CODE)
        digits.append({"name": nm, "box": list(box), "segments": segs, "code": code})

    xs = [d["box"][0] for d in digits]
    ys = [d["box"][1] for d in digits]
    x2 = [d["box"][0] + d["box"][2] for d in digits]
    y2 = [d["box"][1] + d["box"][3] for d in digits]
    return {
        "recognition": dict(R.DEFAULT_RECOG),
        "display_roi": [min(xs), min(ys), max(x2) - min(xs), max(y2) - min(ys)],
        "digits": digits,
    }


def render_raw(names_per_digit, lit=235, bg=14, bleed=0.22, noise=3.0, seed=0) -> np.ndarray:
    """底层渲染：names_per_digit 是三个列表，直接指定每位要点亮哪些段。"""
    rng = np.random.default_rng(seed)
    core = np.zeros((H, W), np.float32)
    raw_geom = [R.segments_from_box(b, 0.14, 0.0) for b in BOXES]

    for i, names in enumerate(names_per_digit):
        for nm in names:
            x, y, w, h = raw_geom[i][nm]
            core[y:y + h, x:x + w] = 1.0

    glow = cv2.GaussianBlur(core, (0, 0), 9.0)
    img = bg + core * (lit - bg) + glow * bleed * (lit - bg) * 0.35
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def render(chars: str, **kw) -> np.ndarray:
    """
    合成数码管图像。
    chars : 3 个字符，空格或空字符串表示该位消隐，例如 "  5" / "100"
    """
    per_digit = []
    for i, ch in enumerate(chars):
        if ch.strip() == "":
            per_digit.append([])
        else:
            per_digit.append(["b", "c"] if i == 0 else R.SEG_CODE[ch])
    return render_raw(per_digit, **kw)


def run() -> tuple[list[str], int]:
    tmpl = build_template()
    lines: list[str] = []
    fail = 0

    def add(s=""):
        lines.append(s)

    normal = [("  5", 5), (" 42", 42), ("100", 100), (" 99", 99),
              ("  1", 1), ("  0", 0), (" 88", 88), ("  9", 9)]

    add("=" * 78)
    add("A. 正常对比度（亮段 235 / 背景 14）")
    add("=" * 78)
    for chars, expect in normal:
        gray = render(chars, seed=100 + sum(map(ord, chars)))
        got, detail = R.read_frame(gray, tmpl)
        ok = (got == expect)
        fail += (not ok)
        add(f"  显示 {chars!r:>8}  期望 {expect:>4}  实得 {str(got):>5}  "
            f"{'OK' if ok else 'FAIL'}   阈值={detail.get('threshold')}")

    add()
    add("=" * 78)
    add("B. 低对比度（亮段 90 / 背景 30）—— 模拟减光片装过头、画面偏暗")
    add("=" * 78)
    for chars, expect in normal[:4]:
        gray = render(chars, lit=90, bg=30, bleed=0.30, noise=4.0, seed=7)
        got, detail = R.read_frame(gray, tmpl)
        ok = (got == expect)
        fail += (not ok)
        add(f"  显示 {chars!r:>8}  期望 {expect:>4}  实得 {str(got):>5}  "
            f"{'OK' if ok else 'FAIL'}   阈值={detail.get('threshold')}")

    add()
    add("=" * 78)
    add("C. 边界与异常（都应该被拦下，绝不出错值）")
    add("=" * 78)
    for chars, why in [("188", "超出 0-100"), ("999", "超出 0-100")]:
        gray = render(chars, seed=11)
        got, detail = R.read_frame(gray, tmpl)
        ok = (got is None)
        fail += (not ok)
        add(f"  显示 {chars!r:>8}（{why}）  实得 {got}  "
            f"{'OK 已拦截' if ok else 'FAIL 未拦截'}   {detail.get('error', '')}")

    # 屏幕在亮、但三位全空白：在数字之间的空隙放一块高亮区抬升 ref，
    # 使其通过"屏幕未点亮"检查，从而单独验证全空白分支
    gray = render("   ", seed=12)
    gray[300:330, 330:390] = 220
    got, detail = R.read_frame(gray, tmpl)
    ok = (got is None)
    fail += (not ok)
    add(f"  三位全空白但屏幕在亮        实得 {got}  "
        f"{'OK 已拦截' if ok else 'FAIL 未拦截'}   {detail.get('error', '')}")

    tmpl_blank0 = build_template()
    tmpl_blank0["recognition"]["blank_value"] = 0
    got2, _ = R.read_frame(gray, tmpl_blank0)
    ok = (got2 == 0)
    fail += (not ok)
    add(f"  同上，但 blank_value=0      实得 {got2}  "
        f"{'OK 视为 0' if ok else 'FAIL'}")

    # 物理不可能状态：左位只有 b 段亮（半位不可能是这个组合）
    gray = render_raw([["b"], [], []], seed=21)
    got, detail = R.read_frame(gray, tmpl)
    ok = (detail.get("exact") is False)
    fail += (not ok)
    add(f"  左位出现非法段组合 {{b}}      精确={detail.get('exact')}  "
        f"{'OK 已标记不精确，会被丢弃' if ok else 'FAIL 未标记'}")

    gray = np.full((H, W), 12, np.uint8)
    got, detail = R.read_frame(gray, tmpl)
    ok = (got is None)
    fail += (not ok)
    add(f"  屏幕全灭                    实得 {got}  "
        f"{'OK 已拦截' if ok else 'FAIL 未拦截'}   {detail.get('error', '')}")

    add()
    add("=" * 78)
    add("D. 段级判读明细（显示 ' 42'，核对每段亮灭与裕量）")
    add("=" * 78)
    gray = render(" 42", seed=3)
    got, detail = R.read_frame(gray, tmpl)
    add(f"  读数 = {got}   阈值 = {detail['threshold']}   参考值 = {detail['ref']}")
    for d in detail["digits"]:
        seg_str = "  ".join(f"{k}:{v:>6.1f}{'*' if k in d['on'] else ' '}"
                            for k, v in sorted(d["means"].items()))
        add(f"  {d['name']:<10} -> {str(d['char'])!r:<4} 精确={d['exact']}   {seg_str}")
    add("  （每个段后面的 * 表示判定为亮）")

    add()
    add("=" * 78)
    add(f"结果：{'全部通过' if fail == 0 else f'{fail} 项失败'}")
    add("=" * 78)
    return lines, fail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out",
                    default=str(Path(__file__).resolve().parent / "selftest_report.txt"),
                    help="报告输出路径，默认写在脚本同目录")
    args = ap.parse_args(argv)

    lines, fail = run()
    text = "\n".join(lines)
    print(text)
    try:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n[报告已写入] {Path(args.out).resolve()}")
    except OSError as e:
        print(f"\n[报告写入失败] {e}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
