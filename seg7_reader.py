#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seg7_reader.py —— 用全局快门黑白 UVC 工业相机读取 3 位数码管（电池电量 0-100）

适配相机：杰锐微通 DCXG100（1280x720 / 全局快门 / 黑白 / UVC 免驱）
场景前提：被测物静止、非接触、单帧读全部 3 位；左位为半位，仅 b / c 两段。

六种模式
--------
  snap      抓一帧存成图片，先看看相机拍到了什么（最常用的第一步）
  diagnose  枚举相机设备、报告图像统计、隔离测试哪个属性写入会破坏画面
  selftest  不接相机，用合成图像验证整条判读链路
  probe     报告当前画面亮度统计（不需要模板），用来判断该不该减光
  calib     交互框选三个数字的外接矩形，自动切成各段 ROI，生成模板 JSON
  read      按模板采集并输出读数（默认连续多帧一致才输出）

推荐顺序：snap → selftest → probe → calib → read
（相机没反应就插一步 diagnose 定位问题，别在错误的方向上调参数）

重要：默认不写曝光/增益属性。实测这台相机的驱动一写曝光画面就变成纯白，
只有显式加 --exposure-write 才会去写。详见方案文档第 12 节。

用法示例
--------
  py seg7_reader.py                       # 打印推荐执行顺序
  py seg7_reader.py snap                   # 抓一张图看看（存为 snapshot.png）
  py seg7_reader.py snap --image a.png     # 指定文件名
  py seg7_reader.py snap --camera 1        # 换设备号
  py seg7_reader.py diagnose               # 枚举所有相机并报告统计
  py seg7_reader.py diagnose --camera 1    # 指定设备做属性写入隔离测试
  py seg7_reader.py --props --camera 0     # 弹驱动属性页，手动验证曝光
  py seg7_reader.py selftest              # 不接相机验算法
  py seg7_reader.py probe                  # 看当前画面亮度
  py seg7_reader.py calib                  # 框选 ROI 生成模板
  py seg7_reader.py read --template seg7_roi.json --frames 3 --agree 3

周期性采集（长时记录放电曲线）
----------------------------
  # 每 60 秒采一次，写进 CSV，读到 100 或 0 连续 5 次就自动收工
  py seg7_reader.py read --template seg7_roi.json --every 60
      --csv battery_log.csv --stop-at 100 0

  # 每 5 分钟一次，跑满 12 轮（1 小时）就停
  py seg7_reader.py read --template seg7_roi.json --every 5m --count 12 --csv log.csv

几个要点
--------
  · --every 支持 60 / 60s / 1m / 5m / 2h；设了它自动持续运行，不用再加 --watch
  · 节奏是「严格间隔」：睡「间隔 − 本轮耗时」，不会越跑越慢
  · --stop-at 的命中判定：失败轮次既不计入、也不重置已累计的连续次数
    （失败=没读到，不代表值变了）
  · --every / --stop-at 也可以写进模板 seg7_roi.json 的 capture.every
  · 相机全程只打开一次，避免反复开关

依赖：opencv-python, numpy
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# 默认配置（可被模板 JSON 或命令行覆盖）
# --------------------------------------------------------------------------

DEFAULT_CAMERA = {
    "index": 0,
    "width": 1280,
    "height": 720,
    "fps": 5,            # 先降帧率，否则 UVC 曝光设不长
    "exposure": 300,     # UVC 原始值，单位随驱动而异 —— 用 probe 模式实测确定
    "gain": 0,
}

DEFAULT_CAPTURE = {
    "frames": 3,         # 融合帧数
    # mean = 逐像素取平均（默认）。**实测结论：静态场景必须用 mean。**
    #   max 是"任一帧抓到该段就算亮"，本意是覆盖动态扫描相位；但代价是它会把
    #   多帧里**最差的那一帧**保留下来——某个灭段偶发被光晕抬到 120 时，
    #   3 帧 max 后仍是 120（越过阈值 → 判读失败），而 3 帧 mean 只有 40（安全）。
    #   实测：同一配置下 max 的成功率 33%，换 mean 后 100%。
    #   帧数越多 max 越糟（越有机会逮到那一次跳变），mean 则越好（平均更充分）。
    #   仅当被测物会动、必须靠短曝光抓拍时才用 max。
    "fusion": "mean",
    "warmup": 5,         # 丢弃的开头帧数
    # 周期性采集的间隔（秒）。0 = 关闭（默认，行为同以前）。
    # 这是"工位固有节奏"，所以放在模板里；命令行 --every 可临时覆盖。
    # 可写成 60 / "1m" / "5m"，也接受纯秒数。
    "every": 0,
}

DEFAULT_RECOG = {
    "ref_percentile": 99.0,   # 亮段参考值取整块数字区的第 99 百分位
    "on_ratio": 0.40,         # 阈值 = on_ratio * ref
    "min_bright_ref": 40,     # ref 低于此值视为屏幕未点亮
    "blank_value": None,      # 三位全空白时的取值；None = 报错不出结果，
                              # 若设备在 0 时确实全消隐，改成 0
}

# 标准 7 段段码表：字符 -> 亮段集合
SEG_CODE = {
    "0": ["a", "b", "c", "d", "e", "f"],
    "1": ["b", "c"],
    "2": ["a", "b", "d", "e", "g"],
    "3": ["a", "b", "c", "d", "g"],
    "4": ["b", "c", "f", "g"],
    "5": ["a", "c", "d", "f", "g"],
    "6": ["a", "c", "d", "e", "f", "g"],
    "7": ["a", "b", "c"],
    "8": ["a", "b", "c", "d", "e", "f", "g"],
    "9": ["a", "b", "c", "d", "f", "g"],
}

# 半位（左位）只有两段竖线
HALF_CODE = {"": [], "1": ["b", "c"]}

# 全 7 段数字额外允许"空白"态：多数电量表会把高位补零消隐，
# 例如 5 实际显示成 "  5"。空字符串表示该位全灭。
FULL_CODE = {"": [], **SEG_CODE}


# --------------------------------------------------------------------------
# 相机
# --------------------------------------------------------------------------

def set_exposure(cap: cv2.VideoCapture, value: float) -> None:
    """设为手动曝光并写入原始值。DirectShow 用 0.25 表示手动档。"""
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_EXPOSURE, float(value))
    time.sleep(0.12)          # 给驱动时间生效
    for _ in range(2):
        cap.read()


def _open_raw(cfg: dict, write_exposure: bool) -> cv2.VideoCapture:
    """打开设备；write_exposure=False 时完全不碰曝光/增益属性。"""
    cap = cv2.VideoCapture(int(cfg["index"]), cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"打不开相机 index={cfg['index']}")

    # YUY2 未压缩输出：避开 MJPEG 在 LED 边缘产生的振铃，对阈值分割很关键
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    except Exception:
        pass  # 后端不支持时退回默认输出，后面统一转灰度也能跑

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])

    # 顺序不能反：必须先降帧率，UVC 才允许设长曝光
    cap.set(cv2.CAP_PROP_FPS, cfg["fps"])

    if write_exposure:
        set_exposure(cap, cfg["exposure"])
        cap.set(cv2.CAP_PROP_GAIN, cfg.get("gain", 0))

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mode = "手动曝光" if write_exposure else "相机自动曝光（未写任何曝光属性）"
    print(f"[相机] {w}x{h}  fps={cap.get(cv2.CAP_PROP_FPS):g}  {mode}")
    return cap


def open_camera(cfg: dict, write_exposure: bool = True) -> cv2.VideoCapture:
    """
    打开相机。写入曝光属性后做一次自检：若整幅画面变成纯色（std≈0），
    说明这台相机的驱动不接受这种写法，自动回退到"完全不写曝光属性"。

    注意：实测发现有驱动被写坏后，**同一次进程内重新打开也恢复不了**，
    必须整个进程重开。所以回退后再检一次，还是坏的就直接报错让人重跑，
    而不是继续拿一张纯色图往下走、最后输出一个看着像模像样的错结果。
    """
    cap = _open_raw(cfg, write_exposure)

    if write_exposure:
        ok, frame = cap.read()
        if ok and frame is not None and frame_stats(to_gray(frame))["std"] < 1.0:
            print("[警告] 写入曝光属性后画面变成纯色（std≈0）：")
            print("       这台相机的驱动不接受这种写法，正在回退到「不写任何曝光属性」…")
            cap.release()
            cap = _open_raw(cfg, False)
            ok, frame = cap.read()
            if ok and frame is not None and frame_stats(to_gray(frame))["std"] < 1.0:
                cap.release()
                raise RuntimeError(
                    "回退后画面仍是纯色。写坏的状态在同一次进程内清不掉，"
                    "需要重新启动进程。\n"
                    "       请重跑并加上 --no-exposure-write：\n"
                    "         py seg7_reader.py probe --no-exposure-write\n"
                    "         py seg7_reader.py calib --no-exposure-write\n"
                    "（加上后本工具完全不会去碰曝光/增益属性，"
                    "相机会用自己的自动曝光）"
                )
            print("       回退成功，画面已恢复正常。下次可直接加 --no-exposure-write 跳过。")

    return cap


# --------------------------------------------------------------------------
# 取帧与融合
# --------------------------------------------------------------------------

def to_gray(frame: np.ndarray) -> np.ndarray:
    """统一转成单通道灰度：YUY2 取 Y 通道，BGR 转灰度。"""
    if frame is None:
        raise RuntimeError("取帧失败")
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 2:          # YUY2 -> 通道 0 是 Y
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def imwrite_unicode(path, img: np.ndarray) -> bool:
    """
    写图片，兼容中文路径。

    cv2.imwrite 在 Windows 上用的是窄字符 fopen，遇到中文路径会**静默失败**
    （返回 False 但不抛异常），而本项目目录名就带中文，所以必须绕开它：
    先用 imencode 编码到内存，再用 Path.write_bytes 写文件。
    """
    try:
        ext = Path(path).suffix or ".png"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(buf.tobytes())
        return True
    except OSError:
        return False


def grab(cap: cv2.VideoCapture, frames: int, fusion: str, warmup: int) -> np.ndarray:
    """
    取 frames 帧并融合成一幅灰度图。

    fusion="mean"（默认）：逐像素平均。抑制偶发的光晕跳变与读出噪声，
        帧数越多越稳。静态场景选它。
    fusion="max"：逐像素取最大。任一帧抓到该段就算亮，用于覆盖动态扫描相位；
        代价是会把多帧里最差的那一帧保留下来，帧数越多反而越糟。
        只在被测物会动、必须短曝光抓拍时使用。
    """
    for _ in range(max(0, int(warmup))):
        cap.read()

    buf = []
    for _ in range(max(1, int(frames))):
        ok, frame = cap.read()
        if ok:
            buf.append(to_gray(frame).astype(np.float32))

    if not buf:
        raise RuntimeError("连续取帧失败，检查相机连接")

    stack = np.stack(buf, axis=0)
    out = stack.max(axis=0) if fusion == "max" else stack.mean(axis=0)
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# 七段几何：从一个数字的外接矩形推出各段的采样框
# --------------------------------------------------------------------------

def shrink_rect(rect, frac: float):
    """以矩形中心为基准内缩 frac 比例，避开光晕与段间漏光。"""
    x, y, w, h = [int(v) for v in rect]
    dx = int(round(w * frac / 2.0))
    dy = int(round(h * frac / 2.0))
    x, y = x + dx, y + dy
    w, h = max(1, w - 2 * dx), max(1, h - 2 * dy)
    return [int(x), int(y), int(w), int(h)]


def segments_from_box(box, thickness: float = 0.14, shrink: float = 0.18) -> dict:
    """
    把数字外接矩形 (x, y, w, h) 切成标准七段采样框。

    thickness  笔画粗细占字高的比例（典型 7 段数码管约 0.12-0.16）
    shrink     每个段采样框的内缩比例

    纵向布局： a | f/b | g | e/c | d  —— 总高 = 2*vl + 3*t = h
    """
    x, y, w, h = [int(v) for v in box]
    t = max(3, int(round(h * thickness)))
    hl = max(3, w - 2 * t)            # 横段长度
    vl = max(3, (h - 3 * t) // 2)     # 竖段长度

    raw = {
        "a": (x + t, y, hl, t),
        "g": (x + t, y + t + vl, hl, t),
        "d": (x + t, y + h - t, hl, t),
        "f": (x, y + t, t, vl),
        "b": (x + w - t, y + t, t, vl),
        "e": (x, y + 2 * t + vl, t, vl),
        "c": (x + w - t, y + 2 * t + vl, t, vl),
    }
    return {k: shrink_rect(v, shrink) for k, v in raw.items()}


# --------------------------------------------------------------------------
# 判读
# --------------------------------------------------------------------------

def measure_segments(gray: np.ndarray, segments: dict) -> dict:
    """对每个段采样框求平均灰度。"""
    h, w = gray.shape[:2]
    out = {}
    for name, (x, y, sw, sh) in segments.items():
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(w, int(x + sw)), min(h, int(y + sh))
        out[name] = float(gray[y0:y1, x0:x1].mean()) if (x1 > x0 and y1 > y0) else 0.0
    return out


def bright_reference(gray: np.ndarray, display_roi, percentile: float) -> float:
    """整块数字区的第 N 百分位灰度，作为亮段参考值。"""
    if display_roi is not None:
        x, y, w, h = [int(v) for v in display_roi]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(gray.shape[1], x + w), min(gray.shape[0], y + h)
        region = gray[y0:y1, x0:x1]
    else:
        region = gray
    return float(np.percentile(region, percentile)) if region.size else 0.0


def decode_digit(on_set: set, code_map: dict):
    """
    把亮段集合比对该数字自己的段码表。
    返回 (字符, 是否精确匹配, 汉明距离)。
    """
    best, best_dist = None, 1 << 30
    for ch, segs in code_map.items():
        dist = len(frozenset(segs) ^ on_set)
        if dist == 0:
            return ch, True, 0
        if dist < best_dist:
            best, best_dist = ch, dist
    return best, False, best_dist


def read_frame(gray: np.ndarray, tmpl: dict):
    """对单帧做完整判读，返回 (读数值或 None, 明细 dict)。"""
    rec = dict(DEFAULT_RECOG)
    rec.update(tmpl.get("recognition", {}))
    ref = bright_reference(gray, tmpl.get("display_roi"), rec["ref_percentile"])
    detail = {"ref": round(ref, 1), "digits": []}

    if ref < rec["min_bright_ref"]:
        detail["error"] = "未检测到点亮（屏幕未亮或相机未对准）"
        return None, detail

    thr = rec["on_ratio"] * ref
    detail["threshold"] = round(thr, 1)

    chars, ok_all = "", True
    for d in tmpl["digits"]:
        means = measure_segments(gray, d["segments"])
        on = {k for k, v in means.items() if v >= thr}
        ch, exact, dist = decode_digit(on, d["code"])
        ok_all &= exact
        chars += ch
        detail["digits"].append({
            "name": d["name"],
            "on": sorted(on),
            "char": ch,
            "exact": exact,
            "hamming": dist,
            "means": {k: round(v, 1) for k, v in means.items()},
        })

    detail["raw"] = chars
    if chars.strip() == "":
        # 三位全空白。默认判定为无效读数（宁可不出结果，也不出错值）；
        # 若你的设备在 0 时确实全消隐，把模板里的 blank_value 设为 0。
        blank_value = rec.get("blank_value")
        if blank_value is None:
            detail["error"] = "三个位全为空白，判定为无效读数"
            value = None
        else:
            value = int(blank_value)
    else:
        try:
            value = int(chars)
        except ValueError:
            detail["error"] = f"结果非数字：{chars!r}"
            value = None
        else:
            if not (0 <= value <= 100):
                detail["error"] = f"取值超出 0-100：{value}"
                value = None

    detail["exact"] = bool(ok_all)
    return value, detail


def annotate(gray: np.ndarray, tmpl: dict, detail: dict) -> np.ndarray:
    """把采样框和判定结果画到图上，便于人工核对。"""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    thr = detail.get("threshold", 0) or 0
    for d in tmpl["digits"]:
        for name, (x, y, w, h) in d["segments"].items():
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(vis.shape[1], x + w), min(vis.shape[0], y + h)
            m = float(gray[y0:y1, x0:x1].mean()) if (x1 > x0 and y1 > y0) else 0.0
            color = (80, 200, 80) if m >= thr else (150, 150, 150)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 1)
            cv2.putText(vis, name, (x0, max(12, y0 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    text = " ".join(f"{d['name']}:{d['char'] or '_'}" for d in detail.get("digits", []))
    cv2.putText(vis, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
    return vis


# --------------------------------------------------------------------------
# 模式：calib
# --------------------------------------------------------------------------

def cmd_calib(args) -> int:
    cam_cfg = dict(DEFAULT_CAMERA)
    cam_cfg.update({"index": args.camera, "fps": args.fps, "exposure": args.exposure})
    cap = open_camera(cam_cfg, write_exposure=args.write_exposure)
    gray = grab(cap, 3, DEFAULT_CAPTURE["fusion"], DEFAULT_CAPTURE["warmup"])
    cap.release()

    scale = args.view_scale
    disp = gray if scale == 1.0 else cv2.resize(
        gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    names = ["左位(半位)", "中位(十位)", "右位(个位)"]
    digits = []
    for i, nm in enumerate(names):
        win = f"calib - {nm}"
        print(f"\n请在窗口里框选【{nm}】的外接矩形，Enter 确认 / c 取消")
        x, y, w, h = cv2.selectROI(win, disp, showCrosshair=True)
        cv2.destroyWindow(win)
        if w <= 0 or h <= 0:
            print("已取消标定")
            return 1

        box = [int(round(x / scale)), int(round(y / scale)),
               int(round(w / scale)), int(round(h / scale))]
        segs = segments_from_box(box, args.thickness, args.shrink)

        if i == 0 and not args.no_first_half:
            segs = {k: v for k, v in segs.items() if k in ("b", "c")}
            code = dict(HALF_CODE)
        else:
            code = dict(FULL_CODE)

        digits.append({"name": nm, "box": box, "segments": segs, "code": code})
        print(f"  box={box}  段数={len(segs)}")

    xs = [d["box"][0] for d in digits]
    ys = [d["box"][1] for d in digits]
    x2 = [d["box"][0] + d["box"][2] for d in digits]
    y2 = [d["box"][1] + d["box"][3] for d in digits]
    display_roi = [min(xs), min(ys), max(x2) - min(xs), max(y2) - min(ys)]

    tmpl = {
        "camera": cam_cfg,
        "capture": dict(DEFAULT_CAPTURE),
        "recognition": dict(DEFAULT_RECOG),
        "display_roi": display_roi,
        "digits": digits,
    }

    out = Path(args.out)
    out.write_text(json.dumps(tmpl, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[完成] 模板已写入 {out.resolve()}")

    preview = out.with_name(out.stem + "_preview.png")
    vis = annotate(gray, tmpl, {"threshold": 0})
    if imwrite_unicode(preview, vis):
        print(f"[核对] 叠加预览图：{preview.resolve()}")
    else:
        print(f"[核对] 预览图写入失败：{preview}")
    print("       请确认每个采样框都落在笔画正中，没有越界到相邻段或间隙。")

    if args.show:
        cv2.imshow("preview", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


# --------------------------------------------------------------------------
# 模式：probe
# --------------------------------------------------------------------------

def cmd_probe(args) -> int:
    """
    扫描曝光值。不依赖模板 —— 找曝光本来就该在标定之前做，
    没有模板时只报告图像统计（含饱和占比），有模板时额外给出判读结果。
    """
    tp = Path(args.template)
    tmpl = json.loads(tp.read_text(encoding="utf-8")) if tp.exists() else None

    cam_cfg = dict(DEFAULT_CAMERA)
    if tmpl:
        cam_cfg.update(tmpl.get("camera", {}))
    cam_cfg.update({"index": args.camera, "fps": args.fps})
    cap = open_camera(cam_cfg, write_exposure=args.write_exposure)

    cap_cfg = dict(DEFAULT_CAPTURE)
    if tmpl:
        cap_cfg.update(tmpl.get("capture", {}))

    if not args.write_exposure:
        scan = [None]
        print("（默认不写曝光属性，曝光由相机自动控制）")
        print("（因此不扫描曝光值，只报告当前画面。亮度请靠减光片或收光圈调整）")
        print("（若你的相机支持手动曝光，加 --exposure-write 再跑这个命令）")
    else:
        scan = args.scan or [50, 100, 200, 300, 500, 800, 1200, 2000, 3000]
    if not tmpl:
        print(f"（未找到模板 {tp}，只报告图像统计。标定完成后重跑会附带判读结果）")

    if tmpl:
        print(f"\n{'曝光值':>8}{'回读':>9}{'均值':>8}{'p99':>7}{'饱和占比':>9}"
              f"{'阈值':>8}{'亮段':>6}   读数")
    else:
        print(f"\n{'曝光值':>8}{'回读':>9}{'均值':>8}{'p99':>7}{'饱和占比':>9}   建议")
    print("-" * 76)

    p99s, sats, means = [], [], []
    for v in scan:
        if v is not None:
            set_exposure(cap, v)
        gray = grab(cap, cap_cfg["frames"], cap_cfg["fusion"], cap_cfg["warmup"])
        p99 = float(np.percentile(gray, 99))
        sat = float((gray >= 250).mean() * 100.0)
        back = cap.get(cv2.CAP_PROP_EXPOSURE) or 0
        p99s.append(p99)
        sats.append(sat)
        means.append(round(gray.mean(), 1))
        label = "auto" if v is None else f"{v:g}"

        if tmpl:
            value, detail = read_frame(gray, tmpl)
            n_on = sum(len(d["on"]) for d in detail.get("digits", []))
            shown = value if value is not None else detail.get("error", "?")
            print(f"{label:>8}{back:>9.0f}{gray.mean():>8.1f}{p99:>7.1f}{sat:>8.1f}%"
                  f"{detail.get('threshold', 0):>8.1f}{n_on:>6}   {shown}")
        else:
            if sat > 5:
                tip = "过曝：加 ND 减光片或收光圈"
            elif p99 > 250:
                tip = "接近饱和，建议减光"
            elif p99 < 60:
                tip = "偏暗：可提高曝光或增益"
            else:
                tip = "亮度合适，可进入标定"
            print(f"{label:>8}{back:>9.0f}{gray.mean():>8.1f}{p99:>7.1f}{sat:>8.1f}%   {tip}")

    cap.release()

    # 诊断：区分"曝光没生效"和"真的过曝"——两者的处理方式完全不同
    if len(scan) > 2 and len(set(p99s)) == 1:
        print("\n[警告] 所有曝光档的亮度完全相同 —— 曝光设置根本没生效。")
        print("       按可能性排查：① 帧率没降下来；② 该相机固件不支持手动曝光；")
        print("       ③ 驱动用了不同的曝光单位（试试差两个数量级的值）；")
        print("       ④ 写曝光属性把画面搞坏了 —— 加 --no-exposure-write 重跑试试。")
        print(f"       py seg7_reader.py probe --camera {args.camera} --no-exposure-write")
    if len(scan) > 2 and len(set(means)) > 1:
        print(f"\n[提示] 画面确实随曝光值变化了（均值 {min(means)} ~ {max(means)}），"
              f"说明曝光可控。")
    if all(s > 5 for s in sats):
        print("\n[警告] 画面过曝 —— 必须加 ND 减光片或收光圈。")
        print("       别靠继续降曝光来解决：曝光短过刷新周期就会重新出现'只拍到一位'的问题。")

    print("\n挑选标准：亮段灰度落在 150-190，饱和像素占比接近 0。")
    print("选定后把该值写进模板，或在 read / calib 时用 --exposure 指定。")
    return 0


# --------------------------------------------------------------------------
# 模式：read
# --------------------------------------------------------------------------

def parse_duration(text) -> float:
    """
    解析时长写法，返回秒。支持 60 / 60s / 1m / 5m / 0.5h，纯数字按秒处理。
    """
    s = str(text).strip().lower()
    if not s:
        return 0.0
    mult = {"s": 1.0, "m": 60.0, "h": 3600.0}.get(s[-1])
    if mult is not None:
        s = s[:-1]
    else:
        mult = 1.0
    try:
        return max(0.0, float(s) * mult)
    except ValueError:
        raise RuntimeError(
            f"无法解析时长 {text!r}，请用 60 / 60s / 1m / 5m / 2h 这类写法")


def cmd_read(args) -> int:
    tmpl = json.loads(Path(args.template).read_text(encoding="utf-8"))
    cam_cfg = dict(DEFAULT_CAMERA)
    cam_cfg.update(tmpl.get("camera", {}))
    cam_cfg.update({"index": args.camera, "fps": args.fps})
    cap = open_camera(cam_cfg, write_exposure=args.write_exposure)

    if args.exposure is not None and args.write_exposure:
        set_exposure(cap, args.exposure)

    cap_cfg = dict(DEFAULT_CAPTURE)
    cap_cfg.update(tmpl.get("capture", {}))
    if args.frames:
        cap_cfg["frames"] = args.frames
    if args.fusion:
        cap_cfg["fusion"] = args.fusion

    # 周期节奏：命令行 --every 优先，其次模板里的 every
    every = parse_duration(args.every) if args.every is not None \
        else float(cap_cfg.get("every", 0) or 0)
    stop_at = set(args.stop_at or [])
    stop_streak = max(1, args.stop_streak)
    # 周期采集与终止条件都只在"持续运行"下才有意义，自动开启 watch
    if every > 0 or stop_at:
        args.watch = True

    csv_path = Path(args.csv) if args.csv else None
    if csv_path is not None:
        if csv_path.parent and not csv_path.parent.exists():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not csv_path.exists():
            csv_path.write_text("时间,电量\n", encoding="utf-8-sig")

    if every > 0:
        print(f"[周期] 每 {every:g} 秒一次"
              + (f"，共 {args.count} 轮" if args.count else "") + "，Ctrl+C 停止")
    elif args.watch:
        print(f"[周期] 连续运行，轮间隔 {args.interval:g} 秒，Ctrl+C 停止")
    if stop_at:
        print("[终止] 连续 {} 次读到 {} 时自动结束".format(
            stop_streak, " 或 ".join(str(v) for v in sorted(stop_at))))
    if csv_path is not None:
        print(f"[日志] 追加写入 {csv_path.resolve()}")

    history, last, streak = [], None, 0
    hit_value, hit_count = None, 0
    attempts, stop_reason = 0, ""
    try:
        while True:
            cycle_t0 = time.time()
            attempts += 1
            gray = grab(cap, cap_cfg["frames"], cap_cfg["fusion"], cap_cfg["warmup"])
            value, detail = read_frame(gray, tmpl)

            if args.debug:
                print(json.dumps(detail, ensure_ascii=False))
            if args.save_debug:
                imwrite_unicode(args.save_debug, annotate(gray, tmpl, detail))

            cycle_value, hit_note = None, ""
            if value is None or not detail.get("exact", True):
                reason = detail.get("error") or "段码未精确匹配"
                print(f"[丢弃] {reason}")
                last, streak = None, 0
                # 失败轮次不计数、也不重置命中计数（失败=没读到，不代表值变了）
            else:
                cycle_value = value
                history.append(value)
                streak = streak + 1 if value == last else 1
                last = value
                if stop_at:
                    if value in stop_at:
                        hit_count = hit_count + 1 if value == hit_value else 1
                        hit_value = value
                        hit_note = f" ｜ 命中 {value}: {hit_count}/{stop_streak}"
                    else:
                        hit_value, hit_count = None, 0
                shown = detail.get("raw") or "0"
                print(f"[读数] {value:>3}（原始 {shown!r}）"
                      f"连续一致 {streak}/{args.agree}{hit_note}")

            if csv_path is not None:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                with csv_path.open("a", encoding="utf-8-sig") as fh:
                    fh.write(f"{ts},{'' if cycle_value is None else cycle_value}\n")

            stable = cycle_value is not None and streak >= max(1, args.agree)

            # 终止条件先判，这样 JSON 里能带上 stopped 原因
            if stable and hit_value is not None and hit_count >= stop_streak:
                print(json.dumps({"value": cycle_value, "stable": True,
                                  "stopped": f"连续 {stop_streak} 次读到 {hit_value}",
                                  "attempts": attempts}, ensure_ascii=False))
                stop_reason = f"连续 {stop_streak} 次读到 {hit_value}"
                break

            if stable:
                print(json.dumps({"value": cycle_value, "stable": True,
                                  "attempts": attempts}, ensure_ascii=False))
                if not args.watch:
                    break

            if args.count and attempts >= args.count:
                print(f"[结束] 已跑满 {args.count} 轮")
                stop_reason = f"跑满 {args.count} 轮"
                break
            if not args.watch and attempts >= args.max_attempts:
                print(f"[超时] 尝试 {attempts} 次仍未达到连续 {args.agree} 帧一致")
                return 3

            # 严格节奏：睡「间隔 − 本轮耗时」，不因采集耗时而越跑越慢
            if every > 0:
                time.sleep(max(0.0, every - (time.time() - cycle_t0)))
            else:
                time.sleep(max(0.0, args.interval))
    except KeyboardInterrupt:
        print("\n已中断")
        stop_reason = "手动中断"
    finally:
        cap.release()

    if len(history) > 1:
        vals, cnts = np.unique(history, return_counts=True)
        print(f"[统计] 有效读数 {len(history)} 次，众数 = {vals[cnts.argmax()]}"
              + (f"，结束原因：{stop_reason}" if stop_reason else ""))
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def print_quickstart() -> None:
    print(
        "\n"
        "============================================================\n"
        " 数码管读数工具 —— 推荐执行顺序\n"
        "============================================================\n"
        " 0) 先抓一张图看看相机拍到了什么\n"
        "      py seg7_reader.py snap\n"
        "\n"
        " 1) 不接相机，先验算法\n"
        "      py seg7_reader.py selftest\n"
        "\n"
        " 2) 找设备号 / 排查画面异常\n"
        "      py seg7_reader.py diagnose\n"
        "\n"
        " 3) 看当前画面亮度（不需要模板）\n"
        "      py seg7_reader.py probe\n"
        "\n"
        " 4) 框选三个数字的 ROI，生成模板\n"
        "      py seg7_reader.py calib\n"
        "\n"
        " 5) 日常读数\n"
        "      py seg7_reader.py read --template seg7_roi.json\n"
        "\n"
        " 6) 周期性采集（长时记录放电曲线）\n"
        "      py seg7_reader.py read --template seg7_roi.json --every 60 \\\n"
        "          --csv battery_log.csv --stop-at 100 0\n"
        "\n"
        " 说明：默认不写曝光/增益属性（这台相机的驱动一写曝光画面就变纯白）。\n"
        "       只有确认相机支持手动曝光，才加 --exposure-write。\n"
        "\n"
        " 完整参数说明：py seg7_reader.py -h\n"
        "============================================================\n"
    )


def cmd_selftest(args) -> int:
    """转调离线自检脚本。注意必须传空的 argv —— 否则 argparse 会读到
    seg7_reader 自己的参数，报 unrecognized arguments。"""
    here = Path(__file__).resolve().parent
    script = here / "selftest_offline.py"
    if not script.exists():
        print(f"找不到 {script}")
        return 2
    sys.path.insert(0, str(here))
    import selftest_offline
    return selftest_offline.main([])


# --------------------------------------------------------------------------
# 模式：diagnose —— 相机没反应 / 画面全白 / 不确定设备号时用
# --------------------------------------------------------------------------

BACKENDS = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", getattr(cv2, "CAP_MSMF", cv2.CAP_ANY))]

CAM_PROP_LIST = [
    ("宽度", cv2.CAP_PROP_FRAME_WIDTH),
    ("高度", cv2.CAP_PROP_FRAME_HEIGHT),
    ("帧率", cv2.CAP_PROP_FPS),
    ("FOURCC", cv2.CAP_PROP_FOURCC),
    ("亮度", cv2.CAP_PROP_BRIGHTNESS),
    ("对比度", cv2.CAP_PROP_CONTRAST),
    ("增益", cv2.CAP_PROP_GAIN),
    ("曝光", cv2.CAP_PROP_EXPOSURE),
    ("自动曝光", cv2.CAP_PROP_AUTO_EXPOSURE),
]

# 属性写入的隔离测试：每项单独写，定位到底是哪个写入把画面搞坏的。
# 关键在于每一项都要**重新打开设备** —— 否则第一次写坏之后就一路坏到底，
# 分辨不出是哪一步的问题（这是第一版诊断脚本踩过的坑）。
ISO_SPECS = [
    ("基准：不写任何属性", {}),
    ("只写 AUTO_EXPOSURE=0.25", {"auto": 0.25}),
    ("只写 EXPOSURE=300", {"exposure": 300}),
    ("只写 GAIN=0", {"gain": 0}),
    ("AUTO=0.25 + EXPOSURE=300（现状做法）", {"auto": 0.25, "exposure": 300}),
    ("AUTO=0.25 + EXPOSURE=300 + GAIN=0", {"auto": 0.25, "exposure": 300, "gain": 0}),
]


def _iso_run(idx: int, bflag, spec: dict, warmup: int = 6):
    """重开设备 -> 抓基准 -> 写属性 -> 再抓。返回 (base_stats, after_stats)。"""
    cap = None
    try:
        cap = cv2.VideoCapture(idx, bflag)
        if not cap.isOpened():
            return None, None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 5)
        for _ in range(warmup):
            cap.read()
        ok, f = cap.read()
        if not ok or f is None:
            return None, None
        base = frame_stats(to_gray(f))

        if "auto" in spec:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(spec["auto"]))
        if "exposure" in spec:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(spec["exposure"]))
        if "gain" in spec:
            cap.set(cv2.CAP_PROP_GAIN, float(spec["gain"]))
        time.sleep(0.15)
        for _ in range(4):
            cap.read()
        ok, f = cap.read()
        if not ok or f is None:
            return base, None
        return base, frame_stats(to_gray(f))
    except cv2.error:
        return None, None
    finally:
        if cap is not None:
            cap.release()


def frame_stats(gray: np.ndarray) -> dict:
    """图像统计。min/max/std 是判断"是否真实图像"的关键。"""
    return {
        "mean": float(gray.mean()),
        "p99": float(np.percentile(gray, 99)),
        "min": int(gray.min()),
        "max": int(gray.max()),
        "std": float(gray.std()),
        "sat": float((gray >= 250).mean() * 100.0),
    }


def _probe_device(idx: int, backend_flag, warmup: int = 8):
    """打开一个设备并抓一帧，返回 (stats, gray) 或 (None, None)。"""
    cap = None
    try:
        cap = cv2.VideoCapture(idx, backend_flag)
        if not cap.isOpened():
            return None, None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 5)
        frame = None
        for _ in range(warmup):
            ok, frame = cap.read()
        if not ok or frame is None:
            return None, None
        gray = to_gray(frame)
        return frame_stats(gray), gray
    except cv2.error:
        return None, None
    finally:
        if cap is not None:
            cap.release()


def _verdict_for(st: dict) -> str:
    """根据统计给出"这像什么"的判断。"""
    if st["std"] < 1.0:
        return "全画面完全均匀（std≈0）—— 不是真实摄像头画面，疑似虚拟设备或驱动占位"
    if st["sat"] > 90:
        return "几乎全饱和 —— 拍到强光源或被正对亮面"
    if st["p99"] > st["mean"] + 30 and st["sat"] < 5:
        return "暗背景上有小片亮点 —— 像是拍到了自发光目标，可能就是数码管"
    return "普通画面"


def cmd_snap(args) -> int:
    """
    抓一帧存成图片 —— "先存一张图看一眼"用这个。

    只打开一次设备，默认不写任何曝光属性（这台相机的驱动一写就变纯白），
    所以是开箱即用的。想看原始未处理图像，这是最直接的方式。
    """
    cam_cfg = dict(DEFAULT_CAMERA)
    cam_cfg.update({"index": args.camera or 0, "fps": args.fps})
    cap = open_camera(cam_cfg, write_exposure=args.write_exposure)

    gray = grab(cap, args.frames or 1, args.fusion or DEFAULT_CAPTURE["fusion"],
                DEFAULT_CAPTURE["warmup"])
    cap.release()

    h, w = gray.shape[:2]
    st = frame_stats(gray)
    out = Path(args.image)
    saved = imwrite_unicode(out, gray)

    print()
    if saved:
        print(f"[已保存] {out.resolve()}")
    else:
        print(f"[保存失败] {out}")

    print(f"[画面] {w}x{h}  均值 {st['mean']:.1f}  p99 {st['p99']:.1f}  "
          f"min {st['min']}  max {st['max']}  std {st['std']:.2f}  "
          f"饱和 {st['sat']:.1f}%")
    print(f"[判断] {_verdict_for(st)}")

    if st["std"] < 1.0:
        print()
        print("画面是纯色（std≈0），说明没有真实图像数据。可能原因：")
        print("  · 设备号不对 —— 换 --camera 1 / 2 试试，或先跑 diagnose 枚举")
        print("  · 刚才是加了 --exposure-write 打开的 —— 去掉它重跑")
    elif st["sat"] > 30:
        print()
        print("画面大面积饱和（饱和像素 > 30%）—— 需要加 ND 减光片或收光圈。")
    elif st["p99"] > st["mean"] + 30 and st["sat"] < 5:
        print()
        print("这个特征（暗背景 + 小片亮点）很像拍到了自发光数码管。")
        print("打开图片确认一下：框里应该是那块 PCB 和蓝色数码管。")

    print()
    print("现在打开这张图看一眼，确认拍到的确实是那块数码管。")
    return 0


def cmd_diagnose(args) -> int:
    print("=" * 78)
    print("阶段 1：枚举相机设备")
    print("=" * 78)
    print("会从 0 号起逐个尝试打开摄像头，过程中设备指示灯可能短暂亮起。\n")
    print(f"{'设备':>4} {'后端':>6} {'分辨率':>10} {'均值':>7}{'p99':>7}"
          f"{'min':>5}{'max':>5}{'std':>7}{'饱和':>8}   判断")
    print("-" * 110)

    opened = []
    # 指定了 --camera 就只探测那一个，避免顺带打开机器上其它摄像头
    max_idx = args.camera if args.camera is not None else args.max_camera
    for idx in range(max_idx + 1):
        for bname, bflag in BACKENDS:
            st, gray = _probe_device(idx, bflag)
            if st is None:
                continue
            opened.append((idx, bname, st))
            h, w = gray.shape[:2]
            print(f"{idx:>4} {bname:>6} {w:>4}x{h:<5} {st['mean']:>7.1f}{st['p99']:>7.1f}"
                  f"{st['min']:>5}{st['max']:>5}{st['std']:>7.2f}{st['sat']:>7.1f}%   "
                  f"{_verdict_for(st)}")
            if args.save:
                given = Path(args.save)
                # 指定了单一设备时直接用给定文件名；枚举多设备时才加后缀区分
                if args.camera is not None and given.suffix:
                    out = given
                else:
                    out = given.with_name(f"diag_cam{idx}_{bname.lower()}.png")
                if imwrite_unicode(out, gray):
                    print(f"       抓图已存：{out}")
                else:
                    print(f"       抓图写入失败：{out}")

    if not opened:
        print("没有找到任何可打开的相机。")
        print("请确认：USB 线插好、相机指示灯亮、没有其它软件（相机 App / 视频会议）占用。")
        return 2

    print()
    print("=" * 78)
    print("怎么读这张表")
    print("=" * 78)
    print("· std ≈ 0 且 min = max → 该设备没有真实图像数据，不要选它")
    print("· 要找的是「暗背景 + 小片亮点」那一行，那才是拍到了数码管")
    print("· 同一台相机在不同后端下表观可能不同，以能控制曝光的那一个为准")

    if args.camera is None:
        print("\n确定设备号后，用 --camera <号> 做下一步：属性表 + 曝光可控性测试")
        print("  py seg7_reader.py diagnose --camera 1")
        return 0

    return _diagnose_one(args)


def cmd_iso_one(args) -> int:
    """
    子进程模式：只做一次「写前 vs 写后」对比，结果以 JSON 打到 stdout。

    为什么要拆成子进程：这类 UVC 驱动在被反复打开/关闭、或写入它不支持的
    属性时会直接把进程搞死（原生层崩溃，Python 的 try/except 拦不住）。
    放在子进程里跑，父进程就能正常地把「这一组合把驱动跑崩了」当成一条
    诊断结论记下来，而不是整个工具一起死掉。
    """
    spec = {}
    if args.iso_index is not None:
        if not (0 <= args.iso_index < len(ISO_SPECS)):
            print(f"--iso-index 超出范围，可选 0-{len(ISO_SPECS) - 1}")
            return 2
        spec = ISO_SPECS[args.iso_index][1]
    elif args.spec:
        # 注意：从 PowerShell/cmd 传 JSON 很容易被剥掉引号，
        # 实际使用请优先用 --iso-index，避免引号转义的坑
        try:
            spec = json.loads(args.spec)
        except json.JSONDecodeError as e:
            print(f"--spec 不是合法 JSON：{e}")
            print(f"建议改用 --iso-index（0-{len(ISO_SPECS) - 1}），或用单引号包住 JSON")
            return 2

    backend = cv2.CAP_MSMF if args.backend == "msmf" else cv2.CAP_DSHOW
    base, after = _iso_run(args.camera, backend, spec)
    print(json.dumps({"base": base, "after": after}, ensure_ascii=False))
    return 0


def _iso_line(name: str, base, after, note: str = "") -> str:
    if base is None:
        return f"    {name:<34}   打不开或取帧失败"
    if after is None:
        return f"    {name:<34}{base['mean']:>10.1f}      取帧失败"
    is_broken = after["std"] < 1.0
    if not note:
        note = "!! 写入后画面变纯色，不可用" if is_broken else "正常"
    return (f"    {name:<34}{base['mean']:>10.1f}{after['mean']:>10.1f}"
            f"{after['std']:>8.2f}{after['sat']:>7.1f}%   {note}")


def _iso_via_subprocess(idx: int, bname: str, ispec_idx: int, timeout: int):
    """
    在独立子进程里跑一次隔离项，返回 (base, after, note)。
    子进程崩溃/超时都当成一条诊断结论，而不是让父进程一起死。

    用 --iso-index 传参而不是传 JSON —— 避免命令行引号被 shell 剥掉的坑。
    """
    cmd = [sys.executable, str(Path(__file__).resolve()), "iso-one",
           "--camera", str(idx), "--backend", bname.lower(),
           "--iso-index", str(ispec_idx)]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return None, None, f"超时（{timeout}s）—— 该组合可能把驱动挂住了"
    except OSError as e:
        return None, None, f"子进程启动失败：{e}"

    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()
        hint = tail[-1][:60] if tail else ""
        return None, None, f"子进程异常退出 rc={p.returncode} —— 该组合可能导致驱动崩溃 {hint}"

    line = ""
    for ln in reversed((p.stdout or "").strip().splitlines()):
        if ln.strip().startswith("{"):
            line = ln.strip()
            break
    if not line:
        return None, None, "子进程没有返回结果"
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None, None, "子进程返回内容无法解析"
    return data.get("base"), data.get("after"), ""


def _diagnose_one(args) -> int:
    idx = args.camera
    print()
    print("=" * 78)
    print(f"阶段 2：{idx} 号设备的属性表与写入隔离测试")
    print("=" * 78)

    verdicts = {}
    # 隔离测试要反复开关设备，很慢，所以默认只测一个后端
    if args.iso_backend == "both":
        iso_backends = list(BACKENDS)
    else:
        iso_backends = [b for b in BACKENDS if b[0].lower() == args.iso_backend]
    print(f"\n（隔离测试会对每个组合单独开一次设备，通常需要 20-60 秒，请稍候）")

    for bname, bflag in iso_backends:
        cap = None
        try:
            cap = cv2.VideoCapture(idx, bflag)
            if not cap.isOpened():
                print(f"\n[{bname}] 打不开，跳过")
                continue
            print(f"\n[{bname}] 属性回读（初始值）")
            for label, prop in CAM_PROP_LIST:
                try:
                    print(f"    {label:<8} = {cap.get(prop):g}")
                except cv2.error:
                    print(f"    {label:<8} = <读不到>")
        except cv2.error as e:
            print(f"\n[{bname}] 读属性出错：{e}")
        finally:
            if cap is not None:
                cap.release()

        print(f"\n[{bname}] 写入隔离测试")
        print("    （每项都重新打开设备，而且跑在独立子进程里 —— 一是避免上一次的写入")
        print("      污染下一次，二是这类驱动被反复开关时可能直接崩掉，隔离子进程后")
        print("      「崩了」本身也能作为一条结论保留下来）")
        print(f"    {'测试项':<34}{'写前均值':>10}{'写后均值':>10}{'std':>8}"
              f"{'饱和':>8}   判定")
        print("    " + "-" * 88)

        broken, tried, crashed = [], [], []
        for i, (name, spec) in enumerate(ISO_SPECS):
            base, after, note = _iso_via_subprocess(idx, bname, i, args.iso_timeout)
            if "崩溃" in note or "超时" in note or "无法解析" in note or "没有返回" in note:
                crashed.append(name)
                print(f"    {name:<34}   {note}")
                continue
            if not spec:
                note = note or "基准"
            elif after and after["std"] < 1.0:
                broken.append(name)
                note = note or "!! 写入后画面变纯色，不可用"
            note = note or "正常"
            if base is not None and after is not None:
                tried.append(name)
            print(_iso_line(name, base, after, note))
        verdicts[bname] = (tried, broken, crashed)

    print()
    print("=" * 78)
    print("结论与下一步")
    print("=" * 78)

    if not verdicts:
        print("两种后端都打不开该设备，无法进一步诊断。")
        return 2

    all_crashed = [n for _, _, c in verdicts.values() for n in c]
    any_broken = [n for _, b, _ in verdicts.values() for n in b]

    if any_broken:
        print("已定位：**写下面这些属性会把画面变成纯色**——")
        for n in any_broken:
            print(f"    · {n}")
        print()
        print("这不是相机坏了，是它的驱动不接受 OpenCV 这种写法。")
        print("→ 直接用不写曝光属性的方式重跑：")
        print(f"     py seg7_reader.py probe --camera {idx} --no-exposure-write")
        print("   相机会用自己的自动曝光。场景偏暗时自动曝光本来就会拉长，")
        print("   很可能已经长到能覆盖数码管的整个刷新周期。")
    elif all_crashed:
        print("所有隔离项都异常结束（崩溃/超时），说明该设备的驱动对反复开关")
        print("非常敏感。这不影响正常使用——实际采集只开一次设备。")
        print("→ 直接试不写曝光属性的方式：")
        print(f"     py seg7_reader.py probe --camera {idx} --no-exposure-write")
    else:
        print("属性写入没有破坏画面。那么 probe 报\"所有档位亮度相同\"的原因是")
        print("**曝光值本身没被驱动接受**，同样建议改用 --no-exposure-write。")
        print()
        print(f"     py seg7_reader.py probe --camera {idx} --no-exposure-write")

    print()
    print("若想手动确认，运行下面这条会弹出驱动原生属性页 + 实时预览：")
    print(f"     py seg7_reader.py --props --camera {idx}")
    return 0


def cmd_props(args) -> int:
    """弹出驱动的原生属性页 + 实时预览，用来手动确认曝光是否真的可调。"""
    idx = args.camera or 0
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"打不开 {idx} 号相机")
        return 2

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 5)

    print("=" * 78)
    print("手动验证模式")
    print("=" * 78)
    print("1. 屏幕上会弹出一个属性对话框（驱动原生页）和一个预览窗口")
    print("2. 在对话框里找到 Exposure / 曝光，关掉自动档，手动拖到底再拖到顶")
    print("3. 盯着预览窗口：画面亮度跟着变 → 曝光可调；毫无变化 → 该驱动控不了")
    print("4. 按 q 或 ESC 退出\n")
    print("提示：预览窗口里画面如果一直是纯白一片，就说明根本没读到真实图像，")
    print("      先去检查 USB 连接和设备号，不要在这里白调。")

    try:
        cap.set(cv2.CAP_PROP_SETTINGS, 1)   # 打开 DirectShow 驱动属性页
    except cv2.error:
        print("（该后端不支持弹出属性页，但预览窗口仍然可用）")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("取帧失败，退出")
                break
            gray = to_gray(frame)
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.putText(vis, f"mean={gray.mean():.1f}  p99={np.percentile(gray, 99):.1f}"
                             f"  std={gray.std():.2f}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.imshow(f"camera {idx}  (press q to quit)", vis)
            if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="全局快门黑白相机读取 3 位数码管（0-100）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("mode", nargs="?", default=None,
                   choices=["snap", "diagnose", "selftest", "probe", "calib",
                            "read", "iso-one"],
                   help="运行模式；不带参数会打印推荐执行顺序（iso-one 为内部使用）")
    p.add_argument("--camera", type=int, default=None,
                   help="相机序号，默认 0。diagnose 不带此参数则只做设备枚举")
    p.add_argument("--fps", type=float, default=5,
                   help="帧率，默认 5。必须先降帧率，否则长曝光设不进去")
    p.add_argument("--exposure", type=float, default=300,
                   help="UVC 曝光原始值。单位随驱动而异，用 probe 模式实测确定")

    p.add_argument("--out", default="seg7_roi.json", help="calib: 模板输出路径")
    p.add_argument("--thickness", type=float, default=0.14,
                   help="calib: 笔画粗细占字高比例，默认 0.14")
    p.add_argument("--shrink", type=float, default=0.18,
                   help="calib: 段采样框内缩比例，默认 0.18")
    p.add_argument("--view-scale", type=float, default=1.0, help="calib: 预览缩放倍数")
    p.add_argument("--no-first-half", action="store_true", help="calib: 左位按 7 段处理")
    p.add_argument("--show", action="store_true", help="calib: 结束后弹窗显示预览")

    p.add_argument("--template", default="seg7_roi.json", help="模板 JSON 路径")
    p.add_argument("--scan", type=float, nargs="+", default=None,
                   help="probe: 要扫描的曝光值。不填则用一组默认值")

    p.add_argument("--frames", type=int, default=None, help="read: 融合帧数")
    p.add_argument("--fusion", choices=["mean", "max"], default=None,
                   help="read: 多帧融合方式，默认 mean（逐像素平均，抗光晕跳变）。"
                        "max 仅在被测物会动、必须短曝光抓拍时使用——它会保留多帧里"
                        "最差的一帧，帧数越多越容易丢帧")
    p.add_argument("--agree", type=int, default=3, help="read: 连续几帧一致才输出，默认 3")
    p.add_argument("--interval", type=float, default=0.1,
                   help="read: 同一个读数内部、两次取帧之间的间隔秒数（配合 --agree 用）。"
                        "想设「多久采集一次」请用 --every")
    p.add_argument("--every", default=None,
                   help="read: 周期性采集的间隔，支持 60 / 60s / 1m / 5m / 2h。"
                        "设置后自动持续运行，并按严格节奏（睡「间隔 − 本轮耗时」），"
                        "不会因为采集本身要花几秒而越跑越慢。"
                        "不设置时沿用模板 seg7_roi.json 里的 capture.every")
    p.add_argument("--csv", default=None,
                   help="read: 每轮读数追加一行「时间,电量」到该 CSV。"
                        "文件不存在时自动写表头；失败的轮次写空值行，便于看出断点")
    p.add_argument("--stop-at", type=int, nargs="+", default=None,
                   help="read: 连续读到这些值就自动终止，例如 --stop-at 100 0。"
                        "设置后自动持续运行")
    p.add_argument("--stop-streak", type=int, default=5,
                   help="read: 需要连续命中多少次 --stop-at 的值才终止，默认 5")
    p.add_argument("--count", type=int, default=0,
                   help="read: 跑满多少轮后退出（每轮 = 一次采集输出），0 = 不限")
    p.add_argument("--max-attempts", type=int, default=30,
                   help="read: 未收敛时的最大尝试次数，默认 30")
    p.add_argument("--watch", action="store_true", help="read: 持续输出，不中途退出")
    p.add_argument("--debug", action="store_true", help="read: 打印每段实测灰度明细")
    p.add_argument("--save-debug", default=None, help="read: 把标注图存到指定路径")

    p.add_argument("--max-camera", type=int, default=5,
                   help="diagnose: 枚举到几号设备，默认枚举 0-5")
    p.add_argument("--props", action="store_true",
                   help="弹出驱动原生属性页 + 实时预览，手动确认曝光是否可调")
    p.add_argument("--save", default=None,
                   help="diagnose: 把抓到的帧存成 PNG（给出路径前缀）")
    p.add_argument("--exposure-write", action="store_true",
                   help="显式要求写入曝光/增益属性。默认不写 —— 实测这台相机的驱动"
                        "一写曝光画面就变纯白（见方案文档第 12 节），所以默认交给"
                        "相机自身的自动曝光。只有确认你的相机支持手动曝光才加它")
    p.add_argument("--no-exposure-write", action="store_true",
                   help="兼容参数：默认行为本来就不写曝光属性，加不加都一样")
    p.add_argument("--image", default="snapshot.png",
                   help="snap: 抓图保存路径，默认 snapshot.png")
    p.add_argument("--iso-timeout", type=int, default=20,
                   help="diagnose: 单个隔离项的超时秒数，默认 20")
    p.add_argument("--iso-backend", choices=["dshow", "msmf", "both"], default="dshow",
                   help="diagnose: 隔离测试跑哪个后端，默认只跑 dshow（这个测试很慢）")
    p.add_argument("--spec", default=None, help="iso-one: 属性组合（JSON，注意 shell 会剥引号）")
    p.add_argument("--iso-index", type=int, default=None,
                   help="iso-one: 按序号选择属性组合（推荐，免去 JSON 转义问题）")
    p.add_argument("--backend", choices=["dshow", "msmf"], default="dshow",
                   help="iso-one: 使用哪个后端")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # 默认不写曝光属性；只有显式 --exposure-write 才写。
    # --no-exposure-write 保留为兼容参数（与默认行为一致）。
    args.write_exposure = bool(args.exposure_write) and not bool(args.no_exposure_write)

    if args.props:
        return cmd_props(args)

    if args.mode is None:
        print_quickstart()
        return 0

    if args.mode == "iso-one":
        return cmd_iso_one(args)

    if args.mode == "snap":
        return cmd_snap(args)

    if args.mode == "diagnose":
        # diagnose 不填 --camera 时只做设备枚举，所以这里不套默认值
        return cmd_diagnose(args)

    # 其余模式默认用 0 号相机
    if args.camera is None:
        args.camera = 0

    try:
        if args.mode == "selftest":
            return cmd_selftest(args)

        if args.mode == "probe":
            return cmd_probe(args)

        if args.mode == "read":
            if not Path(args.template).exists():
                print(f"找不到模板 {args.template}，请先跑：py seg7_reader.py calib")
                return 2
            return cmd_read(args)

        return cmd_calib(args)

    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    except RuntimeError as e:
        print(f"\n[错误] {e}")
        print("提示：相机打不开的话，先跑 py seg7_reader.py diagnose 定位设备号")
        return 2
    except cv2.error as e:
        print(f"\n[OpenCV 错误] {e}")
        print("提示：calib 模式需要图形界面，请在桌面环境下运行，不要用无头终端。")
        return 2


if __name__ == "__main__":
    sys.exit(main())
