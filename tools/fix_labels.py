# -*- coding: utf-8 -*-
"""
定向标签修复：只改 tools/audit_labels.py 审计「确证」的框，绝不做全局改名。

背景：audit_labels.py 已证明 bus 类 135 框并非全部标错（3 框确证是真公交车、
13 框机器无法定论），全局 --rename 会误伤它们。本工具按审计 CSV 逐框白名单改写：
    - 只改 verdict=<to> 且 confidence 在允许档位内的框；
    - 目标类别若不在 data.yaml 的 names 里，则追加为新类别（person 会拿到下标 17）；
    - 默认 dry-run 只打印将发生的改动；加 --apply 才真正写盘；
    - 写盘前把被改的标签文件与 data.yaml 备份到 .eval_cache/<from>_fix_backup/。

注意：本地修复只服务于评估口径；Roboflow 源项目里的标签仍需同步修正后重新导出，
否则下次导出会覆盖本地修复。

用法（在项目根目录运行）：
    # 先看会改什么（默认只改 HIGH/MEDIUM 两档，共 101 框）
    ".venv\\Scripts\\python.exe" tools\\fix_labels.py --from bus --to person
    # 连 LOW 档一起改（119 框）
    ".venv\\Scripts\\python.exe" tools\\fix_labels.py --from bus --to person --confidence HIGH,MEDIUM,LOW
    # 确认无误后真正写盘
    ".venv\\Scripts\\python.exe" tools\\fix_labels.py --from bus --to person --confidence HIGH,MEDIUM,LOW --apply
"""
import os
import sys
import io
import csv
import shutil
import argparse
import collections

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import BASE_DIR
from eval_dataset import find_splits, load_yaml_names

ALL_CONF = ["HIGH", "MEDIUM", "LOW"]


def main():
    parser = argparse.ArgumentParser(description="按审计白名单定向修复标签")
    parser.add_argument("--data",
                        default=os.path.join(BASE_DIR, "datasets",
                                             "world-monitoring-v2-121", "data.yaml"))
    parser.add_argument("--audit",
                        default=os.path.join(BASE_DIR, ".eval_cache", "bus_audit", "audit.csv"))
    parser.add_argument("--from", dest="src", default="bus", help="被审计的源类别名")
    parser.add_argument("--to", dest="dst", default="person", help="审计确证的目标类别名")
    parser.add_argument("--confidence", default="HIGH,MEDIUM",
                        help=f"允许改写的置信档，逗号分隔，可选 {ALL_CONF}")
    parser.add_argument("--apply", action="store_true", help="真正写盘（默认只 dry-run）")
    args = parser.parse_args()

    if not os.path.exists(args.audit):
        print(f"[错误] 找不到审计结果 {args.audit}，请先运行 tools/audit_labels.py")
        sys.exit(1)
    conf_ok = {c.strip().upper() for c in args.confidence.split(",") if c.strip()}
    bad = conf_ok - set(ALL_CONF)
    if bad:
        print(f"[错误] 未知置信档 {sorted(bad)}，可选 {ALL_CONF}")
        sys.exit(1)

    src_root = os.path.dirname(os.path.abspath(args.data))
    names = load_yaml_names(args.data)
    if args.src not in names:
        print(f"[错误] 数据集类别表里没有 {args.src}")
        sys.exit(1)
    src_id = names.index(args.src)
    append_new = args.dst not in names
    dst_id = len(names) if append_new else names.index(args.dst)

    # 白名单：(split, image, line) -> 置信档
    want = {}
    skipped = collections.Counter()
    with open(args.audit, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["verdict"] != args.dst:
                skipped[r["verdict"]] += 1
                continue
            if r["confidence"] not in conf_ok:
                skipped[f"{args.dst}/{r['confidence']}(档位未选)"] += 1
                continue
            want[(r["split"], r["image"], int(r["line"]))] = r["confidence"]

    print(f"审计白名单: {len(want)} 框将 {args.src}(下标{src_id}) → "
          f"{args.dst}(下标{dst_id}{'，新增类别' if append_new else ''})")
    print(f"保持不动: {dict(skipped)}")
    if not want:
        print("[提示] 白名单为空，无需改动")
        return

    # 定位标签文件并按行分组
    label_of = {}
    for sp, img_dir, lbl_dir in find_splits(src_root):
        for r in list(want):
            if r[0] == sp and r[1] not in label_of:
                lf = os.path.join(lbl_dir, r[1] + ".txt")
                if os.path.isfile(lf):
                    label_of[r[1]] = lf
    per_file = collections.defaultdict(list)
    for (sp, img, li), conf in want.items():
        if img not in label_of:
            print(f"[警告] 找不到标签文件 {sp}/{img}.txt，跳过该框")
            continue
        per_file[label_of[img]].append((li, conf))

    n_lines = sum(len(v) for v in per_file.values())
    print(f"涉及标签文件 {len(per_file)} 个 / 共 {n_lines} 行")
    if not args.apply:
        print("\n[dry-run] 未加 --apply，不做任何写盘。样例（前 10 个文件）：")
        for lf, rows in list(per_file.items())[:10]:
            confs = collections.Counter(c for _, c in rows)
            print(f"    {os.path.relpath(lf, src_root)}  改 {len(rows)} 行 {dict(confs)}")
        print("\n确认无误后加 --apply 执行。")
        return

    backup = os.path.join(BASE_DIR, ".eval_cache", f"{args.src}_fix_backup")
    os.makedirs(backup, exist_ok=True)
    for lf, rows in per_file.items():
        rel = os.path.relpath(lf, src_root).replace(os.sep, "__")
        shutil.copy2(lf, os.path.join(backup, rel))
        lines = open(lf, encoding="utf-8").read().splitlines()
        hit = {li for li, _ in rows}
        changed = 0
        for li in hit:
            if li >= len(lines):
                print(f"[警告] {lf} 只有 {len(lines)} 行，跳过第 {li} 行")
                continue
            p = lines[li].split()
            if not p or int(float(p[0])) != src_id:
                print(f"[警告] {lf} 第 {li} 行类别是 {p[0] if p else '空'}，"
                      f"不是 {src_id}，跳过（审计与标签已不同步）")
                continue
            p[0] = str(dst_id)
            lines[li] = " ".join(p)
            changed += 1
        with open(lf, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        print(f"  已改 {changed:>3} 行  {os.path.relpath(lf, src_root)}")

    if append_new:
        shutil.copy2(args.data, os.path.join(backup, "data.yaml"))
        text = open(args.data, encoding="utf-8").read()
        out = []
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("names:") and s.endswith("]"):
                line = line[:line.rindex("]")] + f", '{args.dst}']"
            elif s.startswith("nc:"):
                line = f"nc: {dst_id + 1}"
            out.append(line)
        with open(args.data, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
        print(f"  data.yaml 已追加类别 {args.dst}（下标 {dst_id}，nc={dst_id + 1}）")
    print(f"\n备份目录: {backup}")
    print("提醒: Roboflow 源项目里的标签仍需同步修正后重新导出，否则下次导出会覆盖本地修复。")


if __name__ == "__main__":
    main()
