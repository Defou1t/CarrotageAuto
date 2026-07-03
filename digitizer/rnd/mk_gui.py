r"""GUI-обёртка разделителя MK (Трек 2): вход — изображение ИЛИ папка (наши форматы: скан + парный
<stem>.nlgx рядом), выход — папка с ЦВЕТНЫМИ НАЛОЖЕНИЯМИ (MGZ=красн / MPZ=син) + трассы .npz.
Кнопки выгрузки nlgx/LAS — заглушки (следующий этап). Лёгкий UI: инференс гоняется в venv ComfyUI
отдельным процессом (torch не тащим в UI → отзывчивость + изоляция падений).

  python mk_gui.py            # запуск интерфейса (основной py3.14 с tkinter)
"""
import sys, threading, subprocess, queue, re, csv
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext

HERE = Path(__file__).resolve().parent
VENV = r"D:\ComfyUI\StabilityMatrix\Data\Packages\ComfyUI\venv\Scripts\python.exe"
INFER = str(HERE / "infer_mk.py")
CKPT_DEFAULT = r"F:\nds\output\mk_data\mk_sep_v7.pt"  # production: ансамбль v7+v4+v5 (QC 03.07:
ENS_LIST = "mk_sep_v4.pt,mk_sep_v5.pt"                # v8-ансамбль «перескакивает» — идентичность
CKPT_FAST = r"F:\nds\output\mk_data\mk_sep.pt"        # для эксперта важнее px-следования)
# быстрый режим = v4 соло (v7/v8 соло консервативнее/нестабильнее — только в ансамбле)
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def list_images(inp):
    """Файл → [файл]; папка → отсортированные изображения (без *_overlay.png, чтобы не зациклить выход)."""
    p = Path(inp)
    if p.is_file():
        return [p] if p.suffix.lower() in IMG_EXT else []
    if p.is_dir():
        return sorted(f for f in p.iterdir()
                      if f.suffix.lower() in IMG_EXT and not f.stem.endswith("_overlay"))
    return []


class App:
    def __init__(self, root):
        self.root = root
        root.title("MK разделитель — оцифровка наложений (MGZ/MPZ)")
        root.geometry("820x560")
        self.q = queue.Queue()
        self.busy = False
        pad = {"padx": 6, "pady": 4}

        frm = ttk.Frame(root, padding=10); frm.pack(fill="both", expand=True)
        for i in range(3):
            frm.columnconfigure(1, weight=1)

        # вход
        ttk.Label(frm, text="Вход (изображение или папка):").grid(row=0, column=0, sticky="w", **pad)
        self.inp = tk.StringVar()
        ttk.Entry(frm, textvariable=self.inp).grid(row=0, column=1, sticky="ew", **pad)
        bf = ttk.Frame(frm); bf.grid(row=0, column=2, sticky="e", **pad)
        ttk.Button(bf, text="Файл…", command=self.pick_file).pack(side="left", padx=2)
        ttk.Button(bf, text="Папка…", command=self.pick_indir).pack(side="left", padx=2)

        # выход
        ttk.Label(frm, text="Выходная папка (наложения):").grid(row=1, column=0, sticky="w", **pad)
        self.out = tk.StringVar()
        ttk.Entry(frm, textvariable=self.out).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Папка…", command=self.pick_outdir).grid(row=1, column=2, sticky="e", **pad)

        # чекпойнт
        ttk.Label(frm, text="Модель (.pt):").grid(row=2, column=0, sticky="w", **pad)
        self.ckpt = tk.StringVar(value=CKPT_DEFAULT)
        ttk.Entry(frm, textvariable=self.ckpt).grid(row=2, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Файл…", command=self.pick_ckpt).grid(row=2, column=2, sticky="e", **pad)

        # действия
        act = ttk.Frame(frm); act.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        self.run_btn = ttk.Button(act, text="▶ Распознать → наложения", command=self.start)
        self.run_btn.pack(side="left", padx=4)
        self.quality = tk.BooleanVar(value=True)                 # ансамбль v4+v5 + TTA: −3пп свопов (eval 02.07)
        ttk.Checkbutton(act, text="Качество (медленнее)", variable=self.quality).pack(side="left", padx=6)
        self.refine = tk.BooleanVar(value=True)                  # дотяжка пиков по чернилам (QC 03.07)
        ttk.Checkbutton(act, text="Дотяжка пиков", variable=self.refine).pack(side="left", padx=2)
        ttk.Button(act, text="Открыть выход", command=self.open_out).pack(side="left", padx=4)
        self.nlgx_btn = ttk.Button(act, text="Выгрузить nlgx+bck", command=lambda: self.export(las=False))
        self.nlgx_btn.pack(side="right", padx=4)
        self.las_btn = ttk.Button(act, text="Выгрузить LAS", command=lambda: self.export(las=True))
        self.las_btn.pack(side="right", padx=4)

        self.prog = ttk.Progressbar(frm, mode="determinate")
        self.prog.grid(row=4, column=0, columnspan=3, sticky="ew", **pad)
        self.status = tk.StringVar(value="Готов. Укажите вход и выходную папку.")
        ttk.Label(frm, textvariable=self.status).grid(row=5, column=0, columnspan=3, sticky="w", **pad)

        self.log = scrolledtext.ScrolledText(frm, height=16, font=("Consolas", 9))
        self.log.grid(row=6, column=0, columnspan=3, sticky="nsew", **pad)
        frm.rowconfigure(6, weight=1)
        self._poll()

    # --- выбор путей ---
    def pick_file(self):
        f = filedialog.askopenfilename(title="Скан MK",
                                       filetypes=[("Изображения", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"), ("Все", "*.*")])
        if f:
            self.inp.set(f)
            if not self.out.get():
                self.out.set(str(Path(f).parent / "mk_overlays"))

    def pick_indir(self):
        d = filedialog.askdirectory(title="Папка со сканами")
        if d:
            self.inp.set(d)
            if not self.out.get():
                self.out.set(str(Path(d) / "mk_overlays"))

    def pick_outdir(self):
        d = filedialog.askdirectory(title="Выходная папка")
        if d: self.out.set(d)

    def pick_ckpt(self):
        f = filedialog.askopenfilename(title="Модель", filetypes=[("PyTorch", "*.pt"), ("Все", "*.*")])
        if f: self.ckpt.set(f)

    def open_out(self):
        d = self.out.get()
        if d and Path(d).is_dir():
            import os
            os.startfile(d)  # noqa: проводник Windows

    # --- лог через очередь (tkinter не потокобезопасен) ---
    def emit(self, msg): self.q.put(msg)

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if isinstance(msg, tuple):                       # ("done", n) / ("prog", v) / ("status", s)
                    kind, val = msg
                    if kind == "prog": self.prog["value"] = val
                    elif kind == "status": self.status.set(val)
                    elif kind == "done":
                        self.busy = False; self.run_btn["state"] = "normal"
                        self.status.set(val)
                else:
                    self.log.insert("end", msg); self.log.see("end")
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    # --- запуск ---
    def start(self):
        if self.busy: return
        imgs = list_images(self.inp.get())
        outd = self.out.get().strip()
        if not imgs:
            self.status.set("Не найдено изображений по указанному входу."); return
        if not outd:
            self.status.set("Укажите выходную папку."); return
        Path(outd).mkdir(parents=True, exist_ok=True)
        self.busy = True; self.run_btn["state"] = "disabled"
        self.log.delete("1.0", "end")
        threading.Thread(target=self._work, args=(imgs, outd), daemon=True).start()

    # --- выгрузка nlgx+bck (+LAS): трассы из выходной папки + парный шаблон <stem>.nlgx у скана ---
    def export(self, las=False):
        if self.busy: return
        imgs = list_images(self.inp.get())
        outd = self.out.get().strip()
        if not imgs or not outd:
            self.status.set("Для выгрузки укажите вход и выходную папку (после распознавания)."); return
        jobs = []                                                # (traces.npz, template.nlgx, scan)
        for img in imgs:
            sib = img.with_suffix(".nlgx")
            npz = Path(outd) / f"{img.stem[:40]}_traces.npz"
            if sib.exists() and npz.exists():
                jobs.append((npz, sib, img))
        if not jobs:
            self.status.set("Нет пар трассы+шаблон: сначала «Распознать», и рядом со сканом нужен <имя>.nlgx."); return
        self.busy = True; self.run_btn["state"] = "disabled"
        threading.Thread(target=self._export_work, args=(jobs, outd, las), daemon=True).start()

    def _export_work(self, jobs, outd, las):
        sys.path.insert(0, str(HERE))
        try:
            from mk_export import export_bundle
        except Exception as e:
            self.q.put(("done", f"Ошибка импорта mk_export: {e}")); return
        self.emit(f"\nВыгрузка {'LAS' if las else 'nlgx+bck'}: {len(jobs)} файл(ов) -> {outd}\n")
        ok = 0; rows = []
        for npz, tpl, img in jobs:
            self.emit(f"=== {img.name} ===\n")
            rec = {"файл": img.name, "верификация": "ошибка", "rt_MGZ_px": "", "rt_MPZ_px": ""}
            try:
                rep = export_bundle(npz, tpl, scan_path=img, out_dir=outd,
                                    write_las=las, log=lambda m: self.emit(m + "\n"))
                if rep:
                    rec["верификация"] = rep.get("verify", "?")
                    rec["rt_MGZ_px"] = rep.get("rt_MGZ", ""); rec["rt_MPZ_px"] = rep.get("rt_MPZ", "")
                if rep and rep.get("verify") == "OK":
                    ok += 1
            except Exception as e:
                self.emit(f"  ! ошибка: {e}\n")
            rows.append(rec)
        if rows:
            csvp = Path(outd) / "export_summary.csv"
            with open(csvp, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            self.emit(f"Сводка выгрузки -> {csvp}\n")
        self.q.put(("done", f"Выгружено {ok}/{len(jobs)} в {outd}"))

    def _work(self, imgs, outd):
        self.emit(f"Изображений: {len(imgs)} | выход: {outd}\n")
        self.q.put(("prog", 0))
        ok = 0; rows = []                                        # QC-сводка батча
        for i, img in enumerate(imgs):
            self.q.put(("status", f"[{i+1}/{len(imgs)}] {img.name}"))
            self.emit(f"\n=== [{i+1}/{len(imgs)}] {img.name} ===\n")
            sib = img.with_suffix(".nlgx")
            ck = self.ckpt.get()
            if not self.quality.get() and ck == CKPT_DEFAULT:
                ck = CKPT_FAST                                   # быстрый режим = стабильный v4 соло
            cmd = [VENV, INFER, ck, str(img), "--out", outd]
            if self.quality.get():
                cmd += ["--ens", ENS_LIST, "--tta"]              # ансамбль чекпойнтов + флипы
            if self.refine.get():
                cmd += ["--refine"]                              # дотяжка пиков + сглаживание
            if sib.exists():
                cmd += ["--nlgx", str(sib)]
            else:
                self.emit("  (нет парного .nlgx — frameless: только полное наложение)\n")
            rec = {"файл": img.name, "nlgx": "да" if sib.exists() else "нет",
                   "MGZ_точек": "", "MPZ_точек": "", "вторая_полоса": "", "статус": "ошибка"}
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding="utf-8", errors="replace")
                for line in proc.stdout:
                    self.emit("  " + line)
                    mm = re.search(r"MGZ точек=(\d+) MPZ=(\d+)", line)
                    if mm:
                        rec["MGZ_точек"], rec["MPZ_точек"] = mm.group(1), mm.group(2)
                    if "вторая полоса" in line:
                        rec["вторая_полоса"] = "⚠ есть (5×?)"
                proc.wait()
                if proc.returncode == 0:
                    ok += 1; rec["статус"] = "ок"
                else:
                    self.emit(f"  ! код выхода {proc.returncode}\n")
            except Exception as e:
                self.emit(f"  ! ошибка: {e}\n")
            rows.append(rec)
            self.q.put(("prog", (i + 1) * 100 / len(imgs)))
        if rows:
            csvp = Path(outd) / "qc_summary.csv"
            with open(csvp, "w", newline="", encoding="utf-8-sig") as fh:  # sig: Excel читает кириллицу
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            self.emit(f"\nQC-сводка батча -> {csvp}\n")
        self.q.put(("done", f"Готово: {ok}/{len(imgs)} наложений в {outd}"))


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
