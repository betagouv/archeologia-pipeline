from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from src.pipeline.coords import extract_xy_from_filename, infer_xy_from_file


class CoordsAnalyzerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Archeologia - Analyse coordonnées")
        self.geometry("760x360")

        self.path_var = tk.StringVar(value="")

        top = ttk.Frame(self, padding=12)
        top.pack(fill=tk.BOTH, expand=False)

        ttk.Label(top, text="Fichier à analyser (.laz/.las/.tif/.tiff/.asc/.jpg):").grid(row=0, column=0, sticky="w")

        entry = ttk.Entry(top, textvariable=self.path_var, width=90)
        entry.grid(row=1, column=0, sticky="we", pady=(6, 0))

        btns = ttk.Frame(top)
        btns.grid(row=1, column=1, sticky="e", padx=(8, 0), pady=(6, 0))

        ttk.Button(btns, text="Parcourir…", command=self._browse).pack(side=tk.LEFT)
        ttk.Button(btns, text="Analyser", command=self._analyze).pack(side=tk.LEFT, padx=(8, 0))

        top.columnconfigure(0, weight=1)

        out = ttk.Frame(self, padding=(12, 0, 12, 12))
        out.pack(fill=tk.BOTH, expand=True)

        self.text = tk.Text(out, wrap="word", height=12)
        self.text.pack(fill=tk.BOTH, expand=True)

        self._set_output(
            "Choisis un fichier puis clique sur 'Analyser'.\n"
            "Le résultat affiche:\n"
            "- Coordonnées déduites du nom (XXXX_YYYY)\n"
            "- Coordonnées trouvées via metadata (PDAL pour LAZ/LAS, GDAL/rasterio pour TIF/ASC, world file pour JPG)\n"
        )

    def _set_output(self, s: str) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, s)
        self.text.configure(state="disabled")

    def _browse(self) -> None:
        p = filedialog.askopenfilename(
            title="Sélectionner un fichier",
            filetypes=[
                ("Nuages points", "*.laz *.las *.copc.laz"),
                ("Rasters", "*.tif *.tiff *.asc"),
                ("Images", "*.jpg *.jpeg"),
                ("Tous", "*.*"),
            ],
        )
        if p:
            self.path_var.set(p)

    def _analyze(self) -> None:
        raw = (self.path_var.get() or "").strip()
        if not raw:
            self._set_output("Aucun fichier sélectionné")
            return

        path = Path(raw)
        if not path.exists() or not path.is_file():
            self._set_output(f"Fichier introuvable: {path}")
            return

        name_coords = extract_xy_from_filename(path.name)
        meta_coords = infer_xy_from_file(path)

        lines = []
        lines.append(f"Fichier: {path}")
        lines.append(f"Extension: {path.suffix.lower()}")
        lines.append("")

        lines.append("Coordonnées déduites du nom:")
        if name_coords is None:
            lines.append("  (non trouvé)")
        else:
            lines.append(f"  {name_coords.x_km:04d}_{name_coords.y_km:04d}")

        lines.append("")
        lines.append("Coordonnées trouvées via metadata:")
        if meta_coords is None:
            lines.append("  (non trouvé)")
        else:
            lines.append(f"  {meta_coords.x_km:04d}_{meta_coords.y_km:04d}")

        lines.append("")
        lines.append("Comparaison:")
        if name_coords is None or meta_coords is None:
            lines.append("  (comparaison impossible)")
        else:
            ok = (name_coords.x_km == meta_coords.x_km) and (name_coords.y_km == meta_coords.y_km)
            lines.append("  OK" if ok else "  MISMATCH")

        self._set_output("\n".join(lines))


if __name__ == "__main__":
    app = CoordsAnalyzerApp()
    app.mainloop()
