"""NET-06 (audit v2) : le re-téléchargement d'une dalle VOISINE pendant la
fusion appelait download_one SANS l'argument proxies → requests retombait sur
les variables d'environnement et CONTOURNAIT le proxy QGIS (cas nominal en
réseau d'entreprise, objet du fix proxy récent). Résultat : voisin manquant →
fusion sans marge de ce côté → artefacts de bord.

Le proxy est résolu UNE fois dans prepare_merged_tiles et transporté par la
tâche jusqu'à download_one.
"""
from __future__ import annotations

import pipeline.ign.preprocess as pp

SENTINEL_PROXIES = {"https": "http://proxy.entreprise:3128"}


def test_prepare_merged_tiles_transporte_le_proxy_dans_les_taches(
    tmp_path, monkeypatch
):
    dalles = tmp_path / "dalles"
    dalles.mkdir()
    (dalles / "LHD_FXX_0500_6500_PTS_C.laz").write_bytes(b"x")
    sorted_list = tmp_path / "fichier_tri.txt"
    sorted_list.write_text(
        "LHD_FXX_0500_6500_PTS_C.laz,https://x/a.laz\n", encoding="utf-8"
    )

    monkeypatch.setattr(
        pp, "_get_proxy_config", lambda **k: SENTINEL_PROXIES, raising=False
    )
    monkeypatch.setattr(
        pp, "_extract_coordinates", lambda filename, **kw: ("0500", "6500")
    )

    seen: list = []

    def fake_process(task, file_index, log, cancel):
        seen.append(getattr(task, "proxies", None))
        return pp._PreprocessResult(
            index=task.index, tile_name=task.tile_name,
            merged_path=dalles / "m.laz", success=True,
        )

    monkeypatch.setattr(pp, "_process_single_tile_preprocess", fake_process)

    pp.prepare_merged_tiles(
        sorted_list_file=sorted_list, dalles_dir=dalles,
        output_dir=tmp_path / "out", tile_overlap_percent=20,
    )

    assert seen == [SENTINEL_PROXIES]


def test_download_one_du_voisin_recoit_le_proxy(tmp_path, monkeypatch):
    dalles = tmp_path / "dalles"
    dalles.mkdir()
    central = dalles / "LHD_FXX_0500_6500_PTS_C.laz"
    central.write_bytes(b"x")

    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()

    task = pp._PreprocessTask(
        index=1, total=1, filename=central.name, url="https://x/a.laz",
        tile_name="LHD_FXX_0500_6500", x="0500", y="6500",
        central_path=central, temp_dir=temp_dir, merged_dir=temp_dir,
        margin_m=200, dalles_dir=dalles, proxies=SENTINEL_PROXIES,
    )

    # Un voisin référencé dans l'index mais ABSENT du disque → download_one.
    vx, vy, _place = pp.calculate_neighbor_coordinates(task.x, task.y)[0]
    coord_key = f"{pp.format_coordinate(vx)}_{pp.format_coordinate(vy)}"
    file_index = {coord_key: ("voisin.laz", "https://x/voisin.laz")}

    captured: list = []

    def fake_download_one(url, filename, dalles_dir, **kwargs):
        captured.append(kwargs.get("proxies"))
        return False, False  # échec → voisin ignoré, on continue

    monkeypatch.setattr(pp, "download_one", fake_download_one)
    monkeypatch.setattr(pp, "merge_tiles", lambda **kw: True)

    res = pp._process_single_tile_preprocess(
        task, file_index, lambda m: None, lambda: False
    )

    assert res.success is True
    assert captured == [SENTINEL_PROXIES]
