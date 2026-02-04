"""
Test de détection de fichiers LAZ corrompus vs valides.

Fichiers de test:
- Corrompu: D:\Rambouillet_Archeo\output_dalles\dalles\LHD_FXX_0615_6841_PTS_C_LAMB93_IGN69.copc.laz
- Valide: D:\Rambouillet_Archeo\output_dalles\dalles\LHD_FXX_0615_6839_PTS_C_LAMB93_IGN69.copc.laz
"""

import json
import os
import subprocess
from pathlib import Path
from shutil import which


CORRUPTED_FILE = Path(r"D:\Rambouillet_Archeo\output_dalles\dalles\LHD_FXX_0615_6841_PTS_C_LAMB93_IGN69.copc.laz")
VALID_FILE = Path(r"D:\Rambouillet_Archeo\output_dalles\dalles\LHD_FXX_0615_6839_PTS_C_LAMB93_IGN69.copc.laz")


def get_file_info(path: Path) -> dict:
    """Récupère les informations de base sur le fichier."""
    if not path.exists():
        return {"exists": False, "path": str(path)}
    
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
    }


def run_pdal_info_metadata(path: Path) -> dict:
    """Exécute pdal info --metadata et retourne le résultat."""
    pdal = which("pdal")
    if not pdal:
        return {"error": "pdal not found"}
    
    cmd = [pdal, "info", "--metadata", str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return {
            "returncode": r.returncode,
            "stdout": r.stdout[:5000] if r.stdout else "",
            "stderr": r.stderr[:2000] if r.stderr else "",
        }
    except Exception as e:
        return {"error": str(e)}


def run_pdal_info_all(path: Path) -> dict:
    """Exécute pdal info --all (plus complet mais plus lent)."""
    pdal = which("pdal")
    if not pdal:
        return {"error": "pdal not found"}
    
    cmd = [pdal, "info", "--all", str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return {
            "returncode": r.returncode,
            "stdout_length": len(r.stdout) if r.stdout else 0,
            "stderr": r.stderr[:2000] if r.stderr else "",
        }
    except Exception as e:
        return {"error": str(e)}


def run_pdal_info_summary(path: Path) -> dict:
    """Exécute pdal info --summary pour avoir un résumé."""
    pdal = which("pdal")
    if not pdal:
        return {"error": "pdal not found"}
    
    cmd = [pdal, "info", "--summary", str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return {
            "returncode": r.returncode,
            "stdout": r.stdout[:5000] if r.stdout else "",
            "stderr": r.stderr[:2000] if r.stderr else "",
        }
    except Exception as e:
        return {"error": str(e)}


def run_pdal_info_stats(path: Path) -> dict:
    """Exécute pdal info --stats pour les statistiques."""
    pdal = which("pdal")
    if not pdal:
        return {"error": "pdal not found"}
    
    cmd = [pdal, "info", "--stats", str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return {
            "returncode": r.returncode,
            "stdout": r.stdout[:5000] if r.stdout else "",
            "stderr": r.stderr[:2000] if r.stderr else "",
        }
    except Exception as e:
        return {"error": str(e)}


def check_laz_header(path: Path) -> dict:
    """Vérifie le header LAZ/LAS manuellement."""
    if not path.exists():
        return {"error": "file not found"}
    
    try:
        with open(path, "rb") as f:
            # Lire le header LAS (au moins 227 bytes pour LAS 1.2)
            header = f.read(375)
            
            if len(header) < 100:
                return {"error": "file too small", "size": len(header)}
            
            # Signature LAS: "LASF"
            signature = header[0:4]
            
            # Version
            version_major = header[24]
            version_minor = header[25]
            
            # Point count (offset 107 pour LAS 1.2, différent pour 1.4)
            # Pour LAS 1.4, le point count est à l'offset 247 (8 bytes)
            if version_major == 1 and version_minor >= 4:
                point_count = int.from_bytes(header[247:255], "little")
            else:
                point_count = int.from_bytes(header[107:111], "little")
            
            # Offset to point data
            offset_to_point_data = int.from_bytes(header[96:100], "little")
            
            # Point data format
            point_format = header[104]
            
            # Point record length
            point_record_length = int.from_bytes(header[105:107], "little")
            
            return {
                "signature": signature.decode("ascii", errors="replace"),
                "version": f"{version_major}.{version_minor}",
                "point_count": point_count,
                "offset_to_point_data": offset_to_point_data,
                "point_format": point_format,
                "point_record_length": point_record_length,
                "is_valid_signature": signature == b"LASF",
            }
    except Exception as e:
        return {"error": str(e)}


def parse_metadata_json(stdout: str) -> dict:
    """Parse le JSON de pdal info --metadata."""
    try:
        data = json.loads(stdout)
        metadata = data.get("metadata", {})
        return {
            "count": metadata.get("count"),
            "minx": metadata.get("minx"),
            "miny": metadata.get("miny"),
            "maxx": metadata.get("maxx"),
            "maxy": metadata.get("maxy"),
            "minz": metadata.get("minz"),
            "maxz": metadata.get("maxz"),
            "compressed": metadata.get("compressed"),
            "copc": metadata.get("copc"),
        }
    except Exception as e:
        return {"parse_error": str(e)}


def analyze_file(path: Path, label: str) -> None:
    """Analyse complète d'un fichier LAZ."""
    print(f"\n{'='*80}")
    print(f"ANALYSE: {label}")
    print(f"{'='*80}")
    
    # Info fichier
    file_info = get_file_info(path)
    print(f"\n--- Informations fichier ---")
    for k, v in file_info.items():
        print(f"  {k}: {v}")
    
    if not file_info.get("exists"):
        print("  FICHIER NON TROUVÉ!")
        return
    
    # Header LAZ manuel
    print(f"\n--- Header LAZ (lecture manuelle) ---")
    header_info = check_laz_header(path)
    for k, v in header_info.items():
        print(f"  {k}: {v}")
    
    # PDAL info --metadata
    print(f"\n--- PDAL info --metadata ---")
    metadata_result = run_pdal_info_metadata(path)
    print(f"  returncode: {metadata_result.get('returncode')}")
    if metadata_result.get("stderr"):
        print(f"  stderr: {metadata_result['stderr']}")
    
    if metadata_result.get("returncode") == 0 and metadata_result.get("stdout"):
        parsed = parse_metadata_json(metadata_result["stdout"])
        print(f"  Metadata parsée:")
        for k, v in parsed.items():
            print(f"    {k}: {v}")
    
    # PDAL info --summary
    print(f"\n--- PDAL info --summary ---")
    summary_result = run_pdal_info_summary(path)
    print(f"  returncode: {summary_result.get('returncode')}")
    if summary_result.get("stderr"):
        print(f"  stderr: {summary_result['stderr']}")
    if summary_result.get("stdout"):
        # Afficher les premières lignes
        lines = summary_result["stdout"].strip().split("\n")[:20]
        for line in lines:
            print(f"  {line}")
    
    # PDAL info --stats (peut être lent)
    print(f"\n--- PDAL info --stats ---")
    stats_result = run_pdal_info_stats(path)
    print(f"  returncode: {stats_result.get('returncode')}")
    if stats_result.get("stderr"):
        print(f"  stderr: {stats_result['stderr']}")


def run_pdal_pipeline_count(path: Path) -> dict:
    """
    Essaie de lire tous les points via un pipeline PDAL.
    C'est le test le plus fiable pour détecter une corruption.
    """
    pdal = which("pdal")
    if not pdal:
        return {"error": "pdal not found"}
    
    # Pipeline qui lit tous les points et compte
    import tempfile
    pipeline_json = json.dumps({
        "pipeline": [
            str(path),
            {
                "type": "filters.stats"
            }
        ]
    })
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(pipeline_json)
        pipeline_file = f.name
    
    try:
        cmd = [pdal, "pipeline", pipeline_file, "--metadata", "stdout"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return {
            "returncode": r.returncode,
            "stderr": r.stderr[:3000] if r.stderr else "",
            "stdout_preview": r.stdout[:2000] if r.stdout else "",
        }
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            os.unlink(pipeline_file)
        except:
            pass


def run_pdal_translate_null(path: Path) -> dict:
    """
    Essaie de lire le fichier complet via pdal translate vers /dev/null.
    Détecte les fichiers tronqués.
    """
    pdal = which("pdal")
    if not pdal:
        return {"error": "pdal not found"}
    
    # Sur Windows, utiliser NUL comme destination
    null_dest = "NUL" if os.name == "nt" else "/dev/null"
    
    cmd = [pdal, "translate", str(path), null_dest]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return {
            "returncode": r.returncode,
            "stderr": r.stderr[:3000] if r.stderr else "",
            "stdout": r.stdout[:1000] if r.stdout else "",
        }
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}


def test_read_all_points(path: Path, label: str) -> None:
    """Test de lecture complète des points."""
    print(f"\n{'='*80}")
    print(f"TEST LECTURE COMPLÈTE: {label}")
    print(f"{'='*80}")
    
    file_info = get_file_info(path)
    print(f"Fichier: {path.name}")
    print(f"Taille: {file_info.get('size_mb', '?')} MB")
    
    # Test 1: pdal translate vers null (force lecture complète)
    print(f"\n--- Test pdal translate (lecture complète) ---")
    translate_result = run_pdal_translate_null(path)
    print(f"  returncode: {translate_result.get('returncode')}")
    if translate_result.get("stderr"):
        print(f"  stderr: {translate_result['stderr']}")
    if translate_result.get("error"):
        print(f"  error: {translate_result['error']}")
    
    # Test 2: pipeline avec stats
    print(f"\n--- Test pipeline stats (lecture complète) ---")
    pipeline_result = run_pdal_pipeline_count(path)
    print(f"  returncode: {pipeline_result.get('returncode')}")
    if pipeline_result.get("stderr"):
        print(f"  stderr: {pipeline_result['stderr']}")
    if pipeline_result.get("error"):
        print(f"  error: {pipeline_result['error']}")


def run_pdal_info_pointcount(path: Path) -> dict:
    """
    Utilise pdal info --pointcount pour forcer le comptage réel des points.
    C'est différent de --metadata qui lit juste le header.
    """
    pdal = which("pdal")
    if not pdal:
        return {"error": "pdal not found"}
    
    cmd = [pdal, "info", "--pointcount", str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return {
            "returncode": r.returncode,
            "stdout": r.stdout[:5000] if r.stdout else "",
            "stderr": r.stderr[:3000] if r.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}


def run_pdal_pipeline_stats(path: Path) -> dict:
    """
    Pipeline PDAL avec filters.stats pour forcer la lecture de TOUS les points.
    """
    import tempfile
    pdal = which("pdal")
    if not pdal:
        return {"error": "pdal not found"}
    
    pipeline = {
        "pipeline": [
            {
                "type": "readers.copc",
                "filename": str(path)
            },
            {
                "type": "filters.stats"
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(pipeline, f)
        pipeline_file = f.name
    
    try:
        cmd = [pdal, "pipeline", pipeline_file]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return {
            "returncode": r.returncode,
            "stdout": r.stdout[:2000] if r.stdout else "",
            "stderr": r.stderr[:3000] if r.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            os.unlink(pipeline_file)
        except:
            pass


def run_pdal_info_all(path: Path) -> dict:
    """pdal info --all force la lecture complète."""
    pdal = which("pdal")
    if not pdal:
        return {"error": "pdal not found"}
    
    cmd = [pdal, "info", "--all", str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return {
            "returncode": r.returncode,
            "stdout_len": len(r.stdout) if r.stdout else 0,
            "stderr": r.stderr[:3000] if r.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}


def test_validate_laz_deep():
    """Test de la fonction validate_laz_deep du pipeline."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from pipeline.ign.pdal_validation import validate_laz_deep, validate_las_or_laz_with_pdal
    
    print("="*80)
    print("TEST DE validate_laz_deep()")
    print("="*80)
    
    for path, label, expected_valid in [
        (CORRUPTED_FILE, "CORROMPU", False),
        (VALID_FILE, "VALIDE", True),
    ]:
        print(f"\n--- {label}: {path.name} ---")
        
        # Validation rapide (--metadata)
        ok_fast, msg_fast = validate_las_or_laz_with_pdal(path, use_cache=False)
        print(f"validate_las_or_laz_with_pdal: ok={ok_fast}, msg={msg_fast}")
        
        # Validation profonde (--all)
        ok_deep, msg_deep = validate_laz_deep(path)
        print(f"validate_laz_deep: ok={ok_deep}, msg={msg_deep}")
        
        # Vérification
        if ok_deep == expected_valid:
            print(f"✅ CORRECT: fichier {label} détecté comme {'valide' if ok_deep else 'invalide'}")
        else:
            print(f"❌ ERREUR: fichier {label} devrait être {'valide' if expected_valid else 'invalide'}")
    
    print("\n" + "="*80)


def main():
    test_validate_laz_deep()


if __name__ == "__main__":
    main()
