import pytest

from app.structured_logger import StructuredLogger, create_structured_logger
from pipeline.types import format_params_line


class TestStructuredLogger:
    def test_create_structured_logger(self):
        logs = []
        slog = create_structured_logger(logs.append)
        assert isinstance(slog, StructuredLogger)

    def test_start_pipeline_logs_header(self):
        logs = []
        slog = StructuredLogger(logs.append)
        slog.start_pipeline("ign_laz", "/output/dir")
        
        assert any("DÉMARRAGE DU PIPELINE" in log for log in logs)
        assert any("ign_laz" in log for log in logs)
        assert any("/output/dir" in log for log in logs)

    def test_end_pipeline_success(self):
        logs = []
        slog = StructuredLogger(logs.append)
        slog._start_time = 0
        slog.end_pipeline(success=True, tiles_processed=2, tiles_total=2, products=["MNT", "LD"])
        
        assert any("TERMINÉ AVEC SUCCÈS" in log for log in logs)
        assert any("2/2" in log for log in logs)
        assert any("MNT" in log for log in logs)

    def test_end_pipeline_failure(self):
        logs = []
        slog = StructuredLogger(logs.append)
        slog._start_time = 0
        slog.end_pipeline(success=False)
        
        assert any("TERMINÉ AVEC ERREURS" in log for log in logs)

    def test_section_logs_title(self):
        logs = []
        slog = StructuredLogger(logs.append)
        slog.section("Test Section", "download")
        
        assert any("TEST SECTION" in log for log in logs)

    def test_subsection_logs_title(self):
        logs = []
        slog = StructuredLogger(logs.append)
        slog.subsection("Sub Section")
        
        assert any("Sub Section" in log for log in logs)

    def test_item_with_icon(self):
        logs = []
        slog = StructuredLogger(logs.append)
        slog.item("Test message", "success")
        
        assert any("Test message" in log for log in logs)
        assert any("✅" in log for log in logs)

    def test_success_method(self):
        logs = []
        slog = StructuredLogger(logs.append)
        slog.success("Operation completed")
        
        assert any("✅" in log for log in logs)
        assert any("Operation completed" in log for log in logs)

    def test_error_method(self):
        logs = []
        slog = StructuredLogger(logs.append)
        slog.error("Something failed")
        
        assert any("❌" in log for log in logs)
        assert any("Something failed" in log for log in logs)

    def test_warning_method(self):
        logs = []
        slog = StructuredLogger(logs.append)
        slog.warning("Be careful")
        
        assert any("⚠️" in log for log in logs)
        assert any("Be careful" in log for log in logs)

    def test_info_method(self):
        logs = []
        slog = StructuredLogger(logs.append)
        slog.info("Information message")
        
        assert any("Information message" in log for log in logs)

    def test_progress_bar(self):
        logs = []
        slog = StructuredLogger(logs.append)
        slog.progress(5, 10, "item_name")
        
        assert any("5/10" in log for log in logs)
        assert any("50%" in log for log in logs)
        assert any("item_name" in log for log in logs)

    def test_tile_start(self):
        logs = []
        slog = StructuredLogger(logs.append)
        slog.tile_start(1, 5, "TILE_001")
        
        assert any("DALLE 1/5" in log for log in logs)
        assert any("TILE_001" in log for log in logs)

    def test_tile_end(self):
        logs = []
        slog = StructuredLogger(logs.append)
        slog._section_start = 0
        slog.tile_end("TILE_001", ["MNT", "LD"])
        
        assert any("Terminé" in log for log in logs)
        assert any("MNT" in log for log in logs)

    def test_products_list(self):
        logs = []
        slog = StructuredLogger(logs.append)
        slog.products_list(["MNT", "DENSITE", "LD"], "Generated")
        
        assert any("Generated" in log for log in logs)
        assert any("MNT" in log for log in logs)
        assert any("DENSITE" in log for log in logs)

    def test_preflight_result_ok(self):
        logs = []
        slog = StructuredLogger(logs.append)
        slog.preflight_result("pdal", "OK", "found in PATH")
        
        assert any("✅" in log for log in logs)
        assert any("pdal" in log for log in logs)

    def test_preflight_result_error(self):
        logs = []
        slog = StructuredLogger(logs.append)
        slog.preflight_result("pdal", "ERROR", "not found")
        
        assert any("❌" in log for log in logs)

    def test_format_duration_seconds(self):
        logs = []
        slog = StructuredLogger(logs.append)
        import time
        slog._start_time = time.time() - 30
        result = slog._format_duration(slog._start_time)
        assert "s" in result

    def test_format_duration_minutes(self):
        logs = []
        slog = StructuredLogger(logs.append)
        import time
        slog._start_time = time.time() - 125
        result = slog._format_duration(slog._start_time)
        assert "min" in result

    def test_format_duration_none(self):
        logs = []
        slog = StructuredLogger(logs.append)
        result = slog._format_duration(None)
        assert result == "N/A"

    def test_params_emits_compact_line(self):
        logs = []
        slog = StructuredLogger(logs.append)
        slog.params("MNT", {"resolution_m": 0.5, "filter": "Class = 2"})
        # Une seule ligne de log émise.
        assert len(logs) == 1
        assert logs[0].startswith("[PARAMS MNT] ")
        assert "resolution_m=0.5" in logs[0]
        assert 'filter="Class = 2"' in logs[0]

    def test_params_handles_empty_dict(self):
        logs = []
        slog = StructuredLogger(logs.append)
        slog.params("STEP", {})
        assert logs[0] == "[PARAMS STEP] (aucun paramètre)"


# ----------------------------------------------------------------------
# format_params_line (helper utilisé par pipeline/ sans dépendance app)
# ----------------------------------------------------------------------
class TestFormatParamsLine:
    def test_int_and_float_compact(self):
        line = format_params_line("MNT", {"resolution_m": 0.5, "workers": 4})
        assert "resolution_m=0.5" in line
        assert "workers=4" in line

    def test_string_no_space_unquoted(self):
        line = format_params_line("STEP", {"name": "M_HS"})
        assert "name=M_HS" in line

    def test_string_with_space_quoted(self):
        line = format_params_line("MNT", {"filter": "Class = 2 OR Class = 6"})
        assert 'filter="Class = 2 OR Class = 6"' in line

    def test_bool_formatted_as_python(self):
        line = format_params_line("X", {"flag": True, "save": False})
        assert "flag=True" in line
        assert "save=False" in line

    def test_none_shows_empty_marker(self):
        line = format_params_line("X", {"opt": None})
        assert "opt=∅" in line

    def test_empty_string_shows_empty_marker(self):
        line = format_params_line("X", {"name": ""})
        assert "name=∅" in line

    def test_separator_is_pipe(self):
        line = format_params_line("X", {"a": 1, "b": 2})
        assert "a=1 | b=2" in line

    def test_step_name_in_brackets(self):
        line = format_params_line("RVT/SVF", {})
        assert line.startswith("[PARAMS RVT/SVF]")

    def test_float_format_compact(self):
        # %g évite "0.50000"
        line = format_params_line("X", {"v": 0.500})
        assert "v=0.5" in line

    def test_preserves_insertion_order(self):
        line = format_params_line("X", {"first": 1, "second": 2, "third": 3})
        assert line.index("first") < line.index("second") < line.index("third")
