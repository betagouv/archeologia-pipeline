import pytest

from app.structured_logger import StructuredLogger, create_structured_logger


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
