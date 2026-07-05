from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


HAS_STREAMLIT = importlib.util.find_spec("streamlit") is not None
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAS_ARTIFACTS = (PROJECT_ROOT / "demo" / "artifacts" / "baseline" / "manifest.json").exists()


@unittest.skipUnless(HAS_STREAMLIT and HAS_ARTIFACTS, "Streamlit or baseline artifacts unavailable")
class DashboardTests(unittest.TestCase):
    def test_initial_render_and_controls(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_ROOT / "demo" / "app.py"), default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.selectbox[0].value, "BASELINE")
        self.assertEqual(app.selectbox[1].value, "machine-1-1")
        self.assertEqual([button.label for button in app.button], ["▶ Chạy", "⏸ Dừng", "↺ Reset"])
        self.assertEqual(len(app.metric), 4)

    def test_machine_kpi_evaluation_and_reset(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_ROOT / "demo" / "app.py"), default_timeout=60).run()
        app.selectbox[1].set_value("machine-3-10").run()
        app.selectbox[2].set_value("KPI_38").run()
        app.toggle[0].set_value(True).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.selectbox[1].value, "machine-3-10")
        self.assertTrue(app.toggle[0].value)
        initial = app.slider[0].value
        app.slider[0].set_value(initial + 300).run()
        self.assertEqual(app.slider[0].value, initial + 300)
        app.button[2].click().run()
        self.assertEqual(app.slider[0].value, initial)


if __name__ == "__main__":
    unittest.main()
