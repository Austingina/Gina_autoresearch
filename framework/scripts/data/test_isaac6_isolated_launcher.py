"""Focused CPU-only contracts for the Isaac Sim 6.0.1 launcher."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER_PATH = PROJECT_ROOT / "scripts" / "run_gina_isaac6.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("run_gina_isaac6", LAUNCHER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {LAUNCHER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Isaac6IsolatedLauncherTest(unittest.TestCase):
    def test_build_environment_isolates_vendored_sources_from_legacy_runtime(self):
        launcher = _load_launcher()
        dirty = {
            "PATH": "/usr/bin",
            "PYTHONPATH": "/legacy/isaacsim/5.1:/legacy/IsaacLab",
            "PYTHONHOME": "/legacy/python",
            "CONDA_PREFIX": "/legacy/env_gina_isaac51",
            "CONDA_DEFAULT_ENV": "env_gina_isaac51",
            "VIRTUAL_ENV": "/legacy/venv",
            "LD_LIBRARY_PATH": "/keep/system/lib",
        }

        env = launcher.build_launch_environment(dirty)

        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertEqual(env["PYTHONEXE"], str(launcher.DEPENDENCY_PYTHON))
        self.assertNotIn("PYTHONHOME", env)
        self.assertNotIn("CONDA_PREFIX", env)
        self.assertNotIn("CONDA_DEFAULT_ENV", env)
        self.assertNotIn("VIRTUAL_ENV", env)
        source_paths = env["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(source_paths, [str(path) for path in launcher.VENDORED_SOURCE_PATHS])
        self.assertTrue(all(str(launcher.ISAACLAB_ROOT) in path for path in source_paths))
        self.assertNotIn("/env_gina_isaac51/", env["PYTHONPATH"])
        self.assertNotIn("/AST", env["PYTHONPATH"])
        dependency_lib = "/home/ai-sz-a26394-u1/miniforge3/envs/env_isaaclab/lib"
        self.assertEqual(env["LD_LIBRARY_PATH"].split(os.pathsep)[0], dependency_lib)
        self.assertNotIn("env_gina_isaac51", env["LD_LIBRARY_PATH"])

    def test_cpu_preflight_proves_runtime_api_and_version_contract_without_importing_kit(self):
        launcher = _load_launcher()

        receipt = launcher.cpu_preflight()

        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["isaac_sim_version"], "6.0.1-rc.7+release.42383.32955d8d.gl")
        self.assertEqual(receipt["isaac_lab_release"], "2.3.2")
        self.assertEqual(receipt["isaac_lab_core_version"], "0.54.4")
        self.assertEqual(receipt["dependency_python_version"][:2], [3, 12])
        self.assertEqual(receipt["runtime_python_version"][:2], [3, 12])
        self.assertTrue(receipt["api_contracts"]["omni.kit.commands.register_callback"])
        self.assertTrue(receipt["api_contracts"]["omni.kit.commands.unregister_module_commands"])
        self.assertTrue(receipt["api_contracts"]["omni.usd.get_context"])
        self.assertFalse(receipt["kit_started"])

    def test_isaac6_experience_wrappers_override_version_without_changing_legacy_apps(self):
        app_root = PROJECT_ROOT / "simulators" / "isaaclab" / "apps"
        cases = {
            "gina.isaaclab.python.headless.6_0.kit": "isaaclab.python.headless",
            "gina.isaaclab.python.headless.rendering.6_0.kit": "isaaclab.python.headless.rendering",
        }
        for filename, forbidden_dependency in cases.items():
            with self.subTest(filename=filename):
                text = (app_root / "isaacsim_6_0" / filename).read_text(encoding="utf-8")
                self.assertIn('app.version = "6.0.1"', text)
                # A .kit experience is not a registry extension. Depending on the
                # legacy experience name makes Kit's live dependency solver fail.
                self.assertNotIn(f'"{forbidden_dependency}" = {{}}', text)
                self.assertIn('"isaacsim.simulation_app" = {}', text)
                self.assertIn('"isaaclab" = {order = 1000}', text)
                self.assertIn('"${app}/../../source"', text)
                self.assertIn("kit/prod/110/shared", text)
                if "rendering" in filename:
                    self.assertIn('"omni.replicator.core" = {}', text)
                    self.assertIn("cameras_enabled = true", text)

        for legacy_name in ("isaaclab.python.headless.kit", "isaaclab.python.headless.rendering.kit"):
            legacy_text = (app_root / legacy_name).read_text(encoding="utf-8")
            self.assertIn('app.version = "5.1.0"', legacy_text)

    def test_physics_tensor_import_uses_version_compatible_module(self):
        source_root = PROJECT_ROOT / "simulators" / "isaaclab" / "source" / "isaaclab" / "isaaclab"
        sources = [path.read_text(encoding="utf-8") for path in source_root.rglob("*.py")]
        direct_legacy = "import omni.physics.tensors.impl.api as physx"
        direct_top_level = "import omni.physics.tensors as physx"
        compat_import = "from isaaclab.utils.physics_tensors import physx"
        helper = (source_root / "utils" / "physics_tensors.py").read_text(encoding="utf-8")
        self.assertIn("import omni.physics.tensors.api as physx", helper)
        self.assertIn("import omni.physics.tensors.impl.api as physx", helper)
        non_helper = [text for text in sources if text != helper]
        self.assertFalse(any(direct_legacy in text for text in non_helper))
        self.assertFalse(any(direct_top_level in text for text in non_helper))
        compat_sources = [text for text in non_helper if compat_import in text]
        self.assertEqual(len(compat_sources), 12)
        self.assertTrue(all("from __future__ import annotations" in text for text in compat_sources))

    def test_missing_deformable_schema_is_fail_closed_for_material_binding(self):
        prims = (
            PROJECT_ROOT
            / "simulators"
            / "isaaclab"
            / "source"
            / "isaaclab"
            / "isaaclab"
            / "sim"
            / "utils"
            / "prims.py"
        ).read_text(encoding="utf-8")
        self.assertIn('getattr(PhysxSchema, "PhysxDeformableBodyAPI", None)', prims)
        self.assertNotIn("prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI)", prims)

    def test_headless_offscreen_render_tolerates_missing_viewport(self):
        simulation_context = (
            PROJECT_ROOT
            / "simulators"
            / "isaaclab"
            / "source"
            / "isaaclab"
            / "isaaclab"
            / "sim"
            / "simulation_context.py"
        ).read_text(encoding="utf-8")
        self.assertIn("active_viewport = get_active_viewport()", simulation_context)
        self.assertIn("if active_viewport is not None:", simulation_context)
        self.assertNotIn("get_active_viewport().updates_enabled = False", simulation_context)

    def test_shader_command_uses_version_compatible_name_keyword(self):
        visual_materials = (
            PROJECT_ROOT
            / "simulators"
            / "isaaclab"
            / "source"
            / "isaaclab"
            / "isaaclab"
            / "sim"
            / "spawners"
            / "materials"
            / "visual_materials.py"
        ).read_text(encoding="utf-8")
        self.assertIn('shader_name_key = "prim_name" if get_isaac_sim_version().major >= 6 else "name"', visual_materials)
        self.assertIn('**{shader_name_key: "Shader"}', visual_materials)

    def test_camera_attaches_rtx_hydra_engine_for_isaac_sim_6(self):
        camera = (
            PROJECT_ROOT
            / "simulators"
            / "isaaclab"
            / "source"
            / "isaaclab"
            / "isaaclab"
            / "sensors"
            / "camera"
            / "camera.py"
        ).read_text(encoding="utf-8")
        self.assertIn("get_isaac_sim_version().major >= 6", camera)
        self.assertIn('omni.usd.create_hydra_engine("rtx", usd_context)', camera)


if __name__ == "__main__":
    unittest.main()
