"""
在仿真器中加载场景后，输出 /World 下所有 prim 路径。
用法：
  python scripts/list_scene_prims.py
  python scripts/list_scene_prims.py --usd-path assets/chemistry_lab/lab_003/lab_003.usd
"""
import os
import argparse

_script_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(_script_dir)

parser = argparse.ArgumentParser()
parser.add_argument("--usd-path", default="assets/chemistry_lab/lab_001/lab_001.usd",
                    help="USD 场景路径，默认 lab_001")
args = parser.parse_args()

# 必须最先启动 SimulationApp
from isaacsim import SimulationApp
simulation_config = {"headless": True}
simulation_app = SimulationApp(simulation_config)

import omni
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils import extensions

extensions.enable_extension("omni.physx.bundle")

def main():
    os.chdir(_proj_root)
    usd_path = os.path.join(_proj_root, args.usd_path) if not os.path.isabs(args.usd_path) else args.usd_path
    usd_path = os.path.abspath(usd_path)
    scene_name = args.usd_path

    world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="numpy")
    stage = omni.usd.get_context().get_stage()
    add_reference_to_stage(usd_path=usd_path, prim_path="/World")

    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim.IsValid():
        print("/World 无效")
        return

    paths = []
    def collect(p, depth=0):
        paths.append((p.GetPath().pathString, depth))
        for c in p.GetChildren():
            collect(c, depth + 1)
    collect(world_prim)

    print(f"\n=== 场景 {scene_name} 中 /World 下所有 prim（共 {len(paths)} 个）===\n")
    for path, depth in sorted(paths, key=lambda x: x[0]):
        indent = "  " * depth
        print(f"{indent}{path}")

    print("\n=== Pick 配置常用 prim 存在性 ===")
    for p in ["/World/conical_bottle02", "/World/conical_bottle03", "/World/conical_bottle04",
              "/World/beaker2", "/World/beaker1", "/World/beaker3",
              "/World/graduated_cylinder_03", "/World/volume_flask"]:
        prim = stage.GetPrimAtPath(p)
        print(f"  {p}: {'✓' if prim.IsValid() else '✗ 不存在'}")

    simulation_app.close()

if __name__ == "__main__":
    main()
