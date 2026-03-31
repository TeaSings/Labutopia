from typing import Optional

from pxr import Usd, UsdGeom, Gf, UsdPhysics
from isaacsim.core.utils.stage import get_stage_units
import numpy as np
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
class ObjectUtils:
    _instance = None

    @classmethod
    def get_instance(cls, stage: Usd.Stage = None, default_path: str = "/World") -> "ObjectUtils":
        if cls._instance is None:
            if stage is None:
                raise ValueError("Stage must be provided for first instance")
            cls._instance = cls(stage, default_path)
        return cls._instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ObjectUtils, cls).__new__(cls)
        return cls._instance

    def __init__(self, stage: Usd.Stage, default_path: str = "/World"):
        if not hasattr(self, '_initialized'):
            self._stage = stage
            self._default_path = default_path
            self._pick_height_offsets = {
                "rod": 0.04,
                "tube": 0.01,
                "beaker": 0.0,
                "erlenmeyer flask": 0.018,
                "cylinder": 0.0,
                "petri dish": 0.005,
                "pipette": 0.008,
                "microscope slide": 0.002
            }
            self._initialized = True

    def _get_object_path(self, object_name: str = None, object_path: str = None) -> str:
        if object_path:
            return object_path
        if object_name:
            return f"{self._default_path}/{object_name}"
        raise ValueError("Either object_name or object_path must be provided")

    def get_pick_position(self, object_name: str = None, object_path: str = None) -> np.ndarray:
        """Get the object's pick position with height offset."""
        position = self.get_geometry_center(object_name, object_path)
        if position is None:
            return None

        name = object_name or object_path.split('/')[-1]
        for key, offset in self._pick_height_offsets.items():
            if key in name.lower():
                position[2] += offset / get_stage_units()
                return position

        position[2] += 0.02 / get_stage_units()
        return position

    def get_object_size(self, object_name: str = None, object_path: str = None) -> np.ndarray:
        """Get the world-space size of an object."""
        path = self._get_object_path(object_name, object_path)
        prim = self._stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return None

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(prim)
        min_point = bbox.GetRange().GetMin()
        max_point = bbox.GetRange().GetMax()
        world_min_point = np.array([min_point[0], min_point[1], min_point[2], 1.0])
        world_max_point = np.array([max_point[0], max_point[1], max_point[2], 1.0])
        transform_matrix = bbox.GetMatrix()
        transform_matrix_np = np.array(transform_matrix)
        world_size = world_max_point @ transform_matrix_np - world_min_point @ transform_matrix_np
        return world_size[:3]

    def get_object_xform_position(self, object_path: str) -> np.ndarray:
        """Get the world-space position from the object's transform."""
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            print(f"Object at path {object_path} not found.")
            return None

        xformable = UsdGeom.Xformable(prim)
        transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        position = transform.ExtractTranslation()
        return np.array(position)

    def get_world_rotation_matrix_3x3(self, object_path: str) -> Optional[np.ndarray]:
        """从世界变换矩阵提取 3×3 旋转。PhysX 驱动刚体时，此结果随仿真更新，比只读 xformOp 属性更可靠。"""
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            return None
        xformable = UsdGeom.Xformable(prim)
        m = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        mat = np.zeros((4, 4), dtype=float)
        for i in range(4):
            for j in range(4):
                mat[i, j] = float(m[i][j])
        return mat[:3, :3]
    
    def set_object_position(self, object_path: str, position: np.ndarray, local_position: np.ndarray = None, position_offset: np.ndarray = None) -> None:
        """Set the object's position in world or local space."""
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            print(f"Object at path {object_path} not found.")
            return

        xformable = UsdGeom.Xformable(prim)
        xform_ops = xformable.GetOrderedXformOps()
        if local_position is not None and position_offset is not None:
            new_position = Gf.Vec3d(*(local_position + position_offset))
        else:
            new_position = Gf.Vec3d(*position)

        if xform_ops:
            xform_ops[0].Set(new_position)
        else:
            xformable.AddTranslateOp().Set(new_position)

    def snapshot_object_xform_rotation(self, object_path: str):
        """读取 prim 上当前旋转（`world.reset()` 之后应为资产默认朝向），供改平移后写回。

        许多资产的「竖直」在 USD 里**不是**单位四元数/零欧拉，不可强制 identity。
        返回 `None` 表示未找到常见 xform 旋转属性。
        """
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            return None
        orient_attr = prim.GetAttribute("xformOp:orient")
        if orient_attr.IsValid():
            v = orient_attr.Get()
            if v is not None:
                return ("orient", Gf.Quatf(v))
        rot_attr = prim.GetAttribute("xformOp:rotateXYZ")
        if rot_attr.IsValid():
            v = rot_attr.Get()
            if v is not None:
                return ("rotateXYZ", Gf.Vec3f(v))
        axes = {}
        for name in ("xformOp:rotateX", "xformOp:rotateY", "xformOp:rotateZ"):
            a = prim.GetAttribute(name)
            if a.IsValid() and a.Get() is not None:
                axes[name] = float(a.Get())
        if axes:
            return ("axes", axes)
        return None

    def restore_object_xform_rotation(self, object_path: str, snapshot) -> None:
        """将 `snapshot_object_xform_rotation` 的快照写回（改 `set_object_position` 后调用）。"""
        if snapshot is None:
            return
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            return
        kind, data = snapshot[0], snapshot[1]
        if kind == "orient":
            prim.GetAttribute("xformOp:orient").Set(data)
            return
        if kind == "rotateXYZ":
            prim.GetAttribute("xformOp:rotateXYZ").Set(data)
            return
        if kind == "axes":
            for name, val in data.items():
                a = prim.GetAttribute(name)
                if a.IsValid():
                    a.Set(val)
            
    def get_geometry_center(self, object_name: str = None, object_path: str = None) -> np.ndarray:
        path = self._get_object_path(object_name, object_path)
        prim = self._stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return None

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(prim)
        range_3d = bbox.GetRange()
        center = (np.array(range_3d.GetMin()) + np.array(range_3d.GetMax())) / 2.0
        center_hom = np.append(center, 1.0)
        transform = np.array(bbox.GetMatrix())
        world_center = center_hom @ transform
        return world_center[:3]

    
    def get_transform_quat(self, object_path: str, w_first: bool = False) -> np.ndarray:
        """Get the world-space rotation quaternion from the object's transform.
        
        Args:
            object_path: The USD path to the object.
            w_first: If True, return quaternion in [w, x, y, z] format, else [x, y, z, w].
                    Default is False.
        """
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            print(f"Object at path {object_path} not found.")
            return None

        rotation = prim.GetAttribute("xformOp:orient").Get()
        
        if rotation is None:
            rotation = prim.GetAttribute("xformOp:rotateXYZ").Get()
            rotation = euler_angles_to_quats(rotation, degrees=True)
            # rotation is already in [w, x, y, z] format
            return np.array([rotation[0], rotation[1], rotation[2], rotation[3]]) if w_first else np.array([rotation[1], rotation[2], rotation[3], rotation[0]])

        quat = np.array([rotation.GetImaginary()[0], rotation.GetImaginary()[1], rotation.GetImaginary()[2], rotation.GetReal()])
        if abs(quat[0]) > 0.5 and abs(quat[0]) > abs(quat[3]):
            quat = np.array([quat[1], quat[2], quat[3], quat[0]])
            
        return np.array([quat[3], quat[0], quat[1], quat[2]]) if w_first else quat
        
    def get_revolute_joint_positions(self, joint_path: str) -> np.ndarray:
        joint_prim = self._stage.GetPrimAtPath(joint_path)

        joint_api = UsdPhysics.Joint(joint_prim)
        body1 = joint_api.GetBody1Rel().GetTargets()
        if not body1:
            print("No body1 found!")
            exit()

        body1_prim = self._stage.GetPrimAtPath(body1[0])

        body1_xform = UsdGeom.Xformable(body1_prim)
        body1_world_transform = Gf.Matrix4f(body1_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))

        local_pos1 = joint_api.GetLocalPos1Attr().Get() 
        local_rot1 = joint_api.GetLocalRot1Attr().Get()  

        rotation_matrix = Gf.Matrix3f(local_rot1)
        local_transform = Gf.Matrix4f()
        local_transform.SetTranslateOnly(local_pos1)
        local_transform.SetRotateOnly(rotation_matrix)
        joint_position = local_transform * body1_world_transform
        return np.array(joint_position.ExtractTranslation())
