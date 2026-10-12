"""兼容旧 physics_dynamics 导入；唯一实现是 models.physics。"""
from .physics import *
from . import physics as _implementation


def __getattr__(name):
    """旧调用需要私有辅助函数时转发同一个对象；不复制实现。"""
    return getattr(_implementation, name)
