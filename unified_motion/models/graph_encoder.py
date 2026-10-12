"""可选 GNN 插口：让有明确关系的节点交换信息，而不是把所有节点直接平均。

本模块不猜机器人骨架、不创造接触力、不自动识别物体。
    MessagePassingEncoder：通用图层，调用者明确提供节点、边和有效标记。
    PhysicsRelationGraph：适配当前项目的身体整体—接触关系—物体表示。

当前第一种接线放在独立 DynamicsModel 的物理编码器内，默认关闭。
这样 MLP、GRU、LSTM、Dense、MoE 都能读取经过图编码的 context/z；
没有把 GNN+LSTM+MoE 强制堆到同一个网络，也没有引入 PyG/DGL 新依赖。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass
class GraphBatch:
    """一帧或一段互不混合的图。

    nodes：[B,V,D] 或 [B,T,V,D]，V个节点、每个节点D个数字。
    node_mask：[B,V] 或 [B,T,V]，True=节点真实存在；不是接触发生标记。
    edge_index：[2,E] 整数，第一行是消息来源，第二行是消息去向。
    edge_type：[E] 整数，为每条边选择一个关系类型；不是奖励或动作。
    edge_mask：可选 [...,E] bool，临时不存在的边可以屏蔽。

    一次调用内图拓扑相同；不同环境/时间有自己的特征和mask，绝不会跨环境传消息。
    动态物体身份跟踪仍需环境/感知层提供；不能只交换槽位而不更新拓扑和记忆。
    """
    nodes: torch.Tensor
    node_mask: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_mask: torch.Tensor | None = None


class _MessageLayer(nn.Module):
    """一轮消息传递：沿指定边计算信息，接收节点求平均后更新自身。"""

    def __init__(self, width: int, num_edge_types: int) -> None:
        """输入特征宽度和边类型数；建立本层独立的可训练网络。"""
        super().__init__()
        self.edge_embedding = nn.Embedding(num_edge_types, width)
        self.message = nn.Sequential(nn.Linear(3*width, width), nn.SiLU(),
                                     nn.Linear(width, width))
        self.update = nn.Sequential(nn.Linear(2*width, width), nn.SiLU(),
                                    nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                edges: torch.Tensor, kinds: torch.Tensor,
                edge_mask: torch.Tensor | None) -> torch.Tensor:
        """输入 [...,V,D]，输出同形状；每条边只读取它的两端。

        无效节点在网络计算前清零；无效边不参与求和或分母。
        即使没有任何边，节点仍可经过自身更新，不会出现除零。
        """
        src, dst = edges[0], edges[1]
        edge_valid = mask.index_select(-1, src) & mask.index_select(-1, dst)
        if edge_mask is not None:
            edge_valid = edge_valid & edge_mask
        left, right = x.index_select(-2, src), x.index_select(-2, dst)
        emb = self.edge_embedding(kinds).expand(*x.shape[:-2], -1, -1)
        message_input = torch.where(edge_valid[..., None], torch.cat((left, right, emb), -1), 0.0)
        msg = self.message(message_input)
        msg = torch.where(edge_valid[..., None], msg, 0.0)
        # index_add按目标节点聚合，边列表不扩成 V×V 的稠密邻接矩阵。
        aggregate = torch.zeros_like(x).index_add(-2, dst, msg)
        count = x.new_zeros(*mask.shape).index_add(-1, dst, edge_valid.to(x.dtype))
        aggregate = aggregate / count.clamp_min(1)[..., None]
        y = self.norm(x + self.update(torch.cat((x, aggregate), -1)))
        return torch.where(mask[..., None], y, 0.0)


class MessagePassingEncoder(nn.Module):
    """通用的轻量消息传递 GNN，输出逐节点特征，不强迫立即池化。

    这是真正沿边传消息的图网络，而不是把名字叫 GNN 的普通 MLP。
    输入输出宽度相同，方便以后接 LSTM/GRU/Transformer 或其他研究模块。
    当前无采样邻居、无图级隐状态、无随机Dropout，便于复算 PPO。
    """

    def __init__(self, width: int, *, n_layers: int = 2,
                 num_edge_types: int = 3) -> None:
        """输入正整数 width、消息传递次数和关系类型数；输出可训练图模块。"""
        super().__init__()
        if any(type(n) is not int or n < 1 for n in (width, n_layers, num_edge_types)):
            raise ValueError("GNN宽度、层数和边类型数必须是正整数")
        if width < 2:
            raise ValueError("GNN宽度至少为2，避免一维LayerNorm丢掉信息")
        self.width, self.n_layers, self.num_edge_types = width, n_layers, num_edge_types
        self.layers = nn.ModuleList([_MessageLayer(width, num_edge_types) for _ in range(n_layers)])

    def get_extra_state(self) -> dict:
        """输出图网络结构定义，保存权重时一起核对。"""
        return dict(schema=1, width=self.width, n_layers=self.n_layers,
                    num_edge_types=self.num_edge_types, aggregation='incoming_mean')

    def set_extra_state(self, state: dict) -> None:
        """输入旧定义，结构不匹配就报错；不能仅按权重元素数载入。"""
        if state != self.get_extra_state():
            raise ValueError("GNN checkpoint 结构不一致")

    def forward(self, graph: GraphBatch) -> torch.Tensor:
        """执行多轮消息传递。

        输入：GraphBatch，各张量与模型同设备；节点浮点类型与模型一致。
        输出：与 graph.nodes 同形状的特征，补位节点输出严格为零。
        拒绝越界/负数边、重复边和自环；本层已经通过残差保留节点自身信息。
        """
        if not isinstance(graph, GraphBatch):
            raise TypeError("GNN输入必须是GraphBatch")
        x, mask, edges, kinds = graph.nodes, graph.node_mask, graph.edge_index, graph.edge_type
        p = self.layers[0].message[0].weight
        if (not isinstance(x, torch.Tensor) or x.ndim not in (3, 4)
                or min(x.shape) < 1 or x.shape[-1] != self.width):
            raise ValueError("nodes必须是非空[B,V,D]或[B,T,V,D]")
        if x.device != p.device or x.dtype != p.dtype:
            raise ValueError("nodes必须与GNN同设备、同精度")
        if (not isinstance(mask, torch.Tensor) or mask.shape != x.shape[:-1]
                or mask.dtype != torch.bool or mask.device != x.device):
            raise ValueError("node_mask形状/类型/设备错误")
        if (not isinstance(edges, torch.Tensor) or edges.ndim != 2 or edges.shape[0] != 2
                or edges.dtype != torch.long or edges.device != x.device):
            raise ValueError("edge_index必须是同设备[2,E]的long整数")
        e = edges.shape[1]
        if (not isinstance(kinds, torch.Tensor) or kinds.shape != (e,)
                or kinds.dtype != torch.long or kinds.device != x.device):
            raise ValueError("edge_type必须是同设备[E]的long整数")
        if e:
            if (edges < 0).any().item() or (edges >= x.shape[-2]).any().item():
                raise ValueError("图边索引超出节点范围")
            if (kinds < 0).any().item() or (kinds >= self.num_edge_types).any().item():
                raise ValueError("图边类型超出配置范围")
            if (edges[0] == edges[1]).any().item():
                raise ValueError("不需要重复自环，本层已有自身残差")
            if torch.unique(edges[0]*x.shape[-2] + edges[1]).numel() != e:
                raise ValueError("不允许重复边，否则平均消息的权重会被偷偷改变")
        if graph.edge_mask is not None:
            m = graph.edge_mask
            if (not isinstance(m, torch.Tensor) or m.shape != (*x.shape[:-2], e)
                    or m.dtype != torch.bool or m.device != x.device):
                raise ValueError("edge_mask必须是[...,E]的同设备bool")
        if not torch.isfinite(x).all().item():
            raise ValueError("节点数据含NaN/Inf，包括补位区")
        x = torch.where(mask[..., None], x, 0.0)
        with torch.autocast(device_type=x.device.type, enabled=False):
            for layer in self.layers:
                x = layer(x, mask, edges, kinds, graph.edge_mask)
        if not torch.isfinite(x).all().item():
            raise ValueError("GNN输出含NaN/Inf")
        return x


class PhysicsRelationGraph(nn.Module):
    """当前项目的可选图接线；输入是已有物理Encoder生成的节点，而非生数据。

    节点顺序：身体整体1个 -> C个接触关系 -> M个物体。
    边：身体与所有可见槽位双向连接；接触关系与显式指定的对象双向连接。
    contact_object_indices[j]=k 表示第j条关系的对象是第k个物体；-1表示静态世界。

    这是配置关系图，不是自动识别的机器人骨架图，也不是所有节点都在真实接触。
    实际接触标志仍在输入特征中；无接触时也需要知道目标对象在哪里。
    """

    def __init__(self, width: int, contact_object_indices: Sequence[int],
                 *, n_layers: int = 2) -> None:
        """输入宽度、明确的关系—物体映射和层数；映射会随checkpoint保存。"""
        super().__init__()
        if isinstance(contact_object_indices, (str, bytes)):
            raise TypeError("contact_object_indices必须是整数列表")
        mapping = tuple(contact_object_indices)
        if any(type(i) is not int or i < -1 for i in mapping):
            raise ValueError("对象编号必须是-1或从0开始的整数")
        self.mapping, self.width = mapping, width
        self.node_types = nn.Embedding(3, width)
        self.gnn = MessagePassingEncoder(width, n_layers=n_layers, num_edge_types=3)

    def get_extra_state(self) -> dict:
        """保存槽位拓扑；换对象关系时必须显式开新配置，不能沿用旧记忆。"""
        return dict(schema=1, mapping=self.mapping, layout='body_relations_objects', width=self.width)

    def set_extra_state(self, state: dict) -> None:
        """输入checkpoint拓扑；与当前关系不符则拒绝载入。"""
        if state != self.get_extra_state():
            raise ValueError("GNN关系拓扑与checkpoint不一致")

    def forward(self, body: torch.Tensor, contacts: torch.Tensor, objects: torch.Tensor,
                contact_mask: torch.Tensor, object_mask: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把已有特征临时组成图，消息传递后按原顺序拆回。

        输入：body [...,D]、contacts [...,C,D]、objects [...,M,D]、对应mask。
        输出：更新后的body/contact/object，形状不变、顺序不变。
        不改写环境数据；结果供原来的merge、预测器和历史GRU继续使用。
        """
        c, m = contacts.shape[-2], objects.shape[-2]
        if len(self.mapping) != c or any(i >= m for i in self.mapping):
            raise ValueError("GNN关系映射与当前接触/物体槽位数不一致")
        lead = body.shape[:-1]
        if (contacts.shape != (*lead, c, self.width) or objects.shape != (*lead, m, self.width)
                or contact_mask.shape != (*lead, c) or object_mask.shape != (*lead, m)):
            raise ValueError("物理图各字段的环境/时间/宽度不一致")
        nodes = torch.cat((body[..., None, :], contacts, objects), dim=-2)
        mask = torch.cat((torch.ones((*lead, 1), dtype=torch.bool, device=body.device),
                          contact_mask, object_mask), dim=-1)
        ids = torch.tensor([0] + [1]*c + [2]*m, device=body.device, dtype=torch.long)
        nodes = torch.where(mask[..., None], nodes + self.node_types(ids), 0.0)
        edge_list, edge_types = [], []
        # 边数约线性随槽位增加；没有构建全连接 V×V 矩阵。
        for j in range(c+m):
            edge_list.extend(((0, j+1), (j+1, 0)))
            edge_types.extend((0 if j < c else 1,)*2)
        for j, other in enumerate(self.mapping):
            if other >= 0:
                edge_list.extend(((1+j, 1+c+other), (1+c+other, 1+j)))
                edge_types.extend((2, 2))
        edges = torch.tensor(edge_list, dtype=torch.long, device=body.device).reshape(-1, 2).T.contiguous()
        kinds = torch.tensor(edge_types, dtype=torch.long, device=body.device)
        result = self.gnn(GraphBatch(nodes, mask, edges, kinds))
        return result[..., 0, :], result[..., 1:1+c, :], result[..., 1+c:, :]
