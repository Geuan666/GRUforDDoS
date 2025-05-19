#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Optional, Union
from utils import CLASS_MAP
logger = logging.getLogger(__name__)


class BaseGRUModel(nn.Module):
    """
    分层分类系统的基础GRU模型
    """

    def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_classes=3, dropout_rate=0.3):
        """
        初始化基础GRU模型
        参数:
            input_size: 每个时间步的输入特征数量
            hidden_size: GRU隐藏状态维度
            num_layers: GRU层数
            num_classes: 输出类别数量
            dropout_rate: Dropout概率
        """
        super(BaseGRUModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False  # 使用单向GRU简化模型
        )

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 批归一化层
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # 简化的分类器 - 单层全连接网络
        self.fc = nn.Linear(hidden_size, num_classes)

        # 初始化权重
        self._init_weights()

        logger.info(f"初始化BaseGRUModel: input_size={input_size}, "
                    f"hidden_size={hidden_size}, num_layers={num_layers}, "
                    f"num_classes={num_classes}, dropout_rate={dropout_rate}")

    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    # GRU权重: 使用正交初始化
                    nn.init.orthogonal_(param)
                elif len(param.shape) >= 2:
                    # 线性层权重: 使用Xavier初始化
                    nn.init.xavier_uniform_(param)
                else:
                    # 1维权重: 使用常数初始化
                    nn.init.constant_(param, 1.0)
            elif 'bias' in name:
                # 将偏置初始化为零
                nn.init.zeros_(param)

    def forward(self, x):
        """
        模型的前向传播
        参数:
            x: 输入张量 [batch_size, seq_len, input_size]
        返回:
            output: 类别逻辑值 [batch_size, num_classes]
        """
        # 处理空批次情况
        if x.size(0) == 0:
            return torch.zeros((0, self.num_classes), device=x.device)

        # GRU前向传播
        _, final_hidden = self.gru(x)
        hidden = final_hidden[-1]  # [batch_size, hidden_size]

        # 对于小批次（单样本），临时切换到评估模式
        batch_size = hidden.size(0)
        if batch_size == 1 and self.training:
            # 保存当前训练状态
            was_training = True
            # 临时切换到评估模式
            self.batch_norm.eval()

            # 应用批归一化
            hidden = self.batch_norm(hidden)

            # 恢复训练模式
            self.batch_norm.train()
        else:
            # 正常应用批归一化
            hidden = self.batch_norm(hidden)

        # 应用dropout进行正则化
        hidden = self.dropout(hidden)

        # 通过分类器得到输出
        output = self.fc(hidden)  # [batch_size, num_classes]

        return output


class Level1Classifier(BaseGRUModel):
    """
    一级分类器: BENIGN / DoS / Protocol (3类)
    """

    def __init__(self, input_size=1, hidden_size=128, num_layers=4, dropout_rate=0.3):
        super(Level1Classifier, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=3,  # BENIGN, DoS, Protocol
            dropout_rate=dropout_rate
        )
        logger.info("初始化一级分类器: BENIGN / DoS / Protocol (3类)")


class Level2DosClassifier(BaseGRUModel):
    """
    二级分类器(DoS): DrDoS / Generic-DoS (2类)
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout_rate=0.3):
        super(Level2DosClassifier, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=2,  # DrDoS, Generic-DoS
            dropout_rate=dropout_rate
        )
        logger.info("初始化二级分类器(DoS): DrDoS / Generic-DoS (2类)")


class Level2ProtocolClassifier(BaseGRUModel):
    """
    二级分类器(Protocol): Portmap / TFTP (2类)
    """

    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout_rate=0.3):
        super(Level2ProtocolClassifier, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=2,  # Portmap, TFTP
            dropout_rate=dropout_rate
        )
        logger.info("初始化二级分类器(Protocol): Portmap / TFTP (2类)")


class Level3DrDoSClassifier(BaseGRUModel):
    """
    三级分类器(DrDoS): 8个DrDoS子类型
    """

    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout_rate=0.3):
        super(Level3DrDoSClassifier, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=8,  # DrDoS的8个子类
            dropout_rate=dropout_rate
        )
        logger.info("初始化三级分类器(DrDoS): 8个DrDoS子类型")


class Level3GenericDoSClassifier(BaseGRUModel):
    """
    三级分类器(Generic-DoS): Syn / UDP-lag (2类)
    """

    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout_rate=0.3):
        super(Level3GenericDoSClassifier, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=2,  # Syn, UDP-lag
            dropout_rate=dropout_rate
        )
        logger.info("初始化三级分类器(Generic-DoS): Syn / UDP-lag (2类)")


class HierarchicalDDoSDetector(nn.Module):
    """
    分层DDoS检测器 - 整合所有级别的分类器
    """

    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout_rate=0.3):
        """
        初始化分层DDoS检测器
        参数:
            input_size: 每个时间步的输入特征数量
            hidden_size: GRU隐藏状态维度
            num_layers: GRU层数
            dropout_rate: Dropout概率
        """
        super(HierarchicalDDoSDetector, self).__init__()

        # 创建各级分类器
        self.level1_classifier = Level1Classifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )

        self.level2_dos_classifier = Level2DosClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )

        self.level2_protocol_classifier = Level2ProtocolClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )

        self.level3_drdos_classifier = Level3DrDoSClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )

        self.level3_generic_dos_classifier = Level3GenericDoSClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )

        # 类别映射
        self.level1_classes = ['BENIGN', 'DoS', 'Protocol']
        self.level2_dos_classes = ['DrDoS', 'Generic-DoS']
        self.level2_protocol_classes = ['Portmap', 'TFTP']
        self.level3_drdos_classes = [
            'DNS', 'LDAP', 'MSSQL', 'NTP',
            'NetBIOS', 'SNMP', 'SSDP', 'UDP'
        ]
        self.level3_generic_dos_classes = ['Syn', 'UDP-lag']

        logger.info("初始化分层DDoS检测器")

    def forward_level(self, x, level=1, prev_level_result=None):
        """
        对特定级别进行前向传播
        参数:
            x: 输入特征 [batch_size, seq_len, input_size]
            level: 要预测的级别 (1, 2_dos, 2_protocol, 3_drdos, 3_generic_dos)
            prev_level_result: 上一级别的预测结果 (用于级联预测)
        返回:
            logits: 对应级别的预测logits
        """
        if level == 1:
            return self.level1_classifier(x)
        elif level == '2_dos':
            return self.level2_dos_classifier(x)
        elif level == '2_protocol':
            return self.level2_protocol_classifier(x)
        elif level == '3_drdos':
            return self.level3_drdos_classifier(x)
        elif level == '3_generic_dos':
            return self.level3_generic_dos_classifier(x)
        else:
            raise ValueError(f"不支持的级别: {level}")

    def forward(self, x, return_all_levels=False):
        """
        分层前向传播 - 级联预测
        参数:
            x: 输入特征 [batch_size, seq_len, input_size]
            return_all_levels: 是否返回所有级别的预测logits
        返回:
            结果字典 - 包含预测标签和概率
        """
        batch_size = x.shape[0]

        # 一级分类
        level1_logits = self.level1_classifier(x)
        level1_probs = torch.softmax(level1_logits, dim=1)
        level1_preds = torch.argmax(level1_probs, dim=1)

        results = {
            'level1': {
                'logits': level1_logits,
                'probs': level1_probs,
                'preds': level1_preds
            }
        }

        # 为DoS样本进行二级分类
        dos_mask = (level1_preds == 1)  # DoS的索引为1
        if torch.any(dos_mask):
            dos_samples = x[dos_mask]
            level2_dos_logits = self.level2_dos_classifier(dos_samples)
            level2_dos_probs = torch.softmax(level2_dos_logits, dim=1)
            level2_dos_preds = torch.argmax(level2_dos_probs, dim=1)

            # 创建完整批次的预测结果（未分类为DoS的样本设为-1）
            full_level2_dos_logits = torch.zeros((batch_size, 2), device=x.device) - 1000
            full_level2_dos_probs = torch.zeros((batch_size, 2), device=x.device)
            full_level2_dos_preds = torch.full((batch_size,), -1, device=x.device, dtype=torch.long)

            full_level2_dos_logits[dos_mask] = level2_dos_logits
            full_level2_dos_probs[dos_mask] = level2_dos_probs
            full_level2_dos_preds[dos_mask] = level2_dos_preds

            results['level2_dos'] = {
                'logits': full_level2_dos_logits,
                'probs': full_level2_dos_probs,
                'preds': full_level2_dos_preds
            }

            # 为DrDoS样本进行三级分类
            drdos_mask_local = (level2_dos_preds == 0)  # DrDoS的索引为0
            if torch.any(drdos_mask_local):
                drdos_mask_global = dos_mask.clone()
                drdos_mask_global[dos_mask] = drdos_mask_local

                drdos_samples = x[drdos_mask_global]
                level3_drdos_logits = self.level3_drdos_classifier(drdos_samples)
                level3_drdos_probs = torch.softmax(level3_drdos_logits, dim=1)
                level3_drdos_preds = torch.argmax(level3_drdos_probs, dim=1)

                # 创建完整批次的预测结果
                full_level3_drdos_logits = torch.zeros((batch_size, 8), device=x.device) - 1000
                full_level3_drdos_probs = torch.zeros((batch_size, 8), device=x.device)
                full_level3_drdos_preds = torch.full((batch_size,), -1, device=x.device, dtype=torch.long)

                full_level3_drdos_logits[drdos_mask_global] = level3_drdos_logits
                full_level3_drdos_probs[drdos_mask_global] = level3_drdos_probs
                full_level3_drdos_preds[drdos_mask_global] = level3_drdos_preds

                results['level3_drdos'] = {
                    'logits': full_level3_drdos_logits,
                    'probs': full_level3_drdos_probs,
                    'preds': full_level3_drdos_preds
                }

            # 为Generic-DoS样本进行三级分类
            generic_dos_mask_local = (level2_dos_preds == 1)  # Generic-DoS的索引为1
            if torch.any(generic_dos_mask_local):
                generic_dos_mask_global = dos_mask.clone()
                generic_dos_mask_global[dos_mask] = generic_dos_mask_local

                generic_dos_samples = x[generic_dos_mask_global]
                level3_generic_dos_logits = self.level3_generic_dos_classifier(generic_dos_samples)
                level3_generic_dos_probs = torch.softmax(level3_generic_dos_logits, dim=1)
                level3_generic_dos_preds = torch.argmax(level3_generic_dos_probs, dim=1)

                # 创建完整批次的预测结果
                full_level3_generic_dos_logits = torch.zeros((batch_size, 2), device=x.device) - 1000
                full_level3_generic_dos_probs = torch.zeros((batch_size, 2), device=x.device)
                full_level3_generic_dos_preds = torch.full((batch_size,), -1, device=x.device, dtype=torch.long)

                full_level3_generic_dos_logits[generic_dos_mask_global] = level3_generic_dos_logits
                full_level3_generic_dos_probs[generic_dos_mask_global] = level3_generic_dos_probs
                full_level3_generic_dos_preds[generic_dos_mask_global] = level3_generic_dos_preds

                results['level3_generic_dos'] = {
                    'logits': full_level3_generic_dos_logits,
                    'probs': full_level3_generic_dos_probs,
                    'preds': full_level3_generic_dos_preds
                }

        # 为Protocol样本进行二级分类
        protocol_mask = (level1_preds == 2)  # Protocol的索引为2
        if torch.any(protocol_mask):
            protocol_samples = x[protocol_mask]
            level2_protocol_logits = self.level2_protocol_classifier(protocol_samples)
            level2_protocol_probs = torch.softmax(level2_protocol_logits, dim=1)
            level2_protocol_preds = torch.argmax(level2_protocol_probs, dim=1)

            # 创建完整批次的预测结果
            full_level2_protocol_logits = torch.zeros((batch_size, 2), device=x.device) - 1000
            full_level2_protocol_probs = torch.zeros((batch_size, 2), device=x.device)
            full_level2_protocol_preds = torch.full((batch_size,), -1, device=x.device, dtype=torch.long)

            full_level2_protocol_logits[protocol_mask] = level2_protocol_logits
            full_level2_protocol_probs[protocol_mask] = level2_protocol_probs
            full_level2_protocol_preds[protocol_mask] = level2_protocol_preds

            results['level2_protocol'] = {
                'logits': full_level2_protocol_logits,
                'probs': full_level2_protocol_probs,
                'preds': full_level2_protocol_preds
            }

        return results

    def predict_flat(self, x):
        """
        生成扁平化预测 - 将所有级别的预测合并成最终的17类预测
        参数:
            x: 输入特征 [batch_size, seq_len, input_size]
        返回:
            final_preds: 最终17类别预测 [batch_size]
            final_probs: 最终17类别概率 [batch_size, 17]
        """
        batch_size = x.shape[0]
        device = x.device

        # 获取所有级别的预测
        results = self.forward(x, return_all_levels=True)

        # 创建最终预测数组
        final_preds = torch.full((batch_size,), -1, device=device, dtype=torch.long)
        final_probs = torch.zeros((batch_size, 13), device=device)

        # 直接分类为BENIGN的样本
        benign_mask = (results['level1']['preds'] == 0)
        final_preds[benign_mask] = 0  # BENIGN类别索引为0

        # Portmap 和 TFTP (从Protocol细分)
        if 'level2_protocol' in results:
            protocol_mask = (results['level1']['preds'] == 2)
            portmap_mask = protocol_mask & (results['level2_protocol']['preds'] == 0)
            tftp_mask = protocol_mask & (results['level2_protocol']['preds'] == 1)

            final_preds[portmap_mask] = 6  # Portmap类别索引
            final_preds[tftp_mask] = 10  # TFTP类别索引

        # Syn 和 UDP-lag (从Generic-DoS细分)
        if 'level3_generic_dos' in results:
            generic_dos_mask = (results['level1']['preds'] == 1) & (results['level2_dos']['preds'] == 1)
            syn_mask = generic_dos_mask & (results['level3_generic_dos']['preds'] == 0)
            udp_lag_mask = generic_dos_mask & (results['level3_generic_dos']['preds'] == 1)

            final_preds[syn_mask] = 9  # Syn类别索引
            final_preds[udp_lag_mask] = 12  # UDP-lag类别索引

            # DrDoS子类型 (8类)
            if 'level3_drdos' in results:
                drdos_mask = (results['level1']['preds'] == 1) & (results['level2_dos']['preds'] == 0)
                # 使用正确的映射而不是简单的i+1
                drdos_class_map = {
                    0: 1,  # DNS
                    1: 2,  # LDAP
                    2: 3,  # MSSQL
                    3: 4,  # NTP
                    4: 5,  # NetBIOS
                    5: 7,  # SNMP
                    6: 8,  # SSDP
                    7: 11  # UDP
                }
                for i in range(8):
                    class_mask = drdos_mask & (results['level3_drdos']['preds'] == i)
                    final_preds[class_mask] = drdos_class_map[i]


        return final_preds, final_probs

    def save_models(self, save_dir):
        """
        保存所有级别的模型
        参数:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        # 保存一级分类器
        torch.save(self.level1_classifier.state_dict(), os.path.join(save_dir, 'level1_classifier.pth'))

        # 保存二级分类器
        torch.save(self.level2_dos_classifier.state_dict(), os.path.join(save_dir, 'level2_dos_classifier.pth'))
        torch.save(self.level2_protocol_classifier.state_dict(),
                   os.path.join(save_dir, 'level2_protocol_classifier.pth'))

        # 保存三级分类器
        torch.save(self.level3_drdos_classifier.state_dict(), os.path.join(save_dir, 'level3_drdos_classifier.pth'))
        torch.save(self.level3_generic_dos_classifier.state_dict(),
                   os.path.join(save_dir, 'level3_generic_dos_classifier.pth'))

        logger.info(f"所有级别模型已保存到 {save_dir}")

    def load_models(self, load_dir):
        """
        加载所有级别的模型
        参数:
            load_dir: 加载目录
        """
        # 加载一级分类器
        self.level1_classifier.load_state_dict(torch.load(os.path.join(load_dir, 'level1_classifier.pth')))

        # 加载二级分类器
        self.level2_dos_classifier.load_state_dict(torch.load(os.path.join(load_dir, 'level2_dos_classifier.pth')))
        self.level2_protocol_classifier.load_state_dict(
            torch.load(os.path.join(load_dir, 'level2_protocol_classifier.pth')))

        # 加载三级分类器
        self.level3_drdos_classifier.load_state_dict(torch.load(os.path.join(load_dir, 'level3_drdos_classifier.pth')))
        self.level3_generic_dos_classifier.load_state_dict(
            torch.load(os.path.join(load_dir, 'level3_generic_dos_classifier.pth')))

        logger.info(f"所有级别模型已从 {load_dir} 加载")

    def get_model_by_level(self, level):
        """
        获取特定级别的模型
        参数:
            level: 级别标识符 (1 或 '1', 2_dos, 2_protocol, 3_drdos, 3_generic_dos)
        返回:
            对应级别的模型实例
        """
        if level == 1 or level == '1':  # 接受整数或字符串
            return self.level1_classifier
        elif level == '2_dos':
            return self.level2_dos_classifier
        elif level == '2_protocol':
            return self.level2_protocol_classifier
        elif level == '3_drdos':
            return self.level3_drdos_classifier
        elif level == '3_generic_dos':
            return self.level3_generic_dos_classifier
        else:
            raise ValueError(f"不支持的级别: {level}")


def get_model_classes(level):
    """
    获取特定级别模型的类别列表
    参数:
        level: 级别标识符 (1, 2_dos, 2_protocol, 3_drdos, 3_generic_dos)
    返回:
        类别名称列表
    """
    if level == 1:
        return ['BENIGN', 'DoS', 'Protocol']
    elif level == '2_dos':
        return ['DrDoS', 'Generic-DoS']
    elif level == '2_protocol':
        return ['Portmap', 'TFTP']
    elif level == '3_drdos':
        return [
            'DNS', 'LDAP', 'MSSQL', 'NTP',
            'NetBIOS', 'SNMP', 'SSDP', 'UDP'
        ]
    elif level == '3_generic_dos':
        return ['Syn', 'UDP-lag']
    else:
        raise ValueError(f"不支持的级别: {level}")


# 如果直接运行此文件，则执行测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建分层检测器实例
    detector = HierarchicalDDoSDetector(
        input_size=1,
        hidden_size=128,
        num_layers=2,
        dropout_rate=0.3
    )

    # 测试前向传播
    batch_size = 4
    seq_len = 25  # 对应PCA降维后的特征数
    x = torch.randn(batch_size, seq_len, 1)  # 随机输入数据

    # 测试级联预测
    results = detector.forward(x)
    print("级联预测结果:")
    for level, result in results.items():
        print(f"{level}: {result['preds']}")

    # 测试扁平化预测
    final_preds, _ = detector.predict_flat(x)
    print(f"扁平化预测结果: {final_preds}")

    # 测试模型保存和加载
    save_dir = "test_models"
    detector.save_models(save_dir)
    detector.load_models(save_dir)

    print("测试完成")