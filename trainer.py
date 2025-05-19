#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)


class Trainer:
    """
    DDoS检测模型的训练器类
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            learning_rate: float = 0.001,
            weight_decay: float = 0.001,
            gradient_clip_val: float = 1.0,
            device: str = None,
            checkpoint_dir: str = "./checkpoints"
    ):
        """
        初始化训练器
        参数:
            model: DDoS检测模型
            train_loader: 训练数据的DataLoader
            val_loader: 验证数据的DataLoader
            learning_rate: 优化器的学习率
            weight_decay: L2正则化系数
            gradient_clip_val: 梯度裁剪的最大范数
            device: 运行模型的设备 (cuda/cpu)
            checkpoint_dir: 保存模型检查点的目录
        """
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 优化设置
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val

        # 损失函数: 多类别分类的交叉熵
        self.criterion = nn.CrossEntropyLoss()

        # 优化器: 带有L2正则化的Adam
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 检查点设置
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epochs': []
        }

        logger.info(f"训练器初始化完成，设备: {self.device}, "
                    f"学习率: {learning_rate}, 权重衰减: {weight_decay}, "
                    f"梯度裁剪值: {gradient_clip_val}")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        训练一个epoch
        参数:
            epoch: 当前epoch编号
        返回:
            train_loss: 平均训练损失
            train_acc: 训练准确率
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # 将数据移至设备
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = targets.squeeze()  # 移除目标的额外维度 [batch_size, 1] -> [batch_size]

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # 计算损失
            loss = self.criterion(outputs, targets)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

            # 更新权重
            self.optimizer.step()

            # 记录统计信息
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 打印进度
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(self.train_loader):
                elapsed_time = time.time() - start_time
                logger.info(f'Epoch: {epoch} | Batch: {batch_idx + 1}/{len(self.train_loader)} | '
                            f'Loss: {total_loss / (batch_idx + 1):.4f} | '
                            f'Acc: {100.0 * correct / total:.2f}% | '
                            f'Time: {elapsed_time:.2f}s')

        # 计算平均损失和准确率
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """
        在验证数据集上验证模型
        返回:
            val_loss: 平均验证损失
            val_acc: 验证准确率
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                # 将数据移至设备
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = targets.squeeze()  # 移除额外维度

                # 前向传播
                outputs = self.model(inputs)

                # 计算损失
                loss = self.criterion(outputs, targets)

                # 记录统计信息
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # 计算平均损失和准确率
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        logger.info(f'验证 | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%')

        return avg_loss, accuracy

    def train(self, epochs: int, early_stopping_patience: int = 5) -> Dict:
        """
        训练模型多个epochs
        参数:
            epochs: 要训练的epoch数
            early_stopping_patience: 停止前等待改进的epoch数
        返回:
            history: 训练历史
        """
        logger.info(f"开始训练 {epochs} 个epochs...")

        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            logger.info(f"\nEpoch {epoch}/{epochs}")

            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(epoch)

            # 验证模型
            val_loss, val_acc = self.validate()

            # 更新训练历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['epochs'].append(epoch)

            # 检查是否为目前最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                # 保存最佳模型
                self.save_checkpoint(f'best_model.pth', epoch, val_loss, val_acc)
                logger.info(f"最佳模型已保存，epoch {epoch}，验证损失: {val_loss:.4f}")
            else:
                patience_counter += 1
                logger.info(f"验证损失未改善。耐心计数: {patience_counter}/{early_stopping_patience}")

            # 每5个epoch保存检查点
            if epoch % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch, val_loss, val_acc)

            # 早停
            if patience_counter >= early_stopping_patience:
                logger.info(f"触发早停，在 {epoch} 个epoch后。"
                            f"最佳验证损失: {best_val_loss:.4f}，在epoch {best_epoch}。")
                break

        logger.info(f"训练完成。最佳验证损失: {best_val_loss:.4f}，在epoch {best_epoch}。")
        return self.history

    def save_checkpoint(self, filename: str, epoch: int, val_loss: float, val_acc: float) -> None:
        """
        保存模型检查点
        参数:
            filename: 检查点文件名
            epoch: 当前epoch
            val_loss: 验证损失
            val_acc: 验证准确率
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': self.history
        }

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        加载模型检查点
        参数:
            checkpoint_path: 检查点文件路径
        返回:
            checkpoint: 加载的检查点字典
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 恢复训练历史
        if 'history' in checkpoint:
            self.history = checkpoint['history']

        logger.info(f"已加载检查点，来自epoch {checkpoint['epoch']}，"
                    f"验证损失: {checkpoint['val_loss']:.4f}，"
                    f"验证准确率: {checkpoint['val_acc']:.2f}%")

        return checkpoint


class HierarchicalTrainer:
    """
    用于训练分层DDoS检测模型的训练器类
    """

    def __init__(
            self,
            hierarchical_model,
            dataloaders_dict: Dict[int, Tuple[DataLoader, DataLoader]],
            learning_rate: float = 0.001,
            weight_decay: float = 0.001,
            gradient_clip_val: float = 1.0,
            device: str = None,
            checkpoint_dir: str = "./hierarchical_checkpoints"
    ):
        """
        初始化分层训练器
        参数:
            hierarchical_model: 分层DDoS检测模型
            dataloaders_dict: 数据加载器字典 {level_id: (train_loader, val_loader)}
            learning_rate: 优化器的学习率
            weight_decay: L2正则化系数
            gradient_clip_val: 梯度裁剪的最大范数
            device: 运行模型的设备 (cuda/cpu)
            checkpoint_dir: 保存模型检查点的目录
        """
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = hierarchical_model.to(self.device)
        self.dataloaders_dict = dataloaders_dict

        # 级别映射
        self.level_id_to_name = {
            1: '1',  # 一级分类
            2: '2_dos',  # 二级DoS分类
            3: '2_protocol',  # 二级Protocol分类
            4: '3_drdos',  # 三级DrDoS分类
            5: '3_generic_dos'  # 三级Generic-DoS分类
        }

        # 优化设置
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val

        # 损失函数: 多类别分类的交叉熵
        self.criterion = nn.CrossEntropyLoss()

        # 为每个子模型创建优化器
        self.optimizers = {}
        for level_id, level_name in self.level_id_to_name.items():
            sub_model = self.model.get_model_by_level(level_name)
            self.optimizers[level_id] = optim.Adam(
                sub_model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )

        # 检查点设置
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 训练历史记录 - 为每个级别保存单独的历史
        self.history = {level_id: {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epochs': []
        } for level_id in self.level_id_to_name.keys()}

        logger.info(f"分层训练器初始化完成，设备: {self.device}, "
                    f"学习率: {learning_rate}, 权重衰减: {weight_decay}, "
                    f"梯度裁剪值: {gradient_clip_val}")

    def train_epoch_level(self, level_id: int, epoch: int) -> Tuple[float, float]:
        """
        训练特定级别的一个epoch
        参数:
            level_id: 级别ID
            epoch: 当前epoch编号
        返回:
            train_loss: 平均训练损失
            train_acc: 训练准确率
        """
        level_name = self.level_id_to_name[level_id]
        sub_model = self.model.get_model_by_level(level_name)
        optimizer = self.optimizers[level_id]
        train_loader, _ = self.dataloaders_dict[level_id]

        sub_model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 将数据移至设备
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = targets.squeeze()  # 移除目标的额外维度

            # 前向传播
            optimizer.zero_grad()
            outputs = sub_model(inputs)

            # 计算损失
            loss = self.criterion(outputs, targets)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(sub_model.parameters(), self.gradient_clip_val)

            # 更新权重
            optimizer.step()

            # 记录统计信息
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 打印进度
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                elapsed_time = time.time() - start_time
                logger.info(f'级别 {level_id} | Epoch: {epoch} | Batch: {batch_idx + 1}/{len(train_loader)} | '
                            f'Loss: {total_loss / (batch_idx + 1):.4f} | '
                            f'Acc: {100.0 * correct / total:.2f}% | '
                            f'Time: {elapsed_time:.2f}s')

        # 计算平均损失和准确率
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate_level(self, level_id: int) -> Tuple[float, float]:
        """
        在验证数据集上验证特定级别的模型
        参数:
            level_id: 级别ID
        返回:
            val_loss: 平均验证损失
            val_acc: 验证准确率
        """
        level_name = self.level_id_to_name[level_id]
        sub_model = self.model.get_model_by_level(level_name)
        _, val_loader = self.dataloaders_dict[level_id]

        sub_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                # 将数据移至设备
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = targets.squeeze()  # 移除额外维度

                # 前向传播
                outputs = sub_model(inputs)

                # 计算损失
                loss = self.criterion(outputs, targets)

                # 记录统计信息
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # 计算平均损失和准确率
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        logger.info(f'级别 {level_id} 验证 | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%')

        return avg_loss, accuracy

    def train_level(self, level_id: int, epochs: int, early_stopping_patience: int = 5) -> Dict:
        """
        训练特定级别的模型
        参数:
            level_id: 级别ID
            epochs: 要训练的epoch数
            early_stopping_patience: 停止前等待改进的epoch数
        返回:
            level_history: 该级别的训练历史
        """
        level_name = self.level_id_to_name[level_id]
        logger.info(f"开始训练级别 {level_id} ({level_name}) 模型，共 {epochs} 个epochs...")

        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            logger.info(f"\n级别 {level_id} | Epoch {epoch}/{epochs}")

            # 训练一个epoch
            train_loss, train_acc = self.train_epoch_level(level_id, epoch)

            # 验证模型
            val_loss, val_acc = self.validate_level(level_id)

            # 更新训练历史
            self.history[level_id]['train_loss'].append(train_loss)
            self.history[level_id]['val_loss'].append(val_loss)
            self.history[level_id]['train_acc'].append(train_acc)
            self.history[level_id]['val_acc'].append(val_acc)
            self.history[level_id]['epochs'].append(epoch)

            # 检查是否为目前最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                # 保存最佳模型
                self.save_level_checkpoint(level_id, f'level{level_id}_best_model.pth', epoch, val_loss, val_acc)
                logger.info(f"级别 {level_id} 最佳模型已保存，epoch {epoch}，验证损失: {val_loss:.4f}")
            else:
                patience_counter += 1
                logger.info(f"级别 {level_id} 验证损失未改善。耐心计数: {patience_counter}/{early_stopping_patience}")

            # 早停
            if patience_counter >= early_stopping_patience:
                logger.info(f"级别 {level_id} 触发早停，在 {epoch} 个epoch后。"
                            f"最佳验证损失: {best_val_loss:.4f}，在epoch {best_epoch}。")
                break

        logger.info(f"级别 {level_id} 训练完成。最佳验证损失: {best_val_loss:.4f}，在epoch {best_epoch}。")
        return self.history[level_id]

    def train_all_levels(self, epochs: int, early_stopping_patience: int = 5) -> Dict:
        """
        按顺序训练所有级别的模型
        参数:
            epochs: 每个级别要训练的epoch数
            early_stopping_patience: 每个级别的早停耐心
        返回:
            history: 整体训练历史
        """
        logger.info(f"开始训练所有级别的模型...")

        # 按顺序训练每个级别
        for level_id in sorted(self.level_id_to_name.keys()):
            # 只训练存在数据加载器的级别
            if level_id in self.dataloaders_dict:
                self.train_level(level_id, epochs, early_stopping_patience)
            else:
                logger.warning(f"级别 {level_id} ({self.level_id_to_name[level_id]}) 没有数据加载器，跳过训练")

        # 保存完整的分层模型
        complete_model_path = os.path.join(self.checkpoint_dir, "complete_hierarchical_model.pth")
        self.model.save_models(self.checkpoint_dir)
        logger.info(f"完整的分层模型已保存到 {self.checkpoint_dir}")

        return self.history

    def save_level_checkpoint(self, level_id: int, filename: str, epoch: int, val_loss: float, val_acc: float) -> None:
        """
        保存特定级别的模型检查点
        参数:
            level_id: 级别ID
            filename: 检查点文件名
            epoch: 当前epoch
            val_loss: 验证损失
            val_acc: 验证准确率
        """
        level_name = self.level_id_to_name[level_id]
        sub_model = self.model.get_model_by_level(level_name)
        optimizer = self.optimizers[level_id]

        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': sub_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': self.history[level_id]
        }

        torch.save(checkpoint, checkpoint_path)

    def load_level_checkpoint(self, level_id: int, checkpoint_path: str) -> Dict:
        """
        加载特定级别的模型检查点
        参数:
            level_id: 级别ID
            checkpoint_path: 检查点文件路径
        返回:
            checkpoint: 加载的检查点字典
        """
        level_name = self.level_id_to_name[level_id]
        sub_model = self.model.get_model_by_level(level_name)
        optimizer = self.optimizers[level_id]

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        sub_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 恢复训练历史
        if 'history' in checkpoint:
            self.history[level_id] = checkpoint['history']

        logger.info(f"级别 {level_id} 已加载检查点，来自epoch {checkpoint['epoch']}，"
                    f"验证损失: {checkpoint['val_loss']:.4f}，"
                    f"验证准确率: {checkpoint['val_acc']:.2f}%")

        return checkpoint

    def load_best_models(self):
        """加载所有级别的最佳模型"""
        for level_id in self.level_id_to_name.keys():
            checkpoint_path = os.path.join(self.checkpoint_dir, f'level{level_id}_best_model.pth')
            if os.path.exists(checkpoint_path):
                self.load_level_checkpoint(level_id, checkpoint_path)
            else:
                logger.warning(f"找不到级别 {level_id} 的最佳模型检查点")