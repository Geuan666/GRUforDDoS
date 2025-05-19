#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt

# 导入模块
from data import (
    DDoSDataset, create_dataloader, create_hierarchical_dataloaders,
    LEVEL1_MAPPING, LEVEL2_MAPPING, LEVEL3_MAPPING, CLASS_MAPPINGS
)
from model import (
    BaseGRUModel, Level1Classifier, Level2DosClassifier,
    Level2ProtocolClassifier, Level3DrDoSClassifier,
    Level3GenericDoSClassifier, HierarchicalDDoSDetector, get_model_classes
)
from trainer import Trainer, HierarchicalTrainer
import utils

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ddos_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train_hierarchical_model(
        train_data_path,
        val_data_path,
        output_dir="./hierarchical_outputs",
        batch_size=256,
        epochs=10,
        learning_rate=0.001,
        weight_decay=0.001,
        gradient_clip=1.0
):
    """
    训练分层DDoS检测模型
    参数:
        train_data_path: 训练数据路径
        val_data_path: 验证数据路径
        output_dir: 输出目录
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        gradient_clip: 梯度裁剪值
    返回:
        model: 训练好的分层模型
        history: 训练历史
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 设置预处理器保存路径
    preprocessor_path = os.path.join(output_dir, "hierarchical_preprocessor.pkl")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 创建数据集和数据加载器
    logger.info("创建分层数据加载器...")
    try:
        # 创建分层数据加载器
        hierarchical_loaders = create_hierarchical_dataloaders(
            train_path=train_data_path,
            val_path=val_data_path,
            preprocessor_path=preprocessor_path,
            batch_size=batch_size,
            num_workers=4
        )

        logger.info(f"成功创建 {len(hierarchical_loaders)} 个级别的数据加载器")

        # 样本特征形状
        for level_id, (train_loader, _) in hierarchical_loaders.items():
            inputs, targets = next(iter(train_loader))
            logger.info(f"级别 {level_id} 样本形状: {inputs.shape}, 标签形状: {targets.shape}")
            break

    except Exception as e:
        logger.error(f"创建数据加载器时出错: {str(e)}")
        raise

    # 初始化分层模型
    logger.info("初始化分层DDoS检测器...")
    input_size = 1  # 根据数据集设置
    hidden_size = 128
    num_layers = 2
    dropout_rate = 0.3

    # 创建分层模型
    hierarchical_model = HierarchicalDDoSDetector(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    )

    # 初始化分层训练器
    trainer = HierarchicalTrainer(
        hierarchical_model=hierarchical_model,
        dataloaders_dict=hierarchical_loaders,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip_val=gradient_clip,
        device=device,
        checkpoint_dir=checkpoint_dir
    )

    # 训练模型
    logger.info("开始训练分层模型...")
    history = trainer.train_all_levels(epochs=epochs, early_stopping_patience=5)

    # 加载所有级别的最佳模型
    trainer.load_best_models()

    # 在验证集上评估模型
    logger.info("在验证集上评估分层模型...")

    # 创建整体评估结果
    evaluation_results = {}

    # 对每个级别单独评估
    for level_id, (_, val_loader) in hierarchical_loaders.items():
        level_name = trainer.level_id_to_name[level_id]
        sub_model = hierarchical_model.get_model_by_level(level_name)

        # 评估子模型
        val_loss, val_acc, y_true, y_pred = utils.evaluate_model(
            model=sub_model,
            data_loader=val_loader,
            device=device
        )

        # 保存评估结果
        evaluation_results[level_id] = {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'y_true': y_true,
            'y_pred': y_pred
        }

        # 获取类别名称
        class_names = get_model_classes(level_name)

        # 绘制混淆矩阵
        utils.plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            save_path=os.path.join(output_dir, f"level{level_id}_confusion_matrix.png")
        )

        # 获取分类报告
        report_df = utils.get_classification_report(y_true, y_pred, class_names)
        logger.info(f"级别 {level_id} 分类报告:\n{report_df}")
        report_df.to_csv(os.path.join(output_dir, f"level{level_id}_classification_report.csv"))

        # 绘制训练历史
        utils.plot_training_history(
            history=history[level_id],
            save_path=os.path.join(output_dir, f"level{level_id}_training_history.png")
        )

    # 保存完整模型
    logger.info("保存完整的分层模型...")
    hierarchical_model.save_models(checkpoint_dir)

    logger.info(f"分层模型训练和评估完成。结果保存到 {output_dir}")
    return hierarchical_model, history


def main():
    """运行训练和评估的主函数"""
    # 设置路径
    train_data_path = "C:\\Users\\17380\\train_dataset.csv"  # 替换为您的训练数据路径
    val_data_path = "C:\\Users\\17380\\test_dataset.csv"  # 替换为您的验证数据路径
    output_dir = "./hierarchical_outputs"

    # 训练并评估分层模型
    model, history = train_hierarchical_model(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        output_dir=output_dir,
        batch_size=128,
        epochs=8,
        learning_rate=0.001,
        weight_decay=0.001,
        gradient_clip=1.0
    )

    logger.info("分层DDoS检测系统训练成功完成！")


if __name__ == "__main__":
    main()