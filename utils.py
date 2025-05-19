#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
import pandas as pd
import onnx

# import onnxruntime as ort

# 设置matplotlib参数以避免中文字体问题（在图表中仍使用英文）
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

# 定义类别名称（只使用英文）
# 原始13类标签
CLASS_NAMES = ['BENIGN', 'DNS', 'LDAP', 'MSSQL', 'NTP', 'NetBIOS', 'Portmap', 'SNMP', 'SSDP', 'Syn', 'TFTP', 'UDP',
               'UDP-lag']

CLASS_MAP = {'BENIGN': 0, 'DNS': 1, 'LDAP': 2, 'MSSQL': 3, 'NTP': 4, 'NetBIOS': 5, 'Portmap': 6, 'SNMP': 7, 'SSDP': 8,
             'Syn': 9, 'TFTP': 10, 'UDP': 11, 'UDP-lag': 12}

# 分层类别名称
HIERARCHICAL_CLASS_NAMES = {
    '1': ['BENIGN', 'DoS', 'Protocol'],  # 一级分类
    '2_dos': ['DrDoS', 'Generic-DoS'],  # 二级DoS分类
    '2_protocol': ['Portmap', 'TFTP'],  # 二级Protocol分类
    '3_drdos': ['DrDoS_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NTP',
                'DrDoS_NetBIOS', 'DrDoS_SNMP', 'DrDoS_SSDP', 'DrDoS_UDP'],  # 三级DrDoS分类
    '3_generic_dos': ['Syn', 'UDP-lag']  # 三级Generic-DoS分类
}


def evaluate_model(model: torch.nn.Module,
                   data_loader: torch.utils.data.DataLoader,
                   device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    在数据集上评估模型

    参数:
        model: 训练好的模型
        data_loader: 数据集的DataLoader
        device: 运行评估的设备

    返回:
        loss: 数据集上的平均损失
        accuracy: 数据集上的准确率
        y_true: 真实标签
        y_pred: 预测标签
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            # 跳过空批次
            if inputs.size(0) == 0:
                continue

            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.squeeze()  # 移除额外维度

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, targets)

            # 记录统计信息
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 保存真实和预测标签
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # 计算指标
    avg_loss = total_loss / max(len(data_loader), 1)  # 避免除零错误
    accuracy = 100.0 * correct / max(total, 1)  # 避免除零错误

    logger.info(f'评估 | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%')

    return avg_loss, accuracy, np.array(y_true), np.array(y_pred)


def evaluate_hierarchical_model(model,
                                hierarchical_loaders: Dict[int, Tuple[Any, Any]],
                                device: torch.device) -> Dict[str, Dict[str, Any]]:
    """
    评估分层模型的所有级别

    参数:
        model: 分层模型
        hierarchical_loaders: 分层数据加载器字典 {level_id: (train_loader, val_loader)}
        device: 设备

    返回:
        evaluation_results: 每个级别的评估结果字典
    """
    evaluation_results = {}

    # 级别ID到名称的映射
    level_id_to_name = {
        1: '1',  # 一级分类
        2: '2_dos',  # 二级DoS分类
        3: '2_protocol',  # 二级Protocol分类
        4: '3_drdos',  # 三级DrDoS分类
        5: '3_generic_dos'  # 三级Generic-DoS分类
    }

    # 对每个级别分别进行评估
    for level_id, (_, val_loader) in hierarchical_loaders.items():
        level_name = level_id_to_name[level_id]
        sub_model = model.get_model_by_level(level_name)

        # 评估子模型
        val_loss, val_acc, y_true, y_pred = evaluate_model(
            model=sub_model,
            data_loader=val_loader,
            device=device
        )

        # 保存评估结果
        evaluation_results[level_name] = {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'y_true': y_true,
            'y_pred': y_pred
        }

    # 尝试进行端到端评估（如果有完整的测试数据集）
    if 1 in hierarchical_loaders:
        _, test_loader = hierarchical_loaders[1]  # 使用一级分类的验证集
        results = predict_hierarchical(model, test_loader, device)
        evaluation_results['end_to_end'] = results

    return evaluation_results


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: List[str] = CLASS_NAMES,
                          normalize: bool = True,
                          figsize: Tuple[int, int] = (10, 8),
                          save_path: Optional[str] = None) -> None:
    """
    绘制混淆矩阵

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        normalize: 是否归一化混淆矩阵
        figsize: 图像大小
        save_path: 保存图像的路径，如果为None，则显示图像
    """
    # 如果标签为空，则不绘制
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("标签为空，无法绘制混淆矩阵")
        return

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 如果请求，进行归一化
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)  # 添加小值避免除零
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    # 绘图
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"混淆矩阵已保存至 {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_history(history: Dict[str, List],
                          figsize: Tuple[int, int] = (12, 5),
                          save_path: Optional[str] = None) -> None:
    """
    绘制训练历史

    参数:
        history: 训练历史字典，包含键:
                'train_loss', 'val_loss', 'train_acc', 'val_acc', 'epochs'
        figsize: 图像大小
        save_path: 保存图像的路径，如果为None，则显示图像
    """
    plt.figure(figsize=figsize)

    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history['epochs'], history['train_loss'], 'b-', label='Training Loss')
    plt.plot(history['epochs'], history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(history['epochs'], history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(history['epochs'], history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"训练历史图已保存至 {save_path}")
    else:
        plt.show()

    plt.close()


def plot_hierarchical_training_history(histories: Dict[int, Dict[str, List]],
                                       figsize: Tuple[int, int] = (15, 12),
                                       save_path: Optional[str] = None) -> None:
    """
    绘制所有级别的训练历史

    参数:
        histories: 所有级别的训练历史字典 {level_id: history_dict}
        figsize: 图像大小
        save_path: 保存图像的路径
    """
    # 级别ID到名称的映射
    level_id_to_name = {
        1: 'Level 1 (BENIGN/DoS/Protocol)',
        2: 'Level 2 (DrDoS/Generic-DoS)',
        3: 'Level 2 (Portmap/TFTP)',
        4: 'Level 3 (DrDoS Subtypes)',
        5: 'Level 3 (Syn/UDP-lag)'
    }

    n_levels = len(histories)
    rows = (n_levels + 1) // 2  # 向上取整，每行最多2个级别

    plt.figure(figsize=figsize)

    # 绘制每个级别的训练历史
    for i, (level_id, history) in enumerate(histories.items()):
        # 绘制损失
        plt.subplot(rows, 4, i * 2 + 1)
        plt.plot(history['epochs'], history['train_loss'], 'b-', label='Training Loss')
        plt.plot(history['epochs'], history['val_loss'], 'r-', label='Validation Loss')
        plt.title(f'{level_id_to_name.get(level_id, f"Level {level_id}")} - Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.legend(fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.6)

        # 绘制准确率
        plt.subplot(rows, 4, i * 2 + 2)
        plt.plot(history['epochs'], history['train_acc'], 'b-', label='Training Accuracy')
        plt.plot(history['epochs'], history['val_acc'], 'r-', label='Validation Accuracy')
        plt.title(f'{level_id_to_name.get(level_id, f"Level {level_id}")} - Accuracy', fontsize=12)
        plt.xlabel('Epochs', fontsize=10)
        plt.ylabel('Accuracy (%)', fontsize=10)
        plt.legend(fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"分层训练历史图已保存至 {save_path}")
    else:
        plt.show()

    plt.close()


def plot_roc_curves(y_true: np.ndarray,
                    y_score: np.ndarray,
                    class_names: List[str] = CLASS_NAMES,
                    figsize: Tuple[int, int] = (12, 10),
                    save_path: Optional[str] = None) -> None:
    """
    绘制多类别分类的ROC曲线

    参数:
        y_true: 真实标签（one-hot编码）
        y_score: 预测分数
        class_names: 类别名称列表
        figsize: 图像大小
        save_path: 保存图像的路径，如果为None，则显示图像
    """
    # 如果标签为空，则不绘制
    if len(y_true) == 0 or len(y_score) == 0:
        logger.warning("标签为空，无法绘制ROC曲线")
        return

    # 为每个类别计算ROC曲线和ROC面积
    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # 如果还不是one-hot编码，则进行转换
    if len(y_true.shape) == 1:
        y_true_onehot = np.zeros((y_true.size, n_classes))
        y_true_onehot[np.arange(y_true.size), y_true] = 1
    else:
        y_true_onehot = y_true

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制ROC曲线
    plt.figure(figsize=figsize)

    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC曲线已保存至 {save_path}")
    else:
        plt.show()

    plt.close()


def get_classification_report(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              class_names: List[str] = CLASS_NAMES) -> pd.DataFrame:
    """
    获取分类报告作为DataFrame

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表

    返回:
        report_df: 作为DataFrame的分类报告
    """
    # 如果标签为空，则返回空DataFrame
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("标签为空，无法生成分类报告")
        return pd.DataFrame()

    # 获取分类报告作为文本
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # 转换为DataFrame
    report_df = pd.DataFrame(report).transpose()

    return report_df


def export_model_onnx(model: torch.nn.Module,
                      sample_input: torch.Tensor,
                      file_path: str) -> None:
    """
    将PyTorch模型导出为ONNX格式

    参数:
        model: PyTorch模型
        sample_input: 具有正确形状的样本输入张量
        file_path: 保存ONNX模型的路径
    """
    model.eval()

    # 导出模型
    torch.onnx.export(
        model,  # 正在运行的模型
        sample_input,  # 模型输入
        file_path,  # 保存模型的位置
        export_params=True,  # 将训练好的参数权重存储在模型文件中
        opset_version=12,  # 导出模型的ONNX版本
        do_constant_folding=True,  # 优化
        input_names=['input'],  # 模型的输入名称
        output_names=['output'],  # 模型的输出名称
        dynamic_axes={
            'input': {0: 'batch_size'},  # 可变长度轴
            'output': {0: 'batch_size'}
        }
    )

    logger.info(f"模型已导出为ONNX格式: {file_path}")

    # 验证模型
    onnx_model = onnx.load(file_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX模型验证通过。")


def export_hierarchical_model_onnx(model,
                                   sample_input: torch.Tensor,
                                   output_dir: str) -> None:
    """
    将分层模型的所有子模型导出为ONNX格式

    参数:
        model: 分层模型
        sample_input: 样本输入张量
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 级别名称映射
    level_names = {
        '1': 'level1_classifier',
        '2_dos': 'level2_dos_classifier',
        '2_protocol': 'level2_protocol_classifier',
        '3_drdos': 'level3_drdos_classifier',
        '3_generic_dos': 'level3_generic_dos_classifier'
    }

    # 导出每个子模型
    for level_name, file_name in level_names.items():
        sub_model = model.get_model_by_level(level_name)
        file_path = os.path.join(output_dir, f"{file_name}.onnx")

        export_model_onnx(
            model=sub_model,
            sample_input=sample_input,
            file_path=file_path
        )


def softmax(x, axis=None):
    """
    为x中的每组分数计算softmax值。

    参数:
        x: 输入数组
        axis: 计算softmax的轴

    返回:
        softmax_x: x的softmax
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def predict_batch(model: torch.nn.Module,
                  data_loader: torch.utils.data.DataLoader,
                  device: torch.device,
                  return_probs: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    获取一批数据的预测

    参数:
        model: 训练好的模型
        data_loader: 数据集的DataLoader
        device: 运行预测的设备
        return_probs: 是否返回概率分数

    返回:
        y_true: 真实标签
        y_pred: 预测标签
        y_probs: 预测概率（如果return_probs=True）
    """
    model.eval()

    y_true = []
    y_pred = []
    y_probs = [] if return_probs else None

    with torch.no_grad():
        for inputs, targets in data_loader:
            # 跳过空批次
            if inputs.size(0) == 0:
                continue

            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.squeeze()  # 移除额外维度

            # 前向传播
            outputs = model(inputs)

            # 获取预测
            _, predicted = outputs.max(1)

            # 存储真实和预测标签
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            # 如果请求，存储概率
            if return_probs:
                probs = torch.nn.functional.softmax(outputs, dim=1)
                y_probs.extend(probs.cpu().numpy())

    result = (np.array(y_true), np.array(y_pred))
    if return_probs:
        result += (np.array(y_probs),)

    return result


def predict_hierarchical(model,
                         data_loader: torch.utils.data.DataLoader,
                         device: torch.device) -> Dict[str, Any]:
    """
    使用分层模型进行预测

    参数:
        model: 分层模型
        data_loader: 数据加载器
        device: 设备

    返回:
        results: 预测结果字典
    """
    model.eval()

    all_inputs = []
    all_targets = []
    all_hierarchical_preds = {
        '1': [],  # 一级分类结果
        '2_dos': [],  # 二级DoS分类结果
        '2_protocol': [],  # 二级Protocol分类结果
        '3_drdos': [],  # 三级DrDoS分类结果
        '3_generic_dos': []  # 三级Generic-DoS分类结果
    }
    all_flat_preds = []  # 扁平化最终预测结果

    with torch.no_grad():
        for inputs, targets in data_loader:
            # 跳过空批次
            if inputs.size(0) == 0:
                continue

            inputs, targets = inputs.to(device), targets.to(device)

            # 获取分层预测
            hierarchical_results = model.forward(inputs)

            # 获取扁平化预测
            flat_preds, _ = model.predict_flat(inputs)

            # 存储输入和目标
            all_inputs.extend(inputs.cpu().numpy())
            all_targets.extend(targets.squeeze().cpu().numpy())

            # 存储分层预测结果
            for level_name, level_result in hierarchical_results.items():
                all_hierarchical_preds[level_name].extend(level_result['preds'].cpu().numpy())

            # 存储扁平化预测
            all_flat_preds.extend(flat_preds.cpu().numpy())

    # 转换为NumPy数组
    results = {
        'inputs': np.array(all_inputs),
        'targets': np.array(all_targets),
        'hierarchical_preds': {k: np.array(v) for k, v in all_hierarchical_preds.items()},
        'flat_preds': np.array(all_flat_preds)
    }

    return results


def plot_hierarchical_decision_path(model,
                                    inputs: torch.Tensor,
                                    targets: torch.Tensor,
                                    device: torch.device,
                                    save_path: Optional[str] = None) -> None:
    """
    可视化分层决策路径

    参数:
        model: 分层模型
        inputs: 输入张量 [batch_size, seq_len, input_size]
        targets: 目标张量 [batch_size, 1]
        device: 设备
        save_path: 保存路径
    """
    # 将输入移至设备
    inputs = inputs.to(device)

    # 获取分层预测
    with torch.no_grad():
        results = model.forward(inputs)

    # 准备可视化
    batch_size = inputs.shape[0]

    # 只显示最多10个样本
    n_samples = min(batch_size, 10)

    plt.figure(figsize=(12, n_samples * 2))

    for i in range(n_samples):
        # 获取该样本的各级别预测
        level1_pred = results['level1']['preds'][i].item()
        level1_name = ['BENIGN', 'DoS', 'Protocol'][level1_pred]

        # 初始化决策路径
        path = [f"Level 1: {level1_name}"]

        # 根据一级分类结果继续路径
        if level1_pred == 1:  # DoS
            level2_dos_pred = results['level2_dos']['preds'][i].item()
            if level2_dos_pred != -1:  # 有效预测
                level2_dos_name = ['DrDoS', 'Generic-DoS'][level2_dos_pred]
                path.append(f"Level 2: {level2_dos_name}")

                if level2_dos_pred == 0:  # DrDoS
                    level3_drdos_pred = results['level3_drdos']['preds'][i].item()
                    if level3_drdos_pred != -1:  # 有效预测
                        drdos_classes = ['DrDoS_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NTP',
                                         'DrDoS_NetBIOS', 'DrDoS_SNMP', 'DrDoS_SSDP', 'DrDoS_UDP']
                        level3_drdos_name = drdos_classes[level3_drdos_pred]
                        path.append(f"Level 3: {level3_drdos_name}")

                elif level2_dos_pred == 1:  # Generic-DoS
                    level3_generic_pred = results['level3_generic_dos']['preds'][i].item()
                    if level3_generic_pred != -1:  # 有效预测
                        generic_classes = ['Syn', 'UDP-lag']
                        level3_generic_name = generic_classes[level3_generic_pred]
                        path.append(f"Level 3: {level3_generic_name}")

        elif level1_pred == 2:  # Protocol
            level2_proto_pred = results['level2_protocol']['preds'][i].item()
            if level2_proto_pred != -1:  # 有效预测
                level2_proto_name = ['Portmap', 'TFTP'][level2_proto_pred]
                path.append(f"Level 2: {level2_proto_name}")

        # 绘制决策路径
        plt.subplot(n_samples, 1, i + 1)
        y_pos = np.arange(len(path))
        plt.barh(y_pos, [1] * len(path), align='center', alpha=0.5)
        plt.yticks(y_pos, path)
        target_name = f"Sample {i + 1}, True: {targets[i].item()}"
        plt.title(target_name)
        plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"分层决策路径图已保存至 {save_path}")
    else:
        plt.show()

    plt.close()


def plot_decision_tree_visualization(model,
                                     inputs: torch.Tensor,
                                     device: torch.device,
                                     save_path: Optional[str] = None) -> None:
    """
    生成树形图来可视化分层模型的决策路径

    参数:
        model: 分层模型
        inputs: 输入样本 [batch_size, seq_len, input_size]
        device: 设备
        save_path: 保存路径
    """
    try:
        import networkx as nx
    except ImportError:
        logger.error("请安装networkx库以支持决策树可视化: pip install networkx")
        return

    # 将输入移至设备
    inputs = inputs.to(device)

    # 获取分层预测
    with torch.no_grad():
        results = model.forward(inputs)

    # 创建有向图
    G = nx.DiGraph()

    # 添加根节点
    G.add_node("Root")

    # 添加一级节点
    for name in ["BENIGN", "DoS", "Protocol"]:
        G.add_node(name)
        G.add_edge("Root", name)

    # 添加DoS的二级节点
    for name in ["DrDoS", "Generic-DoS"]:
        G.add_node(name)
        G.add_edge("DoS", name)

    # 添加Protocol的二级节点
    for name in ["Portmap", "TFTP"]:
        G.add_node(name)
        G.add_edge("Protocol", name)

    # 添加DrDoS的三级节点
    for name in ["DrDoS_DNS", "DrDoS_LDAP", "DrDoS_MSSQL", "DrDoS_NTP",
                 "DrDoS_NetBIOS", "DrDoS_SNMP", "DrDoS_SSDP", "DrDoS_UDP"]:
        G.add_node(name)
        G.add_edge("DrDoS", name)

    # 添加Generic-DoS的三级节点
    for name in ["Syn", "UDP-lag"]:
        G.add_node(name)
        G.add_edge("Generic-DoS", name)

    # 计算每个节点的样本数量
    node_counts = {node: 0 for node in G.nodes}
    node_counts["Root"] = len(inputs)

    # 一级分类
    level1_preds = results['level1']['preds'].cpu().numpy()
    for pred in level1_preds:
        if pred == 0:
            node_counts["BENIGN"] += 1
        elif pred == 1:
            node_counts["DoS"] += 1
        elif pred == 2:
            node_counts["Protocol"] += 1

    # 二级分类 - DoS
    if 'level2_dos' in results:
        level2_dos_preds = results['level2_dos']['preds'].cpu().numpy()
        dos_mask = (level1_preds == 1)
        for i, pred in enumerate(level2_dos_preds):
            if dos_mask[i] and pred != -1:
                if pred == 0:
                    node_counts["DrDoS"] += 1
                elif pred == 1:
                    node_counts["Generic-DoS"] += 1

    # 二级分类 - Protocol
    if 'level2_protocol' in results:
        level2_protocol_preds = results['level2_protocol']['preds'].cpu().numpy()
        protocol_mask = (level1_preds == 2)
        for i, pred in enumerate(level2_protocol_preds):
            if protocol_mask[i] and pred != -1:
                if pred == 0:
                    node_counts["Portmap"] += 1
                elif pred == 1:
                    node_counts["TFTP"] += 1

    # 三级分类 - DrDoS
    if 'level3_drdos' in results:
        level3_drdos_preds = results['level3_drdos']['preds'].cpu().numpy()
        drdos_classes = ['DrDoS_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NTP',
                         'DrDoS_NetBIOS', 'DrDoS_SNMP', 'DrDoS_SSDP', 'DrDoS_UDP']

        for i, pred in enumerate(level3_drdos_preds):
            if pred != -1 and i < len(level1_preds) and level1_preds[i] == 1:  # DoS
                if i < len(level2_dos_preds) and level2_dos_preds[i] == 0:  # DrDoS
                    if 0 <= pred < len(drdos_classes):
                        node_counts[drdos_classes[pred]] += 1

    # 三级分类 - Generic-DoS
    if 'level3_generic_dos' in results:
        level3_generic_preds = results['level3_generic_dos']['preds'].cpu().numpy()
        generic_classes = ['Syn', 'UDP-lag']

        for i, pred in enumerate(level3_generic_preds):
            if pred != -1 and i < len(level1_preds) and level1_preds[i] == 1:  # DoS
                if i < len(level2_dos_preds) and level2_dos_preds[i] == 1:  # Generic-DoS
                    if 0 <= pred < len(generic_classes):
                        node_counts[generic_classes[pred]] += 1

    # 设置节点大小和标签
    node_sizes = []
    node_labels = {}

    for node in G.nodes:
        count = node_counts[node]
        node_sizes.append(1000 * (count + 1) / (len(inputs) + 1))  # 归一化并避免大小为0
        node_labels[node] = f"{node}\n({count})"

    # 绘制图
    plt.figure(figsize=(16, 12))
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')  # 需要安装 pygraphviz

    # 设置节点颜色
    node_colors = []
    for node in G.nodes:
        if node == "Root":
            node_colors.append('lightblue')
        elif node in ["BENIGN", "DoS", "Protocol"]:
            node_colors.append('lightgreen')
        elif node in ["DrDoS", "Generic-DoS", "Portmap", "TFTP"]:
            node_colors.append('lightcoral')
        else:
            node_colors.append('lightyellow')

    nx.draw(G, pos, with_labels=True, labels=node_labels,
            node_size=node_sizes, node_color=node_colors,
            font_size=10, font_weight='bold', arrowsize=20)

    plt.title("Hierarchical Model Decision Tree", fontsize=16)

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"决策树可视化已保存至 {save_path}")
    else:
        plt.show()

    plt.close()