#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pickle
import warnings
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
import joblib
import json

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# 定义分层标签映射系统
# 原始标签到一级分类的映射
LEVEL1_MAPPING = {
    "BENIGN": "BENIGN",
    "DrDoS_DNS": "DoS",
    "DrDoS_LDAP": "DoS",
    "DrDoS_MSSQL": "DoS",
    "DrDoS_NTP": "DoS",
    "DrDoS_NetBIOS": "DoS",
    "DrDoS_SNMP": "DoS",
    "DrDoS_SSDP": "DoS",
    "DrDoS_UDP": "DoS",
    "Syn": "DoS",
    "UDP-lag": "DoS",
    "Portmap": "Protocol",
    "TFTP": "Protocol"
}

# 原始标签到二级分类的映射
LEVEL2_MAPPING = {
    "DrDoS_DNS": "DrDoS",
    "DrDoS_LDAP": "DrDoS",
    "DrDoS_MSSQL": "DrDoS",
    "DrDoS_NTP": "DrDoS",
    "DrDoS_NetBIOS": "DrDoS",
    "DrDoS_SNMP": "DrDoS",
    "DrDoS_SSDP": "DrDoS",
    "DrDoS_UDP": "DrDoS",
    "Syn": "Generic-DoS",
    "UDP-lag": "Generic-DoS",
    "Portmap": "Portmap",
    "TFTP": "TFTP"
}

# 原始标签保持不变用于三级分类
LEVEL3_MAPPING = {
    "DrDoS_DNS": "DrDoS_DNS",
    "DrDoS_LDAP": "DrDoS_LDAP",
    "DrDoS_MSSQL": "DrDoS_MSSQL",
    "DrDoS_NTP": "DrDoS_NTP",
    "DrDoS_NetBIOS": "DrDoS_NetBIOS",
    "DrDoS_SNMP": "DrDoS_SNMP",
    "DrDoS_SSDP": "DrDoS_SSDP",
    "DrDoS_UDP": "DrDoS_UDP",
    "Syn": "Syn",
    "UDP-lag": "UDP-lag"
}

# 类别映射字典
CLASS_MAPPINGS = {
    1: {  # 一级分类
        "BENIGN": 0,
        "DoS": 1,
        "Protocol": 2
    },
    2: {  # 二级分类 - DoS
        "DrDoS": 0,
        "Generic-DoS": 1
    },
    3: {  # 二级分类 - Protocol
        "Portmap": 0,
        "TFTP": 1
    },
    4: {  # 三级分类 - DrDoS
        "DrDoS_DNS": 0,
        "DrDoS_LDAP": 1,
        "DrDoS_MSSQL": 2,
        "DrDoS_NTP": 3,
        "DrDoS_NetBIOS": 4,
        "DrDoS_SNMP": 5,
        "DrDoS_SSDP": 6,
        "DrDoS_UDP": 7
    },
    5: {  # 三级分类 - Generic-DoS
        "Syn": 0,
        "UDP-lag": 1
    }
}

# 逆向映射 - 用于从数值标签恢复类别名
INVERSE_CLASS_MAPPINGS = {
    level: {v: k for k, v in mapping.items()}
    for level, mapping in CLASS_MAPPINGS.items()
}


class DataProcessor:
    """
    数据处理类，负责加载、清洗和特征工程
    """

    def __init__(self, data_path: str, n_workers: int = 4, n_components: int = 25):
        """
        初始化数据处理器
        Args:
            data_path: 数据文件路径
            n_workers: 并行处理的工作进程数
            n_components: PCA降维后的维度
        """
        self.data_path = data_path
        self.n_workers = n_workers
        self.column_map = None  # 用于存储列名映射

        # PCA相关参数
        self.n_components = n_components
        self.pca_model = None

        # 存储特征提取器
        self.scalers = {}
        self.encoders = {}

        # 不带空格版本的特征列表
        self.base_features = [
            'Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets',
            'FlowBytes/s', 'FlowPackets/s', 'FwdPacketLengthMax', 'FwdPacketLengthMin',
            'FwdPacketLengthMean', 'FwdPacketLengthStd', 'BwdPacketLengthMax',
            'BwdPacketLengthMin', 'BwdPacketLengthMean', 'BwdPacketLengthStd',
            'PacketLengthVariance', 'FlowIATMin', 'FlowIATMax', 'FlowIATMean',
            'FlowIATStd', 'FwdIATMean', 'FwdIATStd', 'FwdIATMax', 'FwdIATMin',
            'BwdIATMean', 'BwdIATStd', 'BwdIATMax', 'BwdIATMin', 'FwdPSHFlags',
            'BwdPSHFlags', 'FwdURGFlags', 'BwdURGFlags', 'FwdHeaderLength',
            'BwdHeaderLength', 'FwdPackets/s', 'BwdPackets/s', 'Init_Win_bytes_forward',
            'Init_Win_bytes_backward', 'min_seg_size_forward', 'SubflowFwdBytes',
            'SubflowBwdBytes', 'AveragePacketSize', 'AvgFwdSegmentSize',
            'AvgBwdSegmentSize', 'ActiveMean', 'ActiveMin', 'ActiveMax', 'ActiveStd',
            'IdleMean', 'IdleMin', 'IdleMax', 'IdleStd', 'Timestamp',
        ]

        # 需要进行对数转换的特征
        self.log_transform_features_base = [
            'FlowBytes/s', 'FlowPackets/s', 'FwdPackets/s', 'BwdPackets/s',
            'FlowDuration', 'PacketLengthVariance'
        ]

        # 类别特征
        self.categorical_features_base = ['Protocol']

    def normalize_column_names(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        创建标准化的列名映射，将所有列名无空格版本作为键，原始列名作为值
        """
        column_map = {}
        for col in df.columns:
            # 移除所有空格后的列名作为键
            normalized_key = col.replace(" ", "")
            column_map[normalized_key] = col

        logger.info(f"创建了列名映射，共 {len(column_map)} 个列")
        return column_map

    def get_actual_column_name(self, normalized_name: str) -> Optional[str]:
        """根据标准化名称获取数据集中的实际列名"""
        if self.column_map is None:
            logger.warning("列名映射尚未初始化")
            return None
        return self.column_map.get(normalized_name)

    def load_data(self) -> pd.DataFrame:
        """加载CSV数据文件，只读取需要的列"""
        try:
            logger.info(f"开始读取文件: {os.path.basename(self.data_path)}")

            # 先读取文件头，获取列名
            try:
                header_df = pd.read_csv(self.data_path, nrows=0)
            except Exception as e:
                logger.warning(f"默认引擎读取头部失败: {e}，切换 Python 引擎")
                header_df = pd.read_csv(self.data_path, nrows=0, engine='python')

            # 先创建临时列名映射，用于识别需要的列
            temp_column_map = {}
            for col in header_df.columns:
                normalized_key = col.replace(" ", "")
                temp_column_map[normalized_key] = col

            # 确定要读取的列名
            usecols = []

            # 添加特征列
            for base_col in self.base_features:
                if base_col in temp_column_map:
                    usecols.append(temp_column_map[base_col])

            # 添加标签列
            label_col = None
            for label_name in ['Label', 'label']:
                if label_name in temp_column_map:
                    label_col = temp_column_map[label_name]
                    usecols.append(label_col)
                    break

            if not label_col:
                # 尝试其他方式找标签列
                for col in header_df.columns:
                    if 'label' in col.lower():
                        usecols.append(col)
                        logger.info(f"使用替代标签列: {col}")
                        break

            if not usecols:
                logger.error(f"无法识别需要读取的列")
                return pd.DataFrame()

            logger.info(f"将读取 {len(usecols)} 列: {len(usecols) - 1} 个特征列和 1 个标签列")

            # 分块读取，考虑到文件可能很大(150000行左右)
            chunks = None
            try:
                chunks = pd.read_csv(self.data_path, chunksize=10000, usecols=usecols, on_bad_lines='skip')
            except Exception as e:
                logger.warning(f"C 引擎读取失败: {e}，切换 Python 引擎")
                chunks = pd.read_csv(self.data_path, engine='python', chunksize=10000, usecols=usecols,
                                     on_bad_lines='skip')

            chunk_list = []
            for chunk in chunks:
                chunk_list.append(chunk)

            if not chunk_list:
                return pd.DataFrame()

            df = pd.concat(chunk_list, ignore_index=True)
            logger.info(f"文件 {os.path.basename(self.data_path)} 读取完成，shape={df.shape}")

            # 创建列名映射
            self.column_map = self.normalize_column_names(df)

            return df

        except Exception as e:
            logger.error(f"加载文件 {self.data_path} 时出错: {e}")
            return pd.DataFrame()

    def dropna_in_chunks(self, df, chunk_size=100000):
        """分块处理NaN值，避免内存问题"""
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size].dropna()
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗：去重、处理缺失值、异常值处理"""
        logger.info("开始数据清洗")

        # 1. 移除重复记录
        df_clean = df.drop_duplicates()
        logger.info(f"移除重复记录后剩余 {len(df_clean)} 条记录，df_clean.shape={df_clean.shape}")
        after_dedup = len(df_clean)

        # 获取标签列名
        label_col = None
        for possible_label in ['Label', 'label']:
            possible_col = self.get_actual_column_name(possible_label)
            if possible_col and possible_col in df_clean.columns:
                label_col = possible_col
                break

        if not label_col:
            # 尝试其他方式找标签列
            for col in df_clean.columns:
                if 'label' in col.lower():
                    label_col = col
                    logger.info(f"使用替代标签列: {col}")
                    break

        # 2. 处理缺失值
        df_clean = self.dropna_in_chunks(df_clean)
        logger.info(f"删除缺失值后剩余 {len(df_clean)} 条记录 (删除了 {after_dedup - len(df_clean)} 条)")

        # 对分类特征使用众数填充
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        cat_cols = df_clean.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if col != label_col:  # 不处理标签列
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        # 3. 异常值处理（使用IQR方法）
        for col in numeric_cols:
            if col != label_col:  # 不处理标签列
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                # 将异常值限制在边界范围内
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

        logger.info("数据清洗完成")
        return df_clean

    def preprocess_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """特征预处理：独热编码、标准化、归一化、PCA降维"""
        import joblib
        import json
        import os

        logger.info("开始特征预处理")
        df_processed = df.copy()

        # ========== Step 1: 处理类别特征（独热编码） ==========
        for base_col in self.categorical_features_base:
            actual_col = self.get_actual_column_name(base_col)
            if actual_col and actual_col in df_processed.columns:
                already_encoded = any(col.startswith(f"{base_col}_") for col in df_processed.columns)
                if already_encoded:
                    logger.info(f"检测到特征 {base_col} 已经完成独热编码，跳过")
                    continue

                if fit:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded_data = encoder.fit_transform(df_processed[[actual_col]])
                    self.encoders[base_col] = encoder
                else:
                    encoder = self.encoders.get(base_col)
                    if encoder is None:
                        logger.warning(f"找不到特征 {base_col} 的编码器，跳过处理")
                        continue
                    encoded_data = encoder.transform(df_processed[[actual_col]])

                encoded_cols = [f"{base_col}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df_processed.index)

                df_processed = df_processed.drop(actual_col, axis=1)
                df_processed = pd.concat([df_processed, encoded_df], axis=1)

        # ========== Step 2: 获取标签列 ==========
        label_col = None
        for possible_label in ['Label', 'label']:
            actual_col = self.get_actual_column_name(possible_label)
            if actual_col and actual_col in df_processed.columns:
                label_col = actual_col
                break

        if not label_col:
            for col in df_processed.columns:
                if 'label' in col.lower():
                    label_col = col
                    logger.info(f"使用替代标签列: {col}")
                    break

        # ========== Step 3: 数值特征归一化 ==========
        numeric_cols = df_processed.select_dtypes(include=['number']).columns.tolist()
        if label_col and label_col in numeric_cols:
            numeric_cols.remove(label_col)

        os.makedirs('models', exist_ok=True)  # 创建模型目录

        if fit:
            self.numeric_feature_order = numeric_cols
            self.minmax_scaler = MinMaxScaler()
            df_processed[numeric_cols] = self.minmax_scaler.fit_transform(df_processed[numeric_cols])

            # 保存 scaler 和特征顺序
            joblib.dump(self.minmax_scaler, 'models/minmax_scaler.pkl')
            with open('models/numeric_feature_order.json', 'w') as f:
                json.dump(self.numeric_feature_order, f)
        else:
            try:
                self.minmax_scaler = joblib.load('models/minmax_scaler.pkl')
                with open('models/numeric_feature_order.json', 'r') as f:
                    self.numeric_feature_order = json.load(f)
            except Exception as e:
                raise RuntimeError("验证阶段缺少 scaler 或特征顺序，并且加载失败") from e

            numeric_cols = self.numeric_feature_order
            df_processed[numeric_cols] = self.minmax_scaler.transform(df_processed[numeric_cols])

        # ========== Step 4: PCA 降维 ==========
        if fit:
            logger.info(f"执行 PCA 降维: 从 {len(numeric_cols)} 维降至 {self.n_components} 维")
            self.pca_model = PCA(n_components=self.n_components)
            pca_result = self.pca_model.fit_transform(df_processed[numeric_cols])
            explained_var = sum(self.pca_model.explained_variance_ratio_) * 100
            logger.info(f"PCA降维后保留信息量: {explained_var:.2f}%")

            # 保存 PCA 模型
            joblib.dump(self.pca_model, 'models/pca_model.pkl')
        else:
            try:
                self.pca_model = joblib.load('models/pca_model.pkl')
            except Exception as e:
                raise RuntimeError("验证阶段缺少 PCA 模型，并且加载失败") from e

            pca_result = self.pca_model.transform(df_processed[numeric_cols])

        # ========== Step 5: 构造结果 ==========
        pca_columns = [f'pca_component_{i + 1}' for i in range(self.n_components)]
        pca_df = pd.DataFrame(pca_result, columns=pca_columns, index=df_processed.index)

        if label_col:
            result_df = pd.concat([pca_df, df_processed[[label_col]]], axis=1)
        else:
            result_df = pca_df

        logger.info(f"PCA降维完成，最终特征维数: {result_df.shape[1]}")
        return result_df

    def create_hierarchical_labels(self, original_labels: np.ndarray) -> Dict[int, np.ndarray]:
        """
        从原始标签创建分层标签
        参数:
            original_labels: 原始标签数组或列表
        返回:
            分层标签的字典，键为层级ID
        """
        # 将数值标签转换为字符串标签
        label_map = {idx: label for label, idx in self.label_map.items()}
        str_labels = np.array([label_map[label] for label in original_labels])

        # 创建分层标签字典
        hierarchical_labels = {}

        # 第一级: BENIGN / DoS / Protocol
        level1_labels = np.array([LEVEL1_MAPPING.get(label, "Unknown") for label in str_labels])
        hierarchical_labels[1] = np.array([CLASS_MAPPINGS[1].get(label, -1) for label in level1_labels])

        # 第二级: DrDoS / Generic-DoS (仅对DoS标签)
        level2_dos_mask = level1_labels == "DoS"
        if np.any(level2_dos_mask):
            level2_dos_labels = np.full(len(str_labels), -1)  # 默认为-1（不适用）
            dos_str_labels = np.array([LEVEL2_MAPPING.get(label, "Unknown") for label in str_labels[level2_dos_mask]])
            level2_dos_labels[level2_dos_mask] = np.array(
                [CLASS_MAPPINGS[2].get(label, -1) for label in dos_str_labels])
            hierarchical_labels[2] = level2_dos_labels

        # 第二级: Portmap / TFTP (仅对Protocol标签)
        level2_proto_mask = level1_labels == "Protocol"
        if np.any(level2_proto_mask):
            level2_proto_labels = np.full(len(str_labels), -1)  # 默认为-1（不适用）
            proto_str_labels = str_labels[level2_proto_mask]
            level2_proto_labels[level2_proto_mask] = np.array(
                [CLASS_MAPPINGS[3].get(label, -1) for label in proto_str_labels])
            hierarchical_labels[3] = level2_proto_labels

        # 第三级: DrDoS细分为8类
        drdos_mask = np.array([label.startswith("DrDoS_") for label in str_labels])
        if np.any(drdos_mask):
            level3_drdos_labels = np.full(len(str_labels), -1)
            level3_drdos_labels[drdos_mask] = np.array(
                [CLASS_MAPPINGS[4].get(label, -1) for label in str_labels[drdos_mask]])
            hierarchical_labels[4] = level3_drdos_labels

        # 第三级: Generic-DoS细分为2类
        generic_dos_labels = ["Syn", "UDP-lag"]
        generic_dos_mask = np.isin(str_labels, generic_dos_labels)
        if np.any(generic_dos_mask):
            level3_generic_dos_labels = np.full(len(str_labels), -1)
            level3_generic_dos_labels[generic_dos_mask] = np.array(
                [CLASS_MAPPINGS[5].get(label, -1) for label in str_labels[generic_dos_mask]])
            hierarchical_labels[5] = level3_generic_dos_labels

        return hierarchical_labels

    def process_data_pipeline(self, train: bool = True, hierarchical: bool = True) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, Dict[int, np.ndarray]]]:
        """
        完整数据处理流水线
        参数:
            train: 是否为训练模式
            hierarchical: 是否返回分层标签
        返回:
            如果hierarchical=False: (特征, 标签)
            如果hierarchical=True: (特征, 分层标签字典)
        """
        # 1. 加载数据
        df = self.load_data()

        # 2. 数据清洗
        df_clean = self.clean_data(df)

        # 3. 特征预处理
        df_processed = self.preprocess_features(df_clean, fit=train)

        # 4. 提取特征和标签
        # 获取标签列
        label_col = None
        for possible_label in ['Label', 'label']:
            possible_col = self.get_actual_column_name(possible_label)
            if possible_col and possible_col in df_processed.columns:
                label_col = possible_col
                break

        if not label_col:
            for col in df_processed.columns:
                if 'label' in col.lower():
                    label_col = col
                    logger.info(f"使用替代标签列: {col}")
                    break

        if not label_col:
            logger.error("找不到标签列")
            return np.array([]), np.array([]) if not hierarchical else np.array([]), {}

        feature_cols = df_processed.columns.tolist()
        feature_cols.remove(label_col)  # 移除标签列

        # 确保特征全是数值类型
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(df_processed[col]):
                logger.warning(f"将非数值特征 {col} 转换为数值类型")
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        # 提取特征
        X = df_processed[feature_cols].values

        # 确保标签是数值类型
        y_data = df_processed[label_col]

        # 检查标签类型并转换
        if pd.api.types.is_object_dtype(y_data):
            logger.info("检测到标签为非数值类型，进行转换")
            # 创建标签映射字典
            unique_labels = np.unique(y_data)
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
            logger.info(f"标签映射: {self.label_map}")

            # 转换标签为数值
            y = np.array([self.label_map[label] for label in y_data], dtype=np.int64)

            # 保存标签映射
            with open('models/label_map.json', 'w') as f:
                json.dump(self.label_map, f)
        else:
            y = y_data.values.astype(np.int64)
            # 如果已经是数值，尝试加载标签映射
            try:
                with open('models/label_map.json', 'r') as f:
                    self.label_map = json.load(f)
            except:
                logger.warning("找不到标签映射文件，将无法创建分层标签")
                self.label_map = {str(i): i for i in range(len(np.unique(y)))}

        logger.info(f"标签类型: {y.dtype}, 标签唯一值: {np.unique(y)}")

        # 检查特征是否有NaN或无穷值
        if np.isnan(X).any() or np.isinf(X).any():
            logger.warning("特征数据中包含NaN或无穷值，将其替换为0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 根据需要返回分层标签或普通标签
        if hierarchical:
            hierarchical_labels = self.create_hierarchical_labels(y)
            return X, hierarchical_labels
        else:
            return X, y

    def save_preprocessors(self, save_path: str):
        """保存预处理器，包括PCA模型和标签映射"""
        preprocessors = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'pca_model': self.pca_model,
            'n_components': self.n_components,
            'label_map': getattr(self, 'label_map', {})
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(preprocessors, f)

        logger.info(f"预处理器已保存至 {save_path}")

    def load_preprocessors(self, load_path: str):
        """加载预处理器，包括PCA模型和标签映射"""
        with open(load_path, 'rb') as f:
            preprocessors = pickle.load(f)

        self.scalers = preprocessors.get('scalers', {})
        self.encoders = preprocessors.get('encoders', {})
        self.pca_model = preprocessors.get('pca_model')
        self.n_components = preprocessors.get('n_components', 20)
        self.label_map = preprocessors.get('label_map', {})

        logger.info(f"预处理器已从 {load_path} 加载，PCA维度: {self.n_components}")


class DDoSDataset(Dataset):
    """DDoS攻击预测的PyTorch数据集类"""

    def __init__(self,
                 data_path: str,
                 preprocessor_path: Optional[str] = None,
                 train: bool = True,
                 transform: Optional[Any] = None,
                 hierarchical: bool = True,
                 level: int = 1):
        """
        初始化数据集
        参数:
            data_path: 数据文件路径
            preprocessor_path: 预处理器路径
            train: 是否为训练模式
            transform: 数据变换函数
            hierarchical: 是否使用分层标签
            level: 要返回的分层标签级别(1-5)
        """
        self.transform = transform
        self.hierarchical = hierarchical
        self.level = level

        # 初始化处理器
        self.processor = DataProcessor(
            data_path=data_path,
            n_workers=1
        )

        # 如果是训练模式且提供了预处理器路径，在处理后保存预处理器
        if train and preprocessor_path:
            if self.hierarchical:
                self.features, self.hierarchical_labels = self.processor.process_data_pipeline(train=True,
                                                                                               hierarchical=True)
                self.labels = self.hierarchical_labels.get(self.level, np.array([]))
            else:
                self.features, self.labels = self.processor.process_data_pipeline(train=True, hierarchical=False)

            logger.info(f"保存预处理器到: {preprocessor_path}")
            # 确保目录存在
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            self.processor.save_preprocessors(preprocessor_path)
            logger.info("预处理器保存成功")
        # 如果是预测模式且提供了预处理器路径，先加载预处理器再处理数据
        elif not train and preprocessor_path:
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"预处理器文件不存在: {preprocessor_path}")
            logger.info(f"加载预处理器从: {preprocessor_path}")
            self.processor.load_preprocessors(preprocessor_path)
            logger.info("预处理器加载成功")

            if self.hierarchical:
                self.features, self.hierarchical_labels = self.processor.process_data_pipeline(train=False,
                                                                                               hierarchical=True)
                self.labels = self.hierarchical_labels.get(self.level, np.array([]))
            else:
                self.features, self.labels = self.processor.process_data_pipeline(train=False, hierarchical=False)
        else:
            # 无预处理器路径的情况
            if self.hierarchical:
                self.features, self.hierarchical_labels = self.processor.process_data_pipeline(train=train,
                                                                                               hierarchical=True)
                self.labels = self.hierarchical_labels.get(self.level, np.array([]))
            else:
                self.features, self.labels = self.processor.process_data_pipeline(train=train, hierarchical=False)

            if train:
                logger.warning("训练模式未提供预处理器保存路径，将无法在预测时使用一致的预处理")

        # 确保数据不为空
        if len(self.features) == 0 or len(self.labels) == 0:
            raise ValueError("处理数据失败，未能生成有效的特征和标签")

        # 过滤掉不适用的样本（标签为-1的样本）
        if self.hierarchical:
            valid_mask = self.labels != -1
            self.features = self.features[valid_mask]
            self.labels = self.labels[valid_mask]

            # 更新其他层级的标签
            if hasattr(self, 'hierarchical_labels'):
                for level, labels in self.hierarchical_labels.items():
                    self.hierarchical_labels[level] = labels[valid_mask]

        # 检查和打印特征和标签的数据类型
        logger.info(f"特征数据类型: {self.features.dtype}")
        logger.info(f"标签数据类型: {self.labels.dtype}")

        # 明确转换为浮点数(特征)和整数(标签)
        try:
            self.features = np.array(self.features, dtype=np.float32)
            self.labels = np.array(self.labels, dtype=np.int64)

            # 转换为PyTorch张量
            self.features = torch.from_numpy(self.features).float()
            self.labels = torch.from_numpy(self.labels).long().unsqueeze(1)

            logger.info(f"转换后特征形状: {self.features.shape}, 类型: {self.features.dtype}")
            logger.info(f"转换后标签形状: {self.labels.shape}, 类型: {self.labels.dtype}")

        except Exception as e:
            # 如果转换失败，尝试逐行转换
            logger.error(f"标准转换失败: {e}")
            logger.info("尝试逐行转换...")

            # 创建空数组
            float_features = np.zeros((len(self.features), self.features.shape[1]), dtype=np.float32)
            int_labels = np.zeros(len(self.labels), dtype=np.int64)

            # 逐行转换
            for i in range(len(self.features)):
                float_features[i] = [float(val) for val in self.features[i]]

            for i in range(len(self.labels)):
                int_labels[i] = int(self.labels[i])

            # 赋值回原变量并转换为PyTorch张量
            self.features = torch.from_numpy(float_features).float()
            self.labels = torch.from_numpy(int_labels).long().unsqueeze(1)

            logger.info(f"逐行转换后特征形状: {self.features.shape}, 类型: {self.features.dtype}")
            logger.info(f"逐行转换后标签形状: {self.labels.shape}, 类型: {self.labels.dtype}")

    def __len__(self):
        """返回数据集长度"""
        return len(self.features)

    def __getitem__(self, idx):
        """
        获取单个样本
        返回:
            x: 特征张量，形状为 (feature_size, 1)
            y: 标签张量
        """
        x = self.features[idx].unsqueeze(-1)  # 添加最后一个维度，形状变为 (feature_size, 1)
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    def set_level(self, level):
        """设置要返回的分层标签级别"""
        if not self.hierarchical:
            logger.warning("数据集未使用分层标签，无法设置级别")
            return

        if level not in self.hierarchical_labels:
            logger.warning(f"级别 {level} 不存在，可用级别: {list(self.hierarchical_labels.keys())}")
            return

        self.level = level
        self.labels = self.hierarchical_labels[level]

        # 过滤掉不适用的样本（标签为-1的样本）
        valid_mask = self.labels != -1
        self.features = self.features[valid_mask]
        self.labels = self.labels[valid_mask]

        # 转换为PyTorch张量
        self.labels = torch.from_numpy(self.labels).long().unsqueeze(1)
        logger.info(f"已设置标签级别为 {level}，有效样本数: {len(self.labels)}")


def create_dataloader(dataset: DDoSDataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4):
    """创建数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def create_hierarchical_dataloaders(
        train_path: str,
        val_path: str,
        preprocessor_path: str = "models/preprocessors.pkl",
        batch_size: int = 32,
        num_workers: int = 4
) -> Dict[int, Tuple[DataLoader, DataLoader]]:
    """
    为每个分类级别创建训练和验证数据加载器

    参数:
        train_path: 训练数据路径
        val_path: 验证数据路径
        preprocessor_path: 预处理器保存路径
        batch_size: 批次大小
        num_workers: 数据加载线程数

    返回:
        包含每个级别数据加载器的字典 {级别: (训练加载器, 验证加载器)}
    """
    # 创建主数据集并保存预处理器
    train_dataset = DDoSDataset(
        data_path=train_path,
        preprocessor_path=preprocessor_path,
        train=True,
        hierarchical=True
    )

    hierarchy_levels = list(train_dataset.hierarchical_labels.keys())
    logger.info(f"创建分层数据加载器，级别: {hierarchy_levels}")

    dataloaders = {}

    for level in hierarchy_levels:
        # 创建训练数据集
        level_train_dataset = DDoSDataset(
            data_path=train_path,
            preprocessor_path=preprocessor_path,
            train=False,  # 已经训练过预处理器，直接加载
            hierarchical=True,
            level=level
        )

        # 创建验证数据集
        level_val_dataset = DDoSDataset(
            data_path=val_path,
            preprocessor_path=preprocessor_path,
            train=False,
            hierarchical=True,
            level=level
        )

        # 创建数据加载器
        train_loader = create_dataloader(
            level_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        val_loader = create_dataloader(
            level_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        dataloaders[level] = (train_loader, val_loader)

    return dataloaders


# 测试用例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 初始化数据处理器
    data_path = r"C:\Users\17380\processed_dataset_modified.csv"
    processor = DataProcessor(data_path=data_path)

    # 测试分层数据处理
    X, hierarchical_labels = processor.process_data_pipeline(train=True, hierarchical=True)

    print(f"特征数据形状: {X.shape}")
    print("分层标签概述:")
    for level, labels in hierarchical_labels.items():
        unique_values, counts = np.unique(labels[labels != -1], return_counts=True)
        print(f"级别 {level}: 有效样本数 {np.sum(labels != -1)}, 类别数 {len(unique_values)}")
        print(f"类别分布: {dict(zip(unique_values, counts))}")

    # 测试分层数据集
    try:
        dataset = DDoSDataset(data_path=data_path, train=True, hierarchical=True, level=1)
        print(f"级别1数据集大小: {len(dataset)}")
        x, y = dataset[0]
        print(f"样本特征形状: {x.shape}")  # 应该是 (feature_size, 1)
        print(f"样本标签形状: {y.shape}")

        # 切换到其他级别
        dataset.set_level(2)  # 切换到DoS分类
        print(f"级别2数据集大小: {len(dataset)}")

        # 测试数据加载器
        dataloader = create_dataloader(dataset, batch_size=32)
        batch_x, batch_y = next(iter(dataloader))
        print(f"批次特征形状: {batch_x.shape}")  # 应该是 (batch_size, feature_size, 1)
        print(f"批次标签形状: {batch_y.shape}")

        # 测试分层数据加载器创建
        train_path = data_path
        val_path = data_path  # 测试时使用相同数据
        hierarchical_loaders = create_hierarchical_dataloaders(
            train_path=train_path,
            val_path=val_path,
            preprocessor_path="models/test_preprocessors.pkl",
            batch_size=32
        )

        print("分层数据加载器:")
        for level, (train_loader, val_loader) in hierarchical_loaders.items():
            print(f"级别 {level}:")
            print(f"  训练数据大小: {len(train_loader.dataset)}")
            print(f"  验证数据大小: {len(val_loader.dataset)}")

    except Exception as e:
        print(f"数据集测试失败: {str(e)}")