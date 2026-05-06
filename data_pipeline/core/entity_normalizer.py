"""
实体名称标准化和缩写识别模块
用于跨文献识别同一实体（包括缩写和全称）
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

@dataclass
class EntityNameVariant:
    """实体名称变体"""
    canonical_name: str  # 标准名称
    variants: Set[str]  # 所有变体（包括缩写、全称、别名）
    
    def matches(self, name: str) -> bool:
        """检查名称是否匹配此实体"""
        normalized = self._normalize_name(name)
        return normalized in {self._normalize_name(v) for v in self.variants}
    
    @staticmethod
    def _normalize_name(name: str) -> str:
        """标准化名称用于比较"""
        # 转小写，去除空格、连字符、下划线
        normalized = name.lower().strip()
        normalized = re.sub(r'[\s\-_]+', '', normalized)
        # 去除特殊字符但保留字母数字
        normalized = re.sub(r'[^\w]', '', normalized)
        return normalized


class EntityNormalizer:
    """实体名称标准化器"""
    
    def __init__(self):
        """初始化标准化器"""
        # 光电领域常见缩写和全称映射
        self.abbreviation_map = self._build_abbreviation_map()
        
        # 化学式标准化规则
        self.chemical_patterns = self._build_chemical_patterns()
        
        # 实体变体字典（在运行时学习）
        self.entity_variants: Dict[str, EntityNameVariant] = {}
    
    def _build_abbreviation_map(self) -> Dict[str, List[str]]:
        """构建缩写和全称映射表"""
        return {
            # 性能指标 (Metrics)
            "pce": ["Power Conversion Efficiency", "Photovoltaic Efficiency", 
                    "Solar Cell Efficiency", "Conversion Efficiency"],
            "voc": ["Open-Circuit Voltage", "Open Circuit Voltage", "V_oc"],
            "jsc": ["Short-Circuit Current Density", "Short Circuit Current", "J_sc"],
            "ff": ["Fill Factor"],
            "eqe": ["External Quantum Efficiency"],
            "iqe": ["Internal Quantum Efficiency"],
            "ipce": ["Incident Photon-to-Current Efficiency"],
            "plqy": ["Photoluminescence Quantum Yield"],
            
            # 材料 (Materials)
            "mapbi3": ["Methylammonium Lead Iodide", "MAPbI3", "CH3NH3PbI3"],
            "fapbi3": ["Formamidinium Lead Iodide", "FAPbI3", "HC(NH2)2PbI3"],
            "cspbi3": ["Cesium Lead Iodide", "CsPbI3"],
            "ito": ["Indium Tin Oxide", "Indium-Tin Oxide"],
            "fto": ["Fluorine-Doped Tin Oxide", "Fluorine Doped Tin Oxide"],
            "pcbm": ["Phenyl-C61-Butyric Acid Methyl Ester", "[6,6]-Phenyl C61 butyric acid methyl ester"],
            "p3ht": ["Poly(3-hexylthiophene)", "Poly(3-hexylthiophene-2,5-diyl)"],
            "ptb7": ["Poly[[4,8-bis[(2-ethylhexyl)oxy]benzo[1,2-b:4,5-b']dithiophene-2,6-diyl][3-fluoro-2-[(2-ethylhexyl)carbonyl]thieno[3,4-b]thiophenediyl]]"],
            "pm6": ["Poly[(2,6-(4,8-bis(5-(2-ethylhexyl-3-fluoro)thiophen-2-yl)-benzo[1,2-b:4,5-b']dithiophene))-alt-(5,5-(1',3'-di-2-thienyl-5',7'-bis(2-ethylhexyl)benzo[1',2'-c:4',5'-c']dithiophene-4,8-dione)]"],
            "y6": ["2,2'-((2Z,2'Z)-((12,13-bis(2-ethylhexyl)-3,9-diundecyl-12,13-dihydro-[1,2,5]thiadiazolo[3,4-e]thieno[2'',3'':4',5']thieno[2',3':4,5]pyrrolo[3,2-g]thieno[2',3':4,5]thieno[3,2-b]indole-2,10-diyl)bis(methanylylidene))bis(5,6-difluoro-3-oxo-2,3-dihydro-1H-indene-2,1-diylidene))dimalononitrile"],
            "spiro-ometad": ["2,2',7,7'-Tetrakis[N,N-di(4-methoxyphenyl)amino]-9,9'-spirobifluorene", "Spiro-MeOTAD"],
            
            # 层和器件 (Devices/Layers)
            "etl": ["Electron Transport Layer", "Electron-Transport Layer"],
            "htl": ["Hole Transport Layer", "Hole-Transport Layer"],
            "perc": ["Passivated Emitter and Rear Cell", "Passivated Emitter Rear Cell"],
            "hjt": ["Heterojunction", "Hetero-Junction"],
            "tandem": ["Tandem Solar Cell", "Multi-Junction Solar Cell"],
            
            # 工艺 (Processes)
            "pecvd": ["Plasma-Enhanced Chemical Vapor Deposition", "Plasma Enhanced CVD"],
            "ald": ["Atomic Layer Deposition"],
            "pvd": ["Physical Vapor Deposition"],
            "cbd": ["Chemical Bath Deposition"],
            
            # 测量方法 (Measurements)
            "xrd": ["X-Ray Diffraction", "X Ray Diffraction"],
            "sem": ["Scanning Electron Microscopy", "Scanning Electron Microscope"],
            "tem": ["Transmission Electron Microscopy", "Transmission Electron Microscope"],
            "afm": ["Atomic Force Microscopy", "Atomic Force Microscope"],
            "xps": ["X-Ray Photoelectron Spectroscopy"],
            "ftir": ["Fourier Transform Infrared Spectroscopy"],
            "uv-vis": ["Ultraviolet-Visible Spectroscopy", "UV-Vis Spectroscopy"],
            "pl": ["Photoluminescence"],
            "el": ["Electroluminescence"],
            
            # ML算法 (ML Algorithms)
            "rf": ["Random Forest"],
            "svm": ["Support Vector Machine"],
            "nn": ["Neural Network"],
            "cnn": ["Convolutional Neural Network"],
            "rnn": ["Recurrent Neural Network"],
            "gnn": ["Graph Neural Network"],
            "gbdt": ["Gradient Boosting Decision Tree"],
            "xgboost": ["Extreme Gradient Boosting"],
            "dnn": ["Deep Neural Network"],
        }
    
    def _build_chemical_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """构建化学式识别模式"""
        return [
            # 钙钛矿材料模式
            (re.compile(r'(MA|FA|Cs|Rb|K)Pb(I|Br|Cl)3', re.IGNORECASE), 'perovskite'),
            # 氧化物模式
            (re.compile(r'(Ti|Sn|Zn|Ni|Cu)O2?', re.IGNORECASE), 'oxide'),
            # 金属
            (re.compile(r'^(Au|Ag|Al|Cu|Pt|Pd)$', re.IGNORECASE), 'metal'),
        ]
    
    def normalize_entity_name(self, name: str, entity_type: str = None) -> str:
        """
        标准化实体名称
        
        Args:
            name: 原始名称
            entity_type: 实体类型（可选）
        
        Returns:
            标准化后的名称
        """
        # 基本清理
        cleaned = name.strip()
        
        # 去除LaTeX格式
        cleaned = self._remove_latex_formatting(cleaned)
        
        # 检查缩写映射
        normalized_for_lookup = cleaned.lower().replace('-', '').replace('_', '').replace(' ', '')
        
        # 查找缩写映射
        for abbrev, full_names in self.abbreviation_map.items():
            # 检查是否匹配缩写
            if normalized_for_lookup == abbrev:
                return full_names[0]  # 返回首选全称
            
            # 检查是否匹配某个全称
            for full_name in full_names:
                full_normalized = full_name.lower().replace('-', '').replace('_', '').replace(' ', '')
                if normalized_for_lookup == full_normalized:
                    return full_names[0]  # 统一返回首选全称
        
        # 化学式标准化
        for pattern, chem_type in self.chemical_patterns:
            if pattern.match(cleaned):
                # 保持化学式的原始大小写
                return cleaned
        
        # 如果没有找到映射，返回清理后的名称
        return cleaned
    
    def _remove_latex_formatting(self, text: str) -> str:
        """去除LaTeX格式"""
        # 去除$符号
        text = text.replace('$', '')
        # 去除常见LaTeX命令
        text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
        # 处理下标和上标
        text = re.sub(r'_\{([^}]+)\}', r'_\1', text)
        text = re.sub(r'\^\{([^}]+)\}', r'^\1', text)
        # 去除反斜杠
        text = text.replace('\\', '')
        return text.strip()
    
    def are_same_entity(self, name1: str, name2: str, type1: str = None, type2: str = None) -> bool:
        """
        判断两个名称是否指向同一实体
        
        Args:
            name1: 第一个名称
            name2: 第二个名称
            type1: 第一个实体类型
            type2: 第二个实体类型
        
        Returns:
            是否为同一实体
        """
        # 类型不同则不是同一实体
        if type1 and type2 and type1 != type2:
            return False
        
        # 标准化后比较
        norm1 = self.normalize_entity_name(name1, type1)
        norm2 = self.normalize_entity_name(name2, type2)
        
        # 简单比较（不区分大小写）
        if norm1.lower() == norm2.lower():
            return True
        
        # 检查缩写映射
        norm1_key = norm1.lower().replace('-', '').replace('_', '').replace(' ', '')
        norm2_key = norm2.lower().replace('-', '').replace('_', '').replace(' ', '')
        
        # 如果两者都映射到相同的标准名称
        for abbrev, full_names in self.abbreviation_map.items():
            variants = [abbrev] + [fn.lower().replace('-', '').replace('_', '').replace(' ', '') 
                                   for fn in full_names]
            if norm1_key in variants and norm2_key in variants:
                return True
        
        return False
    
    def get_canonical_name(self, name: str, entity_type: str = None) -> str:
        """
        获取标准名称（用于实体ID生成）
        
        Args:
            name: 原始名称
            entity_type: 实体类型
        
        Returns:
            标准名称
        """
        normalized = self.normalize_entity_name(name, entity_type)
        
        # 转换为小写，用于ID生成
        canonical = normalized.lower()
        
        # 去除空格和特殊字符（但保留字母数字和下划线）
        canonical = re.sub(r'[^\w]', '_', canonical)
        
        # 去除连续下划线
        canonical = re.sub(r'_+', '_', canonical)
        
        # 去除首尾下划线
        canonical = canonical.strip('_')
        
        return canonical
    
    def add_entity_variant(self, canonical_name: str, variant_name: str):
        """
        添加实体名称变体（用于学习新的映射）
        
        Args:
            canonical_name: 标准名称
            variant_name: 变体名称
        """
        key = self.get_canonical_name(canonical_name)
        
        if key not in self.entity_variants:
            self.entity_variants[key] = EntityNameVariant(
                canonical_name=canonical_name,
                variants={canonical_name}
            )
        
        self.entity_variants[key].variants.add(variant_name)
    
    def get_all_variants(self, name: str) -> Set[str]:
        """
        获取某个名称的所有已知变体
        
        Args:
            name: 实体名称
        
        Returns:
            所有变体的集合
        """
        key = self.get_canonical_name(name)
        
        if key in self.entity_variants:
            return self.entity_variants[key].variants
        
        return {name}
    
    def merge_entity_names(self, names: List[str], entity_type: str = None) -> str:
        """
        合并多个实体名称，选择最佳名称
        
        Args:
            names: 名称列表
            entity_type: 实体类型
        
        Returns:
            最佳名称
        """
        if not names:
            return ""
        
        if len(names) == 1:
            return names[0]
        
        # 标准化所有名称
        normalized_names = [self.normalize_entity_name(name, entity_type) for name in names]
        
        # 优先选择全称而非缩写
        # 规则：
        # 1. 优先选择较长的名称（通常是全称）
        # 2. 优先选择包含空格的名称（通常是全称）
        # 3. 优先选择首字母大写的名称（更正式）
        
        scored_names = []
        for norm_name in normalized_names:
            score = 0
            score += len(norm_name)  # 长度得分
            score += norm_name.count(' ') * 10  # 包含空格得分更高
            if norm_name and norm_name[0].isupper():
                score += 5  # 首字母大写得分
            scored_names.append((score, norm_name))
        
        # 选择得分最高的
        best_name = max(scored_names, key=lambda x: x[0])[1]
        
        return best_name
    
    def print_statistics(self):
        """打印标准化统计信息"""
        print("\n" + "="*70)
        print("实体名称标准化统计")
        print("="*70)
        print(f"预定义缩写映射: {len(self.abbreviation_map)} 个")
        print(f"学习到的实体变体: {len(self.entity_variants)} 个")
        
        print("\n常见缩写映射示例:")
        for abbrev in list(self.abbreviation_map.keys())[:10]:
            full_names = self.abbreviation_map[abbrev]
            print(f"  {abbrev.upper():10s} → {full_names[0]}")
        
        print("="*70 + "\n")


# 全局实例
_normalizer_instance = None

def get_normalizer() -> EntityNormalizer:
    """获取全局标准化器实例"""
    global _normalizer_instance
    if _normalizer_instance is None:
        _normalizer_instance = EntityNormalizer()
    return _normalizer_instance
