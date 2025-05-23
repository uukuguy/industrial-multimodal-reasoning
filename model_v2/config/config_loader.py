import os
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging
import re

logger = logging.getLogger(__name__)

@dataclass
class ConfigLoader:
    """配置加载器"""
    
    config_path: str
    config: Dict[str, Any] = field(default_factory=dict)
    env_prefix: str = "MODEL_"
    
    def __init__(self, config_path: str, env_prefix: str = "MODEL_"):
        """初始化配置加载器
        
        Args:
            config_path: 配置文件路径
            env_prefix: 环境变量前缀
        """
        self.config_path = config_path
        self.env_prefix = env_prefix
        self.config = self._load_config()
        self._override_from_env()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件
        
        Returns:
            配置字典
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        logger.info(f"成功加载配置文件: {self.config_path}")
        return config
    
    def _override_from_env(self):
        """从环境变量覆盖配置"""
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                # 将环境变量名转换为配置路径
                config_path = key[len(self.env_prefix):].lower()
                config_path = config_path.replace('__', '.')
                
                # 设置配置值
                self._set_nested_value(self.config, config_path.split('.'), value)
                
        logger.info("已从环境变量更新配置")
    
    def _set_nested_value(self, d: Dict[str, Any], path: List[str], value: Any):
        """设置嵌套字典的值
        
        Args:
            d: 目标字典
            path: 路径列表
            value: 要设置的值
        """
        if len(path) == 1:
            # 尝试转换值的类型
            try:
                # 尝试转换为数字
                if value.isdigit():
                    value = int(value)
                elif re.match(r'^-?\d+(\.\d+)?$', value):
                    value = float(value)
                # 尝试转换为布尔值
                elif value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
            except (ValueError, AttributeError):
                pass
                
            d[path[0]] = value
        else:
            if path[0] not in d:
                d[path[0]] = {}
            self._set_nested_value(d[path[0]], path[1:], value)
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置
        
        Returns:
            配置字典
        """
        return self.config
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """获取配置节
        
        Args:
            section: 配置节名称
            
        Returns:
            配置节字典
        """
        if section not in self.config:
            raise KeyError(f"配置节不存在: {section}")
            
        return self.config[section]
    
    def get_value(self, section: str, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            section: 配置节名称
            key: 配置键
            default: 默认值
            
        Returns:
            配置值
        """
        section_config = self.get_section(section)
        return section_config.get(key, default)
    
    def update_config(self, updates: Dict[str, Any]):
        """更新配置
        
        Args:
            updates: 更新字典
        """
        def _update_dict(d: Dict[str, Any], u: Dict[str, Any]):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    _update_dict(d[k], v)
                else:
                    d[k] = v
                    
        _update_dict(self.config, updates)
        logger.info("配置已更新")
    
    def validate_config(self, schema: Dict[str, Any]) -> bool:
        """验证配置
        
        Args:
            schema: 配置模式
            
        Returns:
            验证结果
        """
        def _validate_dict(d: Dict[str, Any], s: Dict[str, Any]) -> bool:
            for k, v in s.items():
                if k not in d:
                    logger.error(f"缺少必需的配置项: {k}")
                    return False
                    
                if isinstance(v, dict):
                    if not isinstance(d[k], dict):
                        logger.error(f"配置项类型错误: {k}")
                        return False
                    if not _validate_dict(d[k], v):
                        return False
                        
            return True
            
        return _validate_dict(self.config, schema)
    
    def save_config(self, path: Optional[str] = None):
        """保存配置
        
        Args:
            path: 保存路径，如果为None则使用原始路径
        """
        save_path = path or self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, sort_keys=False)
            
        logger.info(f"配置已保存到: {save_path}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConfigLoader':
        """从字典创建配置加载器
        
        Args:
            config_dict: 配置字典
            
        Returns:
            配置加载器
        """
        loader = cls("")
        loader.config = config_dict
        return loader 