"""
Конфигурация pytest для проекта Drop Forecaster.
"""

import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
