import unittest
from dataclasses import dataclass, is_dataclass
from src.config import Config


class TestConfig(unittest.TestCase):
    def test_instance_type(self):
        config = Config(project_stub="Potato_Fertilizer_Othello")
        self.assertTrue(is_dataclass(config))


unittest.main()
