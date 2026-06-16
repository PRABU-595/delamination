"""
Unit Tests for Delamination Framework
"""
import unittest
import torch
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.integrated.framework import IntegratedDelaminationFramework

class TestFramework(unittest.TestCase):
    def setUp(self):
        self.config = {
            'snpi_net': {'adaptive_kernel': {'input_dim': 6}},
            'cad_former': {'d_model': 128, 'n_layers': 2},
            'al_vtfd': {}
        }
        self.model = IntegratedDelaminationFramework(self.config)
        self.model.eval()

    def test_inference_shape(self):
        """Test if the model returns correct output shapes."""
        laminate = torch.randn(1, 8, 64)
        loading = torch.randn(1, 100)
        physics = torch.tensor([[1525.0, 3.0, 16.0, 70.0, 25.0, 0.01]])
        
        outputs = self.model.predict_delamination(laminate, loading, physics_inputs=physics)
        
        self.assertIn('delamination_area', outputs)
        self.assertIn('migration_interface', outputs)
        self.assertEqual(outputs['delamination_area'].shape, (1, 1))
        self.assertEqual(outputs['migration_interface'].squeeze().shape, (8,))

    def test_uncertainty_bounds(self):
        """Test if uncertainty is non-negative."""
        laminate = torch.randn(1, 8, 64)
        loading = torch.randn(1, 100)
        physics = torch.tensor([[1525.0, 3.0, 16.0, 70.0, 25.0, 0.01]])
        
        outputs = self.model.predict_delamination(laminate, loading, physics_inputs=physics)
        self.assertTrue(torch.all(outputs['uncertainty'] >= 0))

if __name__ == "__main__":
    unittest.main()
