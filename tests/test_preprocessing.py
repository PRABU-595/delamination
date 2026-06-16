import unittest
import torch
import numpy as np
from src.data.preprocessing import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    def test_fit_transform_numpy(self):
        preprocessor = DataPreprocessor()
        # Create dummy data: 100 samples, 10 features
        data = [np.random.rand(10) * 10 + 5 for _ in range(100)]
        data = np.array(data)
        
        # Mock dataloader (list of batches)
        dataloader = [{'features': data}]
        
        preprocessor.fit(dataloader)
        self.assertTrue(preprocessor.is_fitted)
        
        # Test transform
        sample = np.array([5.0] * 10)
        transformed = preprocessor.transform(sample)
        # Check if it returns a value (not asserting exact value as it depends on random fit)
        self.assertEqual(transformed.shape, sample.shape)
        
    def test_fit_transform_torch(self):
        preprocessor = DataPreprocessor()
        data = torch.randn(100, 10) * 10 + 5
        dataloader = [{'features': data}]
        
        preprocessor.fit(dataloader)
        
        sample = torch.tensor([5.0] * 10, dtype=torch.float32)
        transformed = preprocessor.transform(sample)
        
        self.assertTrue(isinstance(transformed, torch.Tensor))
        self.assertEqual(transformed.shape, sample.shape)

    def test_augment(self):
        config = {'augmentation': True, 'noise_level': 0.1}
        preprocessor = DataPreprocessor(config)
        
        sample = torch.zeros(10)
        augmented = preprocessor.augment(sample)
        
        # Augmented should not be all zeros
        self.assertFalse(torch.all(augmented == 0))
        # Mean should be close to 0 (since noise is zero mean)
        # self.assertAlmostEqual(augmented.mean().item(), 0, delta=0.5)

if __name__ == '__main__':
    unittest.main()
