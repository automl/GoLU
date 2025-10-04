"""
The following file contains unittests for the GoLU activation and it's CUDA kernels
"""

import torch
import unittest
from golu.golu_activation import GoLU  # Adjust the import based on your project structure


class TestGoLU(unittest.TestCase):
    def test_golu_forward(self):
        """
        Test the forward pass of the GoLU activation function with different alpha, beta, gamma values and data types.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")

        # Define the data types to test
        dtypes = [torch.float32, torch.float16, torch.bfloat16]

        # Define values for alpha, beta, gamma
        alpha_values = [0.5, 1.0, 1.5]
        beta_values = [0.5, 1.0, 1.5]
        gamma_values = [0.5, 1.0, 1.5]

        for dtype in dtypes:
            for alpha in alpha_values:
                for beta in beta_values:
                    for gamma in gamma_values:
                        with self.subTest(dtype=dtype, alpha=alpha, beta=beta, gamma=gamma):
                            x = torch.randn(10, requires_grad=True, dtype=dtype, device='cuda')

                            activation_function = GoLU(
                                alpha=alpha,
                                beta=beta,
                                gamma=gamma
                            ).to(dtype=dtype, device='cuda')

                            y1 = activation_function(x)

                            # Check for NaNs in the output
                            self.assertFalse(torch.isnan(y1).any(),
                                            msg=f"Output contains NaNs for dtype={dtype}, alpha={alpha}, beta={beta}, gamma={gamma}")


    def test_golu_backward(self):
        """
        Test the backward pass of the GoLU activation function with different alpha, beta, gamma values and data types.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")

        # Define the data types to test
        dtypes = [torch.float32, torch.float16, torch.bfloat16]

        # Define values for alpha, beta, gamma
        alpha_values = [0.5, 1.0, 1.5]
        beta_values = [0.5, 1.0, 1.5]
        gamma_values = [0.5, 1.0, 1.5]

        for dtype in dtypes:
            for alpha in alpha_values:
                for beta in beta_values:
                    for gamma in gamma_values:
                        with self.subTest(dtype=dtype, alpha=alpha, beta=beta, gamma=gamma):
                            x = torch.randn(10, requires_grad=True, dtype=dtype, device='cuda')

                            activation_function = GoLU(
                                alpha=alpha,
                                beta=beta,
                                gamma=gamma
                            ).cuda()

                            # Cast the activation function to the desired dtype
                            activation_function = activation_function.to(dtype)

                            y = activation_function(x)
                            loss = y.sum()
                            loss.backward()

                            # Check that gradient with respect to x exists and is not NaN
                            self.assertIsNotNone(x.grad)
                            self.assertFalse(torch.isnan(x.grad).any(),
                                             msg=f"NaNs in gradients for dtype={dtype}, alpha={alpha}, beta={beta}, gamma={gamma}")

    def test_edge_cases(self):
        """
        Test edge cases with zeros, large positive, and large negative inputs, including backward pass,
        across different data types.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")

        # Define the data types to test
        dtypes = [torch.float32, torch.float16, torch.bfloat16]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                # Skip bfloat16 if not supported
                if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                    self.skipTest("BFloat16 is not supported on this device.")

                # Initialize the activation function with the appropriate dtype
                activation_function = GoLU().to(dtype=dtype, device='cuda')

                # Test with zeros
                x_zero = torch.zeros(10, requires_grad=True, dtype=dtype, device='cuda')
                y_zero = activation_function(x_zero)
                self.assertFalse(torch.isnan(y_zero).any(),
                                f"Forward pass output contains NaNs for zero input with dtype={dtype}")

                # Backward pass for zero input
                y_zero.sum().backward()
                self.assertIsNotNone(x_zero.grad, f"Gradient is None for zero input with dtype={dtype}")
                self.assertFalse(torch.isnan(x_zero.grad).any(),
                                f"Gradient contains NaNs for zero input with dtype={dtype}")
                x_zero.grad = None  # Reset gradient

                # Determine large positive and negative values based on dtype
                if dtype == torch.float16:
                    # Max finite value for float16 is approximately 65504
                    large_value = 60000.0
                    small_value = -60000.0
                elif dtype == torch.bfloat16:
                    # Max finite value for bfloat16 is approximately 3.3895314e38
                    large_value = 1e6
                    small_value = -1e6
                else:
                    # For float32
                    large_value = 1e6
                    small_value = -1e6

                # Test with large positive inputs
                x_large = torch.full((10,), large_value, requires_grad=True, dtype=dtype, device='cuda')
                y_large = activation_function(x_large)
                self.assertFalse(torch.isnan(y_large).any(),
                                f"Forward pass output contains NaNs for large positive input with dtype={dtype}")

                # Backward pass for large positive input
                y_large.sum().backward()
                self.assertIsNotNone(x_large.grad, f"Gradient is None for large positive input with dtype={dtype}")
                self.assertFalse(torch.isnan(x_large.grad).any(),
                                f"Gradient contains NaNs for large positive input with dtype={dtype}")
                x_large.grad = None  # Reset gradient

                # Test with large negative inputs
                x_negative = torch.full((10,), small_value, requires_grad=True, dtype=dtype, device='cuda')
                y_negative = activation_function(x_negative)
                self.assertFalse(torch.isnan(y_negative).any(),
                                f"Forward pass output contains NaNs for large negative input with dtype={dtype}")

                # Backward pass for large negative input
                y_negative.sum().backward()
                self.assertIsNotNone(x_negative.grad, f"Gradient is None for large negative input with dtype={dtype}")
                self.assertFalse(torch.isnan(x_negative.grad).any(),
                                f"Gradient contains NaNs for large negative input with dtype={dtype}")
                x_negative.grad = None  # Reset gradient


    def test_cuda_extension(self):
        """
        Test that the CUDA extension works correctly on GPU.
        """
        if torch.cuda.is_available():
            x = torch.randn(10, requires_grad=True, device='cuda')
            activation_function = GoLU().cuda()
            y = activation_function(x)
            self.assertTrue(y.is_cuda)
            y.sum().backward()
            self.assertIsNotNone(x.grad)
        else:
            self.skipTest("CUDA is not available.")


if __name__ == '__main__':
    unittest.main()
