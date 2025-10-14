import numpy as np
import math
import numpy.linalg as la
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def evaluate_predictions(true, pred, max_value):
    """Evaluate prediction results"""
    true_product = true * max_value
    pred_product = pred * max_value

    rmse = math.sqrt(mean_squared_error(true_product, pred_product))
    mae = mean_absolute_error(true_product, pred_product)

    true_norm = la.norm(true_product)
    # Calculate F-norm-based accuracy (1 - relative F-norm error)
    F_norm = la.norm(true_product - pred_product) / true_norm if true_norm > 0 else 0.0

    # Handle MAPE: avoid division by zero if true values contain zeros
    if np.any(true_product == 0):
        mape = 0.0
    else:
        mape = mean_absolute_percentage_error(true_product, pred_product)

    return rmse, mae, 1 - F_norm, mape


def analyze_predictions_by_range(true_values, predicted_values, max_value):
    """Analyze prediction performance across different value ranges"""
    true_values = true_values * max_value
    predicted_values = predicted_values * max_value
    value_ranges = [0, 50, 60, 70, 80, 90, 100]

    print("\n Risk Degrees Prediction Analysis:")
    print("=" * 50)

    for i in range(len(value_ranges) - 1):
        lower_bound = value_ranges[i]
        upper_bound = value_ranges[i + 1]

        # Mask values within the current range
        range_mask = (true_values >= lower_bound) & (true_values < upper_bound)

        if np.any(range_mask):
            true_in_range = true_values[range_mask]
            pred_in_range = predicted_values[range_mask]

            # Calculate evaluation metrics
            rmse = math.sqrt(mean_squared_error(true_in_range, pred_in_range))
            mae = mean_absolute_error(true_in_range, pred_in_range)

            # Handle division by zero when calculating F-norm
            true_norm = la.norm(true_in_range)
            F_norm = la.norm(true_in_range - pred_in_range) / true_norm if true_norm > 0 else 0.0
            accuracy = 1 - F_norm

            # Handle MAPE: avoid division by zero; convert to percentage
            if np.any(true_in_range == 0):
                mape = 0.0
            else:
                mape = mean_absolute_percentage_error(true_in_range, pred_in_range) * 100

            print(f"Range [{lower_bound:3d}, {upper_bound:3d}): "
                  f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, "
                  f"MAPE: {mape:.4f}, ACC: {accuracy:.4f}")