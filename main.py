import tensorflow as tf
import pandas as pd
import math
import os
import time
import numpy as np
import numpy.linalg as la
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Import from our modular structure
from configs.default_config import Config
from utils.data_processor import DataProcessor
from utils.metrics import evaluate_predictions, analyze_predictions_by_range
from models.tia_model import motif_gcn


def setup_gpu():
    """Configure GPU settings"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config


def create_model_directory(model_name, config):
    """Create output directory for model results"""
    output_directory = f'output/{model_name}'
    sub_directory = f'{model_name}_lr{config.learning_rate}_batch{config.batch_size}_unit{config.units}_seq{config.seq_len}_pre{config.pre_len}_epoch{config.training_epoch}'
    path = os.path.join(output_directory, sub_directory)

    if not os.path.exists(path):
        os.makedirs(path)
    return path


def build_model(inputs, num_nodes, units, output_dim, adj, k):
    """Build the complete model architecture"""
    # Graph weights and biases
    weights = {
        'out': tf.Variable(tf.random_normal([units, output_dim], mean=1.0), name='weight_out')
    }
    biases = {
        'out': tf.Variable(tf.random_normal([output_dim]), name='bias_out')
    }

    # Build the model
    predictions, motifs, final_states = motif_gcn(inputs, weights, biases, num_nodes, units, output_dim, adj, k)
    return predictions, weights, biases


def compute_loss(predictions, labels, num_nodes, lambda_loss=0.0015):
    """Compute loss function with regularization"""
    labels_flat = tf.reshape(labels, [-1, num_nodes])
    predicted_values = predictions

    # Main loss
    feature_mse = tf.reduce_mean(tf.square(predicted_values - labels_flat))

    # Regularization
    regularization_loss = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

    # Total loss
    loss = tf.reduce_mean(tf.nn.l2_loss(predicted_values - labels_flat) + regularization_loss)
    product_rmse = tf.sqrt(tf.reduce_mean(tf.square(predicted_values - labels_flat)))

    return loss, product_rmse, predicted_values


def train_epoch(sess, optimizer, loss, product_rmse, predicted_values,
                inputs_ph, labels_ph, trainX, trainY, batch_size, totalbatch):
    """Train for one epoch"""
    batch_loss = []

    for m in range(totalbatch):
        mini_batch = trainX[m * batch_size: (m + 1) * batch_size]
        mini_label = trainY[m * batch_size: (m + 1) * batch_size]

        _, loss_val, _, train_output = sess.run(
            [optimizer, loss, product_rmse, predicted_values],
            feed_dict={inputs_ph: mini_batch, labels_ph: mini_label}
        )
        batch_loss.append(loss_val)

    return batch_loss, np.mean(batch_loss)


def test_epoch(sess, loss, product_rmse, predicted_values,
               inputs_ph, labels_ph, testX, testY, batch_size, totaltestbatch):
    """Test for one epoch"""
    test_all = []
    test_pre_all = []
    test_losses = []

    for m in range(totaltestbatch):
        mini_batch = testX[m * batch_size: (m + 1) * batch_size]
        mini_label = testY[m * batch_size: (m + 1) * batch_size]

        loss_val, _, test_output = sess.run(
            [loss, product_rmse, predicted_values],
            feed_dict={inputs_ph: mini_batch, labels_ph: mini_label}
        )
        test_all.append(mini_label)
        test_pre_all.append(test_output)
        test_losses.append(loss_val)

    return test_all, test_pre_all, np.mean(test_losses)


def main():
    """Main training function"""
    # Initialize configuration
    config = Config()

    # Setup
    tf_config = setup_gpu()
    time_start = time.time()

    # Create output directory
    output_path = create_model_directory(config.model_name, config)

    # Initialize data processor
    data_processor = DataProcessor(config)

    # Load and preprocess data
    data, n_matrix, adj = data_processor.load_traffic_data()
    datatxt = data_processor.load_event_data()

    # Data statistics
    time_len = data.shape[0]
    num_nodes = data.shape[1]
    max_value = np.max(data)
    print(f"Data shape: {data.shape}, Max value: {max_value}")

    # Normalize data
    data_normalized = np.mat(data, dtype=np.float32) / max_value

    # Add noise to Pf(E)
    Gauss = np.random.normal(0, 1, size=n_matrix.shape)
    noise_Gauss = DataProcessor.max_min_normalization(Gauss, np.max(Gauss), np.min(Gauss))
    n_matrix = n_matrix + noise_Gauss

    # Process event data
    print("Processing event data...")
    varis_f = ['CrowdQuaries', 'Precipitation', 'Visibility']
    varis_f_num = len(varis_f)

    times_inp = np.zeros((time_len, varis_f_num), dtype='float32')
    spatial_inp = np.zeros((time_len, varis_f_num), dtype='float32')
    values_inp = np.zeros((time_len, varis_f_num), dtype='float32')
    varis_inp = np.zeros((time_len, varis_f_num), dtype='int32')

    for row in tqdm(datatxt.itertuples(), desc="Processing events"):
        time_idx = getattr(row, 'time')
        try:
            # Query count feature
            if float(row.query_count) > 50:
                times_inp[time_idx, 0] = row.time
                spatial_inp[time_idx, 0] = row.wayid
                values_inp[time_idx, 0] = row.query_count
                varis_inp[time_idx, 0] = 1

            # Visibility feature
            if float(row.visibility) < 7:
                times_inp[time_idx, 1] = row.time
                spatial_inp[time_idx, 1] = row.wayid
                values_inp[time_idx, 1] = row.visibility
                varis_inp[time_idx, 1] = 2

            # Precipitation feature
            if float(row.precip_accum) > 0.0001:
                times_inp[time_idx, 2] = row.time
                spatial_inp[time_idx, 2] = row.wayid
                values_inp[time_idx, 2] = row.precip_accum
                varis_inp[time_idx, 2] = 3
        except Exception as e:
            pass

    print("Event data shapes:")
    print(f"Times: {times_inp.shape}, Values: {values_inp.shape}, Variates: {varis_inp.shape}")

    # Prepare training data
    print("Preparing training data...")
    trainX, trainY, testX, testY = data_processor.preprocess_data(
        data_normalized, n_matrix, times_inp, spatial_inp,
        values_inp, varis_inp, time_len, config.train_rate,
        config.seq_len, config.pre_len
    )

    trainY = trainY[..., 0:1]
    testY = testY[..., 0:1]

    print(f"Training data - X: {trainX.shape}, Y: {trainY.shape}")
    print(f"Testing data - X: {testX.shape}, Y: {testY.shape}")

    # Calculate batches
    totalbatch = int(trainX.shape[0] / config.batch_size)
    totaltestbatch = int(testX.shape[0] / config.batch_size)
    print(f"Total batches - Train: {totalbatch}, Test: {totaltestbatch}")

    # Build computational graph
    print("Building model...")
    with tf.Graph().as_default():
        # Placeholders
        inputs_ph = tf.placeholder(tf.float32, shape=[config.batch_size, config.seq_len, num_nodes + 3 * 4, 2])
        labels_ph = tf.placeholder(tf.float32, shape=[config.batch_size, config.pre_len, num_nodes, 1])

        # Build model
        predictions, weights, biases = build_model(
            inputs_ph, num_nodes, config.units, config.pre_len, adj, config.k
        )

        # Compute loss
        loss, product_rmse, predicted_values = compute_loss(
            predictions, labels_ph, num_nodes, config.lambda_loss
        )

        # Optimizer
        optimizer = tf.train.AdamOptimizer(config.learning_rate).minimize(loss)

        # Initialize session
        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())

            # Training history
            train_history = {'loss': [], 'rmse': []}
            test_history = {'loss': [], 'rmse': [], 'mae': [], 'acc': [], 'mape': []}
            best_acc =0
            # Training loop
            print("Starting training...")
            for epoch in range(config.training_epoch):
                # Train
                batch_losses, epoch_train_loss = train_epoch(
                    sess, optimizer, loss, product_rmse, predicted_values,
                    inputs_ph, labels_ph, trainX, trainY, config.batch_size, totalbatch
                )

                # Test
                test_all, test_pre_all, epoch_test_loss = test_epoch(
                    sess, loss, product_rmse, predicted_values,
                    inputs_ph, labels_ph, testX, testY, config.batch_size, totaltestbatch
                )

                # Process test results
                test_label = np.concatenate(test_all, axis=0)
                test_output = np.concatenate(test_pre_all, axis=0)
                test_label = test_label.reshape(-1, num_nodes)
                test_output = test_output.reshape(-1, num_nodes)

                # Calculate metrics
                rmse, mae, acc, mape = evaluate_predictions(test_label, test_output, max_value)

                # Store history
                train_history['loss'].append(epoch_train_loss)
                test_history['loss'].append(epoch_test_loss)
                test_history['rmse'].append(rmse)
                test_history['mae'].append(mae)
                test_history['acc'].append(acc)
                test_history['mape'].append(mape)

                # Print progress
                print(f'Epoch {epoch:4d}: '
                          f'Test MAE: {mae:.4f}, '
                          f'Test RMSE: {rmse:.4f}, '
                          f'Test MAPE: {mape:.4f}, '
                          f'Test ACC: {acc:.4f}')

                # Save best model
                if acc > best_acc:
                    best_acc = acc
                    #print(f'New best model at epoch {epoch}: MAE = {mae:.6f}')
                    analyze_predictions_by_range(test_label, test_output, max_value)
                    saver.save(sess, os.path.join(output_path, f'best_model_epoch_{epoch}'))

            # Final results
            best_idx = np.argmin(test_history['mae'])
            print('\n=== FINAL RESULTS ===')
            print(f'Best Epoch: {best_idx}')
            print(f'Best MAE: {test_history["mae"][best_idx]:.4f}')
            print(f'Best RMSE: {test_history["rmse"][best_idx]:.4f}')
            print(f'Best ACC: {test_history["acc"][best_idx]:.4f}')
            print(f'Best MAPE: {test_history["mape"][best_idx]:.4f}')

            # Save predictions
            test_result = test_history['predictions'][best_idx] if 'predictions' in test_history else None
            if test_result is not None:
                var = pd.DataFrame(test_result)
                var.to_csv(os.path.join(output_path, 'best_predictions.csv'), index=False, header=False)

    # Training time
    time_end = time.time()
    print(f'Total training time: {time_end - time_start:.2f} seconds')


if __name__ == '__main__':
    main()