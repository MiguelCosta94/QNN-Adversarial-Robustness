import numpy as np
from art.defences.preprocessor import SpatialSmoothing
from art.defences.preprocessor.feature_squeezing_INTx import FeatureSqueezing_INTx


def get_l1_norm(x1, x2):
    x1 = np.reshape(x1, newshape=(x1.shape[0], -1))
    x2 = np.reshape(x2, newshape=(x2.shape[0], -1))
    l1_norm = np.linalg.norm(x1-x2, ord=1, axis=-1)

    return l1_norm


def dequantize(data):
    data_f32 = data * (1/32767)
    data_f32 = data_f32.astype(np.float32)
    data_f32 = np.minimum(data_f32, 1)
    data_f32 = np.maximum(data_f32, -1)

    return data_f32


def get_threshold(x, y, fpr, start_bit_depth, compression_bit_depth, spl_smooth_win_size, model, nb_classes):
    # One-hot to categorical
    y_categorical = np.where(y == 1)[1]
    # Build the squeezers
    feature_squeezing = FeatureSqueezing_INTx(clip_values=(-32768, 32767), start_bit_depth=start_bit_depth,
                                                compression_bit_depth=compression_bit_depth, apply_predict=False)
    spatial_smoothing = SpatialSmoothing(window_size=spl_smooth_win_size, clip_values=(-32768, 32767), apply_predict=False)
    thresholds = []

    for i in range(nb_classes):
        # Pick "i" as a reference class to determine the threshold
        neg_idx = np.where(y_categorical == i)[0]
        # Pick the input samples whose class is "i"
        x_neg = x[neg_idx]
        y_neg = y[neg_idx]
        # Apply the squeezers to selected samples
        x_neg_squeeze, _ = feature_squeezing(x_neg)
        x_neg_smooth, _ = spatial_smoothing(x_neg)

        # Get predictions before and after squeezing for selected samples
        _, y_before = b.get_accuracy_quant_model(model, x_neg, y_neg)
        _, y_after_squeeze = b.get_accuracy_quant_model(model, x_neg_squeeze, y_neg)
        _, y_after_smooth = b.get_accuracy_quant_model(model, x_neg_smooth, y_neg)
        y_before = np.array(y_before, np.int16)
        y_after_squeeze = np.array(y_after_squeeze, np.int16)
        y_after_smooth = np.array(y_after_smooth, np.int16)

        # Get the maximum l1 distance between predictions before and after squeezing
        distance_squeeze = get_l1_norm(y_before, y_after_squeeze)
        distance_smooth = get_l1_norm(y_before, y_after_smooth)
        idx_max_dist_squeezing = np.argwhere(distance_squeeze > distance_smooth)
        distance = distance_smooth
        distance[idx_max_dist_squeezing] = distance_squeeze[idx_max_dist_squeezing]

        # Get the minimum amount of samples that must be classfied as 0
        selected_distance_idx = int(np.ceil(len(x_neg) * (1-fpr)))
        # Define the threshold as the x-th biggest distance
        threshold = sorted(distance)[selected_distance_idx-1]
        thresholds.append(threshold)
        #print ("Selected %f as the threshold value." % threshold)

    threshold_max = np.max(thresholds)
    print ("Selected %f as the threshold value." % threshold_max)

    return threshold_max


def squeeze_data(x_adv, y_test, start_bit_depth, compression_bit_depth, spl_smooth_win_size):
    feature_squeezing = FeatureSqueezing_INTx(clip_values=(-32768, 32767), start_bit_depth=start_bit_depth,
                                                compression_bit_depth=compression_bit_depth, apply_predict=False)
    spatial_smoothing = SpatialSmoothing(window_size=spl_smooth_win_size, clip_values=(-32768, 32767), apply_predict=False)

    x_squeeze, _ = feature_squeezing(x_adv, y_test)
    x_smooth, _ = spatial_smoothing(x_adv, y_test)
    distance_squeeze = get_l1_norm(x_adv, x_squeeze)
    distance_smooth = get_l1_norm(x_adv, x_smooth)
    idx_squeeze = np.argwhere(distance_squeeze > distance_smooth)
    x_clean = x_smooth
    x_clean[idx_squeeze] = x_squeeze[idx_squeeze]

    return x_clean


def remove_adv_examples(x, y, preds_before, preds_after, threshold):
    distance = get_l1_norm(preds_before, preds_after)
    clean_idx = np.argwhere(distance < threshold)
    clean_idx = np.reshape(clean_idx, clean_idx.shape[0])
    clean_x = x[clean_idx]
    clean_y = y[clean_idx]

    return clean_x, clean_y