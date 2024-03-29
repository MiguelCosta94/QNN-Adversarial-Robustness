import numpy as np
from art.defences.preprocessor import FeatureSqueezing, SpatialSmoothing


def get_l1_norm(x1, x2):
    x1 = np.reshape(x1, newshape=(x1.shape[0], -1))
    x2 = np.reshape(x2, newshape=(x2.shape[0], -1))
    l1_norm = np.linalg.norm(x1-x2, ord=1, axis=-1)

    return l1_norm


def get_threshold(x, y, fpr, color_bit_depth, spl_smooth_win_size, model, nb_classes):
    # One-hot to categorical
    y_categorical = np.where(y == 1)[1]
    # Build the squeezers
    feature_squeezing = FeatureSqueezing(clip_values=(0, 1), bit_depth=color_bit_depth, apply_predict=False)
    spatial_smoothing = SpatialSmoothing(window_size=spl_smooth_win_size, clip_values=(0, 1), apply_predict=False)
    thresholds = []

    for i in range(nb_classes):
        # Pick "i" as a reference class to determine the threshold
        neg_idx = np.where(y_categorical == i)[0]
        # Pick the input samples whose class is "i"
        x_neg = x[neg_idx]
        # Apply the squeezers to selected samples
        x_neg_squeeze, _ = feature_squeezing(x_neg)
        x_neg_smooth, _ = spatial_smoothing(x_neg)

        # Get predictions before and after squeezing for selected samples
        y_before = model.predict(x_neg)
        y_after_squeeze = model.predict(x_neg_squeeze)
        y_after_smooth = model.predict(x_neg_smooth)

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
        del x_neg, x_neg_smooth, x_neg_squeeze, y_before, y_after_smooth, y_after_squeeze
        #print ("Selected %f as the threshold value." % threshold)

    threshold_max = np.max(thresholds)
    print ("Selected %f as the threshold value." % threshold_max)

    return threshold_max


def squeeze_data(x_adv, y_test, color_bit_depth, spl_smooth_win_size):
    feature_squeezing = FeatureSqueezing(clip_values=(0, 1), bit_depth=color_bit_depth, apply_predict=False)
    spatial_smoothing = SpatialSmoothing(window_size=spl_smooth_win_size, clip_values=(0, 1), apply_predict=False)

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