import math
import numpy as np
def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    """
    feature_map = np.array(feature_map)
    results = []

    for roi in rois:
        x1, y1, x2, y2 = roi
        roi_h = y2 - y1
        roi_w = x2 - x1

        pooled_roi = np.zeros((output_size, output_size))

        for i in range(output_size):
            for j in range(output_size):
                h_start = int(y1 + np.floor(i*roi_h/output_size))
                h_end = int(y1 + np.floor((i+1)*roi_h/output_size))
                w_start = int(x1 + np.floor(j*roi_w/output_size))
                w_end = int(x1 + np.floor((j+1)*roi_w/output_size))

                if h_end==h_start: h_end = h_start+1
                if w_end==w_start: w_end = w_start+1

                region = feature_map[h_start:h_end, w_start:w_end]
                pooled_roi[i,j] = np.max(region)
        results.append(pooled_roi.tolist())
    return results