import numpy as np

def compute_anomaly_score_v2(distances, method='adaptive'):

    if method == 'mean':
        return np.mean(distances)

    elif method == 'top1p':

        top_k = max(1, int(len(distances) * 0.01))
        return np.mean(np.sort(distances)[-top_k:])

    elif method == 'top5p':

        top_k = max(1, int(len(distances) * 0.05))
        return np.mean(np.sort(distances)[-top_k:])

    elif method == 'max':

        return np.max(distances)

    elif method == 'adaptive':

        top1p = np.mean(np.sort(distances)[-max(1, int(len(distances) * 0.01)):])
        top5p = np.mean(np.sort(distances)[-max(1, int(len(distances) * 0.05)):])
        max_val = np.max(distances)
        mean_val = np.mean(distances)

        score = 0.4 * top1p + 0.3 * top5p + 0.2 * max_val + 0.1 * mean_val
        return score

    elif method == 'percentile95':

        return np.percentile(distances, 95)

    else:
        raise ValueError(f"Unknown method: {method}")
