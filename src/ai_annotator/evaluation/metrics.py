def micro_f1_score(true_labels: list, predicted_labels: list) -> tuple[float, float, float]:
    """
    Calculates the micro-averaged precision, recall, and F1 score for a set of predicted labels compared to true labels.

    Args:
        true_labels: A list of sets or lists where each element contains the ground truth labels for a sample.
        predicted_labels: A list of sets or lists where each element contains the predicted labels for a sample.
    """
    
    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = 0  # False Negatives

    for true_set, predicted_set in zip(true_labels, predicted_labels):
        for pred in predicted_set:  # Calculate TP and FP
            if pred in true_set:
                TP += 1
            else:
                FP += 1
        
        for true in true_set:  # Calculate FN
            if true not in predicted_set:
                FN += 1

    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score