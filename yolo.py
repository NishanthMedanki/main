def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):

    box_scores = np.multiply(box_confidence,box_class_probs)

    box_classes = K.argmax(box_scores,axis=-1)
    box_class_scores = K.max(box_scores,axis=-1)

    filtering_mask = K.greater_equal(box_class_scores,threshold)

    scores = tf.boolean_mask(box_class_scores,filtering_mask)
    boxes = tf.boolean_mask(boxes,filtering_mask)
    classes = tf.boolean_mask(box_classes,filtering_mask)
    return scores, boxes, classes