def predict(sess, image_file):

    image, image_data = preprocess_image(image_file, model_image_size = (608, 608))

    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    print('Found {} boxes for {}'.format(len(out_boxes), image_file))

    colors = generate_colors(class_names)

    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

  #  image.save(os.path.join("out", image_file), quality=90)

 #   output_image = scipy.misc.imread(os.path.join("out", image_file))
#    imshow(output_image)
    json_output = {
           "Message": "NULL",
            "Data": [
                {
                    "PersonCount": 0
                },
                {
                    "AnimalCount": 0,
               
                },
                {
                    "ObjectCount": 0
                }
            ]
        }
    json_data = json_output["Data"]
    for i in out_classes:
            if i =='bird' or classes =='cat'or classes =='dog'or classes =='horse'or classes =='sheep'or classes =='cow'or classes =='elephant'or classes =='bear'or classes =='giraffe'or classes =='zebra':
                json_data[1]["AnimalCount"] += 1
       
            elif i == 'person':
                json_data[0]["PersonCount"] += 1
            else:
                json_data[2]["ObjectCount"] += 1
    print(json_output)
    return out_scores, out_boxes, out_classes
