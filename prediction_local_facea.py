
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

#image collage
"""    old_image=[]
    for d in out_boxes:
        (x1,y1,x2,y2)=d
        print(x1)
        print(y2)
    for d in out_boxes:
        (x1,y1,x2,y2)=d
        old_image.append(image[x1:x2,y1:y2])


    resized_imgs = []
    (max_h, max_w, channels) = max([i.shape for i in old_image])
   
    for img in old_image:
        resized_imgs.append(cv2.resize(img,(max_h, max_w), interpolation=cv2.INTER_AREA))
        # resized_imgs.append(img)
    row_imgs = []
   
    if npics == 1:
        return resized_imgs[0]
   
    else :
        for row in range(int(jason_data[0]["PersonCount"]/3)):
            try:
                new_image = np.hstack((resized_imgs[3*row], resized_imgs[3*row + 1]))
                new_image = np.hstack((new_image, resized_imgs[3*row + 2]))
            except:
                pass
            row_imgs.append(new_image)
        if len(row_imgs) <= 1:
            return new_image
       
        black_img = np.zeros((max_h, max_w, channels),dtype=np.uint8)
        if npics % 3 == 1:
            new_image = np.hstack((resized_imgs[-1], black_img))
            new_image = np.hstack((new_image, black_img))
            row_imgs.append(new_image)
           
        elif npics % 3 == 2:
            new_image = np.hstack((resized_imgs[-1], resized_imgs[-2]))
            print(new_image)
            new_image = np.hstack((new_image, black_img))
            row_imgs.append(new_image)
            print(new_image)
       
       
        new_image = np.vstack((row_imgs[0], row_imgs[1]))
       
        for i in range(2,len(row_imgs)):
            new_image = np.vstack((new_image, row_imgs[i]))
        imshow(new_image)
"""