import cv2
import time
import numpy as np

if __name__ == '__main__':
    config_path = '/home/phamvanhanh/PycharmProjects/ComputerVison/IDCardVNRecognition/Test_Opencv/yolotinyv4_cccd.cfg'
    weigth_path = '/home/phamvanhanh/PycharmProjects/ComputerVison/IDCardVNRecognition/Test_Opencv/yolotinyv4_cccd_final.weights'

    net = cv2.dnn.readNetFromDarknet(config_path, weigth_path)

    img_path = '/media/phamvanhanh/3666AAB266AA7273/DATASET/Dataset_CCCD/aligned/aligned553.jpg'
    image = cv2.imread(img_path)
    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence >= 0.5:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
                            0.3)

    labelsPath = '/home/phamvanhanh/PycharmProjects/ComputerVison/IDCardVNRecognition/Test_Opencv/cccd.names'
    LABELS = open(labelsPath).read().strip().split("\n")
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2)
    # show the output image
    image = cv2.resize(image, (0, 0),fx=0.5, fy=0.5)
    cv2.imshow("Image", image)
    cv2.waitKey(0)