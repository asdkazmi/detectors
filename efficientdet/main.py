import argparse
import time

import cv2
import tensorflow as tf

parser = argparse.ArgumentParser(description='Run Inferene of EfficientDet TF-Lite Models')

parser.add_argument('--model-path', required=True, type=str, help='variant of efficient i.e. d0-d7')
args = parser.parse_args()

# Initialize the TFLite interpreter
interpreter = None
def build_interpreter(path):
    global interpreter
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

def predict(interpreter, input_tensor):

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  interpreter.set_tensor(input_details[0]['index'], input_tensor)
  interpreter.invoke()
  try:
      results = interpreter.get_tensor(output_details[0]['index'])
  except ValueError:
      return None

  return results


def preprocess(input_images):
    images = input_images.copy()
    phi = int(args.model_path.split('-d')[1].split('.')[0])
    res = 512 + phi * 128
    images = tf.cast(images, tf.float32)
    images = tf.image.resize_with_pad(images, res, res)
    if len(images.shape) == 3:
        images = tf.expand_dims(images, 0)
        batches = 1
    else:
        batches = images.shape[0]
    images.set_shape((batches, res, res, 3))
    return tf.cast(images, tf.uint8)

def postprocess(pred, width, height):
    input_size = interpreter.get_input_details()[0]['shape'][1]
    max_size = max(height, width)
    box = pred[1:5] / input_size
    box[::2] = box[::2] * max_size - ((max_size - height)/2)
    box[1::2] = box[1::2] * max_size - ((max_size - width)/2)
    box = box.round().astype(int)
    y1, x1, y2, x2 = box
    score = pred[5]
    category = pred[6]
    return (x1, y1), (x2, y2), score, category

def main():

    build_interpreter(args.model_path)
    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret, frame = cap.read()

        height, width, _ = frame.shape

        img = preprocess(frame)

        t = time.time()
        results = predict(interpreter, img)
        # Output Shape: [batch, 100, 7] with each prediction format as [_, ymin, xmin, ymax, xmax, score, class]
        cv2.putText(frame, str(round(time.time() - t, 2)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)

        if results is not None:
            for pred in results[0]:
                p1, p2, score, category = postprocess(pred, width, height)
                if score > 0.5:
                    cv2.rectangle(frame, p1, p2, (0, 255, 0), 0)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()

