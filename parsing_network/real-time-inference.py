import numpy as np
import cv2
import tensorflow as tf
from deeplab_resnet import DeepLabResNetModel, decode_labels, prepare_label
import argparse
import time

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    
NUM_CLASSES = 7
IMG_SIZE = 512

input_feed_shape = (1, IMG_SIZE, IMG_SIZE, 3)

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("--model_weights", type=str, default='./final_model/',
                        help="Path to the file with model weights.")
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    return parser.parse_args()


def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

args = get_arguments()

img_tf = tf.placeholder(dtype=tf.float32, shape=input_feed_shape)

net = DeepLabResNetModel({'data': img_tf}, is_training=False, num_classes=args.num_classes)

restore_var = tf.global_variables()

# Predictions.
raw_output = net.layers['fc1_voc12']
raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(np.zeros( shape=(IMG_SIZE, IMG_SIZE, 3) ) )[0:2,])
raw_output_up = tf.argmax(raw_output_up, dimension=3)
pred = tf.expand_dims(raw_output_up, dim=3)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()

sess.run(init)
    
# Load weights.
ckpt = tf.train.get_checkpoint_state(args.model_weights)
loader = tf.train.Saver(var_list=restore_var)
load(loader, sess, ckpt.model_checkpoint_path)


def process_frame(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    input_img_feed = np.array(frame, dtype=float)
    input_img_feed = np.expand_dims(input_img_feed, axis=0)

    start_time = time.time()
    preds = sess.run(pred, feed_dict={img_tf: input_img_feed})
    elapsed_time = time.time() - start_time
    print("FPS: ", 1 / elapsed_time)
    msk = decode_labels(preds, num_classes=NUM_CLASSES)
    im = msk[0]
    final = cv2.addWeighted(im,0.9,frame,0.7,0)
    return final

def main():

	cap = cv2.VideoCapture(0)

	i = 1
	while(True):
	    i += 1
	    # Capture frame-by-frame
	    ret, frame = cap.read()

	    frame_out = process_frame(frame)
	    #frame_out = cv2.resize(frame, (512,512))
	    # Display the resulting frame
	    cv2.imshow('frame',frame_out)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()
	
if __name__ == '__main__':
    main()
