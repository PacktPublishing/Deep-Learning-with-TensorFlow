from six.moves import xrange  
import tensorflow as tf
import prettytensor as pt
from prettytensor.tutorial import data_utils

tf.app.flags.DEFINE_string('save_path', None, 'Where to save the model checkpoints.')
FLAGS = tf.app.flags.FLAGS

BATCH_SIZE = 50
EPOCH_SIZE = 60000 // BATCH_SIZE
TEST_SIZE = 10000 // BATCH_SIZE

tf.app.flags.DEFINE_string('model', 'full','Choose one of the models, either full or conv')
FLAGS = tf.app.flags.FLAGS
def multilayer_fully_connected(images, labels):
                           images = pt.wrap(images)
                           with pt.defaults_scope(activation_fn=tf.nn.relu,l2loss=0.00001):
                           return (images.flatten().\
                                   fully_connected(100).\
                                   fully_connected(100).\
                                   softmax_classifier(10, labels))

def lenet5(images, labels):
    images = pt.wrap(images)
    with pt.defaults_scope\
         (activation_fn=tf.nn.relu, l2loss=0.00001):
    return (images.conv2d(5, 20).\
            max_pool(2, 2).\
            conv2d(5, 50).\
            max_pool(2, 2).\
            flatten().\
            fully_connected(500).\
            softmax_classifier(10, labels))

def main(_=None):
  image_placeholder = tf.placeholder\
                      (tf.float32, [BATCH_SIZE, 28, 28, 1])
  labels_placeholder = tf.placeholder\
                       (tf.float32, [BATCH_SIZE, 10])

if FLAGS.model == 'full':
    result = multilayer_fully_connected\
             (image_placeholder,\
              labels_placeholder)
  elif FLAGS.model == 'conv':
    result = lenet5(image_placeholder,\
                    labels_placeholder)
else:
    raise ValueError\
              ('model must be full or conv: %s' % FLAGS.model)

accuracy = result.softmax.\
           evaluate_classifier\
           (labels_placeholder,phase=pt.Phase.test)

train_images, train_labels = data_utils.mnist(training=True)
test_images, test_labels = data_utils.mnist(training=False)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_op = pt.apply_optimizer(optimizer,losses=[result.loss])
runner = pt.train.Runner(save_path=FLAGS.save_path)


with tf.Session():
    for epoch in xrange(10):
        train_images, train_labels = \
                      data_utils.permute_data\
                      ((train_images, train_labels))

        runner.train_model(train_op,result.\
                           loss,EPOCH_SIZE,\
                           feed_vars=(image_placeholder,\
                                      labels_placeholder),\
                           feed_data=pt.train.\
                           feed_numpy(BATCH_SIZE,\
                                      train_images,\
                                      train_labels),\
                           print_every=100)
        classification_accuracy = runner.evaluate_model\
                                  (accuracy,\
                                   TEST_SIZE,\
                                   feed_vars=(image_placeholder,\
                                              labels_placeholder),\
                                   feed_data=pt.train.\
                                   feed_numpy(BATCH_SIZE,\
                                              test_images,\
                                              test_labels))
  print('epoch’ , epoch + 1)
  print(‘accuracy’, classification_accuracy )

if __name__ == '__main__':
  tf.app.run()

