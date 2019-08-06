import tensorflow as tf

from mobilenetv3_factory import build_mobilenetv3


def build_mobilenetv3_tpuestimator(use_tpu: bool = True,
                                   params: dict = {},
                                   batch_size: int = 128,
                                   model_dir: str = None,
                                   report_iterations: int = 10,
                                   tpu_name: str = ''):
    params['use_tpu'] = use_tpu

    run_config = tf.contrib.tpu.RunConfig(model_dir=model_dir,
                                          cluster=tf.contrib.cluster_resolver.TPUClusterResolver(tpu_name),
                                          session_config=tf.ConfigProto(allow_soft_placement=True,
                                                                        log_device_placement=True),
                                          tpu_config=tf.contrib.tpu.TPUConfig(report_iterations))

    return tf.estimator.tpu.TPUEstimator(model_fn=model_fn,
                                         params=params,
                                         use_tpu=use_tpu,
                                         eval_on_tpu=False,
                                         train_batch_size=batch_size,
                                         eval_batch_size=batch_size,
                                         predict_batch_size=batch_size,
                                         config=run_config)


def model_fn(features, labels, mode, params):
    model = build_mobilenetv3(model_type=params['model_type'],
                              input_shape=params['input_shape'],
                              num_classes=params['num_classes'],
                              batch_size=params['batch_size'])

    predictions = model(features, training=(mode == tf.estimator.ModeKeys.TRAIN))

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions)

    if params['optimizer'] == 'adam':
        optimizer = tf.compat.v1.train.AdamOptimizer()
    elif params['optimizer'] == 'adagrad':
        optimizer = tf.compat.v1.train.AdagradOptimizer()
    else:
        raise ValueError('Optimizer "{}" not supported.'.format(params['optimizer']))

    if params['use_tpu']:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()(tf.argmax(labels, 1), predictions)

    train_op = optimizer.minimize(loss,
                                  var_list=model.trainable_variables,
                                  global_step=tf.compat.v1.train.get_or_create_global_step())

    def metric_fn(y_true, y_pred):

        predicted_classes = tf.argmax(y_pred, 1)
        accuracy = tf.metrics.accuracy(labels=tf.argmax(y_true, 1),
                                       predictions=predicted_classes,
                                       name="acc_op")
        return {"accuracy": accuracy}

    return tf.estimator.tpu.TPUEstimatorSpec(mode=mode,
                                             predictions=predictions,
                                             loss=loss,
                                             train_op=train_op,
                                             eval_metrics=(metric_fn, [labels, predictions]))
