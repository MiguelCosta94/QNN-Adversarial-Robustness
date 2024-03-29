import tensorflow as tf
from argparse import ArgumentParser
import utils.backend as b
import utils.dataset_loader as ds
import utils.quantize_ann as q
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.defences.trainer import AdversarialTrainer


def load_configs():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--dataset_id", type=str, default=None, help="dataset to use: cifar10, vww, or coffee")
    parser.add_argument("--batch_size", type=int, default=32, help="training batch size")
    parser.add_argument("--epochs", type=int, default=50, help="training epochs")
    parser.add_argument("--workers", type=int, default=1, help="number of workers for training")
    parser.add_argument("--ann_source_float", type=str, default=None, help="path to the source floating-point ANN")
    parser.add_argument("--qnn_source_int16", type=str, default=None, help="path to the source int-16 QNN")
    parser.add_argument("--qnn_source_int8", type=str, default=None, help="path to the source int-8 QNN")
    parser.add_argument("--ann_new_float", type=str, default=None, help="path to the new floating-point ANN")
    parser.add_argument("--qnn_new_int16", type=str, default=None, help="path to the new int-16 QNN")
    parser.add_argument("--qnn_new_int8", type=str, default=None, help="path to the new int-8 QNN")
    parser.add_argument("--pgd_eps", type=float, default=0.008, help="")
    parser.add_argument("--pgd_num_random_init", type=int, default=1, help="")
    parser.add_argument("--pgd_max_iter", type=int, default=10, help="")

    cfgs = parser.parse_args()

    # Check if any required argument is not set
    required_cfgs = ['dataset_id', 'ann_source_float', 'qnn_source_int16', 'qnn_source_int8', 'ann_new_float',
                     'qnn_new_int16', 'qnn_new_int8']
    for arg_name in required_cfgs:
        if getattr(cfgs, arg_name) is None:
            raise ValueError(f"Required argument {arg_name} is not set.")

    run_cfgs = vars(cfgs)

    return run_cfgs


loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD()
def train_step(model, data, labels):
    with tf.GradientTape() as tape:
        predictions = model(data, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def main():
    # Step 1: Load configs
    cfgs = load_configs()

    if cfgs['dataset_id'] == 'cifar10':
        input_shape = (32,32,3)
        num_classes = 10
    elif cfgs['dataset_id'] == 'vww':
        input_shape = (96,96,3)
        num_classes = 2
    elif cfgs['dataset_id'] == 'coffee':
        input_shape = (224,224,3)
        num_classes = 4

    # Step 2: Load source ANN/QNNs
    ann_source_float = tf.keras.models.load_model(cfgs['ann_source_float'])
    qnn_source_int16 = b.get_ml_quant_model(cfgs['qnn_source_int16'])
    scaler_int16, zp_int16 = b.get_input_quant_details(qnn_source_int16)
    qnn_source_int8 = b.get_ml_quant_model(cfgs['qnn_source_int8'])
    scaler_int8, zp_int8 = b.get_input_quant_details(qnn_source_int8)

    # Step 3: Load dataset and generate .bin files
    if cfgs['dataset_id'] == 'cifar10':
        x_train_float, y_train, x_test_float, y_test = ds.get_cifar10_full_ds_f32()
    elif cfgs['dataset_id'] == 'vww':
        x_train_float, y_train, x_test_float, y_test = ds.get_vww_full_ds_f32()
    elif cfgs['dataset_id'] == 'coffee':
        x_train_float, y_train, x_test_float, y_test = ds.get_coffee_full_ds_f32()

    x_test_int16 = b.quantize_dataset_int16(x_test_float, scaler_int16, zp_int16)
    x_test_int8 = b.quantize_dataset_int8(x_test_float, scaler_int8, zp_int8)

    # Step 4: Perform adversarial training
    ann_new_float = tf.keras.models.clone_model(ann_source_float)
    ann_new_float.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics='accuracy')
    ann_new_float.set_weights(ann_source_float.get_weights())
    art_classifier = TensorFlowV2Classifier(model=ann_new_float, clip_values=(0, 1),
                        nb_classes=num_classes, input_shape=input_shape,
                        loss_object=tf.keras.losses.CategoricalCrossentropy(), train_step=train_step)

    pgd = ProjectedGradientDescent(estimator=art_classifier, eps=cfgs['pgd_eps'], eps_step=cfgs['pgd_eps']/10,
                                num_random_init=cfgs['pgd_num_random_init'], max_iter=cfgs['pgd_max_iter'],
                                batch_size=cfgs['batch_size'])

    adv_trainer = AdversarialTrainer(art_classifier, attacks=pgd, ratio=1.0)
    adv_trainer.fit(x_train_float, y_train, batch_size=cfgs['batch_size'], nb_epochs=cfgs['epochs'],
                    workers=cfgs['workers'], use_multiprocessing=True)
    ann_new_float.save(cfgs['ann_new_float'])

    # Step 5: Quantize new ANN to int-8 and int-16 QNNs
    qnn_new_int8 = q.convert_to_tflite_int8(ann_new_float, cfgs['dataset_id'], x_test_float)
    with tf.io.gfile.GFile(cfgs['qnn_new_int8'], 'wb') as f:
        f.write(qnn_new_int8)

    qnn_new_int16 = q.convert_to_tflite_int16(ann_new_float, cfgs['dataset_id'], x_test_float)
    with tf.io.gfile.GFile(cfgs['qnn_new_int16'], 'wb') as f:
        f.write(qnn_new_int16)

    # Step 6: Evaluate the new ANN/QNNs against the old ones
    accuracy = b.get_accuracy_f32(ann_source_float, x_test_float, y_test)
    print("Source ANN -> Float -> ACC: " + str(accuracy))
    accuracy = b.get_accuracy_quant_model(qnn_source_int16, x_test_int16, y_test)
    print("Source QNN -> Int-16 -> ACC: " + str(accuracy))
    accuracy = b.get_accuracy_quant_model(qnn_source_int8, x_test_int8, y_test)
    print("Source QNN -> Int-8 -> ACC: " + str(accuracy))

    qnn_new_int16 = b.get_ml_quant_model(cfgs['qnn_new_int16'])
    qnn_new_int8 = b.get_ml_quant_model(cfgs['qnn_new_int8'])
    scaler_int16, zp_int16 = b.get_input_quant_details(qnn_new_int16)
    scaler_int8, zp_int8 = b.get_input_quant_details(qnn_new_int8)
    x_test_int16 = b.quantize_dataset_int16(x_test_float, scaler_int16, zp_int16)
    x_test_int8 = b.quantize_dataset_int8(x_test_float, scaler_int8, zp_int8)

    accuracy = b.get_accuracy_f32(ann_new_float, x_test_float, y_test)
    print("New ANN -> Float -> ACC: " + str(accuracy))
    accuracy = b.get_accuracy_quant_model(qnn_new_int16, x_test_int16, y_test)
    print("New QNN -> Int-16 -> ACC: " + str(accuracy))
    accuracy = b.get_accuracy_quant_model(qnn_new_int8, x_test_int8, y_test)
    print("New QNN -> Int-8 -> ACC: " + str(accuracy))


if __name__ == '__main__':
    main()