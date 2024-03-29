import tensorflow as tf
from argparse import ArgumentParser
import utils.backend as b
import utils.dataset_loader as ds
import utils.quantize_ann as q
from defenses.distiller import Distiller


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
    parser.add_argument("--softmax_temp", type=int, default=20, help="")
    parser.add_argument("--alpha", type=float, default=0.1, help="")

    cfgs = parser.parse_args()

    # Check if any required argument is not set
    required_cfgs = ['dataset_id', 'ann_source_float', 'qnn_source_int16', 'qnn_source_int8', 'ann_new_float',
                     'qnn_new_int16', 'qnn_new_int8']
    for arg_name in required_cfgs:
        if getattr(cfgs, arg_name) is None:
            raise ValueError(f"Required argument {arg_name} is not set.")

    run_cfgs = vars(cfgs)

    return run_cfgs


def get_teacher_temp_t(ann, temperature):
    teacher = b.clone_ann_without_softmax(ann)
    logits = teacher.output
    logits_T = tf.keras.layers.Lambda(lambda x: x/temperature)(logits)
    probabilities_T = tf.keras.layers.Activation('softmax', name='softmax_prob')(logits_T)

    new_ann = tf.keras.Model(teacher.input, probabilities_T)
    new_ann.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics='accuracy')
    new_ann.set_weights(ann.get_weights())

    return new_ann


def main():
    # Step 1: Load configs
    cfgs = load_configs()

    # Step 2: Load source ANN/QNNs
    ann_source_float = tf.keras.models.load_model(cfgs['ann_source_float'])
    qnn_source_int16 = b.get_ml_quant_model(cfgs['qnn_source_int16'])
    scaler_int16, zp_int16 = b.get_input_quant_details(qnn_source_int16)
    qnn_source_int8 = b.get_ml_quant_model(cfgs['qnn_source_int8'])
    scaler_int8, zp_int8 = b.get_input_quant_details(qnn_source_int8)

    # Step 3: Load dataset and generate .bin files
    if cfgs['dataset_id'] == 'cifar10':
        x_train_float, y_train, x_test_float, y_test = ds.get_cifar10_full_ds_f32(augmentation=True)
    elif cfgs['dataset_id'] == 'vww':
        x_train_float, y_train, x_test_float, y_test = ds.get_vww_full_ds_f32()
    elif cfgs['dataset_id'] == 'coffee':
        x_train_float, y_train, x_test_float, y_test = ds.get_coffee_full_ds_f32()

    x_test_int16 = b.quantize_dataset_int16(x_test_float, scaler_int16, zp_int16)
    x_test_int8 = b.quantize_dataset_int8(x_test_float, scaler_int8, zp_int8)

    ########################################################################
    # Step 4: Train the teacher model at temperature T
    ann_new_float = tf.keras.models.clone_model(ann_source_float)
    ann_new_float.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics='accuracy')
    ann_new_float.set_weights(ann_source_float.get_weights())

    print("\nTRAINING THE TEACHER")
    tmp_model_path = 'defensive_distil_tmp/' + cfgs['dataset_id'] + '_teacher.h5'

    teacher_tmp = get_teacher_temp_t(ann_new_float, cfgs['softmax_temp'])
    teacher_tmp.fit(x_train_float, y_train, batch_size=cfgs['batch_size'], epochs=cfgs['epochs'],
                    validation_data=(x_test_float, y_test), workers=cfgs['workers'], use_multiprocessing=True,
                    callbacks=[
                        tf.keras.callbacks.ModelCheckpoint(filepath=tmp_model_path, save_best_only=True,
                        monitor='val_accuracy', mode='max', save_weights_only=False, save_freq='epoch', verbose=1)]
                    )
    teacher_tmp = tf.keras.models.load_model(tmp_model_path)
    acc_teacher_tmp = b.get_accuracy_f32(teacher_tmp, x_test_float, y_test)
    print("Teacher TMP -> ACC: " + str(acc_teacher_tmp))

    # Remove lambda layer
    teacher = tf.keras.models.clone_model(ann_new_float)
    teacher.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics='accuracy')
    teacher.set_weights(teacher_tmp.get_weights())
    acc_teacher = b.get_accuracy_f32(teacher, x_test_float, y_test)
    print("Teacher -> ACC: " + str(acc_teacher))

    # Step 5: Train the student model at temperature T
    print("\nTRAINING THE STUDENT")
    student_tmp = tf.keras.models.clone_model(teacher_tmp)
    student_tmp.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics='accuracy')
    student_tmp.set_weights(teacher_tmp.get_weights())

    distiller = Distiller(teacher=teacher, student=student_tmp)
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
        student_loss_fn=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=cfgs['alpha'],
        temperature=cfgs['softmax_temp'],
    )
    distiller.fit(x_train_float, y_train, epochs=cfgs['epochs'], batch_size=cfgs['batch_size'],
                    validation_data=(x_test_float, y_test), workers=cfgs['workers'], use_multiprocessing=True)

    acc_student_tmp = b.get_accuracy_f32(student_tmp, x_test_float, y_test)
    print("Student TMP -> ACC: " + str(acc_student_tmp))

    # Remove lambda layer
    ann_new_float.set_weights(student_tmp.get_weights())
    ann_new_float.save(cfgs['ann_new_float'])
    acc_student = b.get_accuracy_f32(ann_new_float, x_test_float, y_test)
    print("Student -> ACC: " + str(acc_student))

    ########################################################################
    # Step 6: Quantize new ANN to int-8 and int-16 QNNs
    qnn_new_int8 = q.convert_to_tflite_int8(ann_new_float, cfgs['dataset_id'], x_test_float)
    with tf.io.gfile.GFile(cfgs['qnn_new_int8'], 'wb') as f:
        f.write(qnn_new_int8)

    qnn_new_int16 = q.convert_to_tflite_int16(ann_new_float, cfgs['dataset_id'], x_test_float)
    with tf.io.gfile.GFile(cfgs['qnn_new_int16'], 'wb') as f:
        f.write(qnn_new_int16)

    # Step 7: Evaluate the new ANN/QNNs against the old ones
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