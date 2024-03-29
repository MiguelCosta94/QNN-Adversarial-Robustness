import tensorflow as tf
import numpy as np
from tqdm import tqdm

def pgd_linf_tf(model, x, y, epsilon, alpha, num_iter, randomize=False):
    # Convert to TensorFlow tensors if not already
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    if randomize:
        delta = tf.random.uniform(tf.shape(x), minval=-epsilon, maxval=epsilon, dtype=tf.float32)
    else:
        delta = tf.zeros_like(x, dtype=tf.float32)

    for t in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(delta)
            # Forward pass to compute the loss
            y_pred = model(x + delta)
            loss = tf.keras.losses.categorical_crossentropy(y, y_pred, from_logits=False)

        # Calculate gradients of the loss with respect to the perturbation
        grad = tape.gradient(loss, delta)

        # Update perturbation using the gradient and the step size
        delta = tf.clip_by_value(delta + alpha * tf.sign(grad), -epsilon, epsilon)

    return delta


def train(epoch, model, train_dataset, optimizer, criterion, epsilon=8./255, alpha=2./255, num_iter=7):
    print(f'\nEpoch: {epoch}')
    train_loss = 0
    correct = 0
    total = 0
    model.trainable = True
    
    pbar = tqdm(desc="Training", total=len(train_dataset))
    
    for batch_idx, (inputs, targets) in enumerate(train_dataset):
        #inputs = tf.convert_to_tensor(inputs.numpy())  # Assuming inputs are in numpy arrays
        #targets = tf.convert_to_tensor(targets.numpy())
        #targets = tf.argmax(targets, axis=1)

        delta = pgd_linf_tf(model, inputs, targets, epsilon=epsilon, alpha=alpha, num_iter=num_iter, randomize=True)
        inputs_adv = inputs + delta
        inputs_adv = tf.clip_by_value(inputs_adv, 0, 1)  # Clip values to [0, 1]
        
        with tf.GradientTape() as tape:
            outputs_adv = model(inputs_adv, training=True)
            outputs = model(inputs, training=True)
            loss_adv, loss_sink = criterion.call(outputs_clean=outputs, outputs_adv=outputs_adv, target=targets)
            loss = loss_adv + loss_sink

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss += loss.numpy()
        mean_loss = train_loss/(batch_idx+1)
        total += tf.shape(targets)[0].numpy()
        correct += np.sum(np.argmax(outputs_adv, axis=1) == np.argmax(targets, axis=1))
        train_acc = (correct / total) * 100

        pbar.update()
        postfix = {'Loss': f'{mean_loss:.3f}', 'Acc': f'{train_acc:.3f}', 'correct': f'{correct}/{total}'}
        postfix['sink'] = loss_sink.numpy()
        pbar.set_postfix(postfix)
    
    pbar.close()
    return train_acc, mean_loss