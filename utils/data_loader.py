import tensorflow as tf
import numpy as np

feature_description = {
    'encoder': tf.io.FixedLenFeature([256 * 256 * 1], tf.float32),
    'decoder': tf.io.FixedLenFeature([256 * 256 * 3], tf.float32),
    'label': tf.io.FixedLenFeature([], tf.string),
}

def parse_tfrecord(example_proto):
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    encoder = tf.reshape(parsed['encoder'], [256, 256, 1])
    decoder = tf.reshape(parsed['decoder'], [256, 256, 3])
    label = parsed['label'] 
    
    label_parts = tf.strings.split(label, '_')
    
    angle_val = tf.strings.to_number(label_parts[1], out_type=tf.float32)
    mach_val = tf.strings.to_number(label_parts[2], out_type=tf.float32)
    
    p = decoder[:, :, 0]
    u = decoder[:, :, 1]
    v = decoder[:, :, 2]
    mask = tf.squeeze(encoder)
    
    return mask, p, u, v, angle_val, mach_val, label

def generate_fluid_point_cloud(mask_np, p_np, u_np, v_np):
    x_coords = np.linspace(-1, 1, 256)
    y_coords = np.linspace(-1, 1, 256)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords, indexing='xy')
    
    X_flat, Y_flat = X_grid.flatten(), Y_grid.flatten()
    p_flat, u_flat, v_flat = p_np.flatten(), u_np.flatten(), v_np.flatten()
    mask_flat = mask_np.flatten()
    
    is_fluid = mask_flat > 0.0
    
    inputs = np.column_stack((X_flat[is_fluid], Y_flat[is_fluid]))
    labels = np.column_stack((u_flat[is_fluid], v_flat[is_fluid], p_flat[is_fluid]))
    
    return inputs.astype(np.float32), labels.astype(np.float32)

def tf_process_wrapper(mask, p, u, v, angle, mach, label): 
    def _py_wrapper(m, pres, vel_x, vel_y):
        inps, labs = generate_fluid_point_cloud(m.numpy().T, pres.numpy().T, vel_x.numpy().T, vel_y.numpy().T)
        return inps, labs

    X, Y = tf.py_function(func=_py_wrapper, inp=[mask, p, u, v], Tout=[tf.float32, tf.float32])
    
    num_pts = tf.shape(X)[0]
    angle_vec = tf.fill([num_pts, 1], angle)
    mach_vec = tf.fill([num_pts, 1], mach)
    
    X_full = tf.concat([X, angle_vec, mach_vec], axis=1)
    
    X_full.set_shape([None, 4]) 
    Y.set_shape([None, 3])
    
    return X_full, Y