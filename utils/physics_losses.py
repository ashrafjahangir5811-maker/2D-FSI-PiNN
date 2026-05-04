import tensorflow as tf

# ==========================================
# 1. CFD Physics Engine (Navier-Stokes)
# ==========================================
@tf.function
def compute_cfd_residual(cfd_model, coords_full, NU):
    """
    Calculates the physics residuals for the fluid domain using the Navier-Stokes 
    and Continuity equations for steady, incompressible flow.
    """
    x_batch = coords_full[:, 0:1]
    y_batch = coords_full[:, 1:2]
    angle_batch = coords_full[:, 2:3]
    mach_batch = coords_full[:, 3:4]
    
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x_batch, y_batch])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x_batch, y_batch])
            
            inputs = tf.concat([x_batch, y_batch, angle_batch, mach_batch], axis=1)
            preds_inner = cfd_model(inputs, training=True)
            
            u_pred = preds_inner[:, 0:1]
            v_pred = preds_inner[:, 1:2]
            p_pred = preds_inner[:, 2:3]
            
        # First Spatial Derivatives
        u_x = tape1.gradient(u_pred, x_batch)
        u_y = tape1.gradient(u_pred, y_batch)
        v_x = tape1.gradient(v_pred, x_batch)
        v_y = tape1.gradient(v_pred, y_batch)
        p_x = tape1.gradient(p_pred, x_batch)
        p_y = tape1.gradient(p_pred, y_batch)
        del tape1
        
    # Second Spatial Derivatives (Laplacians for diffusion)
    u_xx = tape2.gradient(u_x, x_batch)
    u_yy = tape2.gradient(u_y, y_batch)
    v_xx = tape2.gradient(v_x, x_batch)
    v_yy = tape2.gradient(v_y, y_batch)
    del tape2
    
    # Calculate PDEs (Steady-State Incompressible)
    eq_continuity = u_x + v_y
    eq_mom_x = (u_pred * u_x + v_pred * u_y) + p_x - NU * (u_xx + u_yy)
    eq_mom_y = (u_pred * v_x + v_pred * v_y) + p_y - NU * (v_xx + v_yy)
    
    # Compute Physics Residual Loss
    res_loss = tf.reduce_mean(tf.square(eq_continuity)) + \
               tf.reduce_mean(tf.square(eq_mom_x)) + \
               tf.reduce_mean(tf.square(eq_mom_y))
               
    return res_loss


# ==========================================
# 2. FEA Physics Engine (Solid Mechanics)
# ==========================================
@tf.function
def compute_fea_loss(fea_model, pts, pressure_load, E=69e9, NU_SOLID=0.3):
    """
    Calculates the physics residuals for the solid domain using Hooke's Law 
    (Linear Elasticity) and Static Equilibrium equations.
    """
    # Define Lamé parameters based on Young's Modulus (E) and Poisson's ratio (NU)
    E_tf = tf.constant(E, dtype=tf.float32)
    NU_tf = tf.constant(NU_SOLID, dtype=tf.float32)
    LAMBDA = (E_tf * NU_tf) / ((1.0 + NU_tf) * (1.0 - 2.0 * NU_tf))
    MU = E_tf / (2.0 * (1.0 + NU_tf))

    x = pts[:, 0:1]
    y = pts[:, 1:2]
    
    with tf.GradientTape(persistent=True) as tape_spatial:
        tape_spatial.watch([x, y])
        
        inputs = tf.concat([x, y], axis=1)
        preds = fea_model(inputs, training=True)
        
        dx, dy = preds[:, 0:1], preds[:, 1:2]
        s_xx, s_yy, t_xy = preds[:, 2:3], preds[:, 3:4], preds[:, 4:5]
        
    # 1. Kinematics (Strain gradients)
    dx_x = tape_spatial.gradient(dx, x)
    dy_y = tape_spatial.gradient(dy, y)
    dx_y = tape_spatial.gradient(dx, y)
    dy_x = tape_spatial.gradient(dy, x)
    
    # 2. Stress Gradients (Equilibrium gradients)
    s_xx_x = tape_spatial.gradient(s_xx, x)
    t_xy_y = tape_spatial.gradient(t_xy, y)
    t_xy_x = tape_spatial.gradient(t_xy, x)
    s_yy_y = tape_spatial.gradient(s_yy, y)
    del tape_spatial
    
    # 3. Constitutive Law (Hooke's Law) Residuals
    eq_sxx = s_xx - ((LAMBDA + 2.0 * MU) * dx_x + LAMBDA * dy_y)
    eq_syy = s_yy - ((LAMBDA + 2.0 * MU) * dy_y + LAMBDA * dx_x)
    eq_txy = t_xy - (MU * (dx_y + dy_x))
    
    # 4. Equilibrium Residuals (Static body, sum of forces = 0)
    eq_fx = s_xx_x + t_xy_y
    eq_fy = t_xy_x + s_yy_y
    
    # Compute Interior Loss (PDE constraints)
    loss_int = tf.reduce_mean(tf.square(eq_sxx)) + tf.reduce_mean(tf.square(eq_syy)) + \
               tf.reduce_mean(tf.square(eq_txy)) + tf.reduce_mean(tf.square(eq_fx)) + \
               tf.reduce_mean(tf.square(eq_fy))
    
    # Compute Boundary Condition Loss (Surface pressure matching)
    # Assumes downward aerodynamic pressure acts against normal solid stress (sigma_yy)
    loss_bc = tf.reduce_mean(tf.square(s_yy + pressure_load)) 
    
    # Combine losses (Weighted to prevent numerical volume locking)
    total_loss = (100.0 * loss_bc) + (10.0 * loss_int) 
    
    return total_loss
