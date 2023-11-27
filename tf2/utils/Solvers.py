#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 14:54:28 2020

@author: cosmin
"""
import tensorflow as tf

class Poisson2D_coll(tf.keras.Model): 
    def __init__(self, layers, train_op, num_epoch, print_epoch):
        super(Poisson2D_coll, self).__init__()
        self.model_layers = layers
        self.train_op = train_op
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
        self.adam_loss_hist = []
            
    def call(self, X):
        return self.u(X[:,0:1], X[:,1:2])
    
    # Running the model
    @tf.function
    def u(self, xPhys, yPhys):
        X = tf.concat([xPhys, yPhys],1)
        X = 2.0*(X - self.bounds["lb"])/(self.bounds["ub"] - self.bounds["lb"]) - 1.0
        for l in self.model_layers:
            X = l(X)
        return X
    
    # Return the first derivatives
    @tf.function
    def du(self, xPhys, yPhys):        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            u_val = self.u(xPhys, yPhys)
        dudx_val = tape.gradient(u_val, xPhys)
        dudy_val = tape.gradient(u_val, yPhys)
        del tape
        return dudx_val, dudy_val
    
    # Return the second derivative
    @tf.function
    def d2u(self, xPhys, yPhys):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            dudx_val, dudy_val = self.du(xPhys, yPhys)
        d2udx2_val = tape.gradient(dudx_val, xPhys)
        d2udy2_val = tape.gradient(dudy_val, yPhys)
        del tape        
        return d2udx2_val, d2udy2_val
         
    #Custom loss function
    @tf.function
    def get_loss(self,Xint, Yint, Xbnd, Ybnd):
        u_val_bnd = self.call(Xbnd)
        xPhys = Xint[:,0:1]
        yPhys = Xint[:,1:2]
        
        d2udx2_val_int, d2udy2_val_int = self.d2u(xPhys, yPhys)
        f_val_int = -(d2udx2_val_int + d2udy2_val_int)
        int_loss = tf.reduce_mean(tf.math.square(f_val_int - Yint))
        bnd_loss = tf.reduce_mean(tf.math.square(u_val_bnd - Ybnd))
        return int_loss+bnd_loss
      
    # get gradients
    @tf.function
    def get_grad(self, Xint, Yint, Xbnd, Ybnd):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(Xint, Yint, Xbnd, Ybnd)
        g = tape.gradient(L, self.trainable_variables)
        return L, g
      
    # perform gradient descent
    def network_learn(self,Xint,Yint, Xbnd, Ybnd):
        xmin = tf.math.reduce_min(Xint[:,0])
        ymin = tf.math.reduce_min(Xint[:,1])
        xmax = tf.math.reduce_max(Xint[:,0])
        ymax = tf.math.reduce_max(Xint[:,1])
        self.bounds = {"lb" : tf.reshape(tf.stack([xmin, ymin], 0), (1,2)),
                       "ub" : tf.reshape(tf.stack([xmax, ymax], 0), (1,2))}
        for i in range(self.num_epoch):
            L, g = self.get_grad(Xint, Yint, Xbnd, Ybnd)
            self.train_op.apply_gradients(zip(g, self.trainable_variables))
            self.adam_loss_hist.append(L)
            if i%self.print_epoch==0:
                print("Epoch {} loss: {}".format(i, L))
                                
class Poisson2D_DEM(tf.keras.Model): 
    def __init__(self, layers, train_op, num_epoch, print_epoch):
        super(Poisson2D_DEM, self).__init__()
        self.model_layers = layers
        self.train_op = train_op
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
        self.adam_loss_hist = []
            
    def call(self, X):
        return self.u(X[:,0:1], X[:,1:2])
    
    # Running the model
    @tf.function
    def u(self, xPhys, yPhys):
        X = tf.concat([xPhys, yPhys],1)
        X = 2.0*(X - self.bounds["lb"])/(self.bounds["ub"] - self.bounds["lb"]) - 1.0
        for l in self.model_layers:
            X = l(X)
        return X
    
    # Return the first derivatives
    @tf.function
    def du(self, xPhys, yPhys):        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            u_val = self.u(xPhys, yPhys)
        dudx_val = tape.gradient(u_val, xPhys)
        dudy_val = tape.gradient(u_val, yPhys)
        del tape
        return dudx_val, dudy_val
    
    # Return the second derivative
    @tf.function
    def d2u(self, xPhys, yPhys):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            dudx_val, dudy_val = self.du(xPhys, yPhys)
        d2udx2_val = tape.gradient(dudx_val, xPhys)
        d2udy2_val = tape.gradient(dudy_val, yPhys)
        d2udxdy_val = tape.gradient(dudx_val,yPhys)
        del tape        
        return d2udx2_val, d2udy2_val, d2udxdy_val
        
    @tf.function
    def get_loss(self, Xint, Wint, Yint, XbndDir, WbndDir, YbndDir):
        int_loss, bnd_loss = self.get_all_losses(Xint, Wint, Yint,
                                                 XbndDir, WbndDir, YbndDir)
        return int_loss + bnd_loss
 
    #Custom loss function
    @tf.function
    def get_all_losses(self,Xint, Wint, Yint, XbndDir, WbndDir, YbndDir):
        u_val_bnd = self.call(XbndDir)
        xPhys = Xint[:,0:1]
        yPhys = Xint[:,1:2]
        u_val = self.u(xPhys, yPhys)
        dudx_val, dudy_val = self.du(xPhys, yPhys)
        f_val = 0.5*(dudx_val**2+dudy_val**2) - Yint*u_val
        int_loss = tf.reduce_sum(Wint*f_val)
        bnd_loss = tf.reduce_mean(tf.math.square(u_val_bnd - YbndDir)*WbndDir)
        return int_loss, bnd_loss
      
    # get gradients
    @tf.function
    def get_grad(self, Xint, Wint, Yint, Xbnd, Wbnd, Ybnd):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(Xint, Wint, Yint, Xbnd, Wbnd, Ybnd)
        g = tape.gradient(L, self.trainable_variables)
        return L, g
      
    # perform gradient descent
    def network_learn(self, Xint, Wint, Yint, Xbnd, Wbnd, Ybnd):
        xmin = tf.math.reduce_min(Xint[:,0])
        ymin = tf.math.reduce_min(Xint[:,1])
        xmax = tf.math.reduce_max(Xint[:,0])
        ymax = tf.math.reduce_max(Xint[:,1])
        self.bounds = {"lb" : tf.reshape(tf.stack([xmin, ymin], 0), (1,2)),
                       "ub" : tf.reshape(tf.stack([xmax, ymax], 0), (1,2))}
        for i in range(self.num_epoch):
            L, g = self.get_grad(Xint, Wint, Yint, Xbnd, Wbnd, Ybnd)
            self.train_op.apply_gradients(zip(g, self.trainable_variables))
            self.adam_loss_hist.append(L)
            if i%self.print_epoch==0:
                print("Epoch {} loss: {}".format(i, L))
                
                
class Helmholtz2D_coll(tf.keras.Model):
    def __init__(self, layers, train_op, num_epoch, print_epoch, k,
                 real_alpha, imag_alpha):
        super(Helmholtz2D_coll, self).__init__()
        self.model_layers = layers
        self.train_op = train_op
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
        self.k = k
        self.real_alpha = real_alpha
        self.imag_alpha = imag_alpha
        self.adam_loss_hist = []
            
    def call(self, X):
        u_val, v_val = self.w(X[:,0:1], X[:,1:2])        
        return tf.concat([u_val, v_val],1)
    
    # Running the model
    @tf.function
    def w(self, xPhys, yPhys):
        X = tf.concat([xPhys, yPhys],1)
        X = 2.0*(X - self.bounds["lb"])/(self.bounds["ub"] - self.bounds["lb"]) - 1.0
        for l in self.model_layers:
            X = l(X)
        return X[:, 0:1], X[:, 1:2]
    
    # Return the first derivatives
    @tf.function
    def dw(self, xPhys, yPhys):        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            u_val, v_val = self.w(xPhys, yPhys)
        dudx_val = tape.gradient(u_val, xPhys)
        dudy_val = tape.gradient(u_val, yPhys)
        dvdx_val = tape.gradient(v_val, xPhys)
        dvdy_val = tape.gradient(v_val, yPhys)
        del tape
        return dudx_val, dudy_val, dvdx_val, dvdy_val
    
    # Return the second derivatives
    @tf.function
    def d2w(self, xPhys, yPhys):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            dudx_val, dudy_val, dvdx_val, dvdy_val = self.dw(xPhys, yPhys)
        d2udx2_val = tape.gradient(dudx_val, xPhys)
        d2udy2_val = tape.gradient(dudy_val, yPhys)
        d2vdx2_val = tape.gradient(dvdx_val, xPhys)
        d2vdy2_val = tape.gradient(dvdy_val, yPhys)
        del tape        
        return d2udx2_val, d2udy2_val, d2vdx2_val, d2vdy2_val
    
    @tf.function
    def get_loss(self, Xint, Yint, Xbnd_neu, Ybnd_neu, Xbnd_robin, Ybnd_robin):
        int_loss, neu_bnd_loss, robin_bnd_loss = self.get_all_losses(Xint, Yint,
                                     Xbnd_neu, Ybnd_neu, Xbnd_robin, Ybnd_robin)
        return int_loss + neu_bnd_loss + robin_bnd_loss
         
    #Custom loss function
    @tf.function
    def get_all_losses(self, Xint, Yint, Xbnd_neu, Ybnd_neu, Xbnd_robin, Ybnd_robin):
        # Evaluate the interior loss
        xPhys = Xint[:, 0:1]
        yPhys = Xint[:, 1:2]
                
        u_int, v_int = self.w(xPhys, yPhys)
        d2udx2_int, d2udy2_int, d2vdx2_int, d2vdy2_int = self.d2w(xPhys, yPhys)
        f_u_int = (d2udx2_int + d2udy2_int) + self.k**2 * u_int
        f_v_int = (d2vdx2_int + d2vdy2_int) + self.k**2 * v_int
        
        int_loss_u = tf.reduce_mean(tf.math.square(f_u_int - Yint[:,0:1]))
        int_loss_v = tf.reduce_mean(tf.math.square(f_v_int - Yint[:,1:2]))
        int_loss = int_loss_u + int_loss_v
        
        # Evaluate the Neumann boundary loss
        xPhys_neu = Xbnd_neu[:, 0:1]
        yPhys_neu = Xbnd_neu[:, 1:2]
        xPhys_norm_neu = Xbnd_neu[:, 2:3]
        yPhys_norm_neu = Xbnd_neu[:, 3:4]
        
        dudx_neu, dudy_neu, dvdx_neu, dvdy_neu = self.dw(xPhys_neu, yPhys_neu)        
        dudn_neu = dudx_neu*xPhys_norm_neu + dudy_neu*yPhys_norm_neu
        dvdn_neu = dvdx_neu*xPhys_norm_neu + dvdy_neu*yPhys_norm_neu
        
        neu_bnd_loss_u = tf.reduce_mean(tf.math.square(dudn_neu - Ybnd_neu[:, 0:1]))
        neu_bnd_loss_v = tf.reduce_mean(tf.math.square(dvdn_neu - Ybnd_neu[:, 1:2]))
        neu_bnd_loss = neu_bnd_loss_u + neu_bnd_loss_v
        
        # Evaluate the Robin boundary loss                        
        xPhys_robin = Xbnd_robin[:, 0:1]
        yPhys_robin = Xbnd_robin[:, 1:2]
        xPhys_norm_robin = Xbnd_robin[:, 2:3]
        yPhys_norm_robin = Xbnd_robin[:, 3:4]
        
        u_robin, v_robin = self.w(xPhys_robin, yPhys_robin)
        dudx_robin, dudy_robin, dvdx_robin, dvdy_robin = self.dw(xPhys_robin, yPhys_robin)        
        dudn_robin = dudx_robin*xPhys_norm_robin + dudy_robin*yPhys_norm_robin
        dvdn_robin = dvdx_robin*xPhys_norm_robin + dvdy_robin*yPhys_norm_robin
        
        robin_bnd_loss_u = tf.reduce_mean(tf.math.square(dudn_robin + \
                                          self.real_alpha * u_robin - \
                                          self.imag_alpha * v_robin - \
                                          Ybnd_robin[:, 0:1]))
        robin_bnd_loss_v = tf.reduce_mean(tf.math.square(dvdn_robin + \
                                          self.imag_alpha * u_robin + \
                                          self.real_alpha * v_robin - \
                                          Ybnd_robin[:, 1:2]))            
        robin_bnd_loss = robin_bnd_loss_u + robin_bnd_loss_v
        
        return int_loss, neu_bnd_loss, robin_bnd_loss
      
    # get gradients
    @tf.function
    def get_grad(self, Xint, Yint, Xbnd_neu, Ybnd_neu, Xbnd_robin, Ybnd_robin):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(Xint, Yint, Xbnd_neu, Ybnd_neu, Xbnd_robin, Ybnd_robin)
        g = tape.gradient(L, self.trainable_variables)
        return L, g
      
    # perform gradient descent
    def network_learn(self, Xint, Yint, Xbnd_neu, Ybnd_neu, Xbnd_robin, Ybnd_robin):
        xmin = tf.math.reduce_min(Xint[:,0])
        ymin = tf.math.reduce_min(Xint[:,1])
        xmax = tf.math.reduce_max(Xint[:,0])
        ymax = tf.math.reduce_max(Xint[:,1])
        self.bounds = {"lb" : tf.reshape(tf.stack([xmin, ymin], 0), (1,2)),
                       "ub" : tf.reshape(tf.stack([xmax, ymax], 0), (1,2))}
        for i in range(self.num_epoch):
            L, g = self.get_grad(Xint, Yint, Xbnd_neu, Ybnd_neu, Xbnd_robin, Ybnd_robin)
            self.train_op.apply_gradients(zip(g, self.trainable_variables))
            self.adam_loss_hist.append(L)
            if i%self.print_epoch==0:
                print("Epoch {} loss: {}".format(i, L))
                
class Wave1D(tf.keras.Model): 
    def __init__(self, layers, train_op, num_epoch, print_epoch):
        super(Wave1D, self).__init__()
        self.model_layers = layers
        self.train_op = train_op
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
        self.adam_loss_hist = []
            
    def call(self, X):
        return self.u(X[:,0:1], X[:,1:2])
    
    # Running the model
    @tf.function
    def u(self, xPhys, yPhys):
        X = tf.concat([xPhys, yPhys],1)
        X = 2.0*(X - self.bounds["lb"])/(self.bounds["ub"] - self.bounds["lb"]) - 1.0
        for l in self.model_layers:
            X = l(X)
        return X
    
    # Return the first derivatives
    @tf.function
    def du(self, xPhys, yPhys):        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            u_val = self.u(xPhys, yPhys)
        dudx_val = tape.gradient(u_val, xPhys)
        dudy_val = tape.gradient(u_val, yPhys)
        del tape
        return dudx_val, dudy_val
    
    # Return the second derivative
    @tf.function
    def d2u(self, xPhys, yPhys):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            dudx_val, dudy_val = self.du(xPhys, yPhys)
        d2udx2_val = tape.gradient(dudx_val, xPhys)
        d2udy2_val = tape.gradient(dudy_val, yPhys)
        del tape        
        return d2udx2_val, d2udy2_val
         
    #Custom loss function
    @tf.function
    def get_loss(self,Xint, Yint, Xbnd, Ybnd, Xinit, Yinit):
        
        xPhys = Xint[:,0:1]
        tPhys = Xint[:,1:2]
        
        d2udx2_val_int, d2udt2_val_int = self.d2u(xPhys, tPhys)
        f_val_int = d2udx2_val_int - d2udt2_val_int
        int_loss = tf.reduce_mean(tf.math.square(f_val_int - Yint))
        
        u_val_init = self.call(Xinit)
        xPhysInit = Xinit[:,0:1]
        yPhysInit = Xinit[:,1:2]
        _, dudt_val_init = self.du(xPhysInit, yPhysInit)
        init_loss =  tf.reduce_mean(tf.math.square(u_val_init - Yinit[:,0:1])) + \
                     tf.reduce_mean(tf.math.square(dudt_val_init - Yinit[:,1:2]))
        
        xPhysBnd = Xbnd[:,0:1]
        yPhysBnd = Xbnd[:,1:2]
        
        dudx_val_bnd, _ = self.du(xPhysBnd, yPhysBnd)
        bnd_loss = tf.reduce_mean(tf.math.square(dudx_val_bnd - Ybnd))
        return int_loss+init_loss+bnd_loss
      
    # get gradients
    @tf.function
    def get_grad(self, Xint, Yint, Xbnd, Ybnd, Xinit, Yinit):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(Xint, Yint, Xbnd, Ybnd, Xinit, Yinit)
        g = tape.gradient(L, self.trainable_variables)
        return L, g
      
    # perform gradient descent
    def network_learn(self, Xint,Yint, Xbnd, Ybnd, Xinit, Yinit):
        xmin = tf.math.reduce_min(Xint[:,0])
        ymin = tf.math.reduce_min(Xint[:,1])
        xmax = tf.math.reduce_max(Xint[:,0])
        ymax = tf.math.reduce_max(Xint[:,1])
        print("xmin =", xmin)
        print("ymin =", ymin)
        print("xmax =", xmax)
        print("ymax =", ymax)
        self.bounds = {"lb" : tf.reshape(tf.stack([xmin, ymin], 0), (1,2)),
                       "ub" : tf.reshape(tf.stack([xmax, ymax], 0), (1,2))}
        for i in range(self.num_epoch):
            L, g = self.get_grad(Xint, Yint, Xbnd, Ybnd, Xinit, Yinit)
            self.train_op.apply_gradients(zip(g, self.trainable_variables))
            self.adam_loss_hist.append(L)
            if i%self.print_epoch==0:
                print("Epoch {} loss: {}".format(i, L))
                
                
class Elasticity2D_coll_dist(tf.keras.Model): 
    def __init__(self, layers, train_op, num_epoch, print_epoch, model_data, data_type):
        super(Elasticity2D_coll_dist, self).__init__()
        self.model_layers = layers
        self.train_op = train_op
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
        self.data_type = data_type
        self.Emod = model_data["E"]
        self.nu = model_data["nu"]
        if model_data["state"]=="plane strain":
            self.Emat = self.Emod/((1+self.nu)*(1-2*self.nu))*\
                                    tf.constant([[1-self.nu, self.nu, 0], 
                                                 [self.nu, 1-self.nu, 0], 
                                                 [0, 0, (1-2*self.nu)/2]],dtype=data_type)
        elif model_data["state"]=="plane stress":
            self.Emat = self.Emod/(1-self.nu**2)*tf.constant([[1, self.nu, 0], 
                                                              [self.nu, 1, 0], 
                                                              [0, 0, (1-self.nu)/2]],dtype=data_type)
        self.adam_loss_hist = []
    @tf.function                                
    def call(self, X):
        uVal, vVal = self.u(X[:,0:1], X[:,1:2])
        return tf.concat([uVal, vVal],1)
    
    def dirichletBound(self, X, xPhys, yPhys):
        u_val = X[:,0:1]
        v_val = X[:,1:2]
        return u_val, v_val
          
    # Running the model
    @tf.function
    def u(self, xPhys, yPhys):
        X = tf.concat([xPhys, yPhys],1)
        X = 2.0*(X - self.bounds["lb"])/(self.bounds["ub"] - self.bounds["lb"]) - 1.0
        for l in self.model_layers:
            X = l(X)     
            
        # impose the boundary conditions
        u_val, v_val = self.dirichletBound(X, xPhys, yPhys)                          
        return u_val, v_val
    
    # Compute the strains
    @tf.function
    def kinematicEq(self, xPhys, yPhys):        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            u_val, v_val = self.u(xPhys, yPhys)
        eps_xx_val = tape.gradient(u_val, xPhys)
        eps_yy_val = tape.gradient(v_val, yPhys)
        eps_xy_val = tape.gradient(u_val, yPhys) + tape.gradient(v_val, xPhys)
        del tape
        return eps_xx_val, eps_yy_val, eps_xy_val
    
    # Compute the stresses
    @tf.function
    def constitutiveEq(self, xPhys, yPhys):
        eps_xx_val, eps_yy_val, eps_xy_val = self.kinematicEq(xPhys, yPhys)       
        eps_val = tf.transpose(tf.concat([eps_xx_val, eps_yy_val, eps_xy_val],1))
        stress_val = tf.transpose(tf.linalg.matmul(self.Emat, eps_val))
        stress_xx_val = stress_val[:,0:1]
        stress_yy_val = stress_val[:,1:2]
        stress_xy_val = stress_val[:,2:3]
        return stress_xx_val, stress_yy_val, stress_xy_val
    
    # Compute the forces
    @tf.function
    def balanceEq(self, xPhys, yPhys):        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            stress_xx_val, stress_yy_val, stress_xy_val = self.constitutiveEq(xPhys, yPhys)
        dx_stress_xx_val = tape.gradient(stress_xx_val, xPhys)
        dy_stress_yy_val = tape.gradient(stress_yy_val, yPhys)
        dx_stress_xy_val = tape.gradient(stress_xy_val, xPhys)
        dy_stress_xy_val = tape.gradient(stress_xy_val, yPhys)
        del tape
        f_x = dx_stress_xx_val + dy_stress_xy_val
        f_y = dx_stress_xy_val + dy_stress_yy_val
        return f_x, f_y
         
    #Custom loss function
    @tf.function
    def get_all_losses(self,Xint, Yint, Xbnd, Ybnd):
        xPhys = Xint[:,0:1]
        yPhys = Xint[:,1:2]        
        f_x_int, f_y_int = self.balanceEq(xPhys, yPhys)                
        int_loss_x = tf.reduce_mean(tf.math.square(f_x_int - Yint[:,0:1]))
        int_loss_y = tf.reduce_mean(tf.math.square(f_y_int - Yint[:,1:2]))
        sigma_xx, sigma_yy, sigma_xy = self.constitutiveEq(Xbnd[:,0:1], Xbnd[:,1:2])
        trac_x = sigma_xx*Xbnd[:,2:3]+sigma_xy*Xbnd[:,3:4]
        trac_y = sigma_xy*Xbnd[:,2:3]+sigma_yy*Xbnd[:,3:4]
        loss_bnd_tens = tf.where(Xbnd[:,4:5]==0, trac_x - Ybnd, trac_y - Ybnd)
        loss_bnd = tf.reduce_mean(tf.math.square(loss_bnd_tens))        
        return int_loss_x,int_loss_y,loss_bnd
    
    @tf.function
    def get_loss(self,Xint, Yint, Xbnd, Ybnd):
        losses = self.get_all_losses(Xint, Yint, Xbnd, Ybnd)
        return sum(losses)
          
    # get gradients
    @tf.function
    def get_grad(self, Xint, Yint, Xbnd, Ybnd):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(Xint, Yint, Xbnd, Ybnd)
        g = tape.gradient(L, self.trainable_variables)
        return L, g
          
    # perform gradient descent
    def network_learn(self,Xint,Yint, Xbnd, Ybnd):
        xmin = tf.math.reduce_min(Xint[:,0])
        ymin = tf.math.reduce_min(Xint[:,1])
        xmax = tf.math.reduce_max(Xint[:,0])
        ymax = tf.math.reduce_max(Xint[:,1])
        self.bounds = {"lb" : tf.reshape(tf.stack([xmin, ymin], 0), (1,2)),
                       "ub" : tf.reshape(tf.stack([xmax, ymax], 0), (1,2))}
        for i in range(self.num_epoch):
            L, g = self.get_grad(Xint, Yint, Xbnd, Ybnd)
            self.train_op.apply_gradients(zip(g, self.trainable_variables))
            self.adam_loss_hist.append(L)
            if i%self.print_epoch==0:
                int_loss_x, int_loss_y, loss_bond = self.get_all_losses(Xint, Yint, Xbnd, Ybnd)
                L = int_loss_x + int_loss_y + loss_bond
                print("Epoch {} loss: {}, int_loss_x: {}, int_loss_y: {}".format(i, 
                                                                    L, int_loss_x, int_loss_y))
 
                
class Elasticity2D_DEM_dist(tf.keras.Model): 
    def __init__(self, layers, train_op, num_epoch, print_epoch, model_data, data_type):
        super(Elasticity2D_DEM_dist, self).__init__()
        self.model_layers = layers
        self.train_op = train_op
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
        self.data_type = data_type
        self.Emod = model_data["E"]
        self.nu = model_data["nu"]
        if model_data["state"]=="plane strain":
            self.Emat = self.Emod/((1+self.nu)*(1-2*self.nu))*\
                                    tf.constant([[1-self.nu, self.nu, 0], 
                                                 [self.nu, 1-self.nu, 0], 
                                                 [0, 0, (1-2*self.nu)/2]],dtype=data_type)
        elif model_data["state"]=="plane stress":
            self.Emat = self.Emod/(1-self.nu**2)*tf.constant([[1, self.nu, 0], 
                                                              [self.nu, 1, 0], 
                                                              [0, 0, (1-self.nu)/2]],dtype=data_type)
        self.adam_loss_hist = []
    #@tf.function                                
    def call(self, X):
        uVal, vVal = self.u(X[:,0:1], X[:,1:2])
        return tf.concat([uVal, vVal],1)
    
    def dirichletBound(self, X, xPhys, yPhys):
        u_val = X[:,0:1]
        v_val = X[:,1:2]
        return u_val, v_val
          
    # Running the model
    @tf.function
    def u(self, xPhys, yPhys):
        X = tf.concat([xPhys, yPhys],1)
        X = 2.0*(X - self.bounds["lb"])/(self.bounds["ub"] - self.bounds["lb"]) - 1.0
        for l in self.model_layers:
            X = l(X)     
            
        # impose the boundary conditions
        u_val, v_val = self.dirichletBound(X, xPhys, yPhys)                          
        return u_val, v_val
    
    # Compute the strains
    @tf.function
    def kinematicEq(self, xPhys, yPhys):        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            u_val, v_val = self.u(xPhys, yPhys)
        eps_xx_val = tape.gradient(u_val, xPhys)
        eps_yy_val = tape.gradient(v_val, yPhys)
        eps_xy_val = tape.gradient(u_val, yPhys) + tape.gradient(v_val, xPhys)
        del tape        
        return eps_xx_val, eps_yy_val, eps_xy_val
    
    # Compute the stresses
    @tf.function
    def constitutiveEq(self, xPhys, yPhys):
        eps_xx_val, eps_yy_val, eps_xy_val = self.kinematicEq(xPhys, yPhys)       
        eps_val = tf.transpose(tf.concat([eps_xx_val, eps_yy_val, eps_xy_val],1))
        stress_val = tf.transpose(tf.linalg.matmul(self.Emat, eps_val))
        stress_xx_val = stress_val[:,0:1]
        stress_yy_val = stress_val[:,1:2]
        stress_xy_val = stress_val[:,2:3]
        return stress_xx_val, stress_yy_val, stress_xy_val
        
    #Custom loss function
    @tf.function
    def get_all_losses(self, Xint, Wint, Xbnd, Wbnd, Ybnd):
        # calculate the interior loss
        xPhys_int = Xint[:,0:1]
        yPhys_int = Xint[:,1:2]                
        sigma_xx_int, sigma_yy_int, sigma_xy_int = self.constitutiveEq(xPhys_int, yPhys_int)
        eps_xx_int, eps_yy_int, eps_xy_int = self.kinematicEq(xPhys_int, yPhys_int)
        loss_int = tf.reduce_sum(1/2*(eps_xx_int*sigma_xx_int + eps_yy_int*sigma_yy_int + \
                                      eps_xy_int*sigma_xy_int)*Wint)
        
        # calculate the boundary loss corresponding to the Neumann boundary
        xPhys_bnd = Xbnd[:, 0:1]
        yPhys_bnd = Xbnd[:, 1:2]        
        u_val_bnd, v_val_bnd = self.u(xPhys_bnd, yPhys_bnd)                
        loss_bnd = tf.reduce_sum((u_val_bnd*Ybnd[:, 0:1] + v_val_bnd*Ybnd[:, 1:2])*Wbnd)
        return loss_int, loss_bnd
    
    @tf.function
    def get_loss(self,Xint, Wint, Xbnd, Wbnd, Ybnd):
        loss_int, loss_bnd = self.get_all_losses(Xint, Wint, Xbnd, Wbnd, Ybnd)
        return loss_int - loss_bnd
          
    # get gradients
    @tf.function
    def get_grad(self, Xint, Wint, Xbnd, Wbnd, Ybnd):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(Xint, Wint, Xbnd, Wbnd, Ybnd)
        g = tape.gradient(L, self.trainable_variables)
        return L, g    
      
    # perform gradient descent
    def network_learn(self, Xint, Wint, Xbnd, Wbnd, Ybnd):
        xmin = tf.math.reduce_min(Xint[:,0])
        ymin = tf.math.reduce_min(Xint[:,1])
        xmax = tf.math.reduce_max(Xint[:,0])
        ymax = tf.math.reduce_max(Xint[:,1])
        self.bounds = {"lb" : tf.reshape(tf.stack([xmin, ymin], 0), (1,2)),
                       "ub" : tf.reshape(tf.stack([xmax, ymax], 0), (1,2))}
        for i in range(self.num_epoch):
            L, g = self.get_grad(Xint, Wint, Xbnd, Wbnd, Ybnd)
            self.train_op.apply_gradients(zip(g, self.trainable_variables))
            self.adam_loss_hist.append(L)
            if i%self.print_epoch==0:
                loss_int, loss_bnd = self.get_all_losses(Xint,
                                                    Wint, Xbnd, Wbnd, Ybnd)
                L = loss_int - loss_bnd
                print("Epoch {} loss: {}, loss_int: {}, loss_bnd: {}".format(i, 
                                                                    L, loss_int, loss_bnd))
    