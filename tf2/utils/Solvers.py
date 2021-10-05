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
        self.bounds = {"lb" : tf.reshape(tf.concat([xmin, ymin], 0), (1,2)),
                       "ub" : tf.reshape(tf.concat([xmax, ymax], 0), (1,2))}
        for i in range(self.num_epoch):
            L, g = self.get_grad(Xint, Yint, Xbnd, Ybnd)
            self.train_op.apply_gradients(zip(g, self.trainable_variables))
            if i%self.print_epoch==0:
                print("Epoch {} loss: {}".format(i, L))
                
class Poisson2D_DEM(tf.keras.Model): 

    def __init__(self, layers, train_op, num_epoch, print_epoch):
        super(Poisson2D_DEM, self).__init__()
        self.model_layers = layers
        self.train_op = train_op
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
            
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
        self.bounds = {"lb" : tf.reshape(tf.concat([xmin, ymin], 0), (1,2)),
                       "ub" : tf.reshape(tf.concat([xmax, ymax], 0), (1,2))}
        for i in range(self.num_epoch):
            L, g = self.get_grad(Xint, Wint, Yint, Xbnd, Wbnd, Ybnd)
            self.train_op.apply_gradients(zip(g, self.trainable_variables))
            if i%self.print_epoch==0:
                print("Epoch {} loss: {}".format(i, L))
                
class Helmholtz2D_coll(tf.keras.Model): 
    def __init__(self, layers, train_op, num_epoch, print_epoch, k):
        super(Helmholtz2D_coll, self).__init__()
        self.model_layers = layers
        self.train_op = train_op
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
        self.k = k
            
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
    def get_loss(self,Xint, Yint, Xbnd_neu, Ybnd_neu):
        xPhys = Xint[:,0:1]
        yPhys = Xint[:,1:2]
        xPhys_norm = Xint[:,2:3]
        yPhys_norm = Xint[:,3:4]
        u_val_int = self.call(Xint)
        d2udx2_val_int, d2udy2_val_int = self.d2u(xPhys, yPhys)
        f_val_int = (d2udx2_val_int + d2udy2_val_int)+self.k**2*u_val_int
        dudx_val_bnd, dudy_val_bnd = self.du(Xbnd_neu)
        
        int_loss = tf.reduce_mean(tf.math.square(f_val_int - Yint))
        bnd_loss = tf.reduce_mean(tf.math.square(dudx_val_bnd*xPhys_norm + \
                                         dudy_val_bnd*yPhys_norm - Ybnd_neu))
        return int_loss+bnd_loss
      
    # get gradients
    @tf.function
    def get_grad(self, Xint, Yint, Xbnd_neu, Ybnd_neu, Xbnd_rob, Ybnd_rob):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(Xint, Yint, Xbnd_neu, Ybnd_neu, Xbnd_rob, Ybnd_rob)
        g = tape.gradient(L, self.trainable_variables)
        return L, g
      
    # perform gradient descent
    def network_learn(self,Xint,Yint, Xbnd_neu, Ybnd_neu, Xbnd_rob, Ybnd_rob):
        xmin = tf.math.reduce_min(Xint[:,0])
        ymin = tf.math.reduce_min(Xint[:,1])
        xmax = tf.math.reduce_max(Xint[:,0])
        ymax = tf.math.reduce_max(Xint[:,1])
        self.bounds = {"lb" : tf.reshape(tf.concat([xmin, ymin], 0), (1,2)),
                       "ub" : tf.reshape(tf.concat([xmax, ymax], 0), (1,2))}
        for i in range(self.num_epoch):
            L, g = self.get_grad(Xint, Yint, Xbnd_neu, Ybnd_neu, Xbnd_rob, Ybnd_rob)
            self.train_op.apply_gradients(zip(g, self.trainable_variables))
            if i%self.print_epoch==0:
                print("Epoch {} loss: {}".format(i, L))
                
class Wave1D(tf.keras.Model): 
    def __init__(self, layers, train_op, num_epoch, print_epoch):
        super(Wave1D, self).__init__()
        self.model_layers = layers
        self.train_op = train_op
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
            
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
        return int_loss+init_loss+bnd_loss#+tf.sqrt(tf.square(int_loss-bnd_loss)+0.01)
      
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
        self.bounds = {"lb" : tf.reshape(tf.concat([xmin, ymin], 0), (1,2)),
                       "ub" : tf.reshape(tf.concat([xmax, ymax], 0), (1,2))}
        for i in range(self.num_epoch):
            L, g = self.get_grad(Xint, Yint, Xbnd, Ybnd, Xinit, Yinit)
            self.train_op.apply_gradients(zip(g, self.trainable_variables))
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
    
    @tf.function
    def interp_loss(self, Xint, Yint):
        xPhys = Xint[:,0:1]
        yPhys = Xint[:,1:2] 
        ux_val_int, uy_val_int = self.u(xPhys, yPhys)
        intx_loss = tf.sqrt(tf.reduce_mean(tf.math.square(ux_val_int - Yint[:,0:1])))
        inty_loss = tf.sqrt(tf.reduce_mean(tf.math.square(uy_val_int - Yint[:,1:2])))        
        return intx_loss+inty_loss
      
    # get gradients
    @tf.function
    def get_grad(self, Xint, Yint, Xbnd, Ybnd):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(Xint, Yint, Xbnd, Ybnd)
        g = tape.gradient(L, self.trainable_variables)
        return L, g
    
    # get gradients
    @tf.function
    def get_grad_interp(self, Xint, Yint):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.interp_loss(Xint, Yint)
        g = tape.gradient(L, self.trainable_variables)
        return L, g
      
    # perform gradient descent
    def network_learn(self,Xint,Yint, Xbnd, Ybnd):
        xmin = tf.math.reduce_min(Xint[:,0])
        ymin = tf.math.reduce_min(Xint[:,1])
        xmax = tf.math.reduce_max(Xint[:,0])
        ymax = tf.math.reduce_max(Xint[:,1])
        self.bounds = {"lb" : tf.reshape(tf.concat([xmin, ymin], 0), (1,2)),
                       "ub" : tf.reshape(tf.concat([xmax, ymax], 0), (1,2))}
        for i in range(self.num_epoch):
            L, g = self.get_grad(Xint, Yint, Xbnd, Ybnd)
            self.train_op.apply_gradients(zip(g, self.trainable_variables))
            if i%self.print_epoch==0:
                int_loss_x, int_loss_y, loss_bond = self.get_all_losses(Xint, Yint, Xbnd, Ybnd)
                L = int_loss_x + int_loss_y + loss_bond
                print("Epoch {} loss: {}, int_loss_x: {}, int_loss_y: {}".format(i, 
                                                                    L, int_loss_x, int_loss_y))
            # if i%1000==0:
            #     numPtsUTest = 101
            #     numPtsVTest = 101
            #     plot_solution(numPtsUTest, numPtsVTest, domain, self, data_type)
            
    def network_interp(self, Xint, Yint, Xbnd, Ybnd, Yint_interp):
        xmin = tf.math.reduce_min(Xint[:,0])
        ymin = tf.math.reduce_min(Xint[:,1])
        xmax = tf.math.reduce_max(Xint[:,0])
        ymax = tf.math.reduce_max(Xint[:,1])
        self.bounds = {"lb" : tf.reshape(tf.concat([xmin, ymin], 0), (1,2)),
                       "ub" : tf.reshape(tf.concat([xmax, ymax], 0), (1,2))}
        for i in range(self.num_epoch):
            L, g = self.get_grad_interp(Xint, Yint_interp)
            self.train_op.apply_gradients(zip(g, self.trainable_variables))
            if i%self.print_epoch==0:
                int_loss_x, int_loss_y, loss_bond = self.get_all_losses(Xint, Yint, Xbnd, Ybnd)
                L2 = int_loss_x + int_loss_y + loss_bond
                print("Epoch {} loss: {}, int_loss_x: {}, int_loss_y: {}".format(i, 
                                                                    L2, int_loss_x, int_loss_y))
                print("Interp loss: {}".format(L))
                
                