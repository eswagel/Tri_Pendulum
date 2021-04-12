#Constants
g=9.81
L=1
M=1

if __name__=="__main__":
        
    import sympy as sym
    from sympy.abc import t
    from numba import njit
    import dill
    #Required for saving
    dill.settings["recurse"]=True
    
    #Create theta1,theta2,theta3 as functions of time
    theta1, theta2, theta3 = sym.symbols('theta1 theta2 theta3')
    theta1=sym.Function("theta1")(t)
    theta2=sym.Function("theta2")(t)
    theta3=sym.Function("theta3")(t)
    
    #Save their derivatives
    theta1_dot=theta1.diff(t)
    theta1_dot_2=theta1.diff(t, t)
    theta2_dot=theta2.diff(t)
    theta2_dot_2=theta2.diff(t,t)
    theta3_dot=theta3.diff(t)
    theta3_dot_2=theta3.diff(t,t)
    
    #Convert to cartesian coordinates
    x1=L*sym.sin(theta1)
    y1=-L*sym.cos(theta1)
    x2=x1+L*sym.sin(theta2)
    y2=y1-L*sym.cos(theta2)
    x3=x2+L*sym.sin(theta3)
    y3=y2-L*sym.cos(theta3)
    
    #Take derivatives
    x1_dot=x1.diff(t)
    y1_dot=y1.diff(t)
    x2_dot=x2.diff(t)
    y2_dot=y2.diff(t)
    x3_dot=x3.diff(t)
    y3_dot=y3.diff(t)
    
    #Lagrangian=KE-PE
    Lagrangian=1/2*M*(x1_dot**2+y1_dot**2+x2_dot**2+y2_dot**2+x3_dot**2+y3_dot**2)-M*g*(y1+y2+y3)
    
    #Euler-Lagrange equations
    EL1=sym.Eq(sym.simplify(Lagrangian.diff(theta1)-Lagrangian.diff(theta1_dot,t)),0)
    EL2=sym.Eq(sym.simplify(Lagrangian.diff(theta2)-Lagrangian.diff(theta2_dot,t)),0)
    EL3=sym.Eq(sym.simplify(Lagrangian.diff(theta3)-Lagrangian.diff(theta3_dot,t)),0)
    
    #Convert the equations to a matrix
    equations=sym.Matrix([EL1, EL2, EL3])
    #Solve for the second derivatives
    variables=sym.Matrix([theta1_dot_2,theta2_dot_2,theta3_dot_2])
    solution=sym.simplify(sym.solve(equations, variables))
    
    #Now turn the solution for each second derivative into a function of the angle and angular velocity
    compute_theta1=njit(sym.lambdify([theta1,theta2,theta3,theta1_dot,theta2_dot,theta3_dot],solution[theta1_dot_2]))
    compute_theta2=njit(sym.lambdify([theta1,theta2,theta3,theta1_dot,theta2_dot,theta3_dot],solution[theta2_dot_2]))
    compute_theta3=njit(sym.lambdify([theta1,theta2,theta3,theta1_dot,theta2_dot,theta3_dot],solution[theta3_dot_2]))
    
    #Dill the functions
    dill.dump(compute_theta1,open("compute_theta1.pkl","wb"))
    dill.dump(compute_theta2,open("compute_theta2.pkl","wb"))
    dill.dump(compute_theta3,open("compute_theta3.pkl","wb"))
