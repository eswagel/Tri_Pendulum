from Tri_pendulum import *

LOAD_DATA=True #Controls whether to plot saved data or generate data
TAKE_DATA="local" #If this is "local", the LLE will be generated. Otherwise, the MLE will be calculated
PLOT_ANGLE=not True #Either plot vs. angle or plot vs. energy
SAVE=True #If True, save the data

if LOAD_DATA and TAKE_DATA.upper()!="LOCAL":
    """ Load the data
    Keywords:
        - lambda1: first set of lambdas
        - lambda 2: second set of lambdas
        - etc.
        - angs: ndarray of the angle accompanying each lambda
        - dt: the integration step used in calculation
        - tf: the final time used for calculation
    """
    lambdas=np.load("lyapunov_theta.npz")
    equal_lambdas=np.load("lyapunov_equal_angs.npz")
    random_lambdas=np.load("lyapunov_random_theta.npz")
    two_equal=np.load("lyapunov_two_equal_theta.npz")
    
    #Calculate energy
    energies=np.array([[Tri_Pendulum(theta0=[ang,0,0]).initial_energy(),Tri_Pendulum(theta0=[0,ang,0]).initial_energy(), \
                        Tri_Pendulum(theta0=[0,0,ang]).initial_energy(),Tri_Pendulum(theta0=[ang,ang,ang]).initial_energy()] for ang in lambdas["angs"]])
    energies_random=[Tri_Pendulum(theta0=ang).initial_energy() for ang in random_lambdas["angs"]]
    energies_two_equal=np.array([[Tri_Pendulum(theta0=[ang,ang,0]).initial_energy(),Tri_Pendulum(theta0=[0,ang,ang]).initial_energy(), \
                        Tri_Pendulum(theta0=[ang,0,ang]).initial_energy()] for ang in two_equal["angs"]])
    if PLOT_ANGLE:
        plt.scatter(lambdas["angs"][1:-1],lambdas["lambda1"][1:-1],s=1,label="$\\delta \\theta_1$",color="lightblue")
        plt.scatter(lambdas["angs"][1:-1],lambdas["lambda2"][1:-1],s=1,label="$\\delta \\theta_2$",color="blue")
        plt.scatter(lambdas["angs"][1:-1],lambdas["lambda3"][1:-1],s=1,label="$\\delta \\theta_3$",color="darkblue")
        plt.scatter(two_equal["angs"][1:-1],two_equal["lambda1"][1:-1],s=1,label="$\\theta_1=\\theta_2$",color="pink")
        plt.scatter(two_equal["angs"][1:-1],two_equal["lambda2"][1:-1],s=1,label="$\\theta_2=\\theta_3$",color="red")
        plt.scatter(two_equal["angs"][1:-1],two_equal["lambda3"][1:-1],s=1,label="$\\theta_1=\\theta_3$",color="darkred")
        plt.scatter(lambdas["angs"][1:-1],equal_lambdas["lambda1"][1:-1],s=1,label="Equal angles",color="black")
        plt.title("MLE vs. Initial Angle for Multiple Configurations")
        plt.xlabel("Initial Angle (rad)")
    else:
        plt.scatter(energies[1:-1,0],lambdas["lambda1"][1:-1],s=1,label="$\\delta \\theta_1$",color="lightblue")
        plt.scatter(energies[1:-1,1],lambdas["lambda2"][1:-1],s=1,label="$\\delta \\theta_2$",color="blue")
        plt.scatter(energies[1:-1,2],lambdas["lambda3"][1:-1],s=1,label="$\\delta \\theta_3$",color="darkblue")
        plt.scatter(energies_two_equal[1:-1,0],two_equal["lambda1"][1:-1],s=1,label="$\\theta_1=\\theta_2$",color="pink")
        plt.scatter(energies_two_equal[1:-1,1],two_equal["lambda2"][1:-1],s=1,label="$\\theta_2=\\theta_3$",color="red")
        plt.scatter(energies_two_equal[1:-1,2],two_equal["lambda3"][1:-1],s=1,label="$\\theta_1=\\theta_3$",color="darkred")
        plt.scatter(energies[1:-1,3],equal_lambdas["lambda1"][1:-1],s=1,label="Equal angles",color="black")
        plt.scatter(energies_random,random_lambdas["lambda1"],s=1,label="Random initialization",zorder=0,color="gray")
        plt.title("MLE vs. Energy for Multiple Configurations")
        plt.xlabel("Total Energy (J)")
    plt.ylabel("$\\lambda (s^{-1})$")
    leg=plt.legend(markerscale=5)
    plt.show()
    exit()

elif LOAD_DATA:
    lambdas=np.load("local_lyapunov_theta.npz")
    equal_lambdas=np.load("local_lyapunov_equal_angs.npz")
    print(lambdas["tf"])
    energies=np.array([[Tri_Pendulum(theta0=[ang,0,0]).initial_energy(),Tri_Pendulum(theta0=[0,ang,0]).initial_energy(), \
                        Tri_Pendulum(theta0=[0,0,ang]).initial_energy(),Tri_Pendulum(theta0=[ang,ang,ang]).initial_energy()] for ang in lambdas["angs"]])
    if PLOT_ANGLE:
        plt.scatter(lambdas["angs"][1:-1],lambdas["lambda1"][1:-1],s=1,label="$\\delta \\theta_1$",color="lightblue")
        plt.scatter(lambdas["angs"][1:-1],lambdas["lambda2"][1:-1],s=1,label="$\\delta \\theta_2$",color="blue")
        plt.scatter(lambdas["angs"][1:-1],lambdas["lambda3"][1:-1],s=1,label="$\\delta \\theta_3$",color="darkblue")
        plt.scatter(lambdas["angs"][1:-1],equal_lambdas["lambda1"][1:-1],s=1,label="Equal angles",color="black")
        plt.title("Initial Local Lyapunov Exponent vs. Initial Angle for Multiple Configurations")
        plt.xlabel("$\\theta_0$")
    else:
        plt.scatter(energies[1:-1,0],lambdas["lambda1"][1:-1],s=1,label="$\\delta \\theta_1$",color="lightblue")
        plt.scatter(energies[1:-1,1],lambdas["lambda2"][1:-1],s=1,label="$\\delta \\theta_2$",color="blue")
        plt.scatter(energies[1:-1,2],lambdas["lambda3"][1:-1],s=1,label="$\\delta \\theta_3$",color="darkblue")
        plt.scatter(energies[1:-1,3],equal_lambdas["lambda1"][1:-1],s=1,label="Equal angles",color="black")
        plt.title("Initial Local Lyapunov Exponent vs. Energy for Multiple Configurations")
        plt.xlabel("Total Energy (J)")
    plt.ylabel("$\\lambda (s^{-1})$")
    plt.legend()
    plt.show()
    exit()

if TAKE_DATA.upper()!="LOCAL":
    NUM_ANGS=int(np.pi*1000)
    TF=5
    TF0=30
    DT=1e-4
    lyap=[]
    angs=np.linspace(0,np.pi,NUM_ANGS)#np.random.uniform(-np.pi,np.pi,size=(NUM_ANGS,3)) - if generating random
    for i in [0,1,2]:
        theta0=np.zeros(3) #np.zeros(3) - if generating all three angles modified together
        theta0[i]=1
        #theta0[(i+1)%3]=1 - if generating 2 angles modified together
        
        lyapunovs=[] 
        for ang in angs:
            print(theta0*ang)
            p=Tri_Pendulum(theta0=theta0*ang,tf=TF,dt=DT,run_scipy=True)
            lyapunovs+=[p.max_lyapunov(plot_convergence=False)]
        lyap+=[lyapunovs]
        plt.figure()
        plt.scatter(angs,lyapunovs,label="$\\lambda_{max}$",s=2)
        plt.xlabel(f"$\\theta_{i+1}$")
        plt.ylabel("$\\lambda \\ (s^{-1})$")
        plt.legend()
        plt.title(f"Maximum Lyapunov Exponent for Changed $\\theta_{i+1}$")
    
    if SAVE:
        with open(f"lyapunov_theta.npz","wb") as file:
            np.savez(file,**{"lambda"+str(i+1):lyap[i] for i in range(len(lyap))},angs=angs,dt=DT,tf=TF)   
    plt.show()

else:
    NUM_ANGS=int(np.pi*1000)
    TF=1e-3
    DT=1e-6
    lyap=[]
    angs=np.linspace(0,np.pi,NUM_ANGS)
    for i in [0,1,2]:
        theta0=np.zeros(3)
        theta0[i]=1
        
        lyapunovs=[] 
        for ang in angs:
            print(theta0*ang)
            p=Tri_Pendulum(theta0=theta0*ang,tf=TF,dt=DT,run_scipy=True)
            lyapunovs+=[p.local_lyapunov(20)]
        lyap+=[lyapunovs]
        plt.figure()
        plt.scatter(angs,lyapunovs,label="$\\lambda$",s=2)
        plt.xlabel(f"$\\theta_{i+1}$")
        plt.ylabel("$\\lambda \\ (s^{-1})$")
        plt.legend()
        plt.title(f"Local Lyapunov Exponent for Changed $\\theta_{i+1}$")
        plt.savefig(f"Images/local_lyapunov_theta_{i+1}.png")
    
    if SAVE:
        with open(f"local_lyapunov_theta.npz","wb") as file:
            np.savez(file,**{"lambda"+str(i+1):lyap[i] for i in range(len(lyap))},angs=angs,dt=DT,tf=TF)   
    plt.show()