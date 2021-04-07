from Tri_pendulum import *

DT=1e-3
TF=20
TAU=0.01
T=1e-2
SAVE=True

ang=np.pi/2 #Angle at which to generate the plots of the LLE

for i in range(4):
    if i==3:
        theta0=np.ones(3)
    else:
        theta0=np.zeros(3)
        theta0[i]=1
    theta0*=ang
    p=Tri_Pendulum(theta0=theta0,dt=DT,tf=TF,run_scipy=True)
    lles=p.local_lyapunov_trajectory(T=T,dt=1e-4,tau=TAU)*T
    ft_lles=np.abs(np.fft.fft(lles))**2
    fig=plt.figure()
    plt.plot(np.arange(len(lles))*TF/len(lles),lles)
    plt.xlabel("Time (s)")
    plt.ylabel("$\lambda_L (s^{-1})$")
    fig.subplots_adjust(left=0.1)
    if i<3:
        plt.title("LLE vs. Time for $\\theta_{}=\\frac{{\pi}}{{2}}$".format(i+1))
        if SAVE: plt.savefig("Images/LLE_theta_"+str(i)+".png")
    else:
        plt.title("LLE vs. Time for All Angles $\\frac{{\pi}}{{2}}$")
        if SAVE: plt.savefig("Images/LLE_theta_equal.png")
plt.show()