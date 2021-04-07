from Tri_pendulum import *

NUM_ANGS=400
DT=1e-3
TF=20

angs=[[np.pi/4]*3,[0,0,2.43005596104368]]
title_label=[["$\pi/4$"]*3,[0,0,2.43005596104368]]
n=0
for ang in angs:
    print(ang)
    p=Tri_Pendulum(theta0=[ang,0,0],dt=DT,tf=TF,run_scipy=True)
    spect=np.abs(p.spectrum())**2
    plt.figure()
    plt.plot(p.spectrum_axis(),spect[:,0],label="$\\theta_1$")
    plt.plot(p.spectrum_axis(),spect[:,1],label="$\\theta_2$")
    plt.plot(p.spectrum_axis(),spect[:,2],label="$\\theta_3$")
    plt.xlim(-1.5,1.5)
    plt.xlabel("Frequency (1/ms)")
    plt.ylabel("Power (rad^2)")
    plt.title("Power Spectrum $\\theta$=({},{},{})".format(*title_label[n]))
    plt.legend()
    plt.show()
    
    n+=1
    