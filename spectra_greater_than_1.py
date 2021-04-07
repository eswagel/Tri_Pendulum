from Tri_pendulum import *

NUM_ANGS=400
DT=1e-3
TF=20

angs=np.linspace(0,np.pi,NUM_ANGS)
num_at_angle=[]
n=0
for ang in angs:
    print(ang)
    p=Tri_Pendulum(theta0=[ang,0,0],dt=DT,tf=TF,run_scipy=True)
    spect=np.abs(p.spectrum())**2
    num_greater=(np.sum(spect>1))
    #print(num_greater)
    num_at_angle+=[num_greater]
    
    n+=1

plt.figure()
plt.plot(angs,num_at_angle)
plt.xlabel("$\\theta$ (rad)")
plt.ylabel("# Frequencies with Power>1")
plt.title("Discrete Power Spectrum w/ Power>1 for Changed $\\theta_1$")
plt.show()
    