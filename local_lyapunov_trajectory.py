from Tri_pendulum import *
from sklearn.metrics import r2_score

DT=1e-4
TF=10
TAU=0.01
T=5e-3
SAVE=True #Choose whether to save plots and save the lles
LOAD=True #Choose whether to load the lles from saved data

NUMERATOR=1
DENOMINATOR=2
ang=np.pi*NUMERATOR/DENOMINATOR #Angle at which to generate the plots of the LLE

colors=["blue","orange","green","red"]
saved=np.load("lles.npz")
lles_save=[]
tws_save=[]
kes_save=[]

#Create the figure for plotting the lles vs. KE
ke_fig=plt.figure(figsize=(8,4.8))
ke_ax=ke_fig.add_subplot(111)

#Get the saved MLE values
lambdas=np.load("lyapunov_theta_10.npz")
equal_lambdas=np.load("lyapunov_equal_angs_10.npz")

for i in range(4):
    if i==3:
        theta0=np.ones(3)
    else:
        theta0=np.zeros(3)
        theta0[i]=1
    theta0*=ang
    #Generate pendulum and run scipy if not loading data
    p=Tri_Pendulum(theta0=theta0,dt=DT,tf=TF,run_scipy=not LOAD)
    #Get lles from saved values or from pendulum class
    lles=saved["lles"][i] if LOAD else p.local_lyapunov_trajectory(T=T,tau=TAU)
    #Get saved value of MLE
    if i==3:
        MLE=(equal_lambdas["lambda1"][np.argwhere(equal_lambdas["angs"]==ang)])[0][0]
    else: MLE=(lambdas["lambda"+str(i+1)][np.argwhere(lambdas["angs"]==ang)])[0][0]
    print((MLE-lles.mean())/MLE*100)
    kes=saved["kes"][i,::int(saved["kes"].shape[1]/len(lles))] if LOAD else p.KE[::int(TAU/p.dt)]
    if not LOAD:
        lles_save+=[lles]
        tws_save+=[p.tw]
        kes_save+=[p.KE]
    #Plot LLE over time
    fig=plt.figure(figsize=(8,4.8))
    ax1=plt.subplot(1,1,1)
    plt.plot(np.arange(len(lles))*TF/len(lles),lles,color=colors[i])
    plt.plot([0,TF],[MLE,MLE],color="black",linestyle="dashed",lw=1,label="MLE")
    plt.xlabel("Time (s)")
    plt.ylabel("$\lambda_L (s^{-1})$")
    plt.legend()
    
    if i<3:
        plt.title(f"LLE over Time for $\\theta_{i+1}=\\frac{{{'' if NUMERATOR==1 else NUMERATOR}\pi}}{{{DENOMINATOR}}}$")
        fig.tight_layout()
        if SAVE: plt.savefig("Images/LLE_theta_"+str(i)+".png")
    else:
        plt.title(f"LLE over Time for All Angles $\\frac{{{'' if NUMERATOR==1 else NUMERATOR}\pi}}{{{DENOMINATOR}}}$")
        fig.tight_layout()
        if SAVE: plt.savefig("Images/LLE_theta_equal.png")
    #Plot LLE vs. KE
    ke_ax.scatter(kes,lles,s=2,label=f"$\\theta_{i+1}=\\frac{{{'' if NUMERATOR==1 else NUMERATOR}\pi}}{{{DENOMINATOR}}}$" if i<3 else "Equal angles",color=colors[i])

ke_ax.legend(markerscale=5)
ke_ax.set_xlabel("KE (J)")
ke_ax.set_ylabel("$\lambda_L (s^{-1})$")
ke_ax.set_title(f"LLE vs. KE for Four Configurations with $\\theta=\\frac{{{'' if NUMERATOR==1 else NUMERATOR}\pi}}{{{DENOMINATOR}}}$")
ke_ax.set_ylim(14,20)
ke_fig.savefig(f"Images/lle_ke_{NUMERATOR}_{DENOMINATOR}.png")
if SAVE and not LOAD: np.savez("lles.npz",lles=lles_save,tws=tws_save,kes=kes_save,tf=TF,tau=TAU,T=T,dt=DT)
plt.show()
