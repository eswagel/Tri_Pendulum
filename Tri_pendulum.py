import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import odeint
from copy import deepcopy
from numba import jit,njit
from perform_sympy import L,g,M
import dill

#Load the picked functions for computing the second derivatives generated with SymPy in perform_sympy.py
compute_theta1=dill.load(open("compute_theta1.pkl","rb"))
compute_theta2=dill.load(open("compute_theta2.pkl","rb"))
compute_theta3=dill.load(open("compute_theta3.pkl","rb"))

#Projection operation
def proj(u,v):
    return np.dot(u,v)/np.dot(u,u)*u

def gram_schmidt(vs):
    """ Gram-Schmidt Orthonormalization
    Parameters:
    vs: ndarray
    m X n ndarray of n m-dimensional vectors
    Returns:
    ws - m X n ndarray of normalized orthogonal vectors
    alphas - length n ndarray of magnitudes of orthogonalized vectors
    """
    alphas=np.empty(len(vs))
    ws=np.empty_like(vs)
    alphas[0]=np.sqrt(np.sum(vs[0]*vs[0]))
    ws[0]=vs[0]/alphas[0]
    for i in range(1,len(vs)):
        wi=vs[i]-proj(ws[0],vs[i]) #vs[i]-np.sum([proj(ws[j],vs[i]) for j in range(i)],axis=0)
        for j in range(1,i):
            wi=wi-proj(ws[j],vs[i])
        ai=np.sqrt(np.sum(wi*wi))
        ws[i]=wi/ai
        alphas[i]=ai
    return ws,alphas

class Tri_Pendulum:
    def __init__(self,theta0,w0=0,l=L,m=M,g=g,tf=5,dt=0.001,run_scipy=False):
        """
        Triple Pendulum Class. Although the initialization will accept an N-pendulum, analysis will only work for three angles.
        Parameters:
        theta0: iterable
            - Initial angles
        w0: iterable or float (if float, interpreted as an ndarray of same dimension as theta0)
            - Initial angular velocities (default is from rest)
        l: iterable or float (if float, interpreted as an ndarray of same dimension as theta0)
            - Lengths of each pendulum
        m: iterable or float (if float, interpreted as an ndarray of same dimension as theta0)
            - Mass of each pendulum bob
        g: float
            - Gravitational acceleration constant
        tf: float
            - End time of integration
        dt: float
            - Integration step
        run_scipy: Bool
            - If True, run the integration without calling the .scipy_trajectory() method
        """
        
        self.theta0=np.array(theta0)
        self.n=len(theta0)
        self.w0=np.array(w0) if isinstance(w0,(list,np.ndarray)) else np.array([w0]*self.n)
        self.l=np.array(l) if isinstance(l,(list,np.ndarray)) else np.array([l]*self.n)
        self.m=np.array(l) if isinstance(m,(list,np.ndarray)) else np.array([m]*self.n)
        self.g=g
        self.tf=tf
        self.dt=dt
        
        #Number of integration points
        self.npoints=int(tf/dt)
        #Times
        self.tarray = np.linspace(0.0, tf,self.npoints, endpoint = True)
        #Initial phase point (first three elements are the angle, last three are the angular velocity)
        self.tw0=np.ravel(np.array([self.theta0,self.w0]))
        
        if run_scipy:
            self.scipy_trajectory()
    
    def scipy_trajectory(self):
        self.tw = odeint(self.derivative, self.tw0, self.tarray) #shape=(self.npoints,6)
        
        #tw_unwrap holds angles not restricted to the range [-pi,pi]
        self.tw_unwrap=deepcopy(self.tw)
        #Wrap angles to between -pi and pi
        self.tw[:,:len(self.theta0)]=np.where(self.tw[:,:len(self.theta0)]>0,self.tw[:,:len(self.theta0)]%(2*np.pi),self.tw[:,:len(self.theta0)])
        self.tw[:,:len(self.theta0)]=np.where(self.tw[:,:len(self.theta0)]<0,self.tw[:,:len(self.theta0)]%(-2*np.pi),self.tw[:,:len(self.theta0)])
        self.tw[:,:len(self.theta0)]=self.tw[:,:len(self.theta0)]-2*np.pi*(self.tw[:,:len(self.theta0)]>np.pi)
        self.tw[:,:len(self.theta0)]=self.tw[:,:len(self.theta0)]+2*np.pi*(self.tw[:,:len(self.theta0)]<-np.pi)
        
        #Map back to Cartesian coordinates
        self.x=np.cumsum(self.l*np.sin(self.tw[:,:self.n]),axis=1)
        self.y=np.cumsum(-self.l*np.cos(self.tw[:,:self.n]),axis=1)
        self.vx=np.cumsum(self.l*np.cos(self.tw[:,:self.n])*self.tw[:,self.n:],axis=1)
        self.vy=np.cumsum(self.l*np.sin(self.tw[:,:self.n])*self.tw[:,self.n:],axis=1)
        #Calculate kinetic, potential, and total energy at every time
        self.energy()
        #Record the first angle to flip - not currently used
        flip=(np.argwhere(np.any(np.abs(self.tw[1:,:len(self.theta0)])>np.pi))+1)
        if len(flip):
            self.flip=flip[0]
        else:
            self.flip=None
        
    def reverse_trajectory(self):
        #Calculate the trajectory in reverse - not currently used
        self.reverse_tw=odeint(self.derivative,self.tw[-1],-self.tarray)
        
        self.reverse_tw_unwrap=deepcopy(self.reverse_tw)
        
        self.reverse_tw[:,:len(self.theta0)]=np.where(self.reverse_tw[:,:len(self.theta0)]>0,self.reverse_tw[:,:len(self.theta0)]%(2*np.pi),self.reverse_tw[:,:len(self.theta0)])
        self.reverse_tw[:,:len(self.theta0)]=np.where(self.reverse_tw[:,:len(self.theta0)]<0,self.reverse_tw[:,:len(self.theta0)]%(-2*np.pi),self.reverse_tw[:,:len(self.theta0)])
        self.reverse_tw[:,:len(self.theta0)]=self.reverse_tw[:,:len(self.theta0)]-2*np.pi*(self.reverse_tw[:,:len(self.theta0)]>np.pi)
        self.reverse_tw[:,:len(self.theta0)]=self.reverse_tw[:,:len(self.theta0)]+2*np.pi*(self.reverse_tw[:,:len(self.theta0)]<-np.pi)
    
    def derivative(self,tw,t):
        #Calculate the first and second derivatives at every point
        x = tw[:len(self.theta0)]
        v = tw[len(self.theta0):]
        a = self.A(x, v, t)
        return np.ravel(np.array([v, a]))
    
    def A(self,x,v,t):
        #Angular acceleration of each pendulum
        return np.array([compute_theta1(*x,*v),compute_theta2(*x,*v),compute_theta3(*x,*v)])
    
    def energy(self):
        #Calculate the energy of the system
        self.KE=np.sum(0.5*self.m*(self.vx**2+self.vy**2),axis=1)
        self.PE=np.sum(+self.m*self.g*self.y,axis=1)
        self.E=self.KE+self.PE
        
    def initial_energy(self):
        #Calculate the initial energy of the system without first running Scipy integration
        y=np.cumsum(-self.l*np.cos(self.theta0))
        vx=np.cumsum(self.l*np.cos(self.theta0)*self.w0)
        vy=np.cumsum(self.l*np.sin(self.theta0)*self.w0)
        return np.sum(0.5*self.m*(vx**2+vy**2)+self.m*self.g*y)
        
    def spectrum(self):
        #Calculate the FFT of the motion
        return np.fft.fftshift(np.fft.fft(self.tw[:,:self.n],axis=0))
    
    def spectrum_axis(self):
        #Calculate the x-axis for spectrum/power spectrum plots
        return np.fft.fftshift(np.fft.fftfreq(self.npoints,d=self.dt))
    
    def local_lyapunov(self,n=50,h=1e-10):
        """Calculate initial LLE.
        Parameters:
        n: int
            - Number of perturbed trajectories to average
        h: float
            - Magnitude of initial perturbation
        """
        avg_lambda=0
        for _ in range(0,n):
            #Perturbation vector with magnitude h
            epsilon=np.random.normal(size=6)
            epsilon=epsilon/np.sqrt(np.dot(epsilon,epsilon))*h
            #Integrate the perturbed trajectory
            new_p=Tri_Pendulum(theta0=self.theta0+epsilon[:3],tf=self.tf,dt=self.dt,w0=self.w0+epsilon[3:],m=self.m,l=self.l,g=self.g)
            new_p.scipy_trajectory()
            #Calculate the euclidean distance between the trajectories in phase space
            d=new_p.tw_unwrap[1:]-self.tw_unwrap[1:new_p.npoints]
            #Calculate the logarithm of the ratio between the distance and the initial perturbation
            diver=np.log(np.linalg.norm(d,axis=1)/h)
            #Add the linear fit coefficient to the average
            avg_lambda+=np.linalg.lstsq(new_p.tarray[1:,np.newaxis],diver,rcond=None)[0][0]
        print(avg_lambda/n)
        return avg_lambda/n
    def local_lyapunov_trajectory(self,n=20,h=1e-10,T=0.001,dt=None,tau=None):
        """Calculate the LLE along the entire trajectory.
        Parameters:
        n: int
            - Number of perturbed trajectories to average at each timestep
        h: float
            - Magnitude of initial perturbation
        T: float
            - Time between calculating each LLE
        dt: float
            - Integration step for LLE calculation
        tau: float
            - Length of time used for LLE calculation
        """
        if dt is None:
            dt=self.dt
        if tau is None:
            tau=T
        lyap=[]
        #Number of points for which calculating the LLE
        s=int(tau/self.dt)
        for i in range(self.npoints//s):
            lle=0
            for j in range(n):
                epsilon=np.random.normal(size=6)
                epsilon=epsilon/np.sqrt(np.dot(epsilon,epsilon))*h
                new_p=Tri_Pendulum(theta0=self.tw_unwrap[(s)*i,:3]+epsilon[:3],w0=self.tw_unwrap[(s)*i,3:]+epsilon[3:],tf=T,dt=dt,m=self.m,l=self.l,g=self.g,run_scipy=True)
                d=new_p.tw_unwrap[1:]-self.tw_unwrap[1:new_p.npoints]
                diver=np.log(np.linalg.norm(d,axis=1)/h)
                lle+=np.linalg.lstsq(new_p.tarray[1:,np.newaxis],diver,rcond=None)[0][0]
            lyap+=[lle/n]
        return np.array(lyap)
                
    def max_lyapunov(self,s=50,h=1e-10,plot_convergence=False):
        """Calculate the MLE.
        s: int
            - Number of time-steps before renormalization
        h: float
            - Magnitude of initial perturbation
        plot_convergence: bool
            - If True, plot the convergence of the MLE over time
        """
        epsilon=np.random.uniform(size=6)
        epsilon=epsilon/np.sqrt(np.sum(epsilon*epsilon))*h
        perturbed=Tri_Pendulum(theta0=self.theta0+epsilon[:3],w0=self.w0+epsilon[3:],tf=s*self.dt,dt=self.dt,m=self.m,l=self.l,g=self.g,run_scipy=True)
        
        #Slightly repetitive code
        #Divergence at renormalization
        d0_tau=(perturbed.tw_unwrap[s-1]-self.tw_unwrap[s-1])
        mag_d0_tau=np.sqrt(np.dot(d0_tau,d0_tau))
        #Renormalized divergence to begin next integration
        di_0=d0_tau/mag_d0_tau*h
        #Running track of lyapunov exponent at each step
        lyap=[np.log(mag_d0_tau/h)*s*self.dt]
        #Eventual average
        sum_logs=np.log(mag_d0_tau/h)*s
        for i in range(1,self.npoints//s):
            perturbed=Tri_Pendulum(theta0=self.tw_unwrap[(s)*i,:3]+di_0[:3],w0=self.tw_unwrap[(s)*i,3:]+di_0[3:],tf=s*self.dt,dt=self.dt,m=self.m,l=self.l,g=self.g,run_scipy=True)
            #Divergence at renormalization
            di_tau=perturbed.tw_unwrap[-1]-self.tw_unwrap[(s)*(i+1)-1]
            mag_di_tau=np.sqrt(np.dot(di_tau,di_tau))
            di_0=di_tau/mag_di_tau*h
            #Value of lambda for this normalization is the logarithm of the magnitude of the divergence after s steps
            val=np.log(mag_di_tau/h)*s
            sum_logs+=val#*self.dt
            if plot_convergence:
                lyap+=[val]
        print(sum_logs/(self.npoints))
        if plot_convergence:
            lyap=np.cumsum(np.array(lyap))/np.arange(1,len(lyap)+1)/s
            plt.figure()
            plt.plot(np.arange(len(lyap))*s*self.dt,lyap)
            plt.xlabel("Time (s)")
            plt.ylabel("$\\lambda (s^{-1})$")
            #plt.title("MLE convergence for $\\theta=$({},{},{})".format(*["$\\frac{\pi}{3}$"]*3))
            plt.show()
        #Average divergence
        return(sum_logs/(self.npoints))
            
    
    def lyapunov(self,s=150,h=1e-10,plot_convergence=False):
        """Calculate full Lyapunov Characteristic Spectrum. For the Triple pendulum, the Lyapunov vectors collapse too quickly for calculation."""
        sum_logs=np.zeros(6)
        di_0=np.identity(6)#np.random.uniform(size=(6,6))
        #di_0,_=gram_schmidt(di_0)
        di_0*=h
        lyap=np.zeros((self.npoints//s,6))
        
        for i in range(self.npoints//s):
            di_tau=np.empty((6,6))
            for j in range(6):
                perturbed=Tri_Pendulum(theta0=self.tw_unwrap[(s)*i,:3]+di_0[j,:3],w0=self.tw_unwrap[(s)*i,3:]+di_0[j,3:],tf=s*self.dt,dt=self.dt,m=self.m,l=self.l,g=self.g,run_scipy=True)
                di_tau[j]=perturbed.tw_unwrap[-1]-self.tw_unwrap[(s)*(i+1)-1]
            di_0,mag_di_tau=gram_schmidt(di_tau)
            di_0*=h
            val=np.log(np.cumprod(mag_di_tau/h))
            #print(val)
            sum_logs+=val#*self.dt
            if plot_convergence:
                lyap[i]=val
        
        if plot_convergence:
            lyap=np.cumsum(np.array(lyap),axis=0)/np.arange(1,lyap.shape[0]+1)[:,np.newaxis]
            print(lyap.shape)
            plt.figure()
            plt.plot(np.arange(lyap.shape[0])*s*self.dt,lyap)
            plt.xlabel("Time (s)")
            plt.ylabel("$\\lambda (s^{-1})$")
            plt.show()
            
        nspectrum=sum_logs*s/self.npoints#/self.dt
        spectrum=np.empty(6)
        spectrum[0]=nspectrum[0]
        for i in range(1,6):
            spectrum[i]=nspectrum[i]-np.sum(spectrum[:i])
        
        print(nspectrum)
        return spectrum
    
    def poincare(self,transient=4,n=1):
        #Unused
        transient=int(transient/self.dt)
        key=np.abs(((self.tarray[transient:]+self.dt/2) % (2*np.pi/n/np.sqrt(self.g/self.l[0])))-self.dt/2)<self.dt/2
        return self.tw[transient:][key,:]
    
    def plot_poincare(self,transient=1,n=100):
        #Unused
        poincare=self.poincare(transient,n)
        [plt.scatter(poincare[:,i],poincare[:,i+3],s=2,alpha=0.6,label="Pendulum {}".format(i+1)) for i in range(3)]
        plt.legend()
        plt.xlabel("$\\theta$ (rad)")
        plt.ylabel("$\\dot{\\theta}$ (rad/s)")
        plt.xlim(-np.pi,np.pi)
        plt.title("Poincare Section $\\theta_0={}$ \n Sampling frequency={} * $\\sqrt{{\\frac{{g}}{{l}}}}$".format(tuple(np.round(self.theta0,3)),n))
    
    def plot_trajectory(self,dim,plot_tw=True,plot_unwrap=False):
        #Unused
        if plot_tw:
            plt.plot(self.tarray,self.tw[:,dim])
        if plot_unwrap:
            plt.plot(self.tarray,self.tw_unwrap[:,dim])
    
    def autocorr(self,dim=2):
        #Unused
        return np.correlate(self.tw_unwrap[:,2],self.tw_unwrap[:,2],mode="full")
        
    def update_animation(self,frame,bobs,lines,energy_text,time_text,num_frames):
        #Function for updating the animation each frame
        cur=self.npoints//num_frames*frame
        x=self.x[cur]
        y=self.y[cur]
        lines[0].set_data([0,*x],[0,*y])
        bobs[0].set_data(x,y)
        if energy_text:
            energy_text.set_text("Total Energy: {:.2f}J".format(self.E[cur]))
        if time_text:
            time_text.set_text("T: {:.4f}s".format(self.tarray[cur]))
        return bobs[0],lines[0],energy_text
    
    def animate(self,fps=30,time_fraction=1,fig=None,energy=False,time=False):
        """Generate an animation of the triple pendulum.
        Parameters:
        fps: int
            - Frames per second
        time_fraction: float
            - How much faster animation time should move relative to real-world time
        fig: Matplotlib figure
            - If not None, attach the animation to an existing figure
        energy: bool
            - If True, show the energy of the triple pendulum at every timestep
        time: bool
            - If True, show the time in the animation
        """
        sum_l=np.sum(self.l)*1.1
        if fig is None:
            fig=plt.figure()
            ax=plt.axes(xlim=(-sum_l,sum_l),ylim=(-sum_l,sum_l))
        else:
            ax=fig.axes[0]
        bobs=ax.plot([],[],marker="o",markersize=10)
        lines=ax.plot([],[],linewidth=1,color="black")
        if energy:
            energy_text=ax.text(0.02,0.90,'',transform=ax.transAxes)
        if time:
            time_text=ax.text(0.02,0.85,'',transform=ax.transAxes)
        num_frames=round(self.npoints*self.dt*fps*time_fraction)
        return animation.FuncAnimation(fig,self.update_animation,frames=num_frames,interval=2,fargs=(bobs,lines,energy_text if energy else energy,time_text if time else time,num_frames))
 
if __name__=="__main__":  
    print("Running") 
    #If this program is run as __main__, generate an animation of 5 triple pendula released from similar initial conditions
    NUM_ANGS=2
    TF=10
    DT=0.001
    lyap=[]
    angs=[np.pi/10,np.pi/3,np.pi*3/4,np.pi]#np.linspace(0,np.pi,NUM_ANGS)
    anims=[]
    for ang in angs:
        p=Tri_Pendulum(theta0=np.ones(3)*ang,tf=TF,dt=DT,run_scipy=True)
        print("Integrated")
        anim=p.animate(energy=True,time=True,time_fraction=1)
        for i in range(NUM_ANGS-1):
            print(i+2)
            new_p=Tri_Pendulum(theta0=np.ones(3)*ang+np.random.normal(scale=0.001,size=3),tf=TF,dt=DT,run_scipy=True)
            anims+=[new_p.animate(fig=anim._fig,time_fraction=1)]
        plt.show()
        anims=[]
