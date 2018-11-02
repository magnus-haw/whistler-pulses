from Constants import mu0,mass_elec,Kb,elec,eps0,amu,c
from numpy import array,arange,shape,cumsum,reshape,zeros,pi,cos
from numpy import matrix,sqrt,ones,log,exp,correlate,linspace
from scipy.interpolate import interp1d
from numpy.fft import fft,ifft
from file_io_lib import readVME,imacon_times,get_opt_trig
from functions import smooth,myfft,butter_bandpass_filter
from plotting import linplot
import matplotlib.pyplot as plt
import pickle
import mayavi.mlab as mlab
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
q=elec
pnames = ['pc','p1','p2','p3']
recon_shots = [22566,22567,22569,22600]
no_recon_shots = [22564,22565,22568,22570,22572,22573,22575]
maybe_shots = [22571,22574,22576,22577]

sn = 22825
sn = 22588
#sn = 22583
#sn = 22579

dt = 1e-8 #s
lowf = .2e6
highf= 20e6

a=0.5*0.0254 # 'a' is the horizontal dist between central and radial probes
x=array([[           0,   0,-a*3/sqrt(8)],
         [           a,   0, +a/sqrt(8)],
         [ -a/2, a*sqrt(3)/2, +a/sqrt(8)],
         [ -a/2,-a*sqrt(3)/2, +a/sqrt(8)],
         ])

##a=0.5*sqrt(3)*0.0254 # 'a' is the tetrahedron side length
##x1=array([[   0,             0, -a*3/(2*sqrt(6))],
##          [ a/2,-a/(2*sqrt(3)),  a/(2*sqrt(6))],
##          [-a/2,-a/(2*sqrt(3)),  a/(2*sqrt(6))],
##          [   0,     a/sqrt(3),  a/(2*sqrt(6))]])

calMatrix =array([
#Probe #C (Grey)
[[-47838.01367893,   6314.50998734,   4281.12544692],
 [ -7522.4910312,  -48736.74067395,   5387.84482546],
 [ -2655.16707069,   6500.21872864, -49169.24973902]],
#Probe #1 (Grey) 
[[ 44321.89076386, -21144.76387978,   3473.61725319],
 [-25903.029784,   -40613.64154945,  -8017.27491529],
 [ -3729.16347541,   6318.57278939, -48842.12122735]],
#Probe #2 (Blue)
[[ 46918.28677659,  -2604.47066548,  -3920.14360687],
 [ -8069.43905989, -48313.15332326,   2523.71091258],
 [  2863.92700168,   3836.98378501,  50207.58134066]],
#Probe #3 (Pink)
[[ 49370.57276728,  15929.17368495,   -432.10427375],
 [ 13290.79612253,  -44924.17182229, -3378.76560238],
 [   854.33553224,   3829.36550217,  51789.28906524]],
])

def getProbeRawData(path,shot,name,nchan):
    fname = 'quad_whistler_%s_%i.dat'%(name,shot)
    d = readVME(path+fname, cols=2**13, rows=nchan+1)
    time = d[0,:]
    dp = reshape(d[1:,:],(2**13,nchan)).T
    return time,dp

def calibrateProbeData(path,shot,probename,calMatrix=calMatrix):
    time, rawdata = getProbeRawData(path,shot,probename,3)
    n = len(rawdata[0])
    vdata = zeros((3,n))
    vdata[0,:] = rawdata[0]
    vdata[1,:] = rawdata[1]
    vdata[2,:] = rawdata[2]

    c = calMatrix[pnames.index(probename)]
    
    return time,cumsum(array(matrix(c)*vdata)*dt,axis=1)

def get_J(x,B):
    ### B array dimensions [probe, xyz, time]
    B = array(B)
    
    ### Get matrix for inversion
    ###    for calculation of dB/dx,dB/dy,dB/dz
    mat_pos = ones((4,4))
    mat_pos[:,0:3] = x
    m = matrix(mat_pos).I

    dBx = m*B[:,0,:]; dBy = m*B[:,1,:]; dBz = m*B[:,2,:]
    Jx = (dBz[1,:] - dBy[2,:])/mu0
    Jy = (dBx[2,:] - dBz[0,:])/mu0
    Jz = (dBy[0,:] - dBx[1,:])/mu0

    return array([dBx[3,:],dBy[3,:],dBz[3,:]]),array([Jx,Jy,Jz])

def periodic_corr(x, y):
    """Periodic correlation, implemented using the FFT.

    x and y must be real sequences with the same length.
    """
    return ifft(fft(x) * fft(y).conj()).real

def interp(time,path,n):
    f = interp1d(time, path, kind='cubic',axis=0)
    itime = linspace(time[0],time[-1],n)
    return itime,f(itime)

##impath = '/home/magnus/flux/imacon/'
##imname = 'imacon_%i.TIF'%shot_num
##start,exposure = imacon_times(impath+imname)
##print "Imacon times: ",start

for shot_num in [sn]:
    path = '/home/magnus/flux/shots/%i/'%shot_num
##    path = '../data/%i/'%shot_num
    fname = 'quad_whistler_%s_%i.dat'%(pnames[0],shot_num)
    ##vname  = 'spheromak_vgun_%i.dat'%shot_num
    ##iname  = 'spheromak_igun_%i.dat'%shot_num
    optname= 'spheromak_optical_trigger_%i.dat'%shot_num
    ##vout = readVME(path+vname,cols=8192*2,rows=1)
    ##iout = readVME(path+iname,cols=8192*2,rows=1)
    ##optout=readVME(path+optname,cols=8192,rows=1)
    t0 = 19.62#get_opt_trig(path+optname)
    tind = int(t0*100)
    if shot_num == 22569:
        t1= tind+720
        t2= tind+1100
    if shot_num == 22600:
        t1= tind+550
        t2= tind+800
    if shot_num == 22579:
        t1= tind+600
        t2= tind+1000

    #print magnus
    time,Bc = calibrateProbeData(path,shot_num,'pc')
    time,B1 = calibrateProbeData(path,shot_num,'p1')
##    time,B2 = calibrateProbeData(path,shot_num,'p2')
    time,B3 = calibrateProbeData(path,shot_num,'p3')

    ### calib p2-z
    time, dp3 = getProbeRawData(path,shot_num,'p3',3)
    time, dp2 = getProbeRawData(path,shot_num,'p2',3)
    dp2[2]  = dp2[2] #- smooth(dp2[2]-dp3[2],window_len=350)[1:]

    n = len(dp2[0])
    vdata = zeros((3,n))
    vdata[0,:],vdata[1,:],vdata[2,:] = dp2
    B2 = cumsum(array(matrix(calMatrix[2])*vdata)*dt,axis=1)
    
##    plt.plot(time-t0, dp2[2],label='dp3')
##    plt.plot(time-t0, smooth(dp2[2]-dp3[2],window_len=350),label='dp2')
##    plt.legend(loc=0)
##    plt.show()
    Blist = [Bc,B1,B2,B3]

    ### Calculate current 
    B0, J = get_J(x,Blist)

    ### create filtered B-field set
    Blistf = zeros(shape(Blist))
    for p in range(0,4):
        for d in range(0,3):
            Blistf[p,d,:] = butter_bandpass_filter(Blist[p][d,:],lowf,highf,1/dt)

    ################################################
    ############### PLOTTING SECTION ###############
    ################################################
    plot_raw =1
    plot_spectrogram = 0
    plot_Bfield=1
    plot_center_values= 0
    plot_vector_anim = 0
    fourier_plot =0
    kvector =0
    plot_correlation = 0
    plot_hodogram =0
    dlabel = ['x','y','z']
    if plot_raw:
        for pnum in range(0,4):
            time,dp = getProbeRawData(path,shot_num,pnames[pnum],3)
            c = calMatrix[pnum]
            #dp = array(dp)
            dp = array(matrix(c)*dp)
            Bxdiffer = (butter_bandpass_filter(dp[0,:],lowf,highf,1/dt))
            Bydiffer = (butter_bandpass_filter(dp[1,:],lowf,highf,1/dt))
            Bzdiffer = (butter_bandpass_filter(dp[2,:],lowf,highf,1/dt))
            Bmagdiffer = sqrt(Bxdiffer**2 + Bydiffer**2 + Bzdiffer**2)
            plt.figure(3)
            plt.plot(time-t0,Bmagdiffer)
            for i in [2]:
                
                plt.figure(1)
                plt.title('B-dot raw data, single pair, shot #%i'%shot_num)
                plt.plot(time-t0,(dp[i,:]),label='%s %s'%(pnames[pnum],dlabel[i]))
                plt.xlim([0,12])
                plt.ylabel(r'$\dot{B}\ $(T/s)',fontsize=15)
                plt.ylim([-700,50])
                plt.xlabel('Time (us)',fontsize=15)
                plt.legend()
                plt.tight_layout()
                plt.savefig("raw_shot%i.png"%shot_num)

                plt.figure(2)
                Bdiffer = (butter_bandpass_filter(dp[i,:],lowf,highf,1/dt))
                plt.title('B-dot data, (%.1f-%iMHz), shot #%i'%(lowf/1e6,highf/1e6,shot_num))
                plt.plot(time-t0,Bdiffer,label='%s %s'%(pnames[pnum],dlabel[i]))
                plt.xlim([0,12])
                plt.ylim([-400,650])
                plt.ylabel(r'$\dot{B}\ $(T/s)',fontsize=15)
                plt.xlabel('Time (us)',fontsize=15)
                plt.legend()
                plt.tight_layout()
                plt.savefig("filtered_shot%i.png"%shot_num)
                

                plt.figure(4) 
                plt.title("Bdot Spectrogram Shot %i"%shot_num)
                NFFT = 100 # the length of the windowing segments
                Fs = int(1.0/dt)  # the sampling frequency
                xspect = dp[0,:]

                Pxx,freqs,bins,im = plt.specgram(xspect,NFFT=NFFT,Fs=Fs,noverlap=NFFT-1)
        plt.show()
        
    if plot_Bfield:
        plt.figure(figsize=(10,8))
        ax0 = plt.subplot2grid((4,4),(0,0),rowspan=3,colspan=4)
        plt.ylabel('B (T)')
        plt.title("B values, Shot %i, No RT"%shot_num)
        for j in [3]:
            for i in [0,1,2]:
                y = Blist[j][i,:]
                ax0.plot(time-t0,y,label='B%s'%dlabel[i])
            Bmag = sqrt(Blist[j][0,:]**2 + Blist[j][1,:]**2 + Blist[j][2,:]**2 )
            ax0.plot(time-t0,Bmag,'-',label='B mag %i'%j)
        #plt.xlim([(t1-tind)/100,(t2-tind)/100])
        plt.ylim([0,0.005])
        plt.legend(loc=0)
    ##    plt.xlabel('time (us)')
        

        ax1 = plt.subplot2grid((4,4),(3,0),colspan=4, sharex=ax0)
    ##    plt.title("dB values")
        for j in range(0,4):
    ##        for i in [0]:
    ##            y = Blist[j,i,:]
    ##            yf = butter_bandpass_filter(y,lowf,highf,1e8)
    ##            ax1.plot(time-t0,yf,label='dB%s'%dlabel[i])
            Bmag = sqrt(Blist[j][0,:]**2 + Blist[j][1,:]**2 + Blist[j][2,:]**2 )
            dBmag= butter_bandpass_filter(Bmag,lowf,highf,1e8)
            ax1.plot(time-t0,dBmag,'-',label='dB mag %i'%j)
        #plt.xlim([time[t1]-t0,time[t2]-t0])
        plt.ylim([-15e-6,15e-6])
        plt.xlabel('time (us)')
        plt.ylabel('dB (T)')
    ##    plt.legend(loc=2)
        plt.savefig("B-shot%i.png"%shot_num)
        plt.show()
        
    ##    plt.figure() 
    ##    plt.title("dB Spectrum")
    ##    y= B0[0][0,:]
    ##    yf = butter_bandpass_filter(y,lowf,highf,1e8)
    ##    
    ##    w,f= myfft(yf[tind+2600:],1e8)
    ##    F = abs(f)
    ##    plt.plot(w/1e6,F,label=r"$\mathscr{F}$(B)")
    ##    plt.yscale('log')
    ##    plt.xscale('log')
    ##    plt.ylim([.01,100])
    ##    plt.xlim([0,5])
    ##    plt.xlabel('Freq (MHz)')
    ##    plt.ylabel('arb.')

    if plot_center_values:

        ### Calculate current 
        B0, J = get_J(x,Blistf)
        
        plt.figure()
        plt.title("B values")
        for i in range(0,3):
            y = B0[i][0,:]
            plt.plot(time-t0,y,label='B%s'%dlabel[i])
        B0mag = sqrt(B0[0][0,:]**2 + B0[1][0,:]**2 + B0[2][0,:]**2 )
        plt.plot(time-t0,B0mag,'k-',label='B mag')
        plt.xlim([(t1-tind)/100,(t2-tind)/100])
        plt.legend(loc=0)
        plt.xlabel('time (us)')
        plt.ylabel('B (T)')
        
        plt.figure()
        plt.title("J values")
        for i in range(0,3):
            y = J[i][0,:]
            plt.plot(time-t0,y,label='J%s'%dlabel[i])
        Jmag = sqrt(J[0][0,:]**2 + J[1][0,:]**2 + J[2][0,:]**2 )
        plt.plot(time-t0,Jmag,'k-',label='J mag')
        plt.xlim([(t1-tind)/100,(t2-tind)/100])
        plt.legend(loc=0)
        plt.xlabel('time (us)')
        plt.ylabel('J (A/m^2)')
        
##        plt.figure(figsize=(8,4))    
##        plt.title("dB values")
##        for i in range(0,3):
##            y = B0[i][0,:]
##            yf = butter_bandpass_filter(y,lowf,highf,1e8)
##            plt.plot(time-t0,yf,label='dB%s'%dlabel[i])
##        plt.xlim([time[t1]-t0,time[t2]-t0])
##        plt.ylim([yf[t1:t2].min(),yf[t1:t2].max()])
##        plt.xlabel('time (us)')
##        plt.ylabel('dB (T)')
##        plt.legend(loc=0)
##        
##        plt.figure() 
##        plt.title("dB Spectrum")
##        y= B0[0][0,:]
##        yf = butter_bandpass_filter(y,lowf,highf,1e8)
##        
##        w,f= myfft(yf[tind+2600:],1e8)
##        F = abs(f)
##        plt.plot(w/1e6,F,label=r"$\mathscr{F}$(B)")
##        plt.yscale('log')
##        plt.xscale('log')
##        plt.ylim([.01,100])
##        plt.xlim([0,5])
##        plt.xlabel('Freq (MHz)')
##        plt.ylabel('arb.')
        

    ##    w_ind = int(len(w)/2)
    ##    mpp,mperr,res, fit = linplot(log(w[w_ind+1:w_ind+23]),
    ##                                 log(F[w_ind+1:w_ind+23]),plotflag=0)
    ##    mpp1,mperr1,res1,fit1 = linplot(log(w[w_ind+23:w_ind+150]),
    ##                                    log(F[w_ind+23:w_ind+150]),plotflag=0)
    ##
    ##    plt.plot(w[w_ind+1:w_ind+23]/1e6,exp(fit),
    ##             label='slope: %.2f +/- %.2f'%(mpp[1],mperr[1]))
    ##    plt.plot(w[w_ind+23:w_ind+150]/1e6,exp(fit1),
    ##             label='slope: %.2f +/- %.2f'%(mpp1[1],mperr1[1]))
    ##    
        plt.legend(loc=0)
        plt.show()

    if fourier_plot:
        plt.figure(figsize=(8,4))
        pnum=1
        time,dp = getProbeRawData(path,shot_num,pnames[pnum],3)
        c = calMatrix[pnum]
        dB = array(matrix(c)*dp)
        i1,i2 = 600,850
        dBf = (butter_bandpass_filter(dB[0,:],lowf,highf,1/dt))
        ti,dBi = interp(time[i1+tind:i2+tind]-t0,dBf[i1+tind:i2+tind],10000)

        n = 1000
        f_array = ones((n/2,i2-i1))
        for i in range(i1+tind,i2+tind):
            #y = B0[0][0,i:i+100]
            y = dBi[i:i+n]
            w,f= myfft(y,1e8)
            
            f_array[:,i-i1-tind] = (abs(f)[n/2:])
    ##        if i%100 == 0:
    ##            plt.plot(abs(f))
    ##            plt.show()
        plt.imshow(f_array[:,:],extent=[i1/100,i2/100,0,50],origin='lower')
        plt.colorbar()
        plt.show()

    if kvector:
        lf=1e6; hf=20e6
        w,Bx= myfft(butter_bandpass_filter(B0[0,0,t1:t2],lf,hf,1e8),1e8)
        w,By= myfft(butter_bandpass_filter(B0[1,0,t1:t2],lf,hf,1e8),1e8)
        w,Bz= myfft(butter_bandpass_filter(B0[2,0,t1:t2],lf,hf,1e8),1e8)

        w,Jx= myfft(butter_bandpass_filter(J[0,0,t1:t2],lf,hf,1e8),1e8)
        w,Jy= myfft(butter_bandpass_filter(J[1,0,t1:t2],lf,hf,1e8),1e8)
        w,Jz= myfft(butter_bandpass_filter(J[2,0,t1:t2],lf,hf,1e8),1e8)
##        w,Bx= myfft(B0[0,0,t1:t2],1e8)
##        w,By= myfft(B0[1,0,t1:t2],1e8)
##        w,Bz= myfft(B0[2,0,t1:t2],1e8)
##
##        w,Jx= myfft(J[0,0,t1:t2],1e8)
##        w,Jy= myfft(J[1,0,t1:t2],1e8)
##        w,Jz= myfft(J[2,0,t1:t2],1e8)

##        Bmag2 = (abs(Bx)**2 + abs(By)**2 + abs(Bz)**2)/mu0
        Bmag2 = (Bx*Bx.conj() + By*By.conj()+Bz*Bz.conj())/mu0

        kx = ( (complex(0,1.)*(Jy*Bz.conj() - Jz*By.conj())).real )/Bmag2
        ky = ( (complex(0,1.)*(Jz*Bx.conj() - Jx*Bz.conj())).real )/Bmag2
        kz = ( (complex(0,1.)*(Jx*By.conj() - Jy*Bx.conj())).real )/Bmag2
        
##        plt.plot(w/1e6,smooth(kx),'-o',label='kx')
##        plt.plot(w/1e6,smooth(ky),'-o',label='ky')
##        plt.plot(w/1e6,smooth(kz),'-o',label='kz')
##        plt.plot(w/1e6,smooth(sqrt(kx**2 + ky**2+ kz**2)),'-o',label='|k|')

        plt.plot(w/1e6,(kx),'-o',label='kx')
        plt.plot(w/1e6,(ky),'-o',label='ky')
        plt.plot(w/1e6,(kz),'-o',label='kz')
        plt.plot(w/1e6,(sqrt(kx**2 + ky**2+ kz**2)),'-o',label='|k|')
        
        f = 7e6; theta = 0
        B_avg = .00225; c = 2.99e8
        n0 = 2.3e16
        w_pe = sqrt(n0 * q*q/(eps0*mass_elec))
        w_ce = q*B_avg/mass_elec

        ww = w*2*pi
        vA = B_avg/sqrt(n0*amu*mu0)
        
        ksq = (ww/c)**2 - ((w_pe/c)**2)/(1-abs(w_ce*cos(theta)/ww) )
        kalfven = ww/vA
        
        plt.plot(w/1e6,sqrt(ksq),'y--',label="whistler dispersion")
        plt.plot(w/1e6,kalfven,'c--',label="alfven dispersion")
        
    ##    plt.plot(w,smooth(Jx))
        plt.title(r'Wave Pulse Dispersion, shot #%i, %.1f-%.1f $\mu$s'%(sn,(t1-tind)/100.,(t2-tind)/100.),fontsize=20)
        plt.ylabel(r'|k| (m$^{-1}$)',fontsize=15)
        plt.xlabel('Frequency (MHz)',fontsize=15)
        plt.xlim([0,20]); plt.ylim([-55,110])
        plt.legend(loc=0)
        plt.show()

    if plot_correlation:
        dind = 0
        plt.figure()
        thing1 = B[0][dind,t1:t2]; thing2 = B[1,dind,t1:t2]
        cc= correlate(thing1,thing2,mode='same')
        ind = cc.argmax()
        plt.plot(arange(0,len(cc),1)-len(cc)/2,cc)
        plt.plot(ind-len(cc)/2,cc[ind],'ro')
        print( -(ind-len(cc)/2)*.01, 'us phase offset')
        
    ##    plt.xlim([-30,30])

        plt.figure()
        plt.plot(thing1);plt.plot(thing2)
        
        plt.show()

    if plot_hodogram:
        if sn == 22600:
            t1,t2=600+2055,930+2055
        #matplotlib path hodogram
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        pnum = 0 #probe number

        xi,yi,zi = Blist[pnum][0,t1:t2]*1e4,Blist[pnum][1,t1:t2]*1e4,Blist[pnum][2,t1:t2]*1e4
##        xi,yi,zi = Blistf[pnum][0,t1:t2]*1e4,Blistf[pnum][1,t1:t2]*1e4,Blistf[pnum][2,t1:t2]*1e4
##        xi,yi,zi = J[0,0,t1:t2],J[1,0,t1:t2],J[2,0,t1:t2]
##        xi,yi,zi = B0[0,0,t1:t2],B0[1,0,t1:t2],B0[2,0,t1:t2]
        ax.plot(xi,yi,zi, 'bo-',label='B-center')
        ax.plot(xi[0:1],yi[0:1],zi[0:1], 'ro',label='start')
        
        
##        time,dp = getProbeRawData(path,shot_num,pnames[pnum],3)
##        c = calMatrix[pnum]
##        dp = array(matrix(c)*dp)
##        ax.plot(dp[0,t1:t2],
##                dp[1,t1:t2],
##                dp[2,t1:t2], 'bo-',label='dB-center')

        ### Setup equal aspect ratio for all three axes
        
        dx,dy,dz = max(xi)-min(xi),max(yi)-min(yi),max(zi)-min(zi)
        max_range = max(dx,dy,dz)
        mid_x,mid_y,mid_z = min(xi)+dx/2.,min(yi)+dy/2.,min(zi)+dz/2.
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

##        ax.plot(xi, zi, zs=mid_y + max_range/2, zdir='y')
##        ax.plot(yi, zi, zs=mid_x - max_range/2, zdir='x')
##        ax.plot(xi, yi, zs=mid_z - max_range/2, zdir='z')

        ax.set_xlabel('Bx (G)')
        ax.set_ylabel('By (G)')
        ax.set_zlabel('Bz (G)')
        plt.show()

    if plot_vector_anim:
        sf = 40 # scale factor for vectors relative to spatial scale
        if sn == 22600:
            t1,t2=600+2055,930+2055

        #plot probe positions with white spheres
        mlab.figure(size=(800,700))
        x0,y0,z0=[],[],[]
        for i in range(0,4):
            mlab.points3d(x[:,0],x[:,1],x[:,2], scale_factor=.001)
            x0.append(x[:,0]);y0.append(x[:,1]);z0.append(x[:,2]);
        #plot b-vector path
##        bcp = [Blist[0][0,:],Blist[0][1,:],Blist[0][2,:]]
##        bcp1 = [Blist[1][0,:],Blist[1][1,:],Blist[1][2,:]]
##        bcp2 = [Blist[2][0,:],Blist[2][1,:],Blist[2][2,:]]
##        bcp3 = [Blist[3][0,:],Blist[3][1,:],Blist[3][2,:]]

        bcp  = [Blist[0][0,:]-Blist[0][0,t1],Blist[0][1,:]-Blist[0][1,t1],Blist[0][2,:]-Blist[0][2,t1]]
        bcp1 = [Blist[1][0,:]-Blist[1][0,t1],Blist[1][1,:]-Blist[1][1,t1],Blist[1][2,:]-Blist[1][2,t1]]
        bcp2 = [Blist[2][0,:]-Blist[2][0,t1],Blist[2][1,:]-Blist[2][1,t1],Blist[2][2,:]-Blist[2][2,t1]]
        bcp3 = [Blist[3][0,:]-Blist[3][0,t1],Blist[3][1,:]-Blist[3][1,t1],Blist[3][2,:]-Blist[3][2,t1]]

    ##    bcp = [B[0,0,:],B[0,1,:],B[0,2,:]]
    ##    bcp1 = [B[1,0,:],B[1,1,:],B[1,2,:]]
    ##    bcp2 = [B[2,0,:],B[2,1,:],B[2,2,:]]
    ##    bcp3 = [B[3,0,:],B[3,1,:],B[3,2,:]]
        bcpn = sqrt(bcp[0]**2 + bcp[1]**2 + bcp[2]**2.)
        r = .0001
        BcPath = mlab.plot3d(x[0,0]+bcp[0][t1:t2]*sf,
                             x[0,1]+bcp[1][t1:t2]*sf,
                             x[0,2]+bcp[2][t1:t2]*sf,
                             line_width=.0025,tube_radius=r,color=(0,0,1))
        BcPath1 = mlab.plot3d(x[1,0]+bcp1[0][t1:t2]*sf,
                             x[1,1]+bcp1[1][t1:t2]*sf,
                             x[1,2]+bcp1[2][t1:t2]*sf,
                             line_width=.0025,tube_radius=r,color=(1,0,1))
        BcPath2 = mlab.plot3d(x[2,0]+bcp2[0][t1:t2]*sf,
                             x[2,1]+bcp2[1][t1:t2]*sf,
                             x[2,2]+bcp2[2][t1:t2]*sf,
                             line_width=.0025,tube_radius=r,color=(.5,1,.5))
        BcPath3 = mlab.plot3d(x[3,0]+bcp3[0][t1:t2]*sf,
                             x[3,1]+bcp3[1][t1:t2]*sf,
                             x[3,2]+bcp3[2][t1:t2]*sf,
                             line_width=.0025,tube_radius=r,color=(0,0,0))
        pc = BcPath.mlab_source ;p1 = BcPath1.mlab_source
        p2 = BcPath2.mlab_source;p3 = BcPath3.mlab_source
        
        #plot b-vectors at positions
##        for i in range(0,4):
##            pos = x[i,:]
##            B = Blist[i][:,t1]
##            vectors = mlab.quiver3d(pos[0], pos[1], pos[2],
##                                B[0], B[1], B[2],
##                                scale_factor= 1)
##            vectors.glyph.glyph_source.glyph_source.glyph_type = 'thick_arrow'
##            vectors.glyph.glyph_source.glyph_source.filled = True
##            vectors.glyph.glyph_source.glyph_source.cross = True
##            vs = vectors.mlab_source

        # camera position in 3D
        mlab.view(focalpoint=array([0., 0., 0.]),elevation = 65.,azimuth=100.,distance =.065)

        fig = mlab.gcf()
        fig.scene.show_axes = True

        triangles=[(0,1,2),(1,2,3),(2,3,0),(3,0,1)]
        mlab.triangular_mesh(x0, y0, z0, triangles,color=(0.5,0.5,0.5))
##
##        # animate vector motion
##        for t in arange(t1,t2,10):
##            vs.set(u=B[:,0,t], v=B[:,1,t], w=B[:,2,t])
##            fig.scene.render()
##        for t in arange(0,500,1):
##            pc.set(x=x[0,0]+bcp[0][t1+t:t2+t]*sf,
##                   y=x[0,1]+bcp[1][t1+t:t2+t]*sf,
##                   z=x[0,2]+bcp[2][t1+t:t2+t]*sf)
##            p3.set(x=x[3,0]+bcp3[0][t1+t:t2+t]*sf,
##                   y=x[3,1]+bcp3[1][t1+t:t2+t]*sf,
##                   z=x[3,2]+bcp3[2][t1+t:t2+t]*sf)
##            p2.set(x=x[2,0]+bcp2[0][t1+t:t2+t]*sf,
##                   y=x[2,1]+bcp2[1][t1+t:t2+t]*sf,
##                   z=x[2,2]+bcp2[2][t1+t:t2+t]*sf)
##            p1.set(x=x[1,0]+bcp1[0][t1+t:t2+t]*sf,
##                   y=x[1,1]+bcp1[1][t1+t:t2+t]*sf,
##                   z=x[1,2]+bcp1[2][t1+t:t2+t]*sf)
##            vs.set(u=Blist[:,0,t2+t], v=Blist[:,1,t2+t], w=Blist[:,2,t2+t])
##            fig.scene.render()
##    ##        mlab.savefig("quad_movie_%i_%i-%i_frame_%04d.png"%(shot_num,t1-tind,t2-tind+500,t))
        mlab.show()
