#plotting routines
import matplotlib.pyplot as plt
from jlinfit import linfit
#from generalfitter import quadfit,quad
from numpy import array

#List of colors so as to get matching fit/data colors
myc = ['r','b','g','k','y','m','c']

def lin(x,a):
    return x*a[0] + a[1]

def plot_fit(x,y, fit, yerr=None, title="Fit plot", labels=["",""], fit_label=None):
    ax0 = plt.subplot2grid((4,4),(0,0),rowspan=3,colspan=4)
    plt.title(title)
    plt.ylabel(labels[1],fontsize=15)
    ax1 = plt.subplot2grid((4,4),(3,0),colspan=4)
    plt.ylabel("Residuals")
    plt.xlabel(labels[0],fontsize=15)
    ax0.errorbar(x,y,yerr,fmt='o', label='data')
    ax0.plot(x,fit,'-', label=fit_label)
    plt.legend(loc=0)
    yl = ax0.get_ylim()
    xl = ax0.get_xlim()

    ax1.plot(x,y-fit,'o')
    xl = ax1.get_xlim()
    ax1.plot([xl[0],xl[1]],[0,0],'k-')
    
    plt.show()

#plots linear fit & residuals
def linplot(x,y,yerr=None,plotflag=True,
            params = [0, 1], fixed = [False, False],
               limitedmin = [False, False], 
               limitedmax = [False, False],                 
               maxpars = [0,0], minpars = [0,0],    
               quiet = True,
               showme = True,
               guessinitialvals = True,fc='r',
               title="",label=["",""]):
    print(title)
    #Vectorize input so fitting routine works
    x = array(x)
    y = array(y)
    if yerr != None:
        errflag = True
        yerr = array(yerr)
    else:
        errflag = False
        
    mpp, mperr, chi2, fit = linfit(x,y,yerr,params, fixed,
               limitedmin, 
               limitedmax,                 
               maxpars, minpars,    
               quiet,
               showme,
               guessinitialvals)

    consts = [mpp[1], mpp[0]]
    res = y-lin(x,consts)

    if plotflag:
        #plotting commands
        plt.figure()
        ax0 = plt.subplot2grid((4,4),(0,0),rowspan=3,colspan=4)
        plt.title(title)
        
        plt.ylabel(label[1],fontsize=15)
##        plt.axis('equal')
        ax1 = plt.subplot2grid((4,4),(3,0),colspan=4, sharex=ax0)
        plt.ylabel("Residuals")
        plt.xlabel(label[0],fontsize=15)
        ax0.errorbar(x,y,yerr,fmt=fc+'o')
        ax0.plot(x,lin(x,consts),fc+'-')
        yl = ax0.get_ylim()
        xl = ax0.get_xlim()
        ax0.text(xl[0] + .1*(xl[1]-xl[0]),yl[0]+ .8*(yl[1]-yl[0]),
                 r'slope:%.3e $\pm$ %.3e'%(mpp[1],mperr[1]))
        ax0.text(xl[0] + .1*(xl[1]-xl[0]),yl[0]+ .75*(yl[1]-yl[0]),
                 r'incpt:%.3e $\pm$ %.3e'%(mpp[0],mperr[0]))
        ax1.plot(x,y-lin(x,consts),fc+'o')
        xl = ax1.get_xlim()
        ax1.plot([xl[0],xl[1]],[0,0],'k-')
        
##        plt.show()

    return mpp,mperr,res,fit

def quadplot(x,y,yerr=None,plotflag=True,
            p = [1, 0, 0], fixed = [False, False,False],
               limitedmin = [False, False,False], 
               limitedmax = [False, False,False],                 
               maxpars = [0,0,0], minpars = [0,0,0],    
               quiet = True,
               showme = True,
               fc='r'):
    #Vectorize input so fitting routine works
    x = array(x)
    y = array(y)
    if yerr != None:
        errflag = True
        yerr = array(yerr)
    else:
        errflag = False
        
    mpp, mperr, chi2, fit = quadfit(x,y,p,yerr,
                  fixed ,
                  limitedmin,
                  limitedmax,
                  maxpars, minpars,
                  quiet,
                  showme)

    consts = [mpp[1], mpp[0]]
    res = y-lin(x,consts)

    if plotflag:
        #plotting commands
        ax0 = plt.subplot2grid((4,4),(0,0),rowspan=3,colspan=4)
        plt.xticks(visible=False)
        plt.ylabel(r'$\rm{Frequency\ cm}^{-1}$',fontsize=25)
        ax1 = plt.subplot2grid((4,4),(3,0),colspan=4)
        plt.ylabel("Residuals",fontsize=15)
        ax0.errorbar(x,y,yerr,fmt=fc+'o')
        ax0.plot(x,quad(x,*mpp),fc+'-')
        yl = ax0.get_ylim()
        xl = ax0.get_xlim()
        ax0.text(xl[0] + .1*(xl[1]-xl[0]),yl[0]+ .8*(yl[1]-yl[0]),
                 r'sq:%.3e $\pm$ %.3e'%(mpp[0],mperr[0]))
        ax0.text(xl[0] + .1*(xl[1]-xl[0]),yl[0]+ .75*(yl[1]-yl[0]),
                 r'slope:%.3e $\pm$ %.3e'%(mpp[1],mperr[1]))
        ax0.text(xl[0] + .1*(xl[1]-xl[0]),yl[0]+ .7*(yl[1]-yl[0]),
                 r'incpt:%.3e $\pm$ %.3e'%(mpp[2],mperr[2]))
        ax1.plot(x,y-quad(x,*mpp),fc+'o')
        xl = ax1.get_xlim()
        ax1.plot([xl[0],xl[1]],[0,0],'k-')
        plt.subplots_adjust(left=.15,bottom=.11,top=.96,right=.96)
        #plt.show()

    return mpp,mperr,res


def get_contour(cn):
    '''
    Acquire contour paths from plotting object
        input: contour plotting object, cs = plt.contour(yy,zz,v, levels=[levels])

        output: tiered list
                top dimension element -> (list) of paths for each contour level
                next dimension elements -> (2D array of points) single path from given level
    '''
    contours = []
    # for each contour line
    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            paths.append(pp.vertices)
        contours.append(paths)

    return contours

def horiz_img_sequence(folder,fnames,limits,seqname='sequence',
                       annotations=None, save=True, show=False):
    '''
    create image sequence from input images

    folder- path location (string)
    fnames- list of filenames (strings)
    limits- sections of images to extract-> (a,b,c,d)
             a,b are pixel coords of top left,
             c,d coords of lower right
    '''
    n = len(fnames)
    import PIL.Image as Image,PIL.ImageFont as ImageFont,PIL.ImageDraw as ImageDraw

    ### read limits, determine shape of final sequence
    limits = np.array(limits)
    widths = limits[:,2]-limits[:,0]
    heights = limits[:,3]-limits[:,1]
        
    seq = Image.new("RGB", (widths.sum(), heights.max()), "black")

    ### Loop through images and add them to sequence
    pos =0
    for i in range(0,n):
        path = folder+fnames[i]
        im = Image.open(path)
        frame = im.crop(limits[i])
        
        seq.paste(frame,(pos,0,pos+widths[i],heights[i]))
        pos += widths[i]
        if annotations!=None:
            draw  = ImageDraw.Draw(seq)

            #Inscribe time stamp
            font = ImageFont.truetype("arial.ttf",60)
            draw.text((pos+50,50),annotations[i],font=font)
    if show:
        seq.show()
    if save:
        seq.save(folder+'%s.png'%seqname)
    return seq
