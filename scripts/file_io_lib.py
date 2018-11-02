## My file io library
## Mar26, 2013
import array as ar
import numpy as np
import os
import struct
from os.path import isfile, exists

def write_latex(mylist, fname):
    delim = '&'
    fout = open(fname, 'w')
    fout.write("\\begin{tabular}{|c|")
    for k in range(1,len(mylist)):
        fout.write("c|")
    fout.write("}\n")
    for i in range(0,len(mylist)):
        mystr = ""
        for j in range(0,len(mylist[i])-1):
            mystr += str(mylist[i][j]) + delim
        mystr += str(mylist[i][j])
        
        mystr += '\\\\'+'\n'
        mystr += '\hline\n'
        fout.write(mystr)
    fout.write("\end{tabular}")
    fout.close()

def toLatex(fname, delim):
    mydata = np.loadtxt(fname, delimiter=delim,dtype=np.str_,skiprows=1)
    fout = fname.split('.')[0]
    write_latex(mydata, fout + '_latex.txt')

def readVME(fname,cols=8192,rows=2,dtype='f'):
    '''
    Reads data from VME->IDL output files
    fname: full file path,
    cols: number of time steps,
    rows: number variables,
    dtype: desired python data type ('f'-> float)
    '''
    fin = open(fname,'rb')
    a = ar.array('f')
    a.fromfile(fin, cols*rows)
    fin.close()
    ret = []
    for i in range(0,cols*rows,cols):
        ret.append( a[i:i+cols] )
    return np.array(ret)

def get_data(dbase, name, shot, header_suffix='.sph', data_suffix='.dat', row_ind=0, col_ind=1):
    fname = dbase + name+ '_%i'%(shot) + data_suffix
    header_name = dbase + name+ '_header_%i'%(shot) + header_suffix
    head = np.loadtxt(header_name,delimiter=',')

    rows, cols = int(head[row_ind]), int(head[col_ind])
    data = readVME(fname,cols=cols,rows=rows)
    return data, head

def fixfile(f0):
    fin = open(f0,'r')
    f = fin.read()
    if f.find('\n') >=0:
        f = f.replace('\r\n','\n')
    else:
        f = f.replace('\r','\n')
    fin.close()
    fout= open(f0,'w')
    fout.write(f)
    fout.close()
    print("fixed")
        
###Ported from Bao's idl code: extract_imacon_times_from_footer.pro
def imacon_times(path, TIME_CONVERSION=1e3):

    def lindgen(n):
        return np.arange(0,n)
    #-------------------- User Adjustable  constants ---------------------------------------
    #TIME_CONVERSION = 1e3		    # Set at 1 for ns, 1e3 for microsecond, 1e6 for ms.
    #---------------------------------------------------------------------------------------

    #-------------------- Relevant  constants ----------------------------------------------

    BYTE_FOOTER_PACKET_SIZE = 11016	    # Number of bytes / info packet in footer for each CCD.
    BYTE_CCD1_START_TIME = 37651208	    # Byte location of Start time of first frame in ns.
    BYTE_CCD1_EXPOSURE_1 = 37651344
    BYTE_CCD1_EXPOSURE_2 = 37651352
    BYTE_CCD1_DELAY = 37651600
    FRAMES = 14			    # Number of working imacon frames (ie # working CCDs * 2)
    print(type(FRAMES//2))
    #---------------------------------------------------------------------------------------

    # Create appropriate arrays.
    byte_start_time_1_array = BYTE_CCD1_START_TIME + BYTE_FOOTER_PACKET_SIZE*lindgen(FRAMES//2)
    byte_exposure_1_array   = BYTE_CCD1_EXPOSURE_1 + BYTE_FOOTER_PACKET_SIZE*lindgen(FRAMES//2)
    byte_exposure_2_array   = BYTE_CCD1_EXPOSURE_2 + BYTE_FOOTER_PACKET_SIZE*lindgen(FRAMES//2)
    byte_delay_array        = BYTE_CCD1_DELAY      + BYTE_FOOTER_PACKET_SIZE*lindgen(FRAMES//2)


    # Read in the data associated with the tiff.
    fin=open(path,"rb")
    tiffdata= fin.read()

    # Note that imacon tiffs are set in a little-endian style.  Read up about that to understand the following commands.
    # ** I am accessing the data array at the index location denoted by the byte_start_time_array.
    # ** Know that the format is a long thus will need to read in 4 bytes to get the true value.

    start = np.zeros(FRAMES)
    exp1   = np.zeros(FRAMES//2)
    exp2   = np.zeros(FRAMES//2)
    delay  = np.zeros(FRAMES//2)

    for pos in range(0,FRAMES//2):
        i1 = byte_start_time_1_array[pos]
        i2 = byte_exposure_1_array[pos]
        i3 = byte_exposure_2_array[pos]
        i4 = byte_delay_array[pos]

        val = struct.unpack("<L", tiffdata[i1:i1 +4])[0]/TIME_CONVERSION
        start[pos] = val

        val2 = struct.unpack("<L", tiffdata[i2:i2 +4])[0]/TIME_CONVERSION
        exp1[pos] = val2

        val3 = struct.unpack("<L", tiffdata[i3:i3 +4])[0]/TIME_CONVERSION
        exp2[pos] = val3

        val4 = struct.unpack("<L", tiffdata[i4:i4 +4])[0]/TIME_CONVERSION
        delay[pos] = val4

    start[FRAMES//2:] = start[0:FRAMES//2] + exp1 + delay

    return start,np.append(exp1,exp2)   

def get_igun(fname,header_name):
    '''
    Reads in data from igun files, spheromak experiment.
    Applies calibration data from header.
    Returns current in Amps 
    '''
    ###Header data
    print("Reading header data")
    head = np.loadtxt(header_name,delimiter=',')
    dt_sec, ind0, N, mean_bias, atten_dB, calib_factor = head[0],head[1],int(head[2]),head[3],head[4],head[5]

    ### Diagnostic data
    print("Reading data")
    data = readVME(fname,cols=N,rows=1)
    v = data[0,:] - mean_bias
    I = v*calib_factor* 10.**(atten_dB/20.) #Amps

    return I

def get_vgun(fname,header_name):
    '''
    Reads in data from vgun files, spheromak experiment.
    Applies calibration data from header.
    Returns voltage
    '''
    ###Header data
    print("Reading header data")
    head = np.loadtxt(header_name,delimiter=',')
    dt_sec, ind0, N, mean_bias, atten_dB, calib_factor = head[0],head[1],int(head[2]),head[3],head[4],head[5]

    ### Diagnostic data
    print("Reading data")
    data = readVME(fname,cols=N,rows=1)
    v = data[0,:] - mean_bias
    V = v*calib_factor* 10.**(atten_dB/20.) #Amps

    return V

def get_opt_trig(fname):
    '''
    Reads in optical trigger data, finds peak, returns peak time
    '''
    #fname ~ dbase+"spheromak_optical_trigger_%i.dat"%shot_num
    ret = readVME(fname,rows=1)
    v_trigger = ret[0][0:2500]
    dv = np.diff(v_trigger)
    dt = .01 #us
    t0 = (np.argmax(dv) +.5)*dt

    return t0

def get_croft_diag(name,shot_num,dbase="G:\\data\\croft\\shots\\",optical_trig=True):
    dbase += "%i/"%shot_num
    header_name = dbase+name+"_header_%i.meta"%shot_num
    fname = dbase+name+'_%i.dat'%(shot_num)

    ###Header data
    print("Reading header data")
    head = np.loadtxt(header_name,delimiter=',')
    dt_sec, ind0, rows, N, mean_bias, atten_dB, calib_factor = head[0],head[1],int(head[2]),int(head[3]),head[4],head[5],head[6]

    ### Diagnostic data
    print("Reading data")
    data = readVME(fname,cols=N,rows=rows)
    t = data[0,:]
    v = data[1,:]

    ###Optical trigger data
    if optical_trig:
        ret = readVME(dbase+"croft_optical_trigger_%i.dat"%shot_num,rows=1)
        v_trigger = ret[0]
        dt = .01 #us
        ind = np.where(v_trigger > .75)[0][0]
        t0 = (ind +.5)*dt
    else:
        t0 = 18.

    return t-t0, v
