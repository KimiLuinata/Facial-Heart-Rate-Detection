import numpy as np
import time
from scipy import signal

frame_in = np.zeros((10, 10, 3), np.uint8)
frame_ROI = np.zeros((10, 10, 3), np.uint8)
frame_out = np.zeros((10, 10, 3), np.uint8)
buffer_size = 100
times = [] 
data_buffer = []
fps = 0
fft = []
freqs = []
t0 = time.time()
bpm = 0
bpms = []
peaks = []

def resetRun():
    global frame_in, frame_ROI, frame_out, buffer_size, times, data_buffer 
    global fps, fft, freqs, t0, bpm, bpms, peaks
    frame_in = np.zeros((10, 10, 3), np.uint8)
    frame_ROI = np.zeros((10, 10, 3), np.uint8)
    frame_out = np.zeros((10, 10, 3), np.uint8)
    buffer_size = 100
    times = [] 
    data_buffer = []
    fps = 0
    fft = []
    freqs = []
    t0 = time.time()
    bpm = 0
    bpms = []
    peaks = []

def startRun(frame):
    test = 0
    # print(frame)
    run(frame)

    # self.frame = self.process.frame_out #get the frame to show in GUI
    # self.f_fr = self.process.frame_ROI #get the face to show in GUI
    
    # print(bpm)
    if bpms.__len__() >50:
        if(max(bpms-np.mean(bpms))<5): #show HR if it is stable -the change is not over 5 bpm- for 3s
            # print("Heart rate: " + str(float("{:.2f}".format(np.mean(bpms)))) + " bpm")
            return float("{:.2f}".format(np.mean(bpms)))

    return 0
        
def run(frame):
        global frame_in, frame_ROI, frame_out, buffer_size, times, data_buffer 
        global fps, fft, freqs, t0, bpm, bpms, peaks

        L = len(data_buffer)
        
        
        g = extractColor(frame)
        
        # remove sudden change, if the avg value change is over 10, use the mean of the data_buffer
        # if (abs(g - np.mean(data_buffer)) > 10) and (L > 99): 
        #     g = data_buffer[-1]
        
        times.append(time.time() - t0)
        data_buffer.append(g)

        
        #only process in a fixed-size buffer
        if L > buffer_size:
            data_buffer = data_buffer[-buffer_size:]
            times = times[-buffer_size:]
            bpms = bpms[-buffer_size//2:]
            L = buffer_size
    
        
        processed = np.array(data_buffer)
        # processed = processed[:, 0]

        
        # start calculating after the first 10 frames
        if L == buffer_size:
            
            #calculate HR using a true fps of processor of the computer, not the fps the camera provide
            fps = float(L) / (times[-1] - times[0])
            even_times = np.linspace(times[0], times[-1], L)
            
            processed = signal.detrend(processed)#detrend the signal to avoid interference of light change

            interpolated = np.interp(even_times, times, processed) #interpolation by 1
            interpolated = np.hamming(L) * interpolated#make the signal become more periodic (advoid spectral leakage)
            norm = interpolated/np.linalg.norm(interpolated)
            raw = np.fft.rfft(norm*30)#do real fft with the normalization multiplied by 10
            
            freqs = float(fps) / L * np.arange(L / 2 + 1)
            freqs = 60. * freqs
            
            
            fft = np.abs(raw)**2#get amplitude spectrum
        
            idx = np.where((freqs > 50) & (freqs < 180))#the range of frequency that HR is supposed to be within 
            pruned = fft[idx]
            pfreq = freqs[idx]
            
            freqs = pfreq 
            fft = pruned
            
            idx2 = np.argmax(pruned)#max in the range can be HR
            
            bpm = freqs[idx2]
            bpms.append(bpm)
            
            
            processed = butter_bandpass_filter(processed,0.8,3,fps,order = 3)
            #ifft = np.fft.irfft(raw)       

def extractColor(frame):
        
        r = np.mean(frame[:,:,0])
        g = np.mean(frame[:,:,1])

        b = np.mean(frame[:,:,2])
        # return r, g, b
        return g          

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq 
        # print(str(low) + " " + str(high))
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y    

    