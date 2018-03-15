from __future__ import division
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import scipy.stats
import math


def optimalDFTImg(img):
    """Zero-padding sobre img para alcanzar un tamaño óptimo para FFT"""
    h=cv.getOptimalDFTSize(img.shape[0])
    w=cv.getOptimalDFTSize(img.shape[1])
    return cv.copyMakeBorder(img,0,h-img.shape[0],0,w-img.shape[1],cv.BORDER_CONSTANT)

def spectrum(img):
    """Calcula y muestra el módulo logartímico de la DFT de img."""
    #img=optimalDFTImg(img)
    
    imgf=cv.dft(np.float32(img),flags=cv.DFT_COMPLEX_OUTPUT) 
    modulo = np.log(cv.magnitude(imgf[:,:,0],imgf[:,:,1])+1)
    modulo = np.fft.fftshift(modulo) 
    modulo=cv.normalize(modulo, modulo, 0, 1, cv.NORM_MINMAX)
    

    plt.figure()
    plt.imshow(modulo,cmap='gray')
    plt.show()

def rotate(img,angle):
    """Rotación de la imagen sobre el centro"""
    r=cv.getRotationMatrix2D((img.shape[0]/2,img.shape[1]/2),angle,1.0)
    return cv.warpAffine(img,r,img.shape)

def filterImg(img,filtro_magnitud):
    """Filtro para imágenes de un canal"""
    
    
    #como la fase del filtro es 0 la conversión de polar a cartesiano es directa (magnitud->x, fase->y)
    filtro=np.array([filtro_magnitud,np.zeros(filtro_magnitud.shape)]).swapaxes(0,2).swapaxes(0,1)
    imgf=cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
   
    imgf=cv.mulSpectrums(imgf, np.float32(filtro), cv.DFT_ROWS)
    
    return cv.idft(imgf, flags=cv.DFT_REAL_OUTPUT | cv.DFT_SCALE)



def dist(a,b):
    """distancia Euclidea"""
    return np.linalg.norm(np.array(a)-np.array(b))

def filterGaussian(rows,cols,corte):
    """Filtro de magnitud gausiano"""

    magnitud = np.zeros((rows, cols))

    corte *= rows
    for k in range(rows):
        for l in range(cols):
            magnitud[k,l]=np.exp(-dist([k+.5,l+.5],[rows/2,cols/2])/2/corte/corte)
            
    return np.fft.fftshift(magnitud)
    
        
def filterIdeal(rows, cols, corte):
    """filtro de magnitud ideal"""
    magnitud = np.zeros((rows, cols))
    if cols%2==1 and rows%2==1: #impar, el centro cae en un píxel
        magnitud = cv.circle(magnitud, (cols//2, rows//2), int(rows*corte), 1, -1)
    else:
        limit = (corte*rows)**2
        for k in range(rows):
            for l in range(cols):
                d2 = dist([k+.5,l+.5],[rows//2,cols//2])
                if d2 <= limit:
                    magnitud[k,l] = 1
    return np.fft.ifftshift(magnitud)
	
def filterButterworth(rows, cols, corte, order):
    """filtro de magnitud Butterworth"""
    #corte = w en imagen de lado 1
    #1 \over 1 + {D \over w}^{2n}
    magnitud = np.zeros((rows, cols))
    corte *= rows;
    for k in range(rows):
        for l in range(cols):
            d2 = dist([k+.5,l+.5],[rows/2,cols/2])
            magnitud[k,l] = 1.0/(1 + (d2/corte/corte)**order)

    return np.fft.fftshift(magnitud)

    
def motionBlur(size, a,  b):
    """Filtro de movimiento en direcciones a y b"""
    transformation =np.zeros(size)
    rows = size[0]
    cols = size[1]

    #fase exp(j\pi (ua + vb))
    #magnitud \frac{ \sin(\pi(ua+vb)) }{ \pi (ua+vb) }
    for k in range(rows):
        for l in range(cols):
            u = (l-cols/2)/cols
            v = (k-rows/2)/rows

            pi_v = math.pi*(u*a+v*b);
            if pi_v:
                mag = np.sin(pi_v)/pi_v
            else:
                mag=1 #lim{x->0} sin(x)/x
            
            transformation[k,l] = mag*np.exp(complex(0,1)*pi_v);
			

    return np.fft.fftshift(transformation)
    
