from math import floor
import numpy as np
import cv2
import imutils
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from tkinter import *
from tkinter import colorchooser
import colorsys

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

# Exemplos de imagens de parede
# INPUT_IMAGE =  './quadros.jpg'
INPUT_IMAGE =  './parede_rosa.jpg'
# INPUT_IMAGE =  './parede_azul.jpg'
# INPUT_IMAGE =  './parede_real.jpg'


# Função usada para selecionar a parede que será pintada
def selectBackground(event,x,y,flags, param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(sure_bg,(x,y),15,(0,0,0),-1)
            cv2.circle(sure_bg_mask,(x,y),15,(0,0,0),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(sure_bg,(x,y),15,(0,0,0),-1)
        cv2.circle(sure_bg_mask,(x,y),15,(0,0,0),-1)

# Função usada para selecionar os objetos de frente da imagem
def selectForeground(event,x,y,flags, param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(sure_fg,(x,y),15,(255,255,255),-1)
            cv2.circle(sure_fg_mask,(x,y),15,(255,255,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(sure_fg,(x,y),15,(255,255,255),-1)
        cv2.circle(sure_fg_mask,(x,y),15,(255,255,255),-1)

# Leitura da imagem
img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)

# Caso necessário, usado para redimencionar o tamanho da imagem
if INPUT_IMAGE == './parede_rosa.jpg' or INPUT_IMAGE == './parede_real.jpg':
    img = cv2.resize(img, (floor(img.shape[0]/6), floor(img.shape[1]/6)))
elif INPUT_IMAGE == './parede_azul.jpg':
    img = cv2.resize(img, (floor(img.shape[0]/2), floor(img.shape[1]/2)))


# Reliza uma copia da imagem original para a definição do fundo
sure_bg = img.copy()

# Máscara toda branca, onde as áreas pretas são o fundo
sure_bg_mask = np.ones((img.shape[0], img.shape[1]), np.uint8)

# Abre uma janela que terá o reconhecimento do click do mouse
cv2.namedWindow('Pinte a parede que deseja pintar')
cv2.setMouseCallback('Pinte a parede que deseja pintar',selectBackground)

# Entra em um loop até que seja precionado enter ou esc
while(1):
    cv2.imshow('Pinte a parede que deseja pintar',sure_bg)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 13 or k == 27:
        break

cv2.destroyAllWindows ()

# Pinta de branco tudo o que não foi selecionado na máscara
sure_bg_mask = np.where(sure_bg_mask != 0, 255, 0)
sure_bg_mask = sure_bg_mask.astype(np.uint8)
# cv2.imshow('Sure BG', sure_bg_mask)

# Reliza uma copia da imagem original para a definição dos objetos de frente
sure_fg = img.copy()

# Máscara toda branca, onde as áreas brancas não são o fundo
sure_fg_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)

# Abre uma janela que terá o reconhecimento do click do mouse
cv2.namedWindow('Pinte os objetos que não são a parede')
cv2.setMouseCallback('Pinte os objetos que não são a parede',selectForeground)


# Entra em um loop até que seja precionado enter ou esc
while(1):
    cv2.imshow('Pinte os objetos que não são a parede',sure_fg)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 13 or k == 27:
        break

# Pinta de preto tudo o que não foi selecionado na máscara
sure_fg_mask = np.where(sure_fg_mask != 255, 0, 255)
sure_fg_mask = sure_fg_mask.astype(np.uint8)
# cv2.imshow('Sure FG', sure_fg_mask)

cv2.destroyAllWindows ()

# Faz a diferença entre o que se tem certeza que é fundo e
# o que se tem certeza que é frente
unknown = cv2.subtract(sure_bg_mask, sure_fg_mask)

# cv2.imshow('Unknown', unknown.astype(np.uint8))

# Acha os componentes conexos da imagem, colocando labels em cada um
_, markers = cv2.connectedComponents(sure_fg_mask)
markers = markers + 1
markers[unknown == 255] = 0

# Aplica o algoritmo de segmentação watershed
markers = cv2.watershed(img, markers)

# Abre o color picker
cor = colorchooser.askcolor(title = "Tkinter Color Chooser")

# Faz o tratamento da cor selecionada e a conversão para HLS
cor = cor[0]
cor = np.array([[[cor[0],cor[1],cor[2]]]], np.uint8)
cor = cv2.cvtColor(cor, cv2.COLOR_RGB2HLS)

# Converte a imagem para HLS
img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

#
luminancia = img_hls[:,:,1]

luminancia_min = luminancia.min()
luminancia_max = luminancia.max()


for row in range(img.shape[0]):
    for col in range(img.shape[1]):

        # Seleciona apenas o componente conexo com label 1, que se refere à parede
        if markers[row][col]==1 :

            # Tratamento para cores claras
            if cor[0][0][1] > 125 and luminancia[row][col] <160 :
                fator =0.5+1.5*((luminancia[row][col] - luminancia_min)/(luminancia_max - luminancia_min))
            
            # Tratamento para cores claras
            else:    
                fator =1.25*((luminancia[row][col] - luminancia_min)/(luminancia_max - luminancia_min))
           
            # Truncagem para o branco, caso  o valor ultrapasse 255   
            if fator*cor[0][0][1] > 255:
                img_hls[row][col] = [cor[0][0][0],255,cor[0][0][2]]
            else:
                img_hls[row][col] = [cor[0][0][0],(fator*cor[0][0][1]),cor[0][0][2]]

# Reconversão da imagem para BGR
img_hls = cv2.cvtColor(img_hls, cv2.COLOR_HLS2BGR)
cv2.imshow('Original', img)
cv2.imshow('changeColor', img_hls)

# Desenho dos contornos feitos pelo watershed
img[markers == -1] = [255, 0, 0]
cv2.imshow('Contornos', img)


cv2.waitKey ()
cv2.destroyAllWindows ()

