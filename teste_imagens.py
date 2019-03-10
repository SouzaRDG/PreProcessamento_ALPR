

###### TENHAM CERTEZA QUE ESTÃO USANDO PYTHON 3 OU MAIOR



### -------- BIBLIOTECAS --------

import cv2
from skimage import measure
from skimage.transform import resize
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import PlacasTeste
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib


"""

AQUI VAMOS POR OS METODOS QUE SÃO MAIORES QUE UMA LINHA!
ASSIM FICA MAIS LIMPO E FACIL DE IDENTIFICAR O QUE ESTÁ SENDO VIZUALIZADO!!!

"""


def altoCont (imagem):

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imagem, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imagem, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imagem, imgTopHat)
    retorno = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return retorno





image = cv2.imread('cam1.png')
# image = cv2.imread('br2.jpg')
imgGray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
imgAltoCon = altoCont(imgGray)
n,imgThresh = cv2.threshold(imgAltoCon, 27 ,255,cv2.THRESH_BINARY_INV,cv2.THRESH_OTSU)
n,imgThresh2 = cv2.threshold(imgThresh,150,255,cv2.THRESH_BINARY, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, ( 2 , 2 ))
dilated = cv2.dilate(imgThresh2,kernel, iterations = 1)
contours, harch = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)


listaPlacas = list()

for  contour in contours:
    [x,y,w,h] = cv2.boundingRect(contour)
    # if h>250 and w>250:
    #      continue
    if h < 30 or w < 30:
        continue
    if w < 2 * h:
        continue
    if w > 3.2 * h:
        continue
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)

    imgPossivelPlaca = dilated[y:y+h,x:x+w]

    imgResize = cv2.resize(imgPossivelPlaca, (0, 0), fx=2, fy=2)

    novaPlaca = PlacasTeste.PlacasTeste(x, y, w, h, imgResize)

    listaPlacas.append(novaPlaca)

for placa in listaPlacas:

    cv2.imshow(placa.nome,placa.image)
    #cv2.imwrite(placa.nome + '.png', placa.image)

    fig, ax1 = plt.subplots(1)
    ax1.imshow(placa.image, cmap="gray")

    labelled_plate = measure.label(placa.image)

    characters = []
    counter = 0
    column_list = []

    for regions in measure.regionprops(labelled_plate):
        y0, x0, y1, x1 = regions.bbox
        region_height = y1 - y0
        region_width = x1 - x0

        roi = placa.image[y0:y1, x0:x1]

        rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red", linewidth=2, fill=False)
        ax1.add_patch(rect_border)

        resized_char = resize(roi, (20, 20))
        characters.append(resized_char)

        column_list.append(x0)

        plt.show()



# cv2.imshow("image", image)
# # #cv2.imshow("gray",gray)
# # #cv2.imshow("thresh", thresh)
# cv2.imshow("dilated", dilated)












cv2.waitKey()
cv2.destroyAllWindows()

