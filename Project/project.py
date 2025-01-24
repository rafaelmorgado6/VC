import cv2 as cv
import numpy as np
import cv2.aruco as aruco
import string
import json 
import sys

# Verificar se o nome da imagem foi fornecido como argumento
if len(sys.argv) != 2:
    print("Uso: python3 joao_andre.py <nome_da_imagem>")
    sys.exit(1)

# Obter o nome da imagem a partir dos argumentos
image_file = sys.argv[1]

# Carregar a imagem
img = cv.imread(image_file)

# Verificar se a imagem foi carregada corretamente
if img is None:
    print(f"Erro: Não foi possível carregar a imagem '{image_file}'. Verifique o caminho.")
    sys.exit(1)


def recorte(imagem):
    height, width = imagem.shape[:2]
    x_inicial, y_inicial = 50, 50  # Coordenadas do canto superior esquerdo
    x_final, y_final = (width-50), (height-50)     # Coordenadas do canto inferior direito
    recorte = imagem[y_inicial:y_final, x_inicial:x_final]

    return recorte


def enhance_image(image):

    # Amplify the image by resizing it (scaling up by 2x)
    height, width = image.shape[:2]
    enlarged = cv.resize(image, (width * 2, height * 2), interpolation=cv.INTER_CUBIC)

    # Convert to grayscale
    gray = cv.cvtColor(enlarged, cv.COLOR_BGR2GRAY)

    # Apply adaptive histogram equalization to improve contrast
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Strengthen edges using dilation
    kernel = np.ones((3, 3), np.uint8)
    enhanced = cv.dilate(enhanced, kernel, iterations=1)

    return enhanced


def thresh_binary(imagem):

    _, imagem_thresh = cv.threshold(imagem, 130, 255, cv.THRESH_BINARY)

    # Dimensões do elemento estruturante
    tamanho = 20

    # Criar um kernel vazio (zeros)
    kernel = np.zeros((tamanho, tamanho), dtype=np.uint8)

    # Preencher o "X" no kernel
    for i in range(min(tamanho, tamanho)):
        kernel[i, i] = 1  #i Diagonal principal
        kernel[i, tamanho - i - 1] = 1  # Diagonal secundária

    inverted_image = cv.bitwise_not(imagem_thresh)

    # Aplicar a operação de opening
    eroded_image = cv.erode(inverted_image, kernel)

    # Dimensões do elemento estruturante
    tamanho = 60

    # Criar um kernel vazio (zeros)
    kernel = np.zeros((tamanho, tamanho), dtype=np.uint8)

    # Preencher o "X" no kernel
    for i in range(min(tamanho, tamanho)):
        kernel[i, i] = 1  # Diagonal principal
        kernel[i, tamanho - i - 1] = 1  # Diagonal secundária

    opened_image = cv.dilate(eroded_image, kernel)

    return opened_image


def detect_crosses(imagem):

    # Encontrar contornos
    contornos, _ = cv.findContours(imagem, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Obter dimensões da imagem
    altura, largura = imagem.shape
    centro_imagem = (largura // 2, altura // 2)

    # Inicializar variáveis
    melhor_contorno = None
    melhor_distancia_centro = float('inf')

    # Processar cada contorno
    for contorno in contornos:
        # Calcular a área do contorno
        area = cv.contourArea(contorno)
        if area < 100 or area > 5000:  # Ignorar áreas muito pequenas
            continue

        # Aproximar o contorno para simplificar a forma
        perimetro = cv.arcLength(contorno, True)
        approx = cv.approxPolyDP(contorno, 0.02 * perimetro, True)

        # Verificar se o número de vértices é suficiente para um "X"
        if len(approx) < 8:  # Um "X" normalmente terá mais vértices
            continue

        # Calcular o centro do contorno
        M = cv.moments(contorno)
        if M['m00'] == 0:  # Evitar divisão por zero
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Calcular a distância do centro do contorno ao centro da imagem
        # Calcular a distância do centro do contorno ao centro da imagem
        distancia_centro = np.sqrt((cx - centro_imagem[0])**2 + (cy - centro_imagem[1])**2)


        # Verificar simetria aproximada (usando bounding box)
        x, y, w, h = cv.boundingRect(contorno)
        aspect_ratio = w / h
        if aspect_ratio < 0.8 or aspect_ratio > 1.2:  # Um "X" deve ser aproximadamente quadrado
            continue

        # Verificar cruzamento diagonal (analisar ângulos entre segmentos)
        linhas = []
        for i in range(len(approx)):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % len(approx)][0]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            angulo = np.degrees(np.arctan2(dy, dx))
            linhas.append(angulo)

        # Verificar ângulos diagonais típicos de um "X" (45° ou -135°)
        diagonais = [45, -135, 135, -45]
        match_diagonais = sum(1 for angulo in linhas if any(abs(angulo - diag) < 15 for diag in diagonais))
        if match_diagonais < 2:  # Um "X" precisa de pelo menos 2 diagonais
            continue

        # Escolher o contorno mais próximo do centro
        if distancia_centro < melhor_distancia_centro:
            melhor_distancia_centro = distancia_centro
            melhor_contorno = contorno

    # Criar uma cópia da imagem original para desenhar os contornos
    imagem_resultado = cv.cvtColor(imagem, cv.COLOR_GRAY2BGR)

    # Se a cruz 'X' for detectada, desenhar o contorno
    if melhor_contorno is not None:
        cv.drawContours(imagem_resultado, [melhor_contorno], -1, (0, 255, 0), 2)
        cruz_detectada = True
    else:
        cruz_detectada = False

    return imagem_resultado, cruz_detectada


#########################################
perguntas = 42
e_multipla = 4
colunas = 3
na_perguntas = 4
horizontal = 15     # Quantos quadrados tem na horizontal
vertical = 15       # Quantos quadrados tem na vertical
#########################################
total_perguntas = perguntas - na_perguntas


###############################################
### parametros de camara e deteção de arucos###
###############################################

# Load your camera parameters (intrinsics and distortion)
with np.load('camera_capture.npz') as data:
    intrinsics = data['intrinsics']
    distortion = data['distortion']


# Set up AruCo marker detection
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)  # Choose the correct dictionary
parameters = aruco.DetectorParameters()  # Correct way to create DetectorParameters

# Marker side length in meters (you need to specify this value)
markerLength = 0.05  # Example: 5 cm = 0.05 meters

# Define the object points for the corners of the marker in 3D space
objp = np.array([
    [-markerLength / 2, markerLength / 2, 0],
    [markerLength / 2, markerLength / 2, 0],
    [markerLength / 2, -markerLength / 2, 0],
    [-markerLength / 2, -markerLength / 2, 0]
], dtype=np.float32)

# Verificar se o nome da imagem foi fornecido como argumento
if len(sys.argv) != 2:
    print("Try: python3 joao_andre.py <nome_da_imagem>")
    sys.exit(1)

# Obter o nome da imagem a partir dos argumentos
image_file = sys.argv[1]

# Carregar a imagem
img = cv.imread(image_file)

# Verificar se a imagem foi carregada corretamente
if img is None:
    print(f"Erro: Não foi possível carregar a imagem '{image_file}'. Verifique o caminho.")
    sys.exit(1)


#img = cv.imread('Image_2.jpg')

# # Inicializar a captura de vídeo
# cap = cv.VideoCapture(0)  # Use '0' para webcam, ou insira o caminho de um arquivo de vídeo

# captured_image = None

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Não foi possível capturar o vídeo.")
#         break

#     # Exibir o vídeo em tempo real
#     cv.imshow("Video Feed", frame)

#     # Verificar se a tecla 'c' foi pressionada para capturar a imagem
#     key = cv.waitKey(1) & 0xFF
#     if key == ord('c'):
#         captured_image = frame.copy()  # Salva o frame atual
#         print("Imagem capturada.")
#         break  # Sai do loop ao capturar

#     # Pressione 'q' para sair do loop sem capturar
#     if key == ord('q'):
#         print("Saindo sem capturar.")
#         break

# # Liberar o recurso de vídeo
# cap.release()
# cv.destroyAllWindows()

# # Caso uma imagem tenha sido capturada, use-a no restante do código
# if captured_image is not None:
#     img = captured_image
# else:
#     print("Nenhuma imagem foi capturada.")
#     exit()  # Encerra o programa se nenhuma imagem foi capturada


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect AruCo markers
corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

if ids is not None:
    for i in range(len(ids)):
        retval, rvec, tvec = cv.solvePnP(objp, corners[i], intrinsics, distortion)
    if len(ids) == 4:
        cv.imshow('QR Code Detection', img)

# Exibir a imagem processada
cv.imshow('QR Code Detection', img)


######################################################################
### Calculo do centro e vertices dos arucos para recorte de imagem ###
######################################################################
primeiro_array = corners[0]
segundo_array = corners[1]
terceiro_array = corners[2]
quarto_array = corners[3]

x_y_array = []
x_y_array.append(primeiro_array[0][0])
x_y_array.append(segundo_array[0][0])
x_y_array.append(terceiro_array[0][0])
x_y_array.append(quarto_array[0][0])

for i in range(len(ids)):

    # Calcular o centro (média dos pontos)
    centro = np.mean(corners[i], axis=0)

    if (x_y_array[i][0] < 320 and x_y_array[i][1] < 240):
        primeiro_array = corners[i]
    elif (x_y_array[i][0] > 320 and x_y_array[i][1] < 240):
        segundo_array = corners[i]
    elif (x_y_array[i][0] > 320 and x_y_array[i][1] > 240):
        terceiro_array = corners[i]
    else:
        quarto_array = corners[i]

x_y_points = []
x_y_points.append(primeiro_array[0][3])
x_y_points.append(segundo_array[0][2])
x_y_points.append(terceiro_array[0][1])
x_y_points.append(quarto_array[0][0])


pts_src = np.float32(x_y_points)
# print(pts_src)
# print(pts_src[1][0])
# print(pts_src[0][0])
# print(pts_src[2][1])
# print(pts_src[0][1])


width = pts_src[1][0] - pts_src[0][0]  # Largura do recorte
height = pts_src[2][1] - pts_src[0][1]  # Altura do recorte
width = np.int16(width)
height = np.int16(height)


#################################################################
### Recorte da imagem para ficar só a tabela com as respostas ###
#################################################################
# Pontos correspondentes na nova imagem retangular
pts_dst = np.array([
    [0, 0],
    [width, 0],
    [width, height],
    [0, height]
], dtype=np.float32)

M, mask = cv.findHomography(pts_src, pts_dst)

# Aplicar a transformação de perspectiva com o tamanho da imagem original
warped_img = cv.warpPerspective(img, M, (width, height))
warped_img = cv.resize(warped_img, (width*2, height*2), interpolation=cv.INTER_AREA)


# Exibir a imagem corrigida
#cv.imshow('QR Code Detection com centros', img)
cv.imshow("Corrected Perspective", warped_img)

# Obter as dimensões da imagem
altura, largura, _ = warped_img.shape


# Dimensões de cada quadrado
altura_quadrado = altura / vertical      
largura_quadrado = largura / horizontal    


##############################################################################################
### Recorte da imagem quadrado a quadrado e colocados num array de imagens + seus vértices ###
##############################################################################################
# Lista para armazenar os quadrados
quadrados = []
each_center = []
# Iterar sobre os eixos y e x para recortar os quadrados
for y in range(horizontal):                 
    for x in range(vertical):
        # Coordenadas do quadrado atual com ajuste para recortar mais abaixo
        y_inicio = y * altura_quadrado # Ajuste: recortar 10 pixels abaixo
        y_fim = min(altura, (y + 1) * altura_quadrado)  # Ajustar para evitar ultrapassar os limites
        x_inicio = x * largura_quadrado
        x_fim = min(largura, (x + 1) * largura_quadrado)  # Ajustar para evitar perda de pixels

        x_center = x_inicio + (x_fim - x_inicio)/2
        y_center = y_inicio + (y_fim - y_inicio)/2

        each_center.append([x_center, y_center])


        x_y_points = ((x_inicio, y_inicio), (x_fim, y_inicio), (x_fim, y_fim), (x_inicio, y_fim))
        pts_src = np.float32(x_y_points)

        width = x_fim - x_inicio  # Largura do recorte
        height = y_fim - y_inicio  # Altura do recorte
        width = np.int16(width)
        height = np.int16(height)

        # Pontos correspondentes na nova imagem retangular
        pts_dst = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)

        M, mask = cv.findHomography(pts_src, pts_dst)

        # Aplicar a transformação de perspectiva com o tamanho da imagem original
        quadrado = cv.warpPerspective(warped_img, M, (width, height))
        quadrado = cv.resize(quadrado, (200, 200), interpolation=cv.INTER_AREA)

        quadrados.append(quadrado)  # Adicionar à lista


######################################################
### Organizar quadrados e seus vértices respetivos ###
######################################################
pb_cross = []
center_cross = []
sq_inicial = horizontal + 1
for i in range(sq_inicial, len(quadrados), e_multipla+1):
    for j in range(i, i + e_multipla):
        pb_cross.append(quadrados[j])
        center_cross.append(each_center[j])

if len(pb_cross) % 3 != 0:
    raise ValueError(f"O tamanho do array deve ser divisível por {colunas}.")

array1 = []
for j in range(0, len(pb_cross), 12):
    for k in range(4):
        array1.append(pb_cross[j+k])

array2 = []
for j in range(4, len(pb_cross), 12):
    for k in range(4):
        array2.append(pb_cross[j+k])

array3 = []
for j in range(8, len(pb_cross), 12):
    for k in range(4):
        array3.append(pb_cross[j+k])

pb_cross = array1 + array2 + array3
pb_cross = pb_cross[:-(4*na_perguntas)]

if len(center_cross) % 3 != 0:
    raise ValueError(f"O tamanho do array deve ser divisível por {colunas}.")

array1.clear()
for j in range(0, len(center_cross), 12):
    for k in range(4):
        array1.append(center_cross[j+k])

array2.clear()
for j in range(4, len(center_cross), 12):
    for k in range(4):
        array2.append(center_cross[j+k])

array3.clear()
for j in range(8, len(center_cross), 12):
    for k in range(4):
        array3.append(center_cross[j+k])

center_cross = array1 + array2 + array3
center_cross = center_cross[:-(4*na_perguntas)]


##########################################
### Leitura das cruzes + resposta dada ###
##########################################
matriz = []
for i in range(total_perguntas):
    linha = []
    double = 0
    for j in range(e_multipla):
            
        index = i*e_multipla + j
        enhanced_image = enhance_image(pb_cross[index])
        th_binary = thresh_binary(enhanced_image)
        binary = recorte(th_binary)
        processed_image, value = detect_crosses(binary)
        if i == 34 and j == 0:
            cv.imshow(f"Quadrado Processed {i}{j}", processed_image)
            #cv.imshow(f"Quadrado inicial", pb_cross[index])
            #cv.imshow(f"Quadrado p_b", th_binary)
            cv.imshow(f"Quadrado recortada", binary)

        linha.append(value)
        if value == True:
            double = double + 1
            if double > 1:
                linha.clear()
                for k in range(j+1):
                    linha.append(False)

    
    matriz.append(linha)

#print(matriz[0])


#################################################
### Resposta dada pelo aluno + resposta certa ###
#################################################
answer = []
not_answer = []
true_false_a = []
mapping = list(string.ascii_lowercase)
for i in range(len(matriz)):
    if i > 0:
        mult_choice = 0
        not_answer.clear()
    for j in range(e_multipla):
        not_answer.append(matriz[i][j])
        true_false_a.append(matriz[i][j])
        if matriz[i][j] == True:
            if 0 <= j <= len(mapping):
                answer.append(mapping[j])
            
        if j == (e_multipla-1) and True not in not_answer:
            answer.append('NA')

print("\nRespostas detetadas -> " + str(answer))

import json  # Importar biblioteca JSON

# Ler as resoluções do ficheiro JSON
with open('resolutions.json', 'r') as file:
    data = json.load(file)
    resolutions = data["resolutions"]

#resolutions = ['b', 'c', 'a', 'a', 'a', 'a', 'c', 'c', 'd', 'b', 'c', 'a', 'a', 'a', 'a', 'c', 'c', 'd', 'b', 'c', 'a', 'a', 'a', 'a', 'c', 'c', 'd', 'b', 'c', 'a', 'a', 'a', 'a', 'c', 'c', 'd', 'a', 'a']


###########################################################
### Comparação entre a correção e as respostas do aluno ###
###########################################################
#### Corrigir! ####
answer_w = 0
answer_r = 0
for i in range(len(answer)):
    if resolutions[i] == answer[i]:
            answer_r = answer_r + 1
    else:
            answer_w = answer_w + 1

    
print (f"Respostas Certas: {answer_r}\nRespostas Erradas {answer_w}")

# Calcular a porcentagem de acertos
total_questions = len(resolutions)
percentage_score = (answer_r / total_questions) * 100

print(f"Sua nota foi: {percentage_score:.2f}%")


##############################################################
### Meter um ponto verde nas certas e vermelho nas erradas ###
##############################################################
for i in range(total_perguntas):
    for k in range(e_multipla):
        j = i*4+k
        if matriz[i][k] == True and resolutions[i] == answer[i]:
            cv.circle(warped_img, (int(center_cross[j][0]), int(center_cross[j][1])), 5, (0, 255, 0), -1)
        elif matriz[i][k] == True and resolutions[i] != answer[i]:
            cv.circle(warped_img, (int(center_cross[j][0]), int(center_cross[j][1])), 5, (0, 0, 255), -1)

cv.imshow("Corrected Perspective 1", warped_img)

cv.waitKey(0)
cv.destroyAllWindows()