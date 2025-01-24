import cv2
from pyzbar.pyzbar import decode
import numpy as np

# Caminho da imagem
image_path = '/home/rafa/Desktop/VC/Project/tabela.png'

# Carregar a imagem
image = cv2.imread(image_path)

# Função para detectar e decodificar QR codes
def detect_qr_code(image):
    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detectar e decodificar QR codes
    decoded_objects = decode(gray)
    
    # Lista para armazenar os dados dos QR codes
    qr_data = []
    
    # Desenhar os contornos dos QR codes e salvar os dados
    for obj in decoded_objects:
        points = obj.polygon
        if len(points) > 4:  # Para QR codes com formas irregulares
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            points = hull
        points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [points], True, (0, 255, 0), 3)  # Desenhar contorno em verde
        qr_data.append(obj.data.decode('utf-8'))  # Adicionar dados decodificados à lista
    
    return qr_data, image

# Detectar QR codes na imagem
qr_data, processed_image = detect_qr_code(image)

# Exibir a imagem processada com os QR codes detectados
cv2.imshow("QR Code Detection", processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Imprimir os dados dos QR codes
print("Dados dos QR Codes detectados:")
for data in qr_data:
    print(data)
