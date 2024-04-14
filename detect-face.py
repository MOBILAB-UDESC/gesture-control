import cv2  # Importa a biblioteca OpenCV para processamento de imagens
import os  # Importa a biblioteca os para manipulação de diretórios e arquivos do sistema
import numpy as np  # Importa a biblioteca NumPy para manipulação de arrays
import pyrealsense2 as rs  # Importa a biblioteca RealSense para interagir com a câmera RealSense

# Caminho Haarcascade para detecção de rostos
cascPath = 'cascade/haarcascade_frontalface_default.xml'
cascPathOlho = 'cascade/haarcascade-eye.xml'

# Classifier baseado nos Haarcascades para detecção de rostos e olhos
facePath = cv2.CascadeClassifier(cascPath)
facePathOlho = cv2.CascadeClassifier(cascPathOlho)

# Configuração do RealSense para captura de imagens
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Configura o formato e a taxa de frames da câmera

profile = pipeline.start(config)  # Inicia o pipeline do RealSense para captura de imagens

increment = 1  # Variável para contagem do número de amostras capturadas
numMostras = 100 # Número máximo de amostras a serem capturadas
id = input('Digite seu identificador: ')  # Solicita ao usuário para inserir um identificador
width, height = 220, 220  # Dimensões desejadas para a imagem capturada
print('Capturando as faces...')

# Cria o diretório para salvar as imagens capturadas

os.makedirs('fotos', exist_ok=True)

try:
    while True:
        # Captura dos quadros da câmera RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())  # Converte o quadro para um array NumPy
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converte a imagem para escala de cinza

        # Qualidade da luz sobre a imagem capturada
        print(np.average(gray))

        # Realizando detecção de rostos
        face_detect = facePath.detectMultiScale(
            gray,
            scaleFactor=1.5,
            minSize=(35, 35),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in face_detect:
            # Desenhando retângulo na face detectada
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

            # Realizando detecção de olhos
            region = image[y:y + h, x:x + w]  # Região de interesse contendo o rosto detectado
            imageOlhoGray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)  # Converte a região para escala de cinza
            face_detect_olho = facePathOlho.detectMultiScale(imageOlhoGray)

            for (ox, oy, ow, oh) in face_detect_olho:
                # Desenhando retângulo nos olhos detectados
                cv2.rectangle(region, (ox, oy), (ox+ow, oy+oh), (0, 0, 255), 2)

                # Salvando imagem com respectivo id para treinamento
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    '''
                        Caso queira colocar um delimitador para capturar apenas
                        imagens com uma boa qualidade de luz, descomente a linha abaixo
                    '''
                    # if np.average(gray) > 110:

                    face_off = cv2.resize(gray[y:y + h, x:x + w], (width, height))
                    cv2.imwrite('fotos/pessoa.' + str(id) + '.' + str(increment) + '.jpg', face_off)  # Salva a imagem
                    print('[Foto ' + str(increment) + ' capturada com sucesso] - ', np.average(gray))
                    increment += 1

        cv2.imshow('Face', image)  # Mostra a imagem com as detecções
        cv2.waitKey(1)

        if increment > numMostras:
            break

finally:
    pipeline.stop()  # Encerra o pipeline do RealSense
    print('Fotos capturadas com sucesso :)')
    cv2.destroyAllWindows()  # Fecha todas as janelas OpenCV
