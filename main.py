import base64
import json
import os
import sys
import cv2

from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict

# Classe que representa a resposta do modelo.
class Prediction(object):
    def __init__(self, bboxes: [[float]], confidences: [float], displayNames: [str], ids: [str]):
        self.bboxes = bboxes
        self.confidences = confidences
        self.displayNames = displayNames
        self.ids = ids

def predict_image_object_detection_sample(
    project: str,
    endpoint_id: str,
    filename: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):

    client_options = {"api_endpoint": api_endpoint}

    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    with open(filename, "rb") as f:
        file_content = f.read()

    # Codifica-se a imagem em base64.
    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageObjectDetectionPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]

    # Envia-se a imagem até ao endpoint e recebe-se a resposta do modelo.
    parameters = predict.params.ImageObjectDetectionPredictionParams(
        confidence_threshold=0.5, max_predictions=5,
    ).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)

    # Deserializa-se a resposta do modelo.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", str(dict(prediction)))
        deserialized_predict = Prediction(**json.loads(str(dict(prediction)).replace("'", '"')))

    # Desenham-se as bounding boxes e identificam-se as classes detetadas.
    img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR)
    color = (0, 255, 0)
    thickness = 1
    i = 0
    for bbox in deserialized_predict.bboxes:
        start_point = (int(bbox[0] * img.shape[1]), int(bbox[2] * img.shape[0]))
        end_point = (int(bbox[1] * img.shape[1]), int(bbox[3] * img.shape[0]))
        img = cv2.rectangle(img, start_point, end_point, color, thickness)
        cv2.putText(img, deserialized_predict.displayNames[i], (int(bbox[0] * img.shape[1]), int(bbox[2] * img.shape[0]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36, 36, 255), 2)
        i = i + 1

    # Apresenta-se a imagem na UI.
    while True:
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        sys.exit()
    cv2.destroyAllWindows()

# Autenticação através da google service account.
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "C:\\Users\\TECRA\\Desktop\\Uni\\3ano\\ESTAGIO\\" \
                                              "Google_Cloud\\teak-node-386212-87b132b61be4.json"

predict_image_object_detection_sample("teak-node-386212",7036852427533844480,"eee.jpg")