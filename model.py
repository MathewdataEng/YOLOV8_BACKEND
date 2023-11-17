from typing import Optional
import os
from uuid import uuid4
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path, get_single_tag_keys
from ultralytics import YOLO
import cv2
LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8080')
LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN", '4f81bd86324882866758670e58c39e514adc2963')

def get_id() -> str:
    # creates a random ID for your label everytime so no chance for errors
    label_id = str(uuid4())[:4]
    return label_id


def get_image_path(task: dict) -> str:
    raw_img_path = task["data"]["image"]

    img_path = get_image_local_path(
        raw_img_path,
        label_studio_access_token=LABEL_STUDIO_ACCESS_TOKEN,
        label_studio_host=LABEL_STUDIO_HOST,
    )

    return img_path

class NewModel(LabelStudioMLBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config,
            "RectangleLabels",
            "Image",
        )

    def predict(self, tasks: list[dict], context: Optional[dict] = None, **kwargs) -> list[dict]:       
           # Define the model
        model = YOLO("yolov8m.pt")

        # results = []
        for task in tasks:
            img_path = get_image_path(task)
            img = cv2.imread(img_path)
            # print(type(img))
            img_width, img_height,_ = img.shape
            # Get prediction from the model
            results = model.predict(img,classes=[0])
            predictions = []
            # score = 0
            for result in results:
                img_height,img_width = result.orig_shape
                for i,box in enumerate(result.boxes):  # Assuming the first item is the box tensor
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x_pred = x1/img_width *100
                    y_pred = y1/img_height *100
                    w_pred = (x2-x1)/img_width *100
                    h_pred = (y2 - y1)/img_height*100
                    predictions.append({
                        'from_name': self.from_name,
                        'to_name': self.to_name,
                        'type': 'rectanglelabels',
                        'score': box.conf.item(),
                        'id':get_id(),
                        'value': {
                            'rectanglelabels': ['Person'],
                            'x': x_pred,
                            'y': y_pred,
                            'width': w_pred,
                            'height': h_pred
                        },
                        'origin': 'manual'
                    })

        print({
            'result': predictions
        })
        return [{
            'result': predictions
        }]


    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        pass
