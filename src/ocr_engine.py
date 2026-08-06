"""
PaddleOCR recognition-only inference via ONNX Runtime (CPU).

We don't run a text detector — the ROI already localizes the digit field
(drawn once, tracked every frame by the stabilizer), so this is purely
the recognition network: crop in, string + confidence out.

Drop your exported model + dictionary into models/:
  models/en_PP-OCRv5_mobile_rec.onnx
  models/ppocrv5_dict.txt
and point config.yaml -> ocr.model_path / ocr.dict_path at them.
"""

import cv2
import numpy as np
import onnxruntime as ort


class PaddleOCRRecognizer:
    def __init__(self, model_path, dict_path, rec_image_height=48):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.rec_image_height = rec_image_height

        with open(dict_path, "r", encoding="utf-8") as f:
            chars = [line.rstrip("\n") for line in f]
        # index 0 is reserved for CTC blank
        self.charset = ["blank"] + chars + [" "]

    def preprocess(self, img):
        # OpenCV frames are BGR; PP-OCRv5 was trained on RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        ratio = w / float(h) if h > 0 else 1.0
        target_w = max(int(self.rec_image_height * ratio), self.rec_image_height)
        resized = cv2.resize(img, (target_w, self.rec_image_height))
        resized = resized.astype(np.float32) / 255.0
        resized = (resized - 0.5) / 0.5
        resized = resized.transpose(2, 0, 1)
        return np.expand_dims(resized, axis=0).astype(np.float32)

    def recognize(self, img):
        if img is None or img.size == 0:
            return "", 0.0

        inp = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: inp})
        preds = outputs[0][0]  # (seq_len, num_classes)

        pred_ids = np.argmax(preds, axis=1)
        pred_probs = np.max(preds, axis=1)

        text_chars = []
        confs = []
        prev = -1
        for idx, p in zip(pred_ids, pred_probs):
            if idx != 0 and idx != prev:  # skip blank + collapse repeats (CTC greedy decode)
                if idx < len(self.charset):
                    text_chars.append(self.charset[idx])
                    confs.append(p)
            prev = idx

        text = "".join(text_chars)
        avg_conf = float(np.mean(confs)) if confs else 0.0
        return text, avg_conf
