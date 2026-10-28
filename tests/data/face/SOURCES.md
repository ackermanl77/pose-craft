# Face test photo

`portrait_smiling.jpg` is for tests/test_face.py (F12). It comes from Wikimedia Commons:
[Jasmin Moghbeli official portrait.jpg](https://commons.wikimedia.org/wiki/File:Jasmin_Moghbeli_official_portrait.jpg)
by NASA. It is in the **public domain**: a work of the US federal government, with the license read
from the Commons API when downloaded on 2026-10-27. Resized to 640 px on the long edge with the
metadata removed.

The two `.json` files hold reference outputs of the original (non-converted) models:

- `portrait_smiling.mediapipe.json`: MediaPipe 1.2.0's own FaceLandmarker blendshapes for the photo.
- `blendshapes_litert_reference.json`: LiteRT running `face_blendshapes.tflite` on two point sets.
