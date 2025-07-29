1) Models :

    1) FaceNet --> Detect Faces 
    2) FasterRCNN ---> Detect Pedestrain
    3) AGNet --> Onnx Model Detect Age & Gender

2) Pipelines :

    1) get_model.py & get_ob_model --> Download Models
    2) person.py ---> Detect Pedestrian
    3) face.py --> Detect Faces
    4) age_gender.py --> Predict Age and Gender
    5) final_pipeline.py --> Combine All Pipelines

        final-pipelines:
          [input image] -> final_pipeline -> [ output ] 