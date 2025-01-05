## AI-Powered Football Highlights Generator

This is the repository for our TAAC group project.

### Dataset:

We will use the SoccerNet dataset (https://www.soccer-net.org/data) which includes football game videos and labeled actions like goals, free-kicks, red cards, penalties, and others.

### Project idea:

This project aims to create a program using the SoccerNet dataset. The program will take a football game video as input and produce a short summary of the match's key moments. 
For each key moment, it will provide a video clip showing the action, giving a quick and clear replay of the game's highlights. 

To achieve this, we plan to train a deep learning model using the SoccerNet dataset to recognize key football actions from images. 
Once the model is trained, we will develop a program to break the video into individual frames (images) and match them with the corresponding actions identified by the model.

### Data exploration

![image](https://github.com/user-attachments/assets/157e81bc-a648-4b5f-afde-cbc0c218bd06)

![image](https://github.com/user-attachments/assets/621be694-6e49-4d2d-8ddc-e1c58ed5a180)

![image](https://github.com/user-attachments/assets/a8a6b04b-d74e-4135-a782-22969324dbd9)

### Usage 

To run the streamlit app, use the following command at the root of the project :

```
streamlit run src/app.py --server.maxUploadSize 1000
```

Because the app has been hardcoded to use the features from a specific video, you will have to import this one to try the app. the video is the first half of the game in test_folder/.
