🎵 Emotion-Based Music Recommendation System using Facial Expressions



📌 Project Overview

Music plays a significant role in influencing human emotions, mental health, and overall well-being. Traditional music recommendation systems rely heavily on user history, ratings, or genre preferences. However, they fail to adapt dynamically to a user’s current emotional state.

This project presents an Emotion-Based Music Recommendation System (EBMRS) that uses facial expression recognition to detect the user's emotion in real time and recommend suitable music accordingly. The system captures facial expressions through a webcam, classifies emotions using a Convolutional Neural Network (CNN), and integrates with the Spotify API to suggest relevant playlists.

👨‍🎓 Team Details

Institute: Kallam Haranadhareddy Institute of Technology (Autonomous)
Department: Information Technology

Presented By:

Sk. Sohel (218X1A1237)

V. Manideepa (218X1A1243)

A. Sai Pavan (218X1A1248)

D. Vamsi Krishna (218X1A1250)

Project Guide:
Mr. K. R. M. C. Sekhar, M.Tech., (PhD)
Department of Information Technology

🧠 Abstract

Humans share a deep emotional connection with music. Facial expressions play a crucial role in identifying human emotions such as happiness, sadness, anger, fear, and surprise. This project aims to create a personalized and emotionally intelligent music recommendation system by analyzing facial expressions and recommending music that matches the detected emotional state. The system enhances user experience by offering mood-based music in real time.

🚀 Features

🎥 Real-time facial emotion detection using webcam

🧠 CNN-based emotion classification

🎶 Emotion-based music recommendation

🔗 Spotify API integration for playlists

🌐 Web-based interface using Flask

📊 CSV-based playlist data handling

🛠️ Technologies Used

🔹 Programming Language

Python

🔹 Frameworks & Libraries

TensorFlow

Keras

Scikit-learn

OpenCV

NumPy

Pandas

🔹 Backend

Flask

🔹 Frontend

HTML

CSS

JavaScript

🔹 Tools & APIs

Spotify Web API

Haar Cascade Classifier

💻 Hardware Requirements

Processor: Intel i5 or higher

RAM: 8 GB or more

GPU: 4 GB dedicated (recommended)

Webcam

Mouse / Touchpad

📂 Project Structure
Emotion-Based-Music-Recommendation/
│
├── data/
│   ├── train/
│   └── test/
│
├── static/
│   ├── css/
│   └── js/
│
├── templates/
│   └── index.html
│
├── app.py
├── train.py
├── camera.py
├── spotify.py
├── model_weights.weights.h5
├── requirements.txt
└── README.md

🧩 System Architecture

Webcam captures facial image

Face detection using Haar Cascade

Image preprocessing (48x48 grayscale)

CNN model predicts emotion

Detected emotion mapped to music category

Spotify API fetches playlists

Songs displayed on the web interface



😄 Emotions Detected

Happy

Sad

Angry

Fear

Surprise

Disgust

Neutral



🧪 Model Details

Input Size: 48 × 48 grayscale images

Dataset: FER-2013

Model Type: Convolutional Neural Network (CNN)

Activation Function: ReLU

Output Layer: Softmax

Accuracy Achieved: ~70%


⚙️ Installation & Setup


1️⃣ Clone the Repository
git clone https://github.com/your-username/emotion-based-music-recommendation.git
cd emotion-based-music-recommendation


2️⃣ Install Dependencies
pip install -r requirements.txt


3️⃣ Run the Application
python app.py


4️⃣ Open in Browser
http://127.0.0.1:5000/


📸 Output Screens

Live emotion detection

Emotion label display

Recommended Spotify playlists

Music links based on detected emotion


⚠️ Limitations

Accuracy depends on lighting conditions

Works best with frontal face images

Not suitable for visually impaired or hearing-impaired users

Requires stable internet for Spotify integration

🔮 Future Enhancements

Improve emotion detection accuracy using advanced deep learning models

Multimodal emotion detection (voice + facial expressions)

User profile creation and emotion history tracking

Feedback-based recommendation improvement

Mobile application version

📚 Dataset

FER-2013 Facial Expression Dataset
https://www.kaggle.com/datasets/msambare/fer2013

📖 References

Ricci et al., Recommender Systems Handbook

Aggarwal, Recommender Systems: The Textbook

IEEE & ACM Research Papers

Kaggle Datasets

Spotify Developer Documentation

🏁 Conclusion

This project demonstrates the feasibility of using facial expressions as a reliable input for real-time emotion detection and personalized music recommendation. By combining computer vision, deep learning, and web technologies, the system delivers an engaging and intelligent user experience.



# OUTPUT
<img width="1199" height="578" alt="image" src="https://github.com/user-attachments/assets/fbcd9c3a-19fc-4c0f-beb0-0fcb987421b1" />
<img width="1216" height="576" alt="image" src="https://github.com/user-attachments/assets/7827c3e4-093c-4072-87ae-f74410b3c4f4" />
<img width="1218" height="583" alt="image" src="https://github.com/user-attachments/assets/18b12652-9fb0-4dd8-a342-448f1929a155" />
<img width="1221" height="546" alt="image" src="https://github.com/user-attachments/assets/c0c98fb2-f979-4a55-8c76-8df978888c6b" />
<img width="1198" height="554" alt="image" src="https://github.com/user-attachments/assets/a7385fd3-c4bb-4e53-a604-8ab6ab5ff2fd" />
<img width="1183" height="596" alt="image" src="https://github.com/user-attachments/assets/3c578fac-9d6a-496e-a604-e3bcdab47396" />
<img width="1197" height="557" alt="image" src="https://github.com/user-attachments/assets/21edf54b-347c-4345-aad5-f55170ecac60" />
<img width="1221" height="535" alt="image" src="https://github.com/user-attachments/assets/c3756ddd-641b-4028-a473-045a3b22b8bc" />
<img width="1201" height="574" alt="image" src="https://github.com/user-attachments/assets/ba6cbd4f-c74d-45d7-8a13-43c5d176712e" />
<img width="1222" height="569" alt="image" src="https://github.com/user-attachments/assets/2688bcd4-7855-49d7-ba9d-fd4af994115d" />
