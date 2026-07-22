# Facial-Recognition-Model
Step-by-Step guide to implementing a real-time face recognition system using MTCNN, FaceNet (with a MobileFaceNet architecture in mind), and Python in a Jupyter Notebook. This guide is tailored for your specified hardware (GTX 1060/1650 GPU) and camera setup.


Conceptual Overview: The Three Key Components

Before diving into the code, let's understand the role of each model in our pipeline. This system works in a sequence:

1.MTCNN (Multi-task Cascaded Convolutional Networks): This is our face detector. Its sole job is to scan each frame from the camera and find the exact location of any faces. It outputs bounding box coordinates (the x and y of the top-left corner, plus width and height) for every face it finds.[1][2]

2.FaceNet: This is the core recognition engine. It takes the cropped face image provided by MTCNN and converts it into a numerical representation called a "face embedding."[3] This embedding is a vector (a list of numbers, typically 128 or 512) that uniquely represents the facial features. The key principle is that embeddings of the same person's face will be very close to each other in mathematical terms, while embeddings of different people will be far apart.[1][4]

3.MobileFaceNet: This is an efficient and lightweight version of the FaceNet architecture.[5] Given your goal of running this on a GTX 1060/1650 with a 720p-1080p camera feed, using a model like MobileFaceNet is ideal. It provides a great balance between accuracy and performance, allowing for higher frames per second (FPS). While some libraries provide the standard, heavier FaceNet model (like InceptionResnetV1), the implementation logic remains identical.[6][7]
The overall workflow is: Detect Face (MTCNN) -> Generate Embedding (FaceNet/MobileFaceNet) -> Compare Embeddings.[8][9]

Step 1: Setting Up Your Environment
First, we need to install the necessary libraries and ensure your GPU is ready.
1.1. Essential Libraries
We will primarily use facenet-pytorch as it provides excellent, pre-trained PyTorch implementations of MTCNN and InceptionResnetV1 (a common FaceNet architecture).[10][11] We'll also need opencv-python to handle the camera feed.[12]
Open your terminal or an Anaconda Prompt and install the libraries using pip:

<img width="780" height="209" alt="image" src="https://github.com/user-attachments/assets/56af007e-6ea3-4041-a2f0-833b7aaba1f8" />


1.2. GPU and CUDA Setup
For your GTX 1060/1650 to accelerate the process, you need the correct NVIDIA drivers, CUDA Toolkit, and cuDNN installed. The facenet-pytorch library will automatically use your GPU if it detects a valid CUDA installation.[13]
●NVIDIA Driver: Ensure you have the latest drivers for your GPU.
●CUDA Toolkit & cuDNN: PyTorch relies on these. You can find many tutorials online for installing them on Ubuntu. A good starting point is the official NVIDIA documentation or guides specifically for setting up a deep learning environment on Ubuntu.[13][14]
You can verify if PyTorch can see your GPU by running this in Python:

 <img width="780" height="227" alt="image" src="https://github.com/user-attachments/assets/e5ab28b8-0aaf-40a8-a844-a63074918fdf" />


Step 2: Preparing a Database of Known Faces
The system needs to know who to recognize. We will create a simple image database on our computer.
1.Create a root folder named face_database.
2.Inside face_database, create a sub-folder for each person you want to recognize. Name the folder with the person's name (e.g., elon_musk, bill_gates).
3.Place several clear, well-lit pictures of that person inside their respective folder. Use different angles and expressions for better accuracy.
Your folder structure should look like this:


<img width="780" height="549" alt="image" src="https://github.com/user-attachments/assets/6cc1fe04-cc83-4b14-bbb8-150afede282b" />

<img width="780" height="435" alt="image" src="https://github.com/user-attachments/assets/99627a41-c282-4153-8328-97cf78f4fe91" />

<img width="780" height="244" alt="image" src="https://github.com/user-attachments/assets/ccd54f4f-f4a6-4254-b6d6-245dd3db1696" />

3.2. Generate Embeddings for the Known Faces
Next, we'll write a function to process our face_database, detect faces, generate their embeddings, and store them in a dictionary.

<img width="780" height="539" alt="image" src="https://github.com/user-attachments/assets/c08bfc71-c85d-4c9a-b999-07ac84916879" />

<img width="780" height="355" alt="image" src="https://github.com/user-attachments/assets/69defa17-4525-42ad-b05b-74b1eb7769e5" />
          
3.3. Real-Time Recognition from Camera
This is the main part of our application. We will open the camera feed, and in a loop, we will detect and recognize faces in real-time.

<img width="780" height="355" alt="image" src="https://github.com/user-attachments/assets/77aa9ece-09dc-43f1-b829-a44a91e38941" />

<img width="780" height="417" alt="image" src="https://github.com/user-attachments/assets/34ef8513-f502-4c05-ad5c-96a65a7bd5ff" />
 
<img width="780" height="417" alt="image" src="https://github.com/user-attachments/assets/ad2744ee-fabf-474b-980f-c7d3cc646b5b" />
                    
<img width="780" height="474" alt="image" src="https://github.com/user-attachments/assets/fe5dde86-f2a6-4c3a-83da-2b3a8119f256" />
        
<img width="780" height="318" alt="image" src="https://github.com/user-attachments/assets/02190e54-be85-4fd7-a8af-937fd91fe4dd" />

Explanation of the Real-Time Code:

1.Initialization: We set a DISTANCE_THRESHOLD. This value is crucial for accuracy. A lower value makes recognition stricter. You should experiment to find what works best for you.

2. Video Capture: cv2.VideoCapture(0) accesses your primary webcam. We start a while True loop to continuously read frames.

3.Face Detection: In each loop, we convert the frame to an RGB PIL Image, which is the expected input for the facenet-pytorch MTCNN model. mtcnn.detect(img) returns the bounding boxes of all detected faces.

4. Embedding Generation: For each detected face, we use mtcnn(img) again, but this time it returns the cropped and processed face tensor. We pass this tensor to our resnet (FaceNet model) to get the face embedding.

5.Comparison: We iterate through our database of known_faces_embeddings. We calculate the Euclidean distance between the detected face's embedding and each known embedding.

6.Recognition: If the smallest distance found is less than our DISTANCE_THRESHOLD, we assign the corresponding name to the face. Otherwise, it remains "Unknown".

7. Display: Using OpenCV's drawing functions (cv2.rectangle, cv2.putText), we draw the bounding box and the recognized name (along with the distance score) on the original frame.

8.Exiting: cv2.imshow displays the window. The loop breaks when you press the 'q' key, releasing the camera and closing the window.

With this implementation, your GTX 1060/1650 should comfortably handle the 720p stream at a decent frame rate, achieving your goal of a real-time face recognition system.

Sources help

1.medium.com

2.sefiks.com

3.machinelearningmastery.com

4.dev.to

5.kaggle.com

6.youtube.com

7.dev.to

8.aionlinecourse.com

9.medium.com

10.github.com

11.medium.com

12.medium.com

13.medium.com

14.anujonthemove.com

15.youtube.com

16.rayups.com
