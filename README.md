# FormReader
All of the code related to the form reader that I had designed for a practical industrial application.
The Automatic Form Reader that I had designed was designed to satisfy a need in an industrial facility. 
Handwritten forms need to be tabulated, which was being performed manually by our management team.

I used OpenCV and Machine Learning Algorithms do design an automated form reader that accepts a scanned picture of a form and returns the tabulated data by cell. once collected, this data can be returned in tabulated form or entered into a centralized database where the output data can be analyzed with production data through a centralized database.

The Algorithm first calibrates the scanned picture, finding all four corners of the image using OpenCV and returning only the straightened portion of the image that contains the data. This is important for two reasons:
1. Manually scanned images will never be perfectly straight,
2. The program needs to calibrate in order to identify the form cells properly

Once the read section is located, the program goes through an algorithm designed to identify all of the cells to be read. It does this by identifying all contours and then sorting the contours by position on the image like what is found in the image below. 

<img width="1138" alt="Box Detection" src="https://github.com/mklimek25/FormReader/assets/90988711/fe83fb9e-4edb-457b-af60-b760b9746008">

 Once the boxes are detected, they are analyzed for contours. These contours are each reconfigured to fit the MNIST training set and are read using a trained model developed using the MNIST training set. Contours (identified numbers) are separated out from the remainder of the numbers and reconfigured to match the model input which is trained using the MNIST Dataset. The main adjustment needed is an adjustment to image size which is performed by an algorithm that I designed to get the clearest image of the contour and adjusting the size of that image to fit the MNIST input data. This results in an input image shown in the example below:

 <img width="661" alt="Screenshot 2023-10-08 at 1 43 29 PM" src="https://github.com/mklimek25/FormReader/assets/90988711/61850de5-4b15-4f6c-97e6-79a9b04f896f">
 
 These input integers will then be analyzed by the MNIST trained model developed in 'model_generator.py' that was trained and tested based on 60,000 images similar to the input integer displayed. The model will predict each individual contour and return the combination of the predicted values.

 This prediction model performed at a 97% accuracy with digital data, which was impressive. When reading employee handwriting, the accuracy dropped considerably, which is not supprising as I could hardly read the handwriting myself. 

 

 


