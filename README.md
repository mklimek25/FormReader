# FormReader
All of the code related to the form reader that I had designed for a practical industrial application.
The Automatic Form Reader that I had designed was designed to satisfy a need in an industrial facility. 
Handwritten forms need to be tabulated, which was being performed manually by our management team.

I used OpenCV and Machine Learning Algorithms do design an automated form reader that accepts a scanned picture of a form and returns the tabulated data by cell. once collected, this data can be returned in tabulated form or entered into a centralized database where the output data can be analyzed with production data through a centralized database.

The Algorithm first calibrates the scanned picture, finding all four corners of the image using OpenCV and returning only the straightened portion of the image that contains the data. This is important for two reasons:
1. Manually scanned images will never be perfectly straight,
2. The program needs to calibrate in order to identify the form cells properly

Once the read section is located, the program goes through an algorithm designed to identify all of the cells to be read. It does this by identifying all contours and then sorting the contours by position on the image. 

![Alt text](relative%20path/to/img.jpg?raw=true "")
