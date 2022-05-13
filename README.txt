This is the code to run the linear UUV model with the pipeline and dynamic obstacles.


To execute the code, run the following command:
python plotTrajectories.py tanh_4_30x30.h5


The files are as follows:
model_oneHz.mat: Contains the matrices that determine the linear UUV model. You shouldn't need to alter this file at all.
tanh_4_30x30.h5: Contains the NN controller that follows the pipe and avoids obstacles. You shouldn't need to alter this file at all.
Fish_oneHz.py: This file contains the 'World' object, which keeps track of the state of the environment. The important functions here are the reset() and step() functions. Also note that lines 130 and 131 are when noise gets added to the obstacle's velocity (that seems like something we might want to alter during our experiments)
plotTrajectories.py: The file runs the World object from Fish_oneHz.py. The only function we really care about here is main(). It loops through for a set number of trials and for each trial it runs one step of the World (line 151) and computes the next control action (line 153) for a fixed number of iterations. It saves the uuv and obstacle positions into large data arrays and then plots them afterwards.


Let me (Matthew Cleaveland) know if you have any questions or uncertainties about this.
