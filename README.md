# audio-genre-prediction_ML

This project implements classification algorithms for two different types of input. SVM(support vector machine) is implemented for classification. For the input to our algorithms, it experimented with both raw amplitude data as well as transformed mel-spectrograms of that raw amplitude data. The output predicted genre out of 10 common music genres. We found that converting our raw audio into mel-spectrograms produced better results on all our models, with our convolutional neural network surpassing human accuracy.

There are the genre that the model will predict:
* blues 
* classical 
* country 
* disco 
* hiphop 
* jazz 
* metal 
* pop 
* reggae 
* rock  
  
The Data for this project is collected from: [GTZAN Genre Collection](http://marsyasweb.appspot.com/download/data_sets)  
 
 ### Usage
 - Clone the repository
 - intall the prerequisite library(numpy, librosa, torch, sklearn)
 - go to the root folder
 - run get_genre.py file
 - run get_genre.py ../test.mp3


To predict the genre of your music, download the mp3 file song and type run get_genre.py ../your-song-name.mp3
 
