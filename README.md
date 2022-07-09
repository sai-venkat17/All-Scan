# All Scan

## About Project
A web platform in which a user can select the type of MRI scan(Brain or Lung or Kidney) and upload the MRI scan of the same then the website will detect if the MRI scan is tumorous or not and displays the final result along with intermediatory results of the processing which gives user the clear view of where exactly the tumor is present

## Models used
* MLP classifier
* SVM classifier
* Decision Tree 
* Random Forest
* Logistic Regression
* XGB classifier
* CNN
* Ensemble learning model


## Innovation Component
We are using extensive pre-processing techniques to eliminate all kinds of noises with several stages of filtering and thresholding. As the number of samples of Malignant tumors are limited we are going to increase the amount of data by synthesizing new data and this new data will be produced by slight modifications of already existing data, by performing this we will regularize our model thus reducing the overfitting problem and increasing accuracy. We have also proposed a novel Ensemble model with 5 weak learners which is outperforming every other model.

## Preprocessing Steps
1. Cropping
2. Resizing
3. Normalizing
4. Converting image to numpy array
5. Converting image data into a dataframe
6. Applying PCA to reduce the dimensions of the dataframe

## Website screenshots
### Home page
![image not found](https://drive.google.com/uc?export=view&id=1mSQVHgzrZM0yGkRQ9Uj2aFSQPpQzZ-Cb)

### Options
![image not found](https://drive.google.com/uc?export=view&id=1Pl1OkU_SqmkjwX5fm-yN1dx2XlEeXRsK)

### Brain demo
![image not found](https://drive.google.com/uc?export=view&id=1MxJkLR2ZmF_ZpK92hBXGnHaurMtaK4IK)

### Kidney demo
![image not found](https://drive.google.com/uc?export=view&id=1kgIDfoyEEEYGV67KUfjLfUrGu9GKOy5W)

### Lung demo
![image not found](https://drive.google.com/uc?export=view&id=1dJEoFe4FOw0TSBnDTZMueGCg0kUJCsNx)






