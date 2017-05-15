#Image Processing of Arabidopsis Plants

Welcome to the Arabidopsis Image Processing Code. This code has been converted from MatLab to Python OpenCV. The purpose of this code is to speed up the process of measuring the temporal leaf area of arabidopsis plants. 

Temporal images of various genotypes of arabidopsis have been captured and saved to folders. The objectives of the code are the following:

1. Select the image directory
2. Read only the image file by reading from the file extension related with images.
3. Preprocess the image: interactive tool to rotate and crop the image.
4. Crop out individual pot based on predefined dimension.
5. Segment the plant of interest in the pot using hsv and lab color space
6. Remove noise by using: bwareaopen, imdilate, bwconncomp, imerode and imfill functionality.

A code which addresses these objectives was written in MatPlob, but we would like to re-write the code in OpenCV. Due to differences in image processing libraries between MatLab and OpenCV, some of the objectives of the OpenCV code will be carried out differently but should accomplish the same goal.