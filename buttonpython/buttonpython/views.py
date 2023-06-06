from django.shortcuts import render
import cv2
import clip
import subprocess


def button(camera):
    return render(camera,'home.html')


def output(camera):

    import subprocess

# Run the other script
    subprocess.run(["python", "C:/Users/USER/Downloads/Django2/camera.py"])



    return render(camera,'home.html')



    
