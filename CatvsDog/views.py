#coding=utf-8
from django.shortcuts import render
from .form import UploadImageForm
from .models import image
from .test_vgg import pictureclassify

def index(request):
    """图片的上传"""
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            picture = image(photo=request.FILES['image'])
            picture.save()
            lab = pictureclassify(picture)
            return render(request, 'result.html', {'picture': picture, 'label': lab})

    else:
        form = UploadImageForm()

        return render(request, 'index.html', {'form': form})