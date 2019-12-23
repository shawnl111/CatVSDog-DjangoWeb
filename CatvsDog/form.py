from django import forms

class UploadImageForm(forms.Form):
    #text = forms.CharField(max_length=100)
    image = forms.ImageField(
        label='请上传您所需识别的猫/狗的图片:',
    )