from django.http import HttpResponse,HttpResponseRedirect
from django.shortcuts import render
import subprocess
import os


def index(request):
    return render(request, 'index.html')


def predict(request):
    return render(request, "gallery.html", {"predict": predict})


def upload_file(request):
    # 请求方法为POST时，进行处理
    if request.method == "POST":
        # 获取上传的文件，如果没有文件，则默认为None
        File = request.FILES.get("myfile", None)
        if File is None:
            return HttpResponse("没有需要上传的文件")
        else:
            # 打开特定的文件进行二进制的写操作
            # print(os.path.exists('/temp_file/'))
            with open("./template/%s" % File.name, 'wb+') as f:
                # 分块写入文件
                for chunk in File.chunks():
                    f.write(chunk)
                os.system('activate py36&&d:&&cd study\dl\Graduation_Project\object\template&&python segment.py')
                # subprocess.call('activate py36&&d:&&cd study\dl\Graduation_Project\object\template&&python segment.py', shell=True)
                return HttpResponse("上传成功")
    else:
        return render(request, "gallery.html")
