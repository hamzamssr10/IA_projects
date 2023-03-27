import subprocess

# set the URL of the repository you want to clone
url = "https://github.com/WongKinYiu/yolov7"

# run the git clone command
subprocess.run(["git", "clone", url])
