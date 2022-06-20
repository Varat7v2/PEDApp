# import module
import os, sys
import config as myconfig

WORKSPACE_PATH = os.path.abspath(os.getcwd())

# get size
for path, dirs, files in os.walk(WORKSPACE_PATH):
    print(path)
    # Initialize file+folder size
    size = 0
    for item in dirs+files:
        if item not in myconfig.IGNORE_FOLDERS:
            # print(item)
            fp = os.path.join(WORKSPACE_PATH, item)
            size += os.path.getsize(fp)
        else:
            print(item)
   
size_MB = size/1e+6
 
# display size
print(int(size_MB), 'MB')
