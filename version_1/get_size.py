# import module
import os

IGNORE_FOLDERS = [
                      'FreeRouting_Example',
                      'models',
                      '__pycache__',
                      'yolor',
                      'git_push.sh',
                      'get_size.py',
                      'pyspice_examples.py',
                      'test_design.py',
                      'rawread.py',
                      'readFile.py',
                      'skidl_examples.py',
                      'skidl_examples_lib_sklib.py',
                      'skidl_lib_sklib.py',
                      'pyspice_examples.py',
                      'pcbnew_layout.py',
                 ]
 
# assign size
size = 0
 
# assign folder path
Folderpath = '.'
 
# get size
for path, dirs, files in os.walk(Folderpath):
    print(dirs)
#     for f in files:
#         # print(f)
#         fp = os.path.join(path, f)
#         size += os.path.getsize(fp)

# size_MB = size/1e+6
 
# # display size
# print(int(size_MB))