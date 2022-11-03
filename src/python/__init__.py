import os

if os.name=='nt':
    os.add_dll_directory(os.path.join(os.path.dirname(os.path.realpath(__file__)),'Release'))
