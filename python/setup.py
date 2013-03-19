from distutils.core import setup, Extension

setup (
    name = 'SegAnnot',
    version = '1.0',
    description = 'L2-optimal annotation-aware segmentation',
    ##packages=["SegAnnot"],
    ext_modules = [
        Extension('SegAnnot',['interface.c','SegAnnot.c']),
        ],
    )
