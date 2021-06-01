from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files('skimage.feature')
hiddenimports = collect_submodules('skimage.feature')
