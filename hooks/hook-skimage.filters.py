from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files('skimage.filters')
hiddenimports = collect_submodules('skimage.filters')
