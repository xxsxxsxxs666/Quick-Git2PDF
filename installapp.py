from PyInstaller.__main__ import run

if __name__ == '__main__':
    opts = [
        'repo2pdfAPP.py',  # 您的主 Python 文件
        '--windowed',  # 不显示命令行窗口（仅 GUI）
        '--hidden-import=pdfkit',  # 隐藏 pdfkit 模块
        '--hidden-import=PyPDF2',  # 隐藏 pdfkit 模块
        '--hidden-import=tqdm',  # 隐藏 pdfkit 模块
        '--icon=H:\git\Repo2PDF\logo_re.ico',  # 设置图标文件
        '--add-data=.\*.png;.',  # 包含 logo.png 文件
        '--add-data=config.json;.',  # 包含 logo.png 文件
    ]
    run(opts)