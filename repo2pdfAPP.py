import sys
import multiprocessing
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, \
    QFileDialog, QMessageBox, QCheckBox, QSystemTrayIcon
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt
import json

# Import the classes and functions from Repository2Pdf.py
from Repository2Pdf import RepoToPDF, parallel_process
from repo2pdf_visualization_app import TreeView
import os
from git import Repo

# import ctypes
# ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")


class RepositoryToPDFApp(QWidget):
    def __init__(self, config_file):
        super().__init__()
        with open(config_file, 'r') as file:
            config = json.load(file)
        self.wkhtmltopdf = config["wkhtmltopdf"]
        self.output_dir = config["output_dir"]
        self.output_name = config["output_name"]
        self.overwrite = config["overwrite"]
        self.style = config["style"]
        self.icon_path = config["icon_path"]
        self.git_download_path = config.get("git_download_path")
        if not self.git_download_path or self.git_download_path == "":
            self.git_download_path = os.getcwd()

        self.only_read_config = config.get("only_read")
        self.gitignore_flexible_config = config.get("gitignore_flexible_config")

        self.num_multiprocess = config["num_multiprocess"] if config["num_multiprocess"] > 0 \
            else multiprocessing.cpu_count()
        self.setWindowTitle("Repository to PDF Converter")
        self.setWindowIcon(QIcon("logo_re.png"))

        self.tray_icon = QSystemTrayIcon(QIcon(self.icon_path))
        self.tray_icon.show()

        self.resize(400, 200)

        self.create_widgets()
        self.setup_layout()
        self.tree_window = None  # show tree structure

    def create_widgets(self):
        # 添加 Logo
        self.logo_label = QLabel()
        pixmap = QPixmap("logo.png")  # 替换成你的 logo 图片路径
        self.logo_label.setPixmap(pixmap)
        self.logo_label.setAlignment(Qt.AlignCenter)

        self.git_url_label = QLabel("Github URL (Modify config to change downloading path):")
        self.git_url_edit = QLineEdit()

        self.repo_label = QLabel("Repository Path:")
        self.repo_path_edit = QLineEdit()
        self.browse_button_repo = QPushButton("Browse")
        self.convert_button = QPushButton("Convert to PDF")

        self.ignore_file_label = QLabel("ignore file or suffix (For example: test, experiment...):")
        self.ignore_file_edit = QLineEdit()


        self.only_read_label = QLabel("only read suffix (For example: .md, .py...):")
        self.only_read_edit = QLineEdit()


        # 添加可编辑的部件
        self.output_dir_label = QLabel("Output Directory:")
        self.output_dir_edit = QLineEdit(self.output_dir)
        self.output_dir_browse_button = QPushButton("Browse")

        self.output_name_label = QLabel("Output Name:")
        self.output_name_edit = QLineEdit(self.output_name)

        self.information_label = QLabel("Miao~: Welcome to use Quick Git2PDF! I'm ready to convert!")
        self.information_label.setAlignment(Qt.AlignLeft)
        # set color "color: rgb(99, 203, 195);"
        self.information_label.setStyleSheet("color: rgb(99, 203, 195); font-size: 12px; font-weight: bold;")

        # 添加覆盖选项复选框
        self.overwrite_checkbox = QCheckBox("Overwrite Existing PDF")
        self.overwrite_checkbox.setChecked(self.overwrite)

        self.browse_button_repo.clicked.connect(self.browse_repository)
        self.output_dir_browse_button.clicked.connect(self.browse_output)
        self.convert_button.clicked.connect(self.convert_to_pdf)

    def setup_layout(self):
        repo_layout = QHBoxLayout()
        repo_layout.addWidget(self.repo_path_edit)
        repo_layout.addWidget(self.browse_button_repo)

        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_dir_browse_button)

        layout = QVBoxLayout()
        # 将 Logo 添加到布局顶部
        layout.addWidget(self.logo_label)

        layout.addWidget(self.repo_label)
        layout.addLayout(repo_layout)
        layout.addWidget(self.git_url_label)
        layout.addWidget(self.git_url_edit)
        # layout.addWidget(self.repo_label)
        # layout.addWidget(self.repo_path_edit)
        # layout.addWidget(self.browse_button_repo)

        layout.addWidget(self.ignore_file_label)
        layout.addWidget(self.ignore_file_edit)

        layout.addWidget(self.only_read_label)
        layout.addWidget(self.only_read_edit)

        # 添加可编辑的部件到布局
        layout.addWidget(self.output_dir_label)
        layout.addLayout(output_dir_layout)

        layout.addWidget(self.output_name_label)
        layout.addWidget(self.output_name_edit)

        # 添加覆盖选项复选框到布局
        layout.addWidget(self.overwrite_checkbox)

        layout.addWidget(self.convert_button)
        layout.addWidget(self.information_label)
        self.setLayout(layout)

    def browse_repository(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Repository Directory")
        if directory:
            self.repo_path_edit.setText(directory)
            self.output_dir_edit.setText(directory)
            self.output_name_edit.setText(str(Path(directory).name))

    def browse_output(self):
        directory = QFileDialog.getExistingDirectory(self, "Select output Directory")
        if directory:
            self.output_dir_edit.setText(directory)

    def convert_to_pdf(self):
        repo_path = self.repo_path_edit.text()
        url = self.git_url_edit.text()
        use_url = False
        if url.endswith('.git'):
            repo_path = self.download_git_repository()
            use_url = True
        if repo_path:
            try:
                self.information_label.setText("Miao~: Converting to PDF...")
                only_read_config = self.only_read_config + self.get_file_or_suffix_list(self.only_read_edit.text())
                flexible_ignore_config = \
                    self.gitignore_flexible_config + self.get_file_or_suffix_list(self.ignore_file_edit.text())

                repo = RepoToPDF(directory=Path(repo_path), wkhtmltopdf_path=self.wkhtmltopdf,
                                 style=self.style, output_dir=self.output_dir_edit.text(),
                                 output_name=self.output_name_edit.text(),
                                 rewrite=self.overwrite_checkbox.isChecked(),
                                 only_read_config=only_read_config,
                                 flexible_ignore_config=flexible_ignore_config)
                if len(repo.files_to_convert) == 0:
                    QMessageBox.information(self, "Warning!", "No file to convert!")
                else:
                    parallel_process(repo, num_processes=self.num_multiprocess)
                    print(f"{repo.output_tree_path}, {repo_path}")
                    with open(f"{repo.output_tree_path}", "r") as file:
                        file_tree_dict = json.load(file)
                    self.information_label.setText("Miao~: Conversion completed successfully!")
                    QMessageBox.information(self, "Conversion Complete", f"PDF generation completed in "
                                                                         f"{str(repo.output_path)} successfully!")
                    self.show_file_structure(file_tree_dict)
                    self.information_label.setText("Miao~: Ready to convert!")
                    if use_url:
                        os.system(f'rm -rf {repo_path}')

            except Exception as e:
                QMessageBox.warning(self, "Error", f"An error occurred: {str(e)}")
        else:
            QMessageBox.warning(self, "Error", "Please select a repository directory or use correct url.")

    def show_file_structure(self, data):
        self.tree_window = TreeView(data)
        self.tree_window.show()

    @staticmethod
    def get_file_or_suffix_list(text):
        return [part.strip() for part in text.split(',') if len(part.strip()) > 0]

    def download_git_repository(self):
        download_path = self.git_download_path

        url = self.git_url_edit.text()
        assert url.endswith('.git'), 'Please input a valid git repository url'

        if not os.path.exists(download_path):
            os.makedirs(download_path)

        # 获取仓库名称
        repository_name = url.split('/')[-1].replace('.git', '')
        self.information_label.setText(f"Miao~: Downloading {repository_name} from {url}")
        # 拼接本地仓库路径
        local_repository_path = os.path.join(download_path, repository_name)

        # 如果本地仓库路径已经存在，则先移除
        if os.path.exists(local_repository_path):
            os.system(f'rm -rf {local_repository_path}')

        self.output_name_edit.setText(repository_name)
        self.output_dir_edit.setText(download_path)
        # 克隆仓库
        Repo.clone_from(url, local_repository_path)
        self.information_label.setText(f"Miao~: Successfully downloaded {repository_name} from {url}")

        return local_repository_path


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = RepositoryToPDFApp(config_file="config.json")
    window.show()
    sys.exit(app.exec_())