import os
import time
from pathlib import Path
from typing import List

import pdfkit
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
import mimetypes
import multiprocessing

import PyPDF2
from tqdm import tqdm
import json


class RepoToPDF:
    """
    This class convert repository to pdf where this must be already cloned
    """
    # Standard Stuffs
    gitignore_stuff = [
        ".pdf",
        "__init__.py",
        "__pycache__",
        ".gitignore",
        ".git/",
        "LICENSE",
        ".github/",
        "requirements.txt",
        "pyproject.toml", "poetry.lock",
        "Pipfile", "Pipfile.lock",
        ".idea/",
        "env-sample",
        ".flake8",
        "setup.cfg", "Procfile"
        "pytest.ini",
        "file_tree.json",
        "SOURCE.txt",
    ]
    # ".yml", ".xml", ".txt", "README.md",

    # flexible_stuff
    gitignore_flexible = [
    ]

    only_read = [
    ]

    def __init__(self, directory, wkhtmltopdf_path, style="colorful", output_dir=None,
                 output_name=None, rewrite=True, flexible_ignore_config=None, only_read_config=None,
                 only_current_level=False):
        self.directory = Path(str(directory))
        self.style = style
        self.name_repository = str(directory).strip(os.sep).split(os.sep)[-1]
        self.only_current_level = only_current_level
        self.ignored_files = \
            RepoToPDF.gitignore_stuff + self.ignore_files() + RepoToPDF.gitignore_flexible + flexible_ignore_config
        self.only_read = RepoToPDF.only_read + only_read_config
        self.files_to_convert = self.select_files(self.directory)
        self.tree = self.create_tree(self.directory)
        self.wkhtmltopdf_path = wkhtmltopdf_path
        self.output_dir = output_dir if output_dir is not None else self.directory
        self.output_name = output_name if output_name is not None else self.name_repository
        self.output_path = f"{str(self.output_dir)}/{self.output_name}.pdf"
        self.output_tree_path = f"{str(self.output_dir)}/{self.output_name}_tree.json"
        if Path(self.output_path).exists() and not rewrite:
            self.output_path = f"{str(self.output_dir)}/{self.output_name}_1.pdf"
            self.output_tree_path = f"{str(self.output_dir)}/{self.output_name}_tree_1.json"
        self.pbar = None

    def ignore_files(self) -> List[str]:
        """
        Scrap the gitignore file it it exists and prepare all the stuffs that must be ignored.
        :return: List of strings, that depicts the files/folders scrapped from .gitignore file.
        """

        gitignore_content = []
        if self.directory.joinpath(".gitignore").exists():
            with self.directory.joinpath(".gitignore").open() as fp:
                content = fp.readlines()
                for i in content:
                    line = i.strip()
                    if len(line) > 0 and "#" not in line:
                        gitignore_content.append(line.strip("*"))
        else:
            print(
                "Since the .gitignore file doesn't exists, all the files will be considered"
            )
        return gitignore_content

    def add_ignore_files(self, new_ignore_stuff: List[str]) -> None:
        self.ignored_files += new_ignore_stuff

    def select_files(self, directory: Path, files_selected=None) -> List[Path]:
        """
        Discover which files are allowed, then return a list with them.
        :param directory: Path
        :param files_selected: List
        :return: List of paths of selected/filtered files
        """
        if files_selected is None:
            files_selected = []

        mimetypes.add_type('text/typescript', '.ts', True)
        mimetypes.add_type('text/readme', '.md', True)
        mimetypes.add_type('text/json', '.json', True)
        for i in sorted(directory.iterdir()):
            if not self.must_ignore(i) and i.is_dir() and not self.only_current_level:
                self.select_files(i, files_selected)  # 传递新的空列表
            elif self.is_text_files(i):
                if not self.must_ignore(i) and i.is_file() and self.only_read_files(i):
                    files_selected.append(i)
        return files_selected

    @staticmethod
    def is_text_files(filename: Path) -> bool:
        """
        Validates if a file is a text file or not.
        :param filename: Path
        :return: True or False
        """
        mime = mimetypes.guess_type(filename)
        return mime[0] and "text" in mime[0]

    def must_ignore(self, filename: Path) -> bool:
        """
        Validates if a file must be ignored or not from certain logic.
        :param filename: Path
                Ex.: PosixPath('/home/alfonso/PycharmProjects/xml-to-json/manage.py')
        :return: True or False
        """
        if filename.is_dir() and f"{filename.name}/" in self.ignored_files:
            return True
        elif (
                filename.is_file()
                and filename.suffix in self.ignored_files
                or filename.name in self.ignored_files
        ):
            return True

    def only_read_files(self, filename: Path) -> bool:
        """
        Validates if a file must be ignored or not from certain logic. If self.only_read list is [], then we don't use
        this mode.

        :param filename: Path
                Ex.: PosixPath('/home/alfonso/PycharmProjects/xml-to-json/manage.py')
        :return: True or False
        """
        if len(self.only_read) == 0:
            return True
        if filename.is_file() and filename.suffix in self.only_read:
            return True
        else:
            return False

    def create_tree(self, startpath: Path) -> str:
        """
        Create a tree of the folder
        :param startpath:  Path of the folder
        :return: string that depicts the tree of the folder
        """

        # With a criteria (skip hidden files)
        def is_not_hidden(path):
            # filtered_path = False
            ignore = ("__pycache__/", "__init__.py")
            if (
                    path.name.startswith(".")
                    or path.name.startswith("__")
                    or path.name in ignore
            ):
                return False
            return True

        paths = DisplayablePath.make_tree(
            Path(startpath), criteria=is_not_hidden
        )
        tree = ""
        for path in paths:
            tree += path.displayable() + "\n"
        return tree

    def generate_html(self, file: Path, header_path=False) -> tuple[str, int]:
        """
        Generate an HTML file according to a file received in a directory
        :param file: Path. File from the directory input
        :param header_path: boolean. Determines if the file will
                            have the directory in the first line
        :return: str. A string that depicts the html file, int. The length of the content
        """
        path_file = ""
        if header_path:
            for i, parent in enumerate(file.parts):
                if parent == self.name_repository:
                    path_file = "/".join(file.parts[i:])
                    break

        try:
            with open(file, 'r', encoding='utf-8', errors='replace') as fp:
                content = fp.read()
        except UnicodeEncodeError as e:
            print(f"File {file.stem} unreadable\n{e}")
            content = 'Unreadable content'
        except Exception as e:
            print(e)
            content = e

        content_len = len(content) - 1
        content = f'{path_file} \n{content}' if content else f'{file} \nEmpty File'

        # 根据文件扩展名选择合适的词法分析器
        lexer = None
        if file.suffix == '.json':
            lexer = get_lexer_by_name("json", stripall=True)
        elif file.suffix == '.ts':
            lexer = get_lexer_by_name("typescript", stripall=True)
        elif file.suffix == '.txt':
            # 纯文本文件不需要特殊的词法分析器
            lexer = get_lexer_by_name("text", stripall=True)
        elif file.suffix == '.md':
            lexer = get_lexer_by_name("markdown", stripall=True)
        else:
            # 未知文件类型，默认使用文本词法分析器
            lexer = get_lexer_by_name("text", stripall=True)

        # 如果文件是 Python 代码，使用 Python 词法分析器
        if file.suffix == '.py':
            lexer = get_lexer_by_name("python", stripall=True)

        formatter = HtmlFormatter(
            full=True,
            style=self.style,
            filename=str(file),
            linenos=True,
            linenostart=0,
        )

        # 使用 Pygments 高亮代码，并生成 HTML
        if lexer:
            highlighted_code = highlight(content, lexer, formatter)
        else:
            # 如果没有合适的词法分析器，直接显示原始内容
            highlighted_code = f'<pre>{content}</pre>'
        # print(file, content_len, origin_content)
        return highlighted_code, content_len

    def generate_html_from_text(self, content: str, file_name="tree.txt", lexer_type="text") -> str:
        lexer = get_lexer_by_name(lexer_type, stripall=True)
        formatter = HtmlFormatter(
            full=True,
            style=self.style,
            filename=file_name,
            linenos=True,
            linenostart=0,
        )

        # 使用 Pygments 高亮代码，并生成 HTML
        if lexer:
            highlighted_code = highlight(content, lexer, formatter)
        else:
            # 如果没有合适的词法分析器，直接显示原始内容
            highlighted_code = f'<pre>{content}</pre>'

        return highlighted_code

    def text2html2pdf(self):
        empty_html = "<html><head></head><body></body></html>"
        file_len_record = {}
        print("strat_merge")
        tic = time.time()
        for file in self.files_to_convert:
            html_content, file_len = self.generate_html(file, header_path=True)
            file_len_record[file] = file_len
            empty_html = empty_html.replace(
                "</body></html>", html_content + "</body></html>"
            )

        tree_html = self.generate_html_from_text(self.tree)
        empty_html = empty_html.replace(
            "</body></html>", tree_html + "</body></html>"
        )
        print(f"merge time: {time.time() - tic}")

        print("strat")
        tic = time.time()
        pdfkit.from_string(
            input=empty_html,
            output_path=f"{str(self.output_dir)}/{self.name_repository}.pdf",
            options={
                "encoding": "UTF-8",
                "margin-top": "0.15in",
                "margin-right": "0.45in",
                "margin-bottom": "0.15in",
                "margin-left": "0.45in",
            },
            configuration=pdfkit.configuration(wkhtmltopdf=self.wkhtmltopdf_path),
            verbose=True,
        )
        print(f"time: {time.time()-tic}")

    def process_chunk(self, chunk, save_path=None):
        """Process a chunk of files and generate PDFs"""
        chunk_html = "<html><head></head><body></body></html>"
        file_character_len_record = []
        for file in chunk:
            if isinstance(file, Path) and file.is_file():
                html_content, content_len = self.generate_html(file, header_path=True)
                chunk_html = chunk_html.replace("</body></html>", html_content + "</body></html>")
                file_character_len_record.append({'size': content_len, 'path': file})
            elif file == "repository_tree" and isinstance(file, str):
                tree_html = self.generate_html_from_text(self.tree)
                chunk_html = chunk_html.replace("</body></html>", tree_html + "</body></html>")

        pdf_path = f"{self.output_dir}/{os.getpid()}-{time.time()}.pdf" if save_path is None else save_path
        pdfkit.from_string(
            input=chunk_html,
            output_path=pdf_path,
            options={
                "encoding": "UTF-8",
                "margin-top": "0.15in",
                "margin-right": "0.45in",
                "margin-bottom": "0.15in",
                "margin-left": "0.45in",
            },
            configuration=pdfkit.configuration(wkhtmltopdf=self.wkhtmltopdf_path),
            verbose=False,
        )
        return pdf_path, file_character_len_record

    def merge_pdfs(self, pdf_paths, output_path):
        """Merge multiple PDF files into one"""
        # 创建一个空的 PDF writer 对象
        merged_pdf_writer = PyPDF2.PdfWriter()

        # 逐个读取每个 PDF 文件，并将其添加到 merged_pdf_writer 中
        for pdf_path in pdf_paths:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    merged_pdf_writer.add_page(page)

        # 将合并后的 PDF 保存到输出路径
        with open(output_path, 'wb') as merged_pdf_file:
            merged_pdf_writer.write(merged_pdf_file)

    @staticmethod
    def split_list(lst, num_chunks):
        avg_chunk_size = len(lst) // num_chunks
        remainder = len(lst) % num_chunks

        chunks = []
        start = 0
        for i in range(num_chunks):
            chunk_size = avg_chunk_size + 1 if i < remainder else avg_chunk_size
            # chunks_slice.append(slice(start, start + chunk_size, 1))
            chunks.append(lst[start:start + chunk_size])
            start += chunk_size

        return chunks


def error(o):
    print(o)


def callback_function(pbar, o):
    if pbar is not None:
        pbar.update()


def parallel_process(c: RepoToPDF,
                     num_processes: int = 8):
    tic = time.time()
    chunks = c.split_list(lst=c.files_to_convert, num_chunks=min(num_processes, len(c.files_to_convert)))
    chunks[-1].append("repository_tree")

    # 使用进程池并行处理子任务
    pbar = None # tqdm(total=len(chunks), desc="Processing chunks")
    pdf_paths = []
    pool = multiprocessing.Pool(num_processes)
    results = []
    for i, chunk in enumerate(chunks):
        save_path = f"{c.output_dir}/{c.name_repository}_{i}.pdf"
        results.append(pool.apply_async(c.process_chunk, args=(chunk, save_path), error_callback=error,
                       callback=lambda o: callback_function(pbar, o)))
        pdf_paths.append(save_path)

    pool.close()
    pool.join()
    # 将生成的 PDF 文件合并成一个整体 PDF
    c.merge_pdfs(pdf_paths, str(c.output_path))
    file_content_record = []
    for r in results:
        file_content_record += r.get()[1]
    toc = time.time()
    print(f"Total time: {toc - tic} seconds")
    # delete every pdf in pdf_paths
    file_tree_dict = create_file_tree_for_pyqt(file_content_record)
    with open(c.output_tree_path, "w") as file:
        json.dump(file_tree_dict, file, indent=4)

    for pdf_path in pdf_paths:
        os.remove(pdf_path)


def create_file_tree_for_pyqt(file_list_with_size):
    folder_structure = {}

    for file_info in file_list_with_size:
        path = file_info['path']
        size = file_info['size']
        components = Path(path).parts

        current_level = folder_structure
        for component in components[:-1]:
            if component not in current_level:
                current_level[component] = {'size': None, 'sub': {}}
            current_level = current_level[component]['sub']

        filename = components[-1]
        current_level[filename] = {'size': size, 'sub': None}

        # 更新父文件夹的大小
        update_parent_folder_size(folder_structure, components[:-1], size)

    return delete_part_in_path(folder_structure)


def delete_flag(d):
    if d.get("sub") and len(d["sub"]) == 1 and [d["sub"][k] for k in d["sub"].keys()][0]["sub"] is not None:
        return True
    else:
        return False


def delete_part_in_path(folder_structure):
    flag = True
    while folder_structure and len(folder_structure) == 1 and flag:
        for _, items in folder_structure.items():
            if delete_flag(items):
                folder_structure = items["sub"]
            else:
                flag = False
            break

    return folder_structure


def update_parent_folder_size(folder_structure, path_components, size):
    current_level = folder_structure
    for component in path_components:
        if current_level[component]['size'] is None:
            current_level[component]['size'] = 0
        current_level[component]['size'] += size
        current_level = current_level[component]['sub']


class DisplayablePath(object):
    display_filename_prefix_middle = "├──"
    display_filename_prefix_last = "└──"
    display_parent_prefix_middle = "    "
    display_parent_prefix_last = "│   "

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + "/"
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(
            list(path for path in root.iterdir() if criteria(path)),
            key=lambda s: str(s).lower(),
        )
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(
                    path,
                    parent=displayable_root,
                    is_last=is_last,
                    criteria=criteria,
                )
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + "/"
        return self.path.name

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (
            self.display_filename_prefix_last
            if self.is_last
            else self.display_filename_prefix_middle
        )

        parts = ["{!s} {!s}".format(_filename_prefix, self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(
                self.display_parent_prefix_middle
                if parent.is_last
                else self.display_parent_prefix_last
            )
            parent = parent.parent

        return "".join(reversed(parts))


# if __name__ == "__main__":
#     import argparse
#     from repo2pdf_visualization import show_file_structure
#
#     parser = argparse.ArgumentParser(
#         prog="Repository to PDF",
#         description="This program convert a repository to PDF",
#         epilog="Enjoy the program! :)",
#     )
#     parser.add_argument("--dir", default="H:\LLM_project\Metagpt\MetaGPT-main", type=Path, help="Path of the repository.")
#     parser.add_argument(
#         "--style",
#         default="colorful",
#         type=str,
#         help="Style for the PDF. Choose a style -> https://pygments.org/styles/",
#     )
#     parser.add_argument(
#         "--ignore",
#         type=str,
#         nargs='?',
#         help="Add files and/or folders for ignoring.",
#     )
#     parser.add_argument(
#         "--wkhtmltopdf",
#         type=str,
#         default="C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe",
#         help="your wkhtmltopdf.exe path, pdfkit needs it."
#     )
#     parser.add_argument(
#         "--overwrite",
#         type=bool,
#         default=True,
#         help="overwrite the pdf file if it exists."
#     )
#     parser.add_argument(
#         "--num_multiprocess",
#         type=int,
#         default=multiprocessing.cpu_count(),
#         help="Number of processes to use. Default is the number of CPUs in the system."
#     )
#     parser.add_argument(
#         "--output_name",
#         type=int,
#         default=None,
#         help="Name of the output file. Default is the name of the repository."
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=int,
#         default=None,
#         help="Name of the output dir, Default is in the same dictionary of the repository."
#     )
#
#     # Creating a Namespace object
#     args = parser.parse_args()
#     try:
#         if args.style:
#             choices = [
#                 "default", "bw",
#                 "sas", "xcode",
#                 "autumn", "borland",
#                 "arduino", "igor",
#                 "lovelace", "pastie",
#                 "rainbow_dash",
#                 "emacs", "tango",
#                 "colorful",
#                 "rrt", "algol",
#                 "abap",
#             ]
#             if not args.style in choices:
#                 raise NameError
#         if Path(args.dir).exists():
#             repo = RepoToPDF(directory=args.dir, wkhtmltopdf_path=args.wkhtmltopdf,
#                              style=args.style, multi_processing=True, output_dir=args.output_dir,
#                              output_name=args.output_name)
#             # repo.generate_pdf()
#             if not Path(f"{repo.output_path}").exists() or args.overwrite:
#                 parallel_process(repo, num_processes=args.num_multiprocess)
#
#             with open(f"{repo.output_tree_path}", "r") as file:
#                 file_tree_dict = json.load(file)
#             show_file_structure(file_tree_dict)
#
#         else:
#             raise FileNotFoundError
#     except FileNotFoundError:
#         print("Invalid Folder !")
#     except NameError:
#         print(
#             f"Invalid Style, please choose one of nexts :\n{', '.join(choices)}"
#         )
