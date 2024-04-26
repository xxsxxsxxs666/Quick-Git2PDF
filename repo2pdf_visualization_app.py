import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTreeView, QVBoxLayout, QWidget, QLabel, \
    QSpinBox, QStyledItemDelegate
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QColor
from PyQt5.QtCore import QEvent


class ColorDelegate(QStyledItemDelegate):
    def __init__(self, parent=None,
                 model: QStandardItemModel = None,
                 min_size=None, max_size=None):
        super().__init__(parent)
        self.model = model
        self.max_size = max_size
        self.min_size = min_size

    def paint(self, painter, option, index):
        if index.column() == 1:
            first_column_index = index.sibling(index.row(), 0)
            item = self.model.itemFromIndex(first_column_index)  # 获取项
            # if the item is a folder, paint it blue
            if item.hasChildren():
                color = QColor(255, 255, 255)
            else:
                if self.max_size:
                    alpha = (float(index.data())-self.min_size) / (self.max_size - self.min_size + 1e-6) * 0.7
                    color = QColor(255, 22, 22)
                    color.setAlphaF(alpha)
                else:
                    color = QColor(173, 216, 230)
        else:
            color = QColor(255, 255, 255)

        painter.save()
        painter.fillRect(option.rect, color)
        super().paint(painter, option, index)
        painter.restore()


class TreeView(QMainWindow):
    def __init__(self, data, color=True):
        super().__init__()
        self.setWindowTitle("File Sturecture Viewer")
        self.setGeometry(100, 100, 600, 400)  # 设置窗口的初始大小
        max_levels = 10
        min_levels = 1
        self.data = data

        max_size, min_size = self.min_max_calculator(self.data)

        # 创建 QStandardItemModel
        self.model = QStandardItemModel()
        self.header_labels = ["Name", "Character Count"]
        self.model.setHorizontalHeaderLabels(self.header_labels)
        self.add_data_to_model(self.data, max_levels)

        # 创建 QTreeView
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.model)

        # 设置委托, 按照文件大小可视化，从而方便筛选
        self.delegate = ColorDelegate(model=self.model, max_size=max_size, min_size=min_size)
        self.tree_view.setItemDelegate(self.delegate)

        # 创建控制组件
        self.levels_label = QLabel("Levels:")
        self.levels_spinbox = QSpinBox()
        self.levels_spinbox.setMinimum(1)
        self.levels_spinbox.setMaximum(10)  # 设置最大级数
        self.levels_spinbox.setValue(3)  # 默认显示3级
        self.levels_spinbox.valueChanged.connect(self.change_level)

        # 创建布局
        layout = QVBoxLayout()
        layout.addWidget(self.levels_label)
        layout.addWidget(self.levels_spinbox)
        layout.addWidget(self.tree_view)

        # 创建主窗口的中心部件
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.update_tree_view()

        # 设置窗口大小变化时更新列宽度
        self.tree_view.viewport().installEventFilter(self)

    def eventFilter(self, source, event):
        if source == self.tree_view.viewport() and event.type() == QEvent.Resize:
            self.update_column_widths()
        return super().eventFilter(source, event)

    def update_column_widths(self):
        # 获取窗口的宽度
        window_width = self.tree_view.viewport().size().width()

        # 设置每列的宽度为窗口宽度的一半
        column_width = window_width // self.model.columnCount()
        for column in range(self.model.columnCount()):
            self.tree_view.setColumnWidth(column, column_width)

    def change_level(self):
        self.update_tree_view()

    def update_tree_view(self, parent_item=None, current_level=0):
        if parent_item is None:
            parent_item = self.model.invisibleRootItem()

        num_rows = parent_item.rowCount()

        for row in range(num_rows):
            child_item = parent_item.child(row, 0)
            self.tree_view.setExpanded(self.model.indexFromItem(parent_item),
                                       current_level < self.levels_spinbox.value())
            if child_item is not None:
                # 递归遍历子项
                self.update_tree_view(parent_item=child_item, current_level=current_level + 1)

    def add_data_to_model(self, data, max_levels, parent=None, current_level=0):
        if parent is None:
            parent = self.model.invisibleRootItem()

        for key, value in data.items():
            name_item = QStandardItem(key)
            count_item = QStandardItem(str(value["size"]))
            parent.appendRow([name_item, count_item])

            if isinstance(value["sub"], dict):
                self.add_data_to_model(value["sub"], max_levels, name_item, current_level=current_level + 1)

    def collapse_items(self, parent_index, max_levels, current_level=0):
        if current_level >= max_levels:
            return

        for row in range(self.model.rowCount(parent_index)):
            index = self.model.index(row, 0, parent_index)
            self.tree_view.setExpanded(index, False)
            self.collapse_items(index, max_levels, current_level=current_level + 1)

    def min_max_calculator(self, file_structure, max_size=0, min_size=0):
        if file_structure is None:
            return max_size, min_size

        for key, value in file_structure.items():
            if value["sub"] is None:  # 如果是文件
                size = value["size"]
                min_size = min(min_size, size)
                max_size = max(max_size, size)
            else:  # 如果是文件夹
                max_size, min_size = self.min_max_calculator(value["sub"], max_size=max_size, min_size=min_size)
        return max_size, min_size

    def red_gradient(self, value):
        value = (value - self.max_size)/(self.max_size - self.min_size + 1e-7)
        red = int((200-50) * value) + 50
        green, blue = 255, 255
        return red, green, blue


def show_file_structure(data):
    app = QApplication(sys.argv)
    window = TreeView(data)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    # 示例数据
    data = {
        "Meta_gpt": {
            "size": 1050,
            "sub": {
                "Folder1": {
                    "size": 700,
                    "sub": {
                        "File1": {"size": 350, "sub": None},
                        "Folder2": {
                            "size": 350,
                            "sub": {
                                "File2": {"size": 200, "sub": None},
                                "File3": {"size": 150, "sub": None},
                            }
                        }
                    }
                },
                "File4": {"size": 350, "sub": None}
            }
        }
    }
    show_file_structure(data=data)