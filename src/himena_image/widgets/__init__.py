from himena_image.widgets.viewer import HimenaImageViewer
from himena.qt import register_widget
from himena.consts import StandardType

register_widget(StandardType.IMAGE, HimenaImageViewer)
