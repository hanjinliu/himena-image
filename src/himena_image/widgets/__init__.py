from himena_image.widgets.viewer import HimenaImageViewer
from himena.plugins import register_widget_class
from himena.consts import StandardType

register_widget_class(StandardType.IMAGE, HimenaImageViewer)
