from .main import ArcheologiaPipelinePlugin

def classFactory(iface):
    return ArcheologiaPipelinePlugin(iface)