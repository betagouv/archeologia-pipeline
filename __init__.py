try:
    from .main import ArcheologiaPipelinePlugin

    def classFactory(iface):
        return ArcheologiaPipelinePlugin(iface)
except ImportError:
    pass