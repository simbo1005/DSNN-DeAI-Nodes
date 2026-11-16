"""Optional DSNN De-AI runtime acceleration; adds no workflow node IDs."""
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

from .performance import register
register()
