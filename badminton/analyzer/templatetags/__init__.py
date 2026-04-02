"""Custom template tags and filters for the analyzer app."""

from django import template

register = template.Library()

@register.filter
def add_floats(value, arg):
    """Add two floating-point numbers with proper precision."""
    try:
        return float(value) + float(arg)
    except (ValueError, TypeError):
        return 0.0
