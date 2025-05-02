"""Forms for the Analyzer app."""

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.utils.translation import gettext_lazy as _
from .models import VideoAnalysis


class VideoUploadForm(forms.ModelForm):
    """Form for uploading and configuring video analysis."""
    
    class Meta:
        model = VideoAnalysis
        fields = ['input_video', 'shot_type', 'process_interval', 'source_fps']
        widgets = {
            'process_interval': forms.NumberInput(attrs={
                'min': 1,
                'max': 10,
                'step': 1,
            }),
            'source_fps': forms.NumberInput(attrs={
                'step': 0.1,
                'placeholder': 'Auto-detect',
            }),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add Bootstrap classes
        for field_name, field in self.fields.items():
            field.widget.attrs['class'] = 'form-control'
            
        # Customize file input
        self.fields['input_video'].widget.attrs.update({
            'class': 'form-control-file',
            'accept': 'video/*',
        })
        
        # Shot type as radio buttons
        self.fields['shot_type'].widget = forms.RadioSelect(
            choices=VideoAnalysis.SHOT_TYPES
        )
        self.fields['shot_type'].widget.attrs['class'] = 'form-check-input'


class SignUpForm(UserCreationForm):
    """Form for user registration."""
    email = forms.EmailField(max_length=254, help_text=_('Required. Inform a valid email address.'))

    class Meta(UserCreationForm.Meta):
        model = User
        fields = UserCreationForm.Meta.fields + ('email',)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add Bootstrap classes
        for field_name, field in self.fields.items():
            field.widget.attrs['class'] = 'form-control'
