import ollama
import json
import os
from django.conf import settings

# Define the model ID directly
OLLAMA_MODEL_ID = "gemma3:4b" # Gemma 3 4b

def read_context_file():
    # Reads the context file from one directory up from the current base directory
    context_file_path = os.path.join(settings.BASE_DIR, '..', 'context.txt')
    try:
        with open(context_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: context.txt not found at {context_file_path}")
        return ""
    except Exception as e:
        print(f"Error reading context.txt: {e}")
        return ""

def generate_feedback(analysis_summary: dict) -> str:
    if not analysis_summary or analysis_summary.get('status') != 'complete':
        return "Feedback cannot be generated: Analysis is incomplete or missing."

    try:
        # Read the base context/instructions
        base_context = read_context_file()

        # Prepare the analysis data for the prompt
        # Convert dict to a more readable string format (e.g., JSON pretty print)
        analysis_data_str = json.dumps(analysis_summary, indent=2)

        # Construct the prompt
        # Modified prompt to be more explicit about plain text output and no follow-ups
        prompt = f"""
You are an expert badminton coach AI. Analyze the following badminton shot data and provide constructive feedback.
Focus on the grip angles, particularly the pre-hit phase, comparing the user's angles to the ideal angles.
Explain the importance of the correct angles for the specific shot type ({analysis_summary.get('shot_type', 'unknown')}).
Keep the feedback concise, actionable, and encouraging.
DO NOT ASK ANY FOLLOW-UP QUESTIONS.
Provide ONLY the coaching feedback based on the analysis data.
Your response should be plain text, without any formatting (like bolding, bullet points, or lists).
Focus on the form, especially the pre-hit phase, use numbers from the data, explain improvements, and state the target numbers.

Analysis Data ON WHAT THE FEEDBACK SHOULD BE BASED OFF OF:
```json
{analysis_data_str}
```

Context for the analysis:
{base_context}

Coaching Feedback:
"""

        # Call Ollama API
        response = ollama.chat(
            model=OLLAMA_MODEL_ID,
            messages=[{'role': 'user', 'content': prompt}]
        )

        if response and 'message' in response and 'content' in response['message']:
            # Return only the content, stripping leading/trailing whitespace
            return response['message']['content'].strip()
        else:
            return "Feedback generation failed: No content received from Ollama."

    except ollama.ResponseError as e:
        print(f"Ollama API Error: {e.error}")
        return f"Feedback generation failed: Ollama API error - {e.error}"
    except Exception as e:
        print(f"Error generating feedback: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed traceback
        return f"Feedback generation failed due to an unexpected error: {e}"
