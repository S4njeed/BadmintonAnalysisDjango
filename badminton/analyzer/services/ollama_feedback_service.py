import ollama
import os
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

# Configuration
OLLAMA_MODEL_ID = "gemma3:4b"  # Gemma 3 4b
CONTEXT_ENABLED = False  # Set to True if you want to use the context file
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30

@dataclass
class FeedbackMetrics:
    """Structured feedback metrics for better analysis"""
    grip_angle_current: Optional[float] = None
    grip_angle_ideal: Optional[float] = None
    grip_angle_difference: Optional[float] = None
    shoulder_angle_current: Optional[float] = None
    shoulder_angle_ideal: Optional[float] = None
    shoulder_angle_difference: Optional[float] = None
    timing_preparation: Optional[float] = None
    timing_follow_through: Optional[float] = None
    shot_type: Optional[str] = None
    critical_phase: Optional[str] = None

@dataclass
class StructuredFeedback:
    """Structured feedback output with categorized information"""
    overall_assessment: str
    technique_analysis: Dict[str, str]
    improvement_priorities: List[str]
    specific_recommendations: List[str]
    target_metrics: Dict[str, float]
    confidence_score: float

class FeedbackGenerator:
    """Enhanced feedback generator with structured output parsing"""
    
    def __init__(self):
        self.shot_type_priorities = {
            'smash': ['grip_angle', 'shoulder_angle', 'timing'],
            'clear': ['shoulder_angle', 'grip_angle', 'timing'],
            'drop_shot': ['grip_angle', 'timing', 'shoulder_angle']
        }
        
        self.angle_tolerances = {
            'smash': {'grip': 12, 'shoulder': 12},
            'clear': {'grip': 8, 'shoulder': 12},
            'drop_shot': {'grip': 8, 'shoulder': 12}
        }

    def read_context_file(self) -> str:
        """Read context file with improved error handling"""
        context_file_path = os.path.join(settings.BASE_DIR, '..', 'context.txt')
        try:
            with open(context_file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                logger.info(f"Successfully loaded context file: {len(content)} characters")
                return content
        except FileNotFoundError:
            logger.warning(f"Context file not found at {context_file_path}")
            return ""
        except Exception as e:
            logger.error(f"Error reading context file: {e}")
            return ""

    def extract_metrics(self, analysis_summary: Dict) -> FeedbackMetrics:
        """Extract and structure key metrics from analysis summary"""
        metrics = FeedbackMetrics()
        
        try:
            # Extract shot type
            metrics.shot_type = analysis_summary.get('shot_type', 'unknown')
            
            # Extract angles with safe defaults
            angles = analysis_summary.get('angles', {})
            ideal_angles = analysis_summary.get('ideal_angles', {})
            angle_differences = analysis_summary.get('angle_differences', {})
            
            # Focus on pre-hit phase as it's most critical
            metrics.grip_angle_current = angles.get('pre_hit')
            metrics.grip_angle_ideal = ideal_angles.get('pre_hit')
            metrics.grip_angle_difference = angle_differences.get('pre_hit')
            
            # Also check hit phase if pre-hit is missing
            if metrics.grip_angle_current is None:
                metrics.grip_angle_current = angles.get('hit')
                metrics.grip_angle_ideal = ideal_angles.get('hit')
                metrics.grip_angle_difference = angle_differences.get('hit')
                metrics.critical_phase = 'hit'
            else:
                metrics.critical_phase = 'pre_hit'
            
            # Extract timing data
            timing = analysis_summary.get('timing', {})
            metrics.timing_preparation = timing.get('preparation_time')
            metrics.timing_follow_through = timing.get('follow_through_time')
            
            logger.debug(f"Extracted metrics for {metrics.shot_type}: grip={metrics.grip_angle_current}°, ideal={metrics.grip_angle_ideal}°")
            
        except Exception as e:
            logger.error(f"Error extracting metrics: {e}")
            
        return metrics

    def generate_structured_prompt(self, metrics: FeedbackMetrics, context: str) -> Tuple[str, str]:
        """Generate structured prompts for better AI response"""
        
        # Determine focus areas based on shot type and metrics
        focus_areas = self._determine_focus_areas(metrics)
        shot_type = metrics.shot_type or 'unknown'
        
        # Very concise system prompt to reduce response length
        system_prompt = f"""You are a badminton coach. Analyze {shot_type} technique and provide brief feedback.

REQUIREMENTS:
- Keep response under 150 words total
- Use plain text only (no formatting)
- Be specific about angles and timing
- Focus on {', '.join(focus_areas[:2])}

Format:
Overall assessment (1-2 sentences)
Key improvements (2 items max)
Practice tips (2 items max)"""

        # Minimal user prompt
        angle_info = ""
        if metrics.grip_angle_current and metrics.grip_angle_ideal:
            angle_info = f"Grip angle: {metrics.grip_angle_current:.1f}° (should be {metrics.grip_angle_ideal:.1f}°). "
        
        timing_info = ""
        if metrics.timing_preparation:
            timing_info = f"Prep time: {metrics.timing_preparation:.2f}s. "
        
        user_prompt = f"Analyze {shot_type}: {angle_info}{timing_info}Provide concise coaching advice."

        return system_prompt, user_prompt

    def _determine_focus_areas(self, metrics: FeedbackMetrics) -> List[str]:
        """Determine focus areas based on metrics and shot type"""
        focus_areas = []
        shot_type = metrics.shot_type or 'smash'  # Default to smash if None
        tolerance = self.angle_tolerances.get(shot_type, {'grip': 10, 'shoulder': 10})
        
        # Check grip angle deviation
        if (metrics.grip_angle_difference is not None and 
            abs(metrics.grip_angle_difference) > tolerance['grip']):
            focus_areas.append('grip_technique')
        
        # Check timing issues
        if metrics.timing_preparation is not None:
            if metrics.timing_preparation < 0.3:
                focus_areas.append('preparation_timing')
            elif metrics.timing_preparation > 1.0:
                focus_areas.append('shot_efficiency')
        
        # Add shot-specific focus
        if not focus_areas:
            focus_areas.extend(self.shot_type_priorities.get(shot_type, ['general_technique']))
        
        return focus_areas[:3]  # Limit to top 3 focus areas

    def parse_ai_response(self, response: str, metrics: FeedbackMetrics) -> StructuredFeedback:
        """Parse AI response and combine with structured data for better output"""
        # Clean up the response
        response = response.strip()
        
        # Initialize structured feedback
        feedback = StructuredFeedback(
            overall_assessment="",
            technique_analysis={},
            improvement_priorities=[],
            specific_recommendations=[],
            target_metrics={},
            confidence_score=0.8
        )
        
        # Use the AI response directly as overall assessment (first paragraph)
        paragraphs = response.split('\n\n')
        if paragraphs:
            feedback.overall_assessment = paragraphs[0].strip()
        
        # If overall assessment is too short, try first few sentences
        if len(feedback.overall_assessment) < 50:
            sentences = response.split('. ')
            if len(sentences) >= 2:
                feedback.overall_assessment = '. '.join(sentences[:2]).strip()
                if not feedback.overall_assessment.endswith('.'):
                    feedback.overall_assessment += '.'
        
        # Generate improvements based on metrics with specific numbers
        if metrics.grip_angle_difference and abs(metrics.grip_angle_difference) > 3:
            feedback.improvement_priorities.append(
                f"Adjust grip angle from {metrics.grip_angle_current:.1f}° to {metrics.grip_angle_ideal:.1f}°"
            )
        
        if metrics.timing_preparation and metrics.timing_preparation > 0.8:
            feedback.improvement_priorities.append(
                f"Reduce preparation time from {metrics.timing_preparation:.2f}s to 0.6-0.8s"
            )
        
        # Add default improvement if none found
        if not feedback.improvement_priorities:
            if metrics.grip_angle_current and metrics.grip_angle_ideal:
                feedback.improvement_priorities.append(
                    f"Maintain grip angle near {metrics.grip_angle_ideal:.1f}°"
                )
            else:
                feedback.improvement_priorities.append("Continue practicing current technique")
        
        # Generate practice recommendations
        feedback.specific_recommendations = [
            "Practice shadow swings focusing on grip positioning",
            "Work on smooth, efficient preparation movements"
        ]
        
        # Add target metrics - ensure we always have meaningful targets
        if metrics.grip_angle_ideal:
            feedback.target_metrics['target_grip_angle'] = metrics.grip_angle_ideal
        
        # Add timing targets
        if metrics.timing_preparation and metrics.timing_preparation > 0.5:
            feedback.target_metrics['target_preparation_time'] = 0.7  # Target 0.7s preparation
        
        # Add shot type specific targets
        shot_type = metrics.shot_type or 'smash'
        if shot_type == 'smash':
            if 'target_grip_angle' not in feedback.target_metrics:
                feedback.target_metrics['target_grip_angle'] = 15.0  # Default smash grip angle
            feedback.target_metrics['target_power_transfer'] = 85.0  # Target % power transfer
        elif shot_type == 'clear':
            if 'target_grip_angle' not in feedback.target_metrics:
                feedback.target_metrics['target_grip_angle'] = 20.0  # Default clear grip angle
            feedback.target_metrics['target_arc_height'] = 4.5  # Target arc height in meters
        elif shot_type == 'drop':
            if 'target_grip_angle' not in feedback.target_metrics:
                feedback.target_metrics['target_grip_angle'] = 12.0  # Default drop grip angle
            feedback.target_metrics['target_control_accuracy'] = 90.0  # Target % accuracy
        
        # Ensure we always have at least one target metric
        if not feedback.target_metrics:
            feedback.target_metrics['target_consistency'] = 80.0  # Target % consistency
        
        return feedback

    def format_output(self, structured_feedback: StructuredFeedback) -> str:
        """Format structured feedback into clean, simple text output - always complete"""
        output_parts = []
        
        # Overall Assessment - ensure we always have this
        output_parts.append("OVERALL ASSESSMENT:")
        if structured_feedback.overall_assessment:
            output_parts.append(structured_feedback.overall_assessment)
        else:
            output_parts.append("Your technique shows areas for improvement that can enhance performance.")
        output_parts.append("")  # Empty line for spacing
        
        # Key Improvement Areas - ensure we always have this
        output_parts.append("KEY IMPROVEMENT AREAS:")
        if structured_feedback.improvement_priorities:
            for i, priority in enumerate(structured_feedback.improvement_priorities[:3], 1):
                # Clean up any existing numbering
                clean_priority = priority.strip()
                if clean_priority.startswith(('1.', '2.', '3.', '-')):
                    clean_priority = clean_priority[2:].strip()
                output_parts.append(f"{i}. {clean_priority}")
        else:
            output_parts.append("1. Continue practicing to maintain and improve current form")
        output_parts.append("")
        
        # Practice Recommendations - ensure we always have this
        output_parts.append("PRACTICE RECOMMENDATIONS:")
        if structured_feedback.specific_recommendations:
            for i, rec in enumerate(structured_feedback.specific_recommendations[:3], 1):
                # Clean up any existing formatting
                clean_rec = rec.strip()
                if clean_rec.startswith(('-', '•')):
                    clean_rec = clean_rec[1:].strip()
                output_parts.append(f"{i}. {clean_rec}")
        else:
            output_parts.append("1. Practice shadow swings to improve muscle memory")
            output_parts.append("2. Focus on consistent form during training")
        output_parts.append("")
        
        # Target Metrics - ensure we always complete this section
        output_parts.append("TARGET METRICS:")
        if structured_feedback.target_metrics:
            for metric, value in structured_feedback.target_metrics.items():
                metric_name = metric.replace('target_', '').replace('_', ' ').title()
                
                if isinstance(value, float):
                    # Format based on metric type
                    if 'angle' in metric.lower():
                        output_parts.append(f"- {metric_name}: {value:.1f}°")
                    elif 'time' in metric.lower():
                        output_parts.append(f"- {metric_name}: {value:.2f}s")
                    elif 'height' in metric.lower():
                        output_parts.append(f"- {metric_name}: {value:.1f}m")
                    elif any(word in metric.lower() for word in ['power', 'accuracy', 'consistency', 'transfer']):
                        output_parts.append(f"- {metric_name}: {value:.0f}%")
                    else:
                        output_parts.append(f"- {metric_name}: {value:.1f}")
                else:
                    output_parts.append(f"- {metric_name}: {value}")
        else:
            # Fallback targets - this should never happen now
            output_parts.append("- Focus on maintaining consistent form and timing")
        
        return "\n".join(output_parts)

def generate_feedback(analysis_summary: dict) -> str:
    """Enhanced feedback generation with structured output and fallback"""
    if not analysis_summary or analysis_summary.get('status') != 'complete':
        return "Feedback cannot be generated: Analysis is incomplete or missing."
    
    generator = FeedbackGenerator()
    
    try:
        # Extract structured metrics
        metrics = generator.extract_metrics(analysis_summary)
        
        # Read context if enabled
        context = generator.read_context_file() if CONTEXT_ENABLED else ""
        
        # Generate structured prompts
        system_prompt, user_prompt = generator.generate_structured_prompt(metrics, context)
        
        # Call Ollama API with retries
        response_content = None
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Attempting to generate feedback (attempt {attempt + 1}/{MAX_RETRIES})")
                
                response = ollama.chat(
                    model=OLLAMA_MODEL_ID,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ]
                )
                
                if response and 'message' in response and 'content' in response['message']:
                    response_content = response['message']['content'].strip()
                    # Check if response seems complete (not cut off)
                    # Look for common cutoff indicators
                    is_complete = (
                        len(response_content) > 100 and  # Minimum reasonable length
                        not response_content.endswith((':', 'TARGET', 'METRICS', 'RECOMMEND')) and
                        not response_content.rstrip().endswith(('TARGET METRICS:', 'PRACTICE RECOMMENDATIONS:'))
                    )
                    
                    if is_complete:
                        logger.info(f"Received complete response: {len(response_content)} chars")
                        break
                    else:
                        logger.warning(f"Response appears incomplete on attempt {attempt + 1}: {len(response_content)} chars, ends with: '{response_content[-50:]}'")
                else:
                    logger.warning(f"Empty response on attempt {attempt + 1}")
                    
            except ollama.ResponseError as e:
                logger.error(f"Ollama API error on attempt {attempt + 1}: {e.error}")
                if attempt == MAX_RETRIES - 1:
                    # Use fallback if all attempts fail
                    return _generate_fallback_feedback(metrics)
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == MAX_RETRIES - 1:
                    return _generate_fallback_feedback(metrics)
        
        if not response_content or len(response_content) < 100:
            logger.warning("No valid response received, using fallback")
            return _generate_fallback_feedback(metrics)
        
        # Post-process response to handle incomplete responses
        response_content = _fix_incomplete_response(response_content, metrics)
        
        # Parse and structure the response
        structured_feedback = generator.parse_ai_response(response_content, metrics)
        
        # Format final output
        formatted_output = generator.format_output(structured_feedback)
        
        # Ensure output is complete and not empty
        if not formatted_output.strip() or not _is_output_complete(formatted_output):
            logger.warning("Incomplete formatted output, using fallback")
            return _generate_fallback_feedback(metrics)
        
        logger.info("Successfully generated structured feedback")
        return formatted_output
        
    except Exception as e:
        logger.error(f"Error in feedback generation pipeline: {e}")
        # Try to generate fallback feedback
        try:
            metrics = generator.extract_metrics(analysis_summary)
            return _generate_fallback_feedback(metrics)
        except:
            return f"Feedback generation failed due to an unexpected error: {e}"

def _generate_fallback_feedback(metrics: FeedbackMetrics) -> str:
    """Generate simple fallback feedback when AI fails"""
    output_parts = []
    
    output_parts.append("OVERALL ASSESSMENT:")
    shot_type = metrics.shot_type or "shot"
    
    if metrics.grip_angle_difference and abs(metrics.grip_angle_difference) > 5:
        assessment = f"Your {shot_type} technique shows room for improvement, particularly in grip positioning."
    elif metrics.timing_preparation and metrics.timing_preparation > 0.8:
        assessment = f"Your {shot_type} preparation timing needs to be more efficient for better shot execution."
    else:
        assessment = f"Your {shot_type} technique is generally good with minor adjustments needed."
    
    output_parts.append(assessment)
    output_parts.append("")
    
    output_parts.append("KEY IMPROVEMENT AREAS:")
    improvement_count = 1
    
    if metrics.grip_angle_difference and abs(metrics.grip_angle_difference) > 5:
        output_parts.append(f"{improvement_count}. Adjust grip angle from {metrics.grip_angle_current:.1f}° to {metrics.grip_angle_ideal:.1f}°")
        improvement_count += 1
    
    if metrics.timing_preparation and metrics.timing_preparation > 0.8:
        output_parts.append(f"{improvement_count}. Reduce preparation time from {metrics.timing_preparation:.2f}s to 0.6-0.8s")
        improvement_count += 1
    
    if improvement_count == 1:
        output_parts.append("1. Continue practicing to maintain current good form")
    
    output_parts.append("")
    
    output_parts.append("PRACTICE RECOMMENDATIONS:")
    output_parts.append("1. Practice shadow swings focusing on grip positioning")
    output_parts.append("2. Work on smoother, faster preparation movements")
    output_parts.append("")
    
    # Always include target metrics in fallback (this section is now fixed)
    output_parts.append("TARGET METRICS:")
    
    # Add grip angle target if available
    if metrics.grip_angle_ideal:
        output_parts.append(f"- Grip Angle: {metrics.grip_angle_ideal:.1f}°")
    else:
        # Default targets based on shot type
        shot_type = metrics.shot_type or 'smash'
        if shot_type == 'smash':
            output_parts.append("- Grip Angle: 15.0°")
            output_parts.append("- Power Transfer: 85%")
        elif shot_type == 'clear':
            output_parts.append("- Grip Angle: 20.0°")
            output_parts.append("- Arc Height: 4.5m")
        elif shot_type == 'drop':
            output_parts.append("- Grip Angle: 12.0°")
            output_parts.append("- Control Accuracy: 90%")
        else:
            output_parts.append("- Consistency: 80%")
    
    # Add timing target if we have timing issues
    if metrics.timing_preparation and metrics.timing_preparation > 0.8:
        output_parts.append("- Preparation Time: 0.7s")
    
    return "\n".join(output_parts)

def _fix_incomplete_response(response: str, metrics: FeedbackMetrics) -> str:
    """Fix incomplete AI responses by adding missing sections"""
    # Check if response ends abruptly
    response = response.strip()
    
    # If response ends with incomplete section headers, truncate to last complete section
    incomplete_endings = [
        'TARGET METRICS:',
        'PRACTICE RECOMMENDATIONS:',
        'KEY IMPROVEMENT AREAS:',
        'SPECIFIC RECOMMENDATIONS:',
        'TARGET',
        'PRACTICE',
        'KEY'
    ]
    
    for ending in incomplete_endings:
        if response.endswith(ending):
            # Find the last complete section before this
            lines = response.split('\n')
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()
                if line and not any(end in line for end in incomplete_endings):
                    # Keep everything up to this line
                    response = '\n'.join(lines[:i+1])
                    break
            break
    
    return response

def _is_output_complete(output: str) -> bool:
    """Check if the formatted output appears complete"""
    # Check for empty sections that might indicate incomplete processing
    lines = output.split('\n')
    
    # Look for section headers followed immediately by another section header (indicating empty section)
    section_headers = ['OVERALL ASSESSMENT:', 'KEY IMPROVEMENT AREAS:', 'PRACTICE RECOMMENDATIONS:', 'TARGET METRICS:']
    
    for i, line in enumerate(lines):
        if line.strip() in section_headers:
            # Check if this section has content
            has_content = False
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                if not next_line:  # Empty line
                    continue
                elif next_line in section_headers:  # Next section found
                    break
                else:  # Found content
                    has_content = True
                    break
            
            # If OVERALL ASSESSMENT or KEY IMPROVEMENT AREAS is empty, consider incomplete
            if not has_content and line.strip() in ['OVERALL ASSESSMENT:', 'KEY IMPROVEMENT AREAS:']:
                return False
    
    # Basic completeness check - should have at least assessment and one improvement
    return ('OVERALL ASSESSMENT:' in output and 
            ('KEY IMPROVEMENT AREAS:' in output or 'PRACTICE RECOMMENDATIONS:' in output))
