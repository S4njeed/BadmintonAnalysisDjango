try:
    import ollama  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency for local test environments
    class _OllamaStub:
        class ResponseError(Exception):
            pass

        @staticmethod
        def chat(*args, **kwargs):
            raise ModuleNotFoundError("ollama is required for AI feedback generation")

    ollama = _OllamaStub()  # type: ignore[assignment]
import os
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

# Configuration
OLLAMA_MODEL_ID = "gemma4:e4b"  # Gemma 4 E4B
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
    footwork_overall_assessment: Optional[str] = None
    footwork_movement_pattern: Optional[str] = None
    footwork_pre_hit_stance_ratio: Optional[float] = None
    footwork_hit_stance_ratio: Optional[float] = None
    footwork_follow_through_stance_ratio: Optional[float] = None
    footwork_pre_hit_knee_angle: Optional[float] = None
    footwork_hit_knee_angle: Optional[float] = None
    footwork_follow_through_knee_angle: Optional[float] = None
    footwork_recovery_ratio_target: Optional[float] = None
    footwork_ready_stance_ratio_low: Optional[float] = None
    footwork_ready_stance_ratio_high: Optional[float] = None
    footwork_ideal_ready_stance_ratio: Optional[float] = None
    footwork_ideal_knee_angle: Optional[float] = None

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
            'smash': ['grip_angle', 'shoulder_angle', 'timing', 'footwork_recovery'],
            'clear': ['shoulder_angle', 'grip_angle', 'timing', 'recovery_positioning'],
            'drop_shot': ['grip_angle', 'timing', 'shoulder_angle', 'quick_lunge']
        }
        
        self.angle_tolerances = {
            'smash': {'grip': 12, 'shoulder': 12},
            'clear': {'grip': 8, 'shoulder': 12},
            'drop_shot': {'grip': 8, 'shoulder': 12}
        }

        self.footwork_priorities = {
            'smash': ['explosive split step', 'scissor kick landing', 'fast recovery'],
            'clear': ['directional split step', 'chasse and pivot', 'rear-court recovery'],
            'drop_shot': ['small split step', 'short lunge', 'controlled recovery']
        }

        self.footwork_targets = {
            'smash': {
                'ready_stance_ratio_low': 0.95,
                'ready_stance_ratio_high': 1.35,
                'ideal_ready_stance_ratio': 1.10,
                'ideal_knee_angle': 145.0,
                'recovery_ratio_target': 0.65,
            },
            'clear': {
                'ready_stance_ratio_low': 1.00,
                'ready_stance_ratio_high': 1.40,
                'ideal_ready_stance_ratio': 1.15,
                'ideal_knee_angle': 150.0,
                'recovery_ratio_target': 0.75,
            },
            'drop_shot': {
                'ready_stance_ratio_low': 0.80,
                'ready_stance_ratio_high': 1.15,
                'ideal_ready_stance_ratio': 0.95,
                'ideal_knee_angle': 135.0,
                'recovery_ratio_target': 0.85,
            },
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

            # Extract footwork summary and phase metrics
            footwork = analysis_summary.get('footwork', {}) or {}
            metrics.footwork_overall_assessment = footwork.get('overall_assessment') or footwork.get('assessment')
            metrics.footwork_movement_pattern = footwork.get('movement_pattern')

            phase_metrics = footwork.get('phase_metrics', {}) or {}
            pre_hit_footwork = phase_metrics.get('pre_hit', {}) or {}
            hit_footwork = phase_metrics.get('hit', {}) or {}
            follow_through_footwork = phase_metrics.get('follow_through', {}) or {}

            metrics.footwork_pre_hit_stance_ratio = pre_hit_footwork.get('stance_width_ratio')
            metrics.footwork_hit_stance_ratio = hit_footwork.get('stance_width_ratio')
            metrics.footwork_follow_through_stance_ratio = follow_through_footwork.get('stance_width_ratio')

            metrics.footwork_pre_hit_knee_angle = pre_hit_footwork.get('average_knee_angle')
            metrics.footwork_hit_knee_angle = hit_footwork.get('average_knee_angle')
            metrics.footwork_follow_through_knee_angle = follow_through_footwork.get('average_knee_angle')

            footwork_targets = footwork.get('targets', {}) or {}
            metrics.footwork_ready_stance_ratio_low = footwork_targets.get('ready_stance_ratio_low')
            metrics.footwork_ready_stance_ratio_high = footwork_targets.get('ready_stance_ratio_high')
            metrics.footwork_ideal_ready_stance_ratio = footwork_targets.get('ideal_ready_stance_ratio')
            metrics.footwork_ideal_knee_angle = footwork_targets.get('ideal_knee_angle')
            metrics.footwork_recovery_ratio_target = footwork_targets.get('recovery_ratio_target')
            
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
        system_prompt = f"""You are a badminton coach. Analyze {shot_type} technique and footwork together.

REQUIREMENTS:
    - Keep response under 180 words total
- Use plain text only (no formatting)
    - Be specific about angles, timing, balance, loading, landing, and recovery
    - Give advice for both shot execution and footwork/recovery
    - Focus on {', '.join(focus_areas[:3])}

Format:
Overall assessment (1-2 sentences)
    Shot technique (1 short paragraph)
    Footwork and recovery (1 short paragraph)
    Key improvements (2 items max)
    Practice tips (2 items max)"""

        # Minimal user prompt
        angle_info = ""
        if metrics.grip_angle_current and metrics.grip_angle_ideal:
            angle_info = f"Grip angle: {metrics.grip_angle_current:.1f}° (should be {metrics.grip_angle_ideal:.1f}°). "
        
        timing_info = ""
        if metrics.timing_preparation:
            timing_info = f"Prep time: {metrics.timing_preparation:.2f}s. "

        footwork_parts = []
        if metrics.footwork_overall_assessment:
            footwork_parts.append(f"Footwork summary: {metrics.footwork_overall_assessment}")
        if metrics.footwork_movement_pattern:
            footwork_parts.append(f"Movement pattern: {metrics.footwork_movement_pattern}")
        if metrics.footwork_pre_hit_stance_ratio is not None:
            footwork_parts.append(f"Pre-hit stance width: {metrics.footwork_pre_hit_stance_ratio:.2f}x shoulder width")
        if metrics.footwork_hit_knee_angle is not None:
            footwork_parts.append(f"Hit-phase knee angle: {metrics.footwork_hit_knee_angle:.1f}°")
        if metrics.footwork_follow_through_stance_ratio is not None:
            footwork_parts.append(f"Follow-through stance width: {metrics.footwork_follow_through_stance_ratio:.2f}x shoulder width")

        footwork_info = ""
        if footwork_parts:
            footwork_info = " ".join(footwork_parts) + ". "
        
        user_prompt = f"Analyze {shot_type}: {angle_info}{timing_info}{footwork_info}Provide concise coaching advice for both the shot and the footwork."

        if context:
            user_prompt = f"{user_prompt}\n\nReference context:\n{context[:2000]}"

        return system_prompt, user_prompt

    def _determine_focus_areas(self, metrics: FeedbackMetrics) -> List[str]:
        """Determine focus areas based on metrics and shot type"""
        focus_areas = []
        shot_type = metrics.shot_type or 'smash'  # Default to smash if None
        tolerance = self.angle_tolerances.get(shot_type, {'grip': 10, 'shoulder': 10})
        footwork_target = self.footwork_targets.get(shot_type, self.footwork_targets['smash'])
        
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

        if metrics.footwork_pre_hit_stance_ratio is not None:
            if metrics.footwork_pre_hit_stance_ratio < footwork_target['ready_stance_ratio_low']:
                focus_areas.append('split_step_balance')
            elif metrics.footwork_pre_hit_stance_ratio > footwork_target['ready_stance_ratio_high']:
                focus_areas.append('first_step_explosiveness')

        if metrics.footwork_hit_knee_angle is not None:
            if metrics.footwork_hit_knee_angle > footwork_target['ideal_knee_angle'] + 10:
                focus_areas.append('loading_depth')

        if metrics.footwork_follow_through_stance_ratio is not None:
            if metrics.footwork_follow_through_stance_ratio < footwork_target['ready_stance_ratio_low']:
                focus_areas.append('recovery_positioning')
            elif metrics.footwork_follow_through_stance_ratio > footwork_target['ready_stance_ratio_high'] + 0.15:
                focus_areas.append('recovery_speed')
        
        # Add shot-specific focus
        if not focus_areas:
            focus_areas.extend(self.shot_type_priorities.get(shot_type, ['general_technique']))

        # Add a footwork-oriented priority if the shot-specific logic did not surface one.
        if not any('footwork' in area or 'recovery' in area or 'split' in area for area in focus_areas):
            focus_areas.extend(self.footwork_priorities.get(shot_type, ['split step timing', 'recovery speed']))
        
        return focus_areas[:3]  # Limit to top 3 focus areas

    def _build_shot_technique_section(self, metrics: FeedbackMetrics) -> str:
        """Create a concise shot-technique summary from the extracted metrics."""
        shot_type = (metrics.shot_type or 'shot').replace('_', ' ')
        parts = []

        if metrics.grip_angle_current is not None and metrics.grip_angle_ideal is not None:
            parts.append(
                f"Your {shot_type} grip is around {metrics.grip_angle_current:.1f}°, compared with the target of {metrics.grip_angle_ideal:.1f}°."
            )
        elif metrics.grip_angle_current is not None:
            parts.append(f"Your {shot_type} grip is around {metrics.grip_angle_current:.1f}°.")

        if metrics.timing_preparation is not None:
            parts.append(f"Preparation time is {metrics.timing_preparation:.2f}s, so keep the swing compact and decisive.")

        if metrics.timing_follow_through is not None:
            parts.append(f"Follow-through timing sits at {metrics.timing_follow_through:.2f}s, which should stay controlled enough for the next movement.")

        if not parts:
            parts.append(f"The {shot_type} contact looks serviceable, but it still needs cleaner timing and a more repeatable finish.")

        return ' '.join(parts)

    def _build_footwork_section(self, metrics: FeedbackMetrics) -> str:
        """Create a concise footwork summary from the extracted metrics."""
        shot_type = metrics.shot_type or 'smash'
        footwork_target = self.footwork_targets.get(shot_type, self.footwork_targets['smash'])
        parts = []

        if metrics.footwork_overall_assessment:
            parts.append(metrics.footwork_overall_assessment)

        if metrics.footwork_movement_pattern:
            parts.append(f"Pattern: {metrics.footwork_movement_pattern}.")

        if metrics.footwork_pre_hit_stance_ratio is not None:
            stance_ratio = metrics.footwork_pre_hit_stance_ratio
            if stance_ratio < footwork_target['ready_stance_ratio_low']:
                parts.append("Your split step is a little too narrow for a clean first push-off.")
            elif stance_ratio > footwork_target['ready_stance_ratio_high']:
                parts.append("Your ready stance is too wide, which will slow the first movement.")
            else:
                parts.append("Your ready stance is in a usable range for this shot.")

        if metrics.footwork_hit_knee_angle is not None:
            knee_angle = metrics.footwork_hit_knee_angle
            if knee_angle > footwork_target['ideal_knee_angle'] + 10:
                parts.append("You are too upright at contact; lower the legs so they can drive the shot.")
            elif knee_angle < footwork_target['ideal_knee_angle'] - 15:
                parts.append("You are sitting very deep; stay low but keep the movement compact and balanced.")

        if metrics.footwork_follow_through_stance_ratio is not None:
            recovery_ratio = metrics.footwork_follow_through_stance_ratio
            if recovery_ratio < footwork_target['ready_stance_ratio_low']:
                parts.append("The recovery step is too tight; reopen the base faster after contact.")
            elif recovery_ratio > footwork_target['ready_stance_ratio_high'] + 0.15:
                parts.append("The recovery landing is still too spread out; get back to base faster.")

        if not parts:
            parts.append("The footwork needs clearer split-step timing, a stronger landing, and a faster return to base.")

        return ' '.join(parts)

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

        feedback.technique_analysis = {
            'Shot Technique': self._build_shot_technique_section(metrics),
            'Footwork': self._build_footwork_section(metrics)
        }
        
        # Generate improvements based on metrics with specific numbers
        if metrics.grip_angle_difference and abs(metrics.grip_angle_difference) > 3:
            feedback.improvement_priorities.append(
                f"Adjust grip angle from {metrics.grip_angle_current:.1f}° to {metrics.grip_angle_ideal:.1f}°"
            )
        
        if metrics.timing_preparation and metrics.timing_preparation > 0.8:
            feedback.improvement_priorities.append(
                f"Reduce preparation time from {metrics.timing_preparation:.2f}s to 0.6-0.8s"
            )

        if metrics.footwork_pre_hit_stance_ratio is not None:
            footwork_target = self.footwork_targets.get(metrics.shot_type or 'smash', self.footwork_targets['smash'])
            if metrics.footwork_pre_hit_stance_ratio < footwork_target['ready_stance_ratio_low']:
                feedback.improvement_priorities.append("Widen the split step so you can push off more explosively")
            elif metrics.footwork_pre_hit_stance_ratio > footwork_target['ready_stance_ratio_high']:
                feedback.improvement_priorities.append("Tighten the ready stance so the first step is faster")

        if metrics.footwork_hit_knee_angle is not None:
            footwork_target = self.footwork_targets.get(metrics.shot_type or 'smash', self.footwork_targets['smash'])
            if metrics.footwork_hit_knee_angle > footwork_target['ideal_knee_angle'] + 10:
                feedback.improvement_priorities.append("Sink lower into the lunge or scissor at contact")

        if metrics.footwork_follow_through_stance_ratio is not None:
            footwork_target = self.footwork_targets.get(metrics.shot_type or 'smash', self.footwork_targets['smash'])
            if metrics.footwork_follow_through_stance_ratio < footwork_target['ready_stance_ratio_low']:
                feedback.improvement_priorities.append("Recover to a neutral base sooner after the shot")
        
        # Add default improvement if none found
        if not feedback.improvement_priorities:
            if metrics.grip_angle_current and metrics.grip_angle_ideal:
                feedback.improvement_priorities.append(
                    f"Maintain grip angle near {metrics.grip_angle_ideal:.1f}°"
                )
            else:
                feedback.improvement_priorities.append("Continue practicing the shot and footwork pattern together")
        
        # Generate practice recommendations
        feedback.specific_recommendations = [
            "Practice shadow swings while matching the correct split step and first movement for the shot.",
            "Pair each contact with an immediate recovery step back to base."
        ]

        if metrics.shot_type == 'smash':
            feedback.specific_recommendations.insert(0, 'Drill scissor-kick smash entries so you land wide and balanced before recovering.')
        elif metrics.shot_type == 'clear':
            feedback.specific_recommendations.insert(0, 'Drill chasse-and-pivot rear-court movement before each clear.')
        elif metrics.shot_type == 'drop_shot':
            feedback.specific_recommendations.insert(0, 'Drill short, sharp lunge entries for the drop shot and return quickly to base.')
        
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

        if metrics.footwork_ideal_ready_stance_ratio is not None:
            feedback.target_metrics['target_ready_stance_ratio'] = metrics.footwork_ideal_ready_stance_ratio
        if metrics.footwork_ideal_knee_angle is not None:
            feedback.target_metrics['target_knee_angle'] = metrics.footwork_ideal_knee_angle
        if metrics.footwork_recovery_ratio_target is not None:
            feedback.target_metrics['target_recovery_ratio'] = metrics.footwork_recovery_ratio_target
        
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

        # Technique Analysis - always include shot and footwork guidance when available
        output_parts.append("TECHNIQUE ANALYSIS:")
        if structured_feedback.technique_analysis:
            for label, text in structured_feedback.technique_analysis.items():
                clean_text = str(text).strip()
                if clean_text:
                    output_parts.append(f"{label.upper()}:")
                    output_parts.append(clean_text)
        else:
            output_parts.append("SHOT TECHNIQUE:")
            output_parts.append("Keep the contact, timing, and racket path consistent.")
            output_parts.append("FOOTWORK:")
            output_parts.append("Stay light on your feet, split step early, and recover after contact.")
        output_parts.append("")
        
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
                    elif 'ratio' in metric.lower() or 'stance' in metric.lower():
                        output_parts.append(f"- {metric_name}: {value:.2f}x")
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
        
        # Call Ollama API with retries and format the result into structured coaching.
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
                    if response_content:
                        logger.info(f"Received Ollama response: {len(response_content)} chars")
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
        
        if not response_content:
            logger.warning("No valid response received, using fallback")
            return _generate_fallback_feedback(metrics)

        logger.info("Successfully generated feedback from Ollama")
        structured_feedback = generator.parse_ai_response(response_content, metrics)
        return generator.format_output(structured_feedback)
        
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
    shot_type = (metrics.shot_type or "shot").replace('_', ' ')
    shot_key = metrics.shot_type or 'smash'
    footwork_targets = {
        'smash': (0.95, 1.35, 1.10, 145.0, 0.65),
        'clear': (1.00, 1.40, 1.15, 150.0, 0.75),
        'drop_shot': (0.80, 1.15, 0.95, 135.0, 0.85),
    }
    ready_low, ready_high, ideal_ratio, ideal_knee, recovery_target = footwork_targets.get(shot_key, footwork_targets['smash'])

    output_parts.append("OVERALL ASSESSMENT:")
    if metrics.footwork_overall_assessment:
        assessment = metrics.footwork_overall_assessment
    elif metrics.grip_angle_difference and abs(metrics.grip_angle_difference) > 5:
        assessment = f"Your {shot_type} technique shows room for improvement, particularly in grip positioning."
    elif metrics.timing_preparation and metrics.timing_preparation > 0.8:
        assessment = f"Your {shot_type} preparation timing needs to be more efficient for better shot execution."
    else:
        assessment = f"Your {shot_type} technique is generally good with minor adjustments needed."

    output_parts.append(assessment)
    output_parts.append("")

    output_parts.append("TECHNIQUE ANALYSIS:")
    output_parts.append("SHOT TECHNIQUE:")
    if metrics.grip_angle_current is not None and metrics.grip_angle_ideal is not None:
        output_parts.append(
            f"Your {shot_type} grip sits around {metrics.grip_angle_current:.1f}°, compared with the target of {metrics.grip_angle_ideal:.1f}°."
        )
    elif metrics.grip_angle_current is not None:
        output_parts.append(f"Your {shot_type} grip sits around {metrics.grip_angle_current:.1f}°.")
    else:
        output_parts.append(f"Keep the {shot_type} contact clean and the swing compact.")

    if metrics.timing_preparation is not None:
        output_parts.append(f"Preparation time is {metrics.timing_preparation:.2f}s, so keep the swing compact and decisive.")

    output_parts.append("FOOTWORK:")
    if metrics.footwork_movement_pattern:
        output_parts.append(f"Pattern: {metrics.footwork_movement_pattern}.")
    if metrics.footwork_pre_hit_stance_ratio is not None:
        if metrics.footwork_pre_hit_stance_ratio < ready_low:
            output_parts.append("Your split step is too narrow; widen the base so you can push off more explosively.")
        elif metrics.footwork_pre_hit_stance_ratio > ready_high:
            output_parts.append("Your ready stance is too wide; tighten it so the first step is faster.")
        else:
            output_parts.append("Your ready stance is in a useful range for this shot.")
    if metrics.footwork_hit_knee_angle is not None:
        if metrics.footwork_hit_knee_angle > ideal_knee + 10:
            output_parts.append("You are too upright at contact; sink lower into the lunge or scissor.")
        elif metrics.footwork_hit_knee_angle < ideal_knee - 15:
            output_parts.append("You are sitting very deep; stay low but keep the movement compact.")
    if metrics.footwork_follow_through_stance_ratio is not None:
        if metrics.footwork_follow_through_stance_ratio < ready_low:
            output_parts.append("Recovery is late; push back to base sooner after contact.")
        elif metrics.footwork_follow_through_stance_ratio > ready_high + 0.15:
            output_parts.append("The recovery landing is still too spread out; get back to base faster.")
    if not metrics.footwork_movement_pattern and metrics.footwork_pre_hit_stance_ratio is None:
        output_parts.append("Stay light on your feet, split step early, and recover immediately after the shot.")

    output_parts.append("")

    output_parts.append("KEY IMPROVEMENT AREAS:")
    improvement_count = 1

    if metrics.grip_angle_difference is not None and abs(metrics.grip_angle_difference) > 5:
        output_parts.append(f"{improvement_count}. Adjust grip angle from {metrics.grip_angle_current:.1f}° to {metrics.grip_angle_ideal:.1f}°")
        improvement_count += 1

    if metrics.timing_preparation is not None and metrics.timing_preparation > 0.8:
        output_parts.append(f"{improvement_count}. Reduce preparation time from {metrics.timing_preparation:.2f}s to 0.6-0.8s")
        improvement_count += 1

    if metrics.footwork_pre_hit_stance_ratio is not None:
        if metrics.footwork_pre_hit_stance_ratio < ready_low:
            output_parts.append(f"{improvement_count}. Widen the split step so the first push-off is more explosive")
            improvement_count += 1
        elif metrics.footwork_pre_hit_stance_ratio > ready_high:
            output_parts.append(f"{improvement_count}. Tighten the ready stance so the first step is faster")
            improvement_count += 1

    if metrics.footwork_hit_knee_angle is not None and metrics.footwork_hit_knee_angle > ideal_knee + 10:
        output_parts.append(f"{improvement_count}. Sink lower into the lunge or scissor at contact")
        improvement_count += 1

    if metrics.footwork_follow_through_stance_ratio is not None and metrics.footwork_follow_through_stance_ratio < ready_low:
        output_parts.append(f"{improvement_count}. Recover to a neutral base sooner after the shot")
        improvement_count += 1

    if improvement_count == 1:
        output_parts.append("1. Continue practicing the shot and footwork pattern together")

    output_parts.append("")

    output_parts.append("PRACTICE RECOMMENDATIONS:")
    if shot_key == 'smash':
        output_parts.append("1. Practice smash shadow swings with a scissor-kick landing and instant recovery.")
        output_parts.append("2. Work on keeping the split step compact before exploding to the shuttle.")
    elif shot_key == 'clear':
        output_parts.append("1. Drill chasse-and-pivot entries so you get behind the shuttle for clears.")
        output_parts.append("2. Recover back to the middle immediately after each clear.")
    elif shot_key == 'drop_shot':
        output_parts.append("1. Drill short, sharp lunge entries for drop shots and keep the landing controlled.")
        output_parts.append("2. Push off the front leg quickly so you can get back to base.")
    else:
        output_parts.append("1. Practice shadow swings focusing on grip positioning and first-step timing.")
        output_parts.append("2. Work on smoother, faster preparation movements and recovery steps.")

    output_parts.append("")

    output_parts.append("TARGET METRICS:")
    if metrics.grip_angle_ideal:
        output_parts.append(f"- Grip Angle: {metrics.grip_angle_ideal:.1f}°")
    else:
        if shot_key == 'smash':
            output_parts.append("- Grip Angle: 15.0°")
            output_parts.append("- Power Transfer: 85%")
        elif shot_key == 'clear':
            output_parts.append("- Grip Angle: 20.0°")
            output_parts.append("- Arc Height: 4.5m")
        elif shot_key in ('drop', 'drop_shot'):
            output_parts.append("- Grip Angle: 12.0°")
            output_parts.append("- Control Accuracy: 90%")
        else:
            output_parts.append("- Consistency: 80%")

    if metrics.timing_preparation and metrics.timing_preparation > 0.8:
        output_parts.append("- Preparation Time: 0.7s")

    output_parts.append(f"- Ready Stance Ratio: {ideal_ratio:.2f}x")
    output_parts.append(f"- Knee Angle: {ideal_knee:.1f}°")
    output_parts.append(f"- Recovery Ratio: {recovery_target:.2f}x")

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
