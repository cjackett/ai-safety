"""
Complete safety pipeline orchestrating all guardrail layers.

Implements defense-in-depth: access control → input guardrails → model → output guardrails → logging
"""

import time
import uuid
import yaml
import ollama
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from access_control import AccessControl
from input_guardrails import InputGuardrails
from output_guardrails import OutputGuardrails


@dataclass
class Response:
    """Pipeline response."""
    success: bool
    content: str
    request_id: str
    blocked_at: Optional[str] = None
    reason: str = ""
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SafetyPipeline:
    """Complete inference pipeline with all safety layers."""

    def __init__(self, config_path: str):
        """
        Initialize safety pipeline from config file.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)

        # Initialize components
        self.access_control = AccessControl(self.config.get("access_control", {}))
        self.input_guardrails = InputGuardrails(self.config.get("input_guardrails", {}))
        self.output_guardrails = OutputGuardrails(self.config.get("output_guardrails", {}))

        # Model config
        self.model_config = self.config.get("model", {})
        self.default_model = self.model_config.get("default_model", "llama3.2:3b")

    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def process_request(self, api_key: str, prompt: str, model: Optional[str] = None) -> Response:
        """
        Process inference request through all safety layers.

        Pipeline:
        1. Access control (auth, rate limits)
        2. Input guardrails (jailbreak detection, filtering)
        3. Model inference
        4. Output guardrails (harm detection, PII redaction)
        5. Audit logging
        6. Return response

        Args:
            api_key: User API key
            prompt: User prompt
            model: Optional model override

        Returns:
            Response with success status and content
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        try:
            # Layer 1: Access Control
            auth_result = self.access_control.authorize_request(api_key)

            if not auth_result.allowed:
                return Response(
                    success=False,
                    content="",
                    request_id=request_id,
                    blocked_at="access_control",
                    reason=auth_result.reason
                )

            # Layer 2: Input Guardrails
            input_check = self.input_guardrails.check_prompt(prompt)

            if input_check.verdict == "BLOCK":
                self.access_control.audit_logger.log_blocked_request(
                    request_id=request_id,
                    user_tier=auth_result.user_tier,
                    prompt_hash=str(hash(prompt)),
                    blocked_by=input_check.blocked_by,
                    confidence=input_check.confidence
                )

                return Response(
                    success=False,
                    content="",
                    request_id=request_id,
                    blocked_at="input_guardrails",
                    reason=input_check.reason,
                    metadata={
                        "blocked_by": input_check.blocked_by,
                        "confidence": input_check.confidence
                    }
                )

            # Use sanitized prompt if available
            safe_prompt = input_check.sanitized_prompt or prompt

            # Layer 3: Model Inference
            model_to_use = model or self.default_model
            model_response = self._generate_response(safe_prompt, model_to_use)

            # Layer 4: Output Guardrails
            output_check = self.output_guardrails.validate_response(
                response=model_response,
                prompt=safe_prompt
            )

            if output_check.verdict == "BLOCK":
                self.access_control.audit_logger.log_harmful_output(
                    request_id=request_id,
                    user_tier=auth_result.user_tier,
                    violations=output_check.violations
                )

                return Response(
                    success=False,
                    content=output_check.filtered_response,  # Safe refusal message
                    request_id=request_id,
                    blocked_at="output_guardrails",
                    reason="Generated content violates safety policies",
                    metadata={
                        "violations": output_check.violations,
                        "output_compliance_type": output_check.compliance_type,
                        "output_compliance_confidence": output_check.compliance_confidence,
                        "output_classifier_votes": output_check.classifier_votes,
                        "output_classifier_reasoning": output_check.classifier_reasoning
                    }
                )

            # Layer 5: Audit Logging
            latency = time.time() - start_time
            self.access_control.audit_logger.log_successful_request(
                request_id=request_id,
                user_tier=auth_result.user_tier,
                prompt_length=len(prompt),
                response_length=len(output_check.filtered_response),
                latency=latency
            )

            # Return safe response
            return Response(
                success=True,
                content=output_check.filtered_response,
                request_id=request_id,
                metadata={
                    "latency_ms": latency * 1000,
                    "input_sanitized": input_check.sanitized_prompt is not None,
                    "output_redacted": output_check.redacted,
                    "model": model_to_use,
                    # Output guardrail debug information
                    "output_compliance_type": output_check.compliance_type,
                    "output_compliance_confidence": output_check.compliance_confidence,
                    "output_classifier_votes": output_check.classifier_votes,
                    "output_classifier_reasoning": output_check.classifier_reasoning
                }
            )

        except Exception as e:
            self.access_control.audit_logger.log_error(request_id, str(e))
            return Response(
                success=False,
                content="",
                request_id=request_id,
                blocked_at="error",
                reason=f"Internal error: {str(e)}"
            )

    def _generate_response(self, prompt: str, model: str) -> str:
        """Generate response from model."""
        try:
            result = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": self.model_config.get("temperature", 0.7),
                    "num_predict": self.model_config.get("max_tokens", 2048)
                }
            )
            return result["response"]
        except Exception as e:
            raise RuntimeError(f"Model generation failed: {e}")

    def create_test_user(self, user_id: str, tier: str = "research") -> str:
        """
        Create a test API key for evaluation.

        Args:
            user_id: User identifier
            tier: User tier (free, research, enterprise)

        Returns:
            API key
        """
        expiry_days = self.config.get("access_control", {}).get("key_expiry_days", 90)
        return self.access_control.key_manager.generate_key(user_id, tier, expiry_days)
