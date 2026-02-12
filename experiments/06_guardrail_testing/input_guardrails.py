"""
Input guardrails for detecting attacks BEFORE they reach the model.

Implements jailbreak detection, encoding detection, and prompt injection detection.
"""

import base64
import binascii
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

ROT13_WORD_MIN_LENGTH = 5
ROT13_CLUSTER_THRESHOLD = 3


@dataclass
class GuardrailResult:
    """Result of guardrail check."""

    verdict: str  # "ALLOW", "BLOCK", "WARN"
    blocked_by: str | None = None
    confidence: float = 0.0
    reason: str = ""
    sanitized_prompt: str | None = None


class JailbreakDetector:
    """Detect common jailbreak attack patterns."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize jailbreak detector configuration."""
        self.enabled = config.get("enabled", True)
        self.threshold = config.get("threshold", 0.75)
        self.block_patterns = config.get("block_patterns", [])

    def check(self, prompt: str) -> GuardrailResult:
        """Check for jailbreak patterns."""
        if not self.enabled:
            return GuardrailResult(verdict="ALLOW")

        prompt_lower = prompt.lower()

        # Pattern matching
        matches = [pattern for pattern in self.block_patterns if pattern.lower() in prompt_lower]

        if matches:
            confidence = min(1.0, len(matches) * 0.3)  # More matches = higher confidence

            if confidence >= self.threshold:
                return GuardrailResult(
                    verdict="BLOCK",
                    blocked_by="jailbreak_detector",
                    confidence=confidence,
                    reason=f"Detected jailbreak patterns: {', '.join(matches)}",
                )

        return GuardrailResult(verdict="ALLOW")


class EncodingDetector:
    """Detect encoded harmful content (Base64, ROT13, leetspeak, etc.)."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize encoding detector configuration."""
        self.enabled = config.get("enabled", True)
        self.block_base64 = config.get("block_base64", True)
        self.block_rot13 = config.get("block_rot13", False)
        self.block_leetspeak = config.get("block_leetspeak", False)
        self.block_unicode_tricks = config.get("block_unicode_tricks", True)

    def check(self, prompt: str) -> GuardrailResult:
        """Check for encoded content."""
        if not self.enabled:
            return GuardrailResult(verdict="ALLOW")

        # Check for Base64
        if self.block_base64 and self._detect_base64(prompt):
            return GuardrailResult(
                verdict="BLOCK",
                blocked_by="encoding_detector",
                confidence=0.8,
                reason="Detected Base64 encoded content",
            )

        # Check for ROT13
        if self.block_rot13 and self._detect_rot13(prompt):
            return GuardrailResult(
                verdict="BLOCK",
                blocked_by="encoding_detector",
                confidence=0.7,
                reason="Detected ROT13 encoded content",
            )

        # Check for leetspeak
        if self.block_leetspeak and self._detect_leetspeak(prompt):
            return GuardrailResult(
                verdict="WARN",
                blocked_by="encoding_detector",
                confidence=0.5,
                reason="Detected possible leetspeak",
            )

        # Check for Unicode tricks
        if self.block_unicode_tricks and self._detect_unicode_tricks(prompt):
            return GuardrailResult(
                verdict="BLOCK",
                blocked_by="encoding_detector",
                confidence=0.9,
                reason="Detected Unicode homoglyph or zero-width characters",
            )

        return GuardrailResult(verdict="ALLOW")

    def _detect_base64(self, text: str) -> bool:
        """Detect Base64 encoded strings."""
        # Look for long Base64-like strings
        base64_pattern = r"[A-Za-z0-9+/]{20,}={0,2}"
        matches = re.findall(base64_pattern, text)

        for match in matches:
            try:
                # Try to decode - if successful, likely Base64
                base64.b64decode(match, validate=True)
            except (binascii.Error, ValueError) as exc:
                logger.debug("Base64 decode failed: %s", exc)
            else:
                return True

        return False

    def _detect_rot13(self, text: str) -> bool:
        """Detect ROT13 patterns (high consonant-vowel ratio anomaly)."""
        # ROT13 text often has unusual character distributions
        # This is a heuristic - not perfect
        words = text.split()
        suspicious_words = 0

        for word in words:
            if len(word) > ROT13_WORD_MIN_LENGTH and re.search(
                r"[bcdfghjklmnpqrstvwxyz]{4,}",
                word,
                re.IGNORECASE,
            ):
                suspicious_words += 1

        return suspicious_words >= ROT13_CLUSTER_THRESHOLD

    def _detect_leetspeak(self, text: str) -> bool:
        """Detect leetspeak (character substitution)."""
        leetspeak_patterns = [
            r"h[4a]ck",
            r"cr[4a]ck",
            r"p[4a]ssw[0o]rd",
            r"3xpl[0o][1i]t",
            r"m[4a]lw[4a]r[3e]",
        ]

        return any(re.search(pattern, text, re.IGNORECASE) for pattern in leetspeak_patterns)

    def _detect_unicode_tricks(self, text: str) -> bool:
        """Detect Unicode homoglyphs and zero-width characters."""
        # Zero-width characters
        zero_width_chars = ["\u200b", "\u200c", "\u200d", "\ufeff"]
        if any(char in text for char in zero_width_chars):
            return True

        # Cyrillic homoglyphs (common in phishing)
        cyrillic_lookalikes = [
            "\u0430",
            "\u0435",
            "\u043e",
            "\u0440",
            "\u0441",
            "\u0443",
            "\u0445",
        ]  # Look like Latin a,e,o,p,c,y,x
        return any(char in text for char in cyrillic_lookalikes)


class InjectionDetector:
    """Detect prompt injection attempts."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize injection detector configuration."""
        self.enabled = config.get("enabled", True)
        self.max_prompt_length = config.get("max_prompt_length", 4000)
        self.block_system_tags = config.get("block_system_tags", True)
        self.system_override_patterns = config.get("system_override_patterns", [])

    def check(self, prompt: str) -> GuardrailResult:
        """Check for prompt injection."""
        if not self.enabled:
            return GuardrailResult(verdict="ALLOW")

        # Check length
        if len(prompt) > self.max_prompt_length:
            return GuardrailResult(
                verdict="BLOCK",
                blocked_by="injection_detector",
                confidence=0.6,
                reason=f"Prompt exceeds maximum length ({self.max_prompt_length} chars)",
            )

        # Check for system tags
        if self.block_system_tags:
            system_tags = ["<system>", "</system>", "<|system|>", "<|endoftext|>"]
            for tag in system_tags:
                if tag.lower() in prompt.lower():
                    return GuardrailResult(
                        verdict="BLOCK",
                        blocked_by="injection_detector",
                        confidence=0.9,
                        reason=f"Detected system tag: {tag}",
                    )

        # Check for system override patterns
        for pattern in self.system_override_patterns:
            if pattern.lower() in prompt.lower():
                return GuardrailResult(
                    verdict="BLOCK",
                    blocked_by="injection_detector",
                    confidence=0.85,
                    reason=f"Detected injection pattern: {pattern}",
                )

        return GuardrailResult(verdict="ALLOW")


class InputGuardrails:
    """Orchestrate all input guardrail checks."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize input guardrails."""
        self.jailbreak_detector = JailbreakDetector(
            config.get("jailbreak_detector", {}),
        )
        self.encoding_detector = EncodingDetector(
            config.get("encoding_detector", {}),
        )
        self.injection_detector = InjectionDetector(
            config.get("injection_detector", {}),
        )

    def check_prompt(self, prompt: str) -> GuardrailResult:
        """
        Run all input checks and return verdict.

        Returns:
            GuardrailResult with verdict and details
        """
        # Run all checks
        checks = [
            self.jailbreak_detector.check(prompt),
            self.encoding_detector.check(prompt),
            self.injection_detector.check(prompt),
        ]

        # If any check blocks, return that result
        for result in checks:
            if result.verdict == "BLOCK":
                return result

        # If any check warns, return warning
        for result in checks:
            if result.verdict == "WARN":
                result.verdict = "ALLOW"  # Warnings don't block, just log
                return result

        # All passed
        return GuardrailResult(verdict="ALLOW", sanitized_prompt=prompt)
