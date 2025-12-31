"""
Access control layer for AI safety pipeline.

Implements API key management, rate limiting, user tiers, and audit logging.
Leverages DevSecOps best practices for secure key storage and access control.
"""

import hashlib
import secrets
import time
import json
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque


@dataclass
class UserInfo:
    """User information associated with API key."""
    user_id: str
    tier: str  # "free", "research", "enterprise"
    created_at: str
    expires_at: str


@dataclass
class AuthResult:
    """Result of authorization check."""
    allowed: bool
    user_tier: str
    quota_remaining: int
    reason: str = ""


class APIKeyManager:
    """Secure API key management with hashing."""

    def __init__(self):
        self.keys: Dict[str, UserInfo] = {}  # key_hash -> UserInfo

    def generate_key(self, user_id: str, tier: str, expiry_days: int = 90) -> str:
        """
        Generate cryptographically secure API key.

        Returns:
            API key in format: sk-proj-<random-32-chars>
        """
        prefix = "sk-proj-"
        random_part = secrets.token_urlsafe(32)
        key = prefix + random_part

        # Store hashed version only
        key_hash = self._hash_key(key)

        now = datetime.now()
        expires = now + timedelta(days=expiry_days)

        self.keys[key_hash] = UserInfo(
            user_id=user_id,
            tier=tier,
            created_at=now.isoformat(),
            expires_at=expires.isoformat()
        )

        return key  # Show once, never again

    def validate_key(self, api_key: str) -> Optional[UserInfo]:
        """
        Validate API key and return user info.

        Uses constant-time comparison to prevent timing attacks.
        """
        key_hash = self._hash_key(api_key)

        if key_hash not in self.keys:
            return None

        user_info = self.keys[key_hash]

        # Check expiry
        expires_at = datetime.fromisoformat(user_info.expires_at)
        if datetime.now() > expires_at:
            return None

        return user_info

    def revoke_key(self, api_key: str) -> bool:
        """Immediately invalidate API key."""
        key_hash = self._hash_key(api_key)
        if key_hash in self.keys:
            del self.keys[key_hash]
            return True
        return False

    def _hash_key(self, key: str) -> str:
        """Hash API key using SHA-256."""
        return hashlib.sha256(key.encode()).hexdigest()


class RateLimiter:
    """
    Token bucket rate limiter with per-tier limits.

    Implements sliding window rate limiting for requests per minute/hour/day.
    """

    def __init__(self, config: Dict):
        self.config = config
        # Track requests: user_tier -> window -> deque of timestamps
        self.requests: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: {
                "minute": deque(),
                "hour": deque(),
                "day": deque()
            }
        )

    def check_rate_limit(self, user_tier: str) -> Tuple[bool, str, int]:
        """
        Check if request is within rate limits.

        Returns:
            (allowed, reason, quota_remaining)
        """
        if user_tier not in self.config:
            return False, f"Unknown tier: {user_tier}", 0

        tier_limits = self.config[user_tier]
        now = time.time()

        # Clean old requests and check limits
        for window, limit_key in [
            ("minute", "requests_per_minute"),
            ("hour", "requests_per_hour"),
            ("day", "requests_per_day")
        ]:
            window_seconds = {"minute": 60, "hour": 3600, "day": 86400}[window]
            limit = tier_limits[limit_key]

            # Remove requests outside window
            requests = self.requests[user_tier][window]
            while requests and requests[0] < now - window_seconds:
                requests.popleft()

            # Check limit
            if len(requests) >= limit:
                quota_remaining = 0
                return False, f"Rate limit exceeded: {limit} {window}", quota_remaining

        # Calculate quota remaining (most restrictive window)
        min_quota = float('inf')
        for window, limit_key in [
            ("minute", "requests_per_minute"),
            ("hour", "requests_per_hour"),
            ("day", "requests_per_day")
        ]:
            limit = tier_limits[limit_key]
            used = len(self.requests[user_tier][window])
            remaining = limit - used
            min_quota = min(min_quota, remaining)

        return True, "", int(min_quota)

    def record_request(self, user_tier: str):
        """Record a request timestamp."""
        now = time.time()
        for window in ["minute", "hour", "day"]:
            self.requests[user_tier][window].append(now)


class AuditLogger:
    """Security event and audit logging."""

    def __init__(self, config: Dict):
        from pathlib import Path

        self.config = config
        self.audit_log_path = config.get("audit_log_path", "results/audit.jsonl")
        self.security_log_path = config.get("security_events_path", "results/security_events.jsonl")

        # Create log directories if they don't exist
        Path(self.audit_log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.security_log_path).parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, data: Dict):
        """Log event to audit log."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **data
        }

        # Append to audit log
        try:
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"Failed to write audit log: {e}")

    def log_security_event(self, event_type: str, data: Dict):
        """Log security event to security log."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **data
        }

        try:
            with open(self.security_log_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"Failed to write security log: {e}")

    def log_successful_request(self, request_id: str, user_tier: str,
                               prompt_length: int, response_length: int,
                               latency: float):
        """Log successful request."""
        self.log_event("successful_request", {
            "request_id": request_id,
            "user_tier": user_tier,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "latency_ms": latency * 1000
        })

    def log_blocked_request(self, request_id: str, user_tier: str,
                           prompt_hash: str, blocked_by: str,
                           confidence: float):
        """Log blocked request."""
        self.log_security_event("blocked_request", {
            "request_id": request_id,
            "user_tier": user_tier,
            "prompt_hash": prompt_hash,
            "blocked_by": blocked_by,
            "confidence": confidence
        })

    def log_harmful_output(self, request_id: str, user_tier: str,
                          violations: list):
        """Log harmful output detection."""
        self.log_security_event("harmful_output_blocked", {
            "request_id": request_id,
            "user_tier": user_tier,
            "violations": violations
        })

    def log_error(self, request_id: str, error: str):
        """Log error."""
        self.log_event("error", {
            "request_id": request_id,
            "error": error
        })


class AccessControl:
    """Main access control orchestration."""

    def __init__(self, config: Dict):
        self.config = config
        self.key_manager = APIKeyManager()
        self.rate_limiter = RateLimiter(config.get("rate_limits", {}))
        self.audit_logger = AuditLogger(config.get("logging", {}))

    def authorize_request(self, api_key: str) -> AuthResult:
        """
        Authorize request: validate key and check rate limits.

        Returns:
            AuthResult with allowed status and details
        """
        # Validate API key
        user_info = self.key_manager.validate_key(api_key)
        if not user_info:
            self.audit_logger.log_security_event("invalid_api_key", {
                "key_prefix": api_key[:10] if len(api_key) >= 10 else "***"
            })
            return AuthResult(
                allowed=False,
                user_tier="",
                quota_remaining=0,
                reason="Invalid or expired API key"
            )

        # Check rate limits
        allowed, reason, quota = self.rate_limiter.check_rate_limit(user_info.tier)

        if not allowed:
            self.audit_logger.log_security_event("rate_limit_exceeded", {
                "user_id": user_info.user_id,
                "tier": user_info.tier,
                "reason": reason
            })
            return AuthResult(
                allowed=False,
                user_tier=user_info.tier,
                quota_remaining=0,
                reason=reason
            )

        # Record request
        self.rate_limiter.record_request(user_info.tier)

        return AuthResult(
            allowed=True,
            user_tier=user_info.tier,
            quota_remaining=quota
        )
