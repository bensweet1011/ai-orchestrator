"""
Rate limiting with human-like randomization.
Enforces minimum delays between browser actions to avoid detection and abuse.
"""

import time
import random
from datetime import datetime
from typing import Optional


class RateLimiter:
    """
    Enforces minimum delays between browser actions.

    Uses randomization to appear more human-like and avoid
    detection/rate limiting by websites.

    Default delays: 3-5 seconds between actions.
    """

    def __init__(
        self,
        min_delay: float = 3.0,
        max_delay: float = 5.0,
        burst_protection: bool = True,
        burst_threshold: int = 10,
        burst_cooldown: float = 30.0,
    ):
        """
        Initialize rate limiter.

        Args:
            min_delay: Minimum seconds between actions (default 3)
            max_delay: Maximum seconds between actions (default 5)
            burst_protection: Enable protection against rapid bursts
            burst_threshold: Actions in window before cooldown triggered
            burst_cooldown: Cooldown period after burst (seconds)
        """
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.burst_protection = burst_protection
        self.burst_threshold = burst_threshold
        self.burst_cooldown = burst_cooldown

        self._last_action: Optional[datetime] = None
        self._action_times: list = []  # Track recent actions for burst detection
        self._burst_window = 60.0  # 1 minute window for burst detection
        self._total_wait_time = 0.0  # Track total wait time for stats

    def wait(self) -> float:
        """
        Wait appropriate time before next action.

        Enforces minimum delay with random jitter.

        Returns:
            Actual wait time in seconds
        """
        now = datetime.utcnow()
        wait_time = 0.0

        # Burst protection check
        if self.burst_protection:
            self._cleanup_old_actions(now)
            if len(self._action_times) >= self.burst_threshold:
                wait_time = self.burst_cooldown
                time.sleep(self.burst_cooldown)
                self._action_times.clear()

        # Enforce minimum delay with randomization
        if self._last_action:
            elapsed = (now - self._last_action).total_seconds()
            required_delay = self.get_recommended_delay()

            if elapsed < required_delay:
                additional_wait = required_delay - elapsed
                time.sleep(additional_wait)
                wait_time += additional_wait

        # Record this action
        self._last_action = datetime.utcnow()
        self._action_times.append(self._last_action)
        self._total_wait_time += wait_time

        return wait_time

    def _cleanup_old_actions(self, now: datetime):
        """Remove actions outside the burst window."""
        cutoff = now.timestamp() - self._burst_window
        self._action_times = [
            t for t in self._action_times if t.timestamp() > cutoff
        ]

    def get_recommended_delay(self) -> float:
        """Get a random delay within the configured range."""
        return random.uniform(self.min_delay, self.max_delay)

    def reset(self):
        """Reset the rate limiter state."""
        self._last_action = None
        self._action_times.clear()
        self._total_wait_time = 0.0

    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        return {
            "total_wait_time": self._total_wait_time,
            "actions_in_window": len(self._action_times),
            "min_delay": self.min_delay,
            "max_delay": self.max_delay,
            "burst_threshold": self.burst_threshold,
            "last_action": (
                self._last_action.isoformat() if self._last_action else None
            ),
        }

    def would_trigger_burst(self) -> bool:
        """Check if next action would trigger burst protection."""
        if not self.burst_protection:
            return False
        self._cleanup_old_actions(datetime.utcnow())
        return len(self._action_times) >= self.burst_threshold - 1

    def time_until_next_action(self) -> float:
        """
        Calculate time until next action is allowed.

        Returns:
            Seconds until next action (0 if immediate)
        """
        if self._last_action is None:
            return 0.0

        elapsed = (datetime.utcnow() - self._last_action).total_seconds()
        remaining = self.min_delay - elapsed

        return max(0.0, remaining)


class AdaptiveRateLimiter(RateLimiter):
    """
    Rate limiter that adapts based on response patterns.

    Increases delays when errors are detected,
    decreases (within bounds) when successful.
    """

    def __init__(
        self,
        min_delay: float = 3.0,
        max_delay: float = 5.0,
        adaptive_min: float = 2.0,
        adaptive_max: float = 15.0,
        **kwargs,
    ):
        """
        Initialize adaptive rate limiter.

        Args:
            min_delay: Starting minimum delay
            max_delay: Starting maximum delay
            adaptive_min: Absolute minimum delay floor
            adaptive_max: Absolute maximum delay ceiling
        """
        super().__init__(min_delay=min_delay, max_delay=max_delay, **kwargs)
        self.adaptive_min = adaptive_min
        self.adaptive_max = adaptive_max
        self._consecutive_errors = 0
        self._consecutive_successes = 0

    def report_success(self):
        """Report a successful action."""
        self._consecutive_errors = 0
        self._consecutive_successes += 1

        # Gradually decrease delay after sustained success
        if self._consecutive_successes >= 5:
            decrease = 0.1 * (self._consecutive_successes // 5)
            self.min_delay = max(self.adaptive_min, self.min_delay - decrease)
            self.max_delay = max(
                self.min_delay + 1.0, self.max_delay - decrease
            )

    def report_error(self, is_rate_limit: bool = False):
        """
        Report a failed action.

        Args:
            is_rate_limit: True if error was a rate limit response
        """
        self._consecutive_successes = 0
        self._consecutive_errors += 1

        # Increase delays on errors
        increase = 1.0 if is_rate_limit else 0.5
        increase *= self._consecutive_errors

        self.min_delay = min(self.adaptive_max - 2.0, self.min_delay + increase)
        self.max_delay = min(self.adaptive_max, self.max_delay + increase)

    def reset_adaptation(self):
        """Reset to initial delay values."""
        self._consecutive_errors = 0
        self._consecutive_successes = 0
