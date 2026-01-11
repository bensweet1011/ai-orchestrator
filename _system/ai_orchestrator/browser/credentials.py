"""
Secure credential storage for browser automation.
Encrypts credentials at rest using Fernet (AES-128-CBC).
"""

import os
import json
import base64
import warnings
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class CredentialManager:
    """
    Secure storage for browser session credentials.

    Features:
    - Encrypts credentials at rest using Fernet (AES-128-CBC)
    - Per-site credential isolation
    - Storage state management (cookies, localStorage)
    - Master key derived from environment variable
    """

    def __init__(
        self,
        storage_path: Path,
        master_key: Optional[str] = None,
    ):
        """
        Initialize credential manager.

        Args:
            storage_path: Directory for credential storage
            master_key: Encryption key (defaults to BROWSER_MASTER_KEY env var)
        """
        self.storage_path = Path(storage_path)
        self._master_key = master_key or os.environ.get("BROWSER_MASTER_KEY")
        self._ephemeral_key = False

        if not self._master_key:
            # Generate ephemeral key - credentials won't persist across restarts
            warnings.warn(
                "BROWSER_MASTER_KEY not set. Credentials will not persist across restarts. "
                "Set BROWSER_MASTER_KEY environment variable for persistent storage."
            )
            self._master_key = Fernet.generate_key().decode()
            self._ephemeral_key = True

        self._fernet = self._create_fernet(self._master_key)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _create_fernet(self, key: str) -> Fernet:
        """Create Fernet cipher from key string."""
        # Derive a proper key from the master key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"orchestrator_browser_salt_v1",  # Fixed salt for deterministic derivation
            iterations=100000,
        )
        derived_key = base64.urlsafe_b64encode(kdf.derive(key.encode()))
        return Fernet(derived_key)

    def _get_site_path(self, site_id: str, suffix: str = "") -> Path:
        """Get file path for site credentials."""
        # Sanitize site_id for filename
        safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in site_id)
        filename = f"{safe_id}{suffix}.enc"
        return self.storage_path / filename

    def _encrypt(self, data: Dict[str, Any]) -> bytes:
        """Encrypt data dictionary."""
        json_bytes = json.dumps(data).encode()
        return self._fernet.encrypt(json_bytes)

    def _decrypt(self, encrypted: bytes) -> Optional[Dict[str, Any]]:
        """Decrypt data, return None on failure."""
        try:
            decrypted = self._fernet.decrypt(encrypted)
            return json.loads(decrypted.decode())
        except (InvalidToken, json.JSONDecodeError):
            return None

    # -------------------------------------------------------------------------
    # Browser Storage State (cookies, localStorage)
    # -------------------------------------------------------------------------

    def save_storage_state(self, site_id: str, storage_state: Dict[str, Any]):
        """
        Save browser storage state (cookies, localStorage) encrypted.

        Args:
            site_id: Identifier for the site (e.g., "github.com")
            storage_state: Playwright storage state dict
        """
        data = {
            "site_id": site_id,
            "type": "storage_state",
            "storage_state": storage_state,
            "saved_at": datetime.utcnow().isoformat(),
        }

        encrypted = self._encrypt(data)
        path = self._get_site_path(site_id, "_state")
        path.write_bytes(encrypted)

    def get_storage_state(self, site_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve stored browser state.

        Args:
            site_id: Site identifier

        Returns:
            Storage state dict or None if not found
        """
        path = self._get_site_path(site_id, "_state")
        if not path.exists():
            return None

        data = self._decrypt(path.read_bytes())
        if data:
            return data.get("storage_state")
        return None

    def delete_storage_state(self, site_id: str) -> bool:
        """Delete stored browser state."""
        path = self._get_site_path(site_id, "_state")
        if path.exists():
            path.unlink()
            return True
        return False

    # -------------------------------------------------------------------------
    # Login Credentials (username/password)
    # -------------------------------------------------------------------------

    def save_credentials(
        self,
        site_id: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        extra: Optional[Dict[str, str]] = None,
    ):
        """
        Save login credentials (encrypted).

        Args:
            site_id: Site identifier
            username: Username/email
            password: Password
            api_key: API key if applicable
            extra: Additional credential fields
        """
        data = {
            "site_id": site_id,
            "type": "credentials",
            "username": username,
            "password": password,
            "api_key": api_key,
            "extra": extra or {},
            "saved_at": datetime.utcnow().isoformat(),
        }

        encrypted = self._encrypt(data)
        path = self._get_site_path(site_id, "_creds")
        path.write_bytes(encrypted)

    def get_credentials(self, site_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve stored credentials.

        Args:
            site_id: Site identifier

        Returns:
            Credentials dict or None if not found
        """
        path = self._get_site_path(site_id, "_creds")
        if not path.exists():
            return None

        data = self._decrypt(path.read_bytes())
        if data:
            # Return only credential fields, not metadata
            return {
                "username": data.get("username"),
                "password": data.get("password"),
                "api_key": data.get("api_key"),
                "extra": data.get("extra", {}),
            }
        return None

    def delete_credentials(self, site_id: str) -> bool:
        """Delete stored credentials for a site."""
        deleted = False
        for suffix in ["_creds", "_state"]:
            path = self._get_site_path(site_id, suffix)
            if path.exists():
                path.unlink()
                deleted = True
        return deleted

    # -------------------------------------------------------------------------
    # Site Management
    # -------------------------------------------------------------------------

    def list_sites(self) -> List[Dict[str, Any]]:
        """
        List all sites with stored credentials.

        Returns:
            List of dicts with site_id and stored types
        """
        sites: Dict[str, Dict[str, bool]] = {}

        for path in self.storage_path.glob("*.enc"):
            filename = path.stem  # Remove .enc
            if filename.endswith("_creds"):
                site_id = filename[:-6]  # Remove _creds
                if site_id not in sites:
                    sites[site_id] = {"has_credentials": False, "has_state": False}
                sites[site_id]["has_credentials"] = True
            elif filename.endswith("_state"):
                site_id = filename[:-6]  # Remove _state
                if site_id not in sites:
                    sites[site_id] = {"has_credentials": False, "has_state": False}
                sites[site_id]["has_state"] = True

        return [
            {"site_id": site_id, **info}
            for site_id, info in sorted(sites.items())
        ]

    def has_credentials(self, site_id: str) -> bool:
        """Check if credentials exist for a site."""
        return self._get_site_path(site_id, "_creds").exists()

    def has_storage_state(self, site_id: str) -> bool:
        """Check if storage state exists for a site."""
        return self._get_site_path(site_id, "_state").exists()

    # -------------------------------------------------------------------------
    # Status and Info
    # -------------------------------------------------------------------------

    def is_persistent(self) -> bool:
        """Check if credentials will persist across restarts."""
        return not self._ephemeral_key

    def get_status(self) -> Dict[str, Any]:
        """Get credential manager status."""
        sites = self.list_sites()
        return {
            "persistent": self.is_persistent(),
            "storage_path": str(self.storage_path),
            "total_sites": len(sites),
            "sites_with_credentials": sum(
                1 for s in sites if s["has_credentials"]
            ),
            "sites_with_state": sum(1 for s in sites if s["has_state"]),
        }

    def clear_all(self):
        """Remove all stored credentials and states."""
        for path in self.storage_path.glob("*.enc"):
            path.unlink()
