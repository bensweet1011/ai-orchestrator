"""
Product Registry for tracking deployed products.
Provides centralized management of all deployments across platforms.
"""

import json
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

from ..config import DEPLOY_REGISTRY_PATH


class ProductType(Enum):
    """Types of deployed products."""

    STREAMLIT = "streamlit"  # Streamlit Cloud app
    VERCEL = "vercel"  # Vercel deployment (Next.js)
    STATIC = "static"  # Static site
    API = "api"  # API service


class ProductStatus(Enum):
    """Deployment status."""

    DRAFT = "draft"  # Not yet deployed
    BUILDING = "building"  # Build in progress
    DEPLOYED = "deployed"  # Successfully deployed
    FAILED = "failed"  # Deployment failed
    ARCHIVED = "archived"  # No longer active


@dataclass
class DeploymentRecord:
    """Record of a single deployment."""

    id: str
    timestamp: str
    status: str
    url: Optional[str] = None
    commit_sha: Optional[str] = None
    build_time_ms: Optional[int] = None
    error: Optional[str] = None


@dataclass
class ProductEntry:
    """Entry for a deployed product."""

    id: str
    name: str
    description: str
    product_type: str  # ProductType value
    github_repo: str
    deploy_url: Optional[str]
    status: str  # ProductStatus value
    created_at: str
    updated_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    deployment_history: List[DeploymentRecord] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class ProductRegistry:
    """
    Central registry for tracking all deployed products.

    Features:
    - Register and track products across platforms
    - Deployment history with timestamps
    - Status tracking
    - Tag-based organization
    - Persistent storage
    """

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize product registry.

        Args:
            registry_path: Path to registry file (defaults to config path)
        """
        self.registry_path = Path(registry_path or DEPLOY_REGISTRY_PATH)
        self._products: Dict[str, ProductEntry] = {}
        self._load()

    def _load(self) -> None:
        """Load registry from file."""
        if self.registry_path.exists():
            try:
                data = json.loads(self.registry_path.read_text())
                products_data = data.get("products", {})

                for pid, pdata in products_data.items():
                    # Convert deployment history
                    history = [
                        DeploymentRecord(**h)
                        for h in pdata.get("deployment_history", [])
                    ]
                    pdata["deployment_history"] = history
                    self._products[pid] = ProductEntry(**pdata)
            except Exception:
                self._products = {}

    def _save(self) -> None:
        """Save registry to file."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing data to preserve other sections
        if self.registry_path.exists():
            try:
                data = json.loads(self.registry_path.read_text())
            except Exception:
                data = {}
        else:
            data = {}

        # Update products section
        data["products"] = {
            pid: {
                **asdict(product),
                "deployment_history": [
                    asdict(h) for h in product.deployment_history
                ],
            }
            for pid, product in self._products.items()
        }

        self.registry_path.write_text(json.dumps(data, indent=2))

    def _generate_id(self, name: str, github_repo: str) -> str:
        """Generate unique product ID."""
        content = f"{name}-{github_repo}-{datetime.utcnow().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    # =========================================================================
    # Product Operations
    # =========================================================================

    def register_product(
        self,
        name: str,
        product_type: ProductType,
        github_repo: str,
        description: str = "",
        deploy_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> ProductEntry:
        """
        Register a new product.

        Args:
            name: Product name
            product_type: Type of product (streamlit, vercel, etc.)
            github_repo: GitHub repository (owner/repo)
            description: Product description
            deploy_url: Deployment URL (if already deployed)
            metadata: Additional metadata
            tags: Tags for organization

        Returns:
            Created product entry
        """
        product_id = self._generate_id(name, github_repo)
        now = datetime.utcnow().isoformat()

        product = ProductEntry(
            id=product_id,
            name=name,
            description=description,
            product_type=product_type.value,
            github_repo=github_repo,
            deploy_url=deploy_url,
            status=ProductStatus.DRAFT.value if not deploy_url else ProductStatus.DEPLOYED.value,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
            deployment_history=[],
            tags=tags or [],
        )

        self._products[product_id] = product
        self._save()

        return product

    def update_product(
        self,
        product_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        deploy_url: Optional[str] = None,
        status: Optional[ProductStatus] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[ProductEntry]:
        """
        Update a product.

        Args:
            product_id: Product ID
            name: New name (optional)
            description: New description (optional)
            deploy_url: New URL (optional)
            status: New status (optional)
            metadata: Metadata to merge (optional)
            tags: New tags (optional)

        Returns:
            Updated product or None if not found
        """
        if product_id not in self._products:
            return None

        product = self._products[product_id]

        if name is not None:
            product.name = name
        if description is not None:
            product.description = description
        if deploy_url is not None:
            product.deploy_url = deploy_url
        if status is not None:
            product.status = status.value
        if metadata is not None:
            product.metadata.update(metadata)
        if tags is not None:
            product.tags = tags

        product.updated_at = datetime.utcnow().isoformat()

        self._save()
        return product

    def get_product(self, product_id: str) -> Optional[ProductEntry]:
        """Get product by ID."""
        return self._products.get(product_id)

    def find_product_by_repo(self, github_repo: str) -> Optional[ProductEntry]:
        """Find product by GitHub repository."""
        for product in self._products.values():
            if product.github_repo == github_repo:
                return product
        return None

    def list_products(
        self,
        product_type: Optional[ProductType] = None,
        status: Optional[ProductStatus] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ProductEntry]:
        """
        List products with optional filters.

        Args:
            product_type: Filter by type
            status: Filter by status
            tags: Filter by tags (any match)

        Returns:
            List of matching products
        """
        products = list(self._products.values())

        if product_type:
            products = [p for p in products if p.product_type == product_type.value]

        if status:
            products = [p for p in products if p.status == status.value]

        if tags:
            products = [
                p for p in products
                if any(t in p.tags for t in tags)
            ]

        return sorted(products, key=lambda p: p.updated_at, reverse=True)

    def delete_product(self, product_id: str) -> bool:
        """
        Delete a product from registry.

        Args:
            product_id: Product ID

        Returns:
            True if deleted
        """
        if product_id in self._products:
            del self._products[product_id]
            self._save()
            return True
        return False

    # =========================================================================
    # Deployment History
    # =========================================================================

    def record_deployment(
        self,
        product_id: str,
        status: str,
        url: Optional[str] = None,
        commit_sha: Optional[str] = None,
        build_time_ms: Optional[int] = None,
        error: Optional[str] = None,
    ) -> Optional[DeploymentRecord]:
        """
        Record a deployment for a product.

        Args:
            product_id: Product ID
            status: Deployment status
            url: Deployment URL
            commit_sha: Git commit SHA
            build_time_ms: Build time in milliseconds
            error: Error message if failed

        Returns:
            Deployment record or None if product not found
        """
        if product_id not in self._products:
            return None

        product = self._products[product_id]
        now = datetime.utcnow().isoformat()

        deployment_id = hashlib.sha256(
            f"{product_id}-{now}".encode()
        ).hexdigest()[:8]

        record = DeploymentRecord(
            id=deployment_id,
            timestamp=now,
            status=status,
            url=url,
            commit_sha=commit_sha,
            build_time_ms=build_time_ms,
            error=error,
        )

        product.deployment_history.append(record)
        product.updated_at = now

        # Update product status based on deployment
        if status == "success":
            product.status = ProductStatus.DEPLOYED.value
            if url:
                product.deploy_url = url
        elif status == "failed":
            product.status = ProductStatus.FAILED.value
        elif status == "building":
            product.status = ProductStatus.BUILDING.value

        self._save()
        return record

    def get_deployment_history(
        self,
        product_id: str,
        limit: int = 10,
    ) -> List[DeploymentRecord]:
        """
        Get deployment history for a product.

        Args:
            product_id: Product ID
            limit: Max records to return

        Returns:
            List of deployment records (newest first)
        """
        product = self._products.get(product_id)
        if not product:
            return []

        history = product.deployment_history
        return sorted(history, key=lambda r: r.timestamp, reverse=True)[:limit]

    def get_latest_deployment(
        self,
        product_id: str,
    ) -> Optional[DeploymentRecord]:
        """Get most recent deployment for a product."""
        history = self.get_deployment_history(product_id, limit=1)
        return history[0] if history else None

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dict with counts and breakdowns
        """
        products = list(self._products.values())

        by_type = {}
        by_status = {}
        total_deployments = 0

        for p in products:
            by_type[p.product_type] = by_type.get(p.product_type, 0) + 1
            by_status[p.status] = by_status.get(p.status, 0) + 1
            total_deployments += len(p.deployment_history)

        return {
            "total_products": len(products),
            "total_deployments": total_deployments,
            "by_type": by_type,
            "by_status": by_status,
        }


# Singleton instance
_registry: Optional[ProductRegistry] = None


def get_product_registry() -> ProductRegistry:
    """Get or create ProductRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = ProductRegistry()
    return _registry


def reset_product_registry() -> None:
    """Reset the singleton."""
    global _registry
    _registry = None
