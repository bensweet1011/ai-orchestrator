"""
Notion integration for AI Orchestrator.
Provides read/write access to Notion pages and databases.
"""

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from notion_client import Client
from notion_client.errors import APIResponseError


class NotionClient:
    """
    Client for Notion API operations.

    Features:
    - Read pages and databases
    - Create and update pages
    - Add comments to pages
    - Log pipeline runs to a database
    - Create project pages from templates
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Notion client.

        Args:
            api_key: Notion API key (defaults to NOTION_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("NOTION_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Notion API key required. Set NOTION_API_KEY environment variable."
            )

        self.client = Client(auth=self.api_key)

    # =========================================================================
    # Page Operations
    # =========================================================================

    def get_page(self, page_id: str) -> Dict[str, Any]:
        """
        Get a Notion page by ID.

        Args:
            page_id: Notion page ID or URL

        Returns:
            Page data including properties and content
        """
        page_id = self._extract_id(page_id)
        page = self.client.pages.retrieve(page_id=page_id)

        # Get page content (blocks)
        blocks = self._get_all_blocks(page_id)
        content = self._blocks_to_text(blocks)

        # Extract title from properties
        title = self._extract_title(page.get("properties", {}))

        return {
            "id": page["id"],
            "title": title,
            "content": content,
            "properties": page.get("properties", {}),
            "url": page.get("url", ""),
            "created_time": page.get("created_time", ""),
            "last_edited_time": page.get("last_edited_time", ""),
        }

    def get_page_content(self, page_id: str) -> str:
        """
        Get just the text content of a page.

        Args:
            page_id: Notion page ID or URL

        Returns:
            Plain text content
        """
        page_id = self._extract_id(page_id)
        blocks = self._get_all_blocks(page_id)
        return self._blocks_to_text(blocks)

    def create_page(
        self,
        parent_id: str,
        title: str,
        content: str = "",
        properties: Optional[Dict[str, Any]] = None,
        is_database: bool = True,
    ) -> Dict[str, Any]:
        """
        Create a new Notion page.

        Args:
            parent_id: Parent page or database ID
            title: Page title
            content: Page content (markdown-like text)
            properties: Additional properties for database pages
            is_database: Whether parent is a database (True) or page (False)

        Returns:
            Created page data
        """
        parent_id = self._extract_id(parent_id)

        # Build parent reference
        if is_database:
            parent = {"database_id": parent_id}
            page_properties = {"Name": {"title": [{"text": {"content": title}}]}}
        else:
            parent = {"page_id": parent_id}
            page_properties = {"title": {"title": [{"text": {"content": title}}]}}

        # Merge additional properties
        if properties:
            page_properties.update(properties)

        # Convert content to blocks
        children = self._text_to_blocks(content) if content else []

        page = self.client.pages.create(
            parent=parent,
            properties=page_properties,
            children=children,
        )

        return {
            "id": page["id"],
            "url": page.get("url", ""),
            "title": title,
        }

    def update_page(
        self,
        page_id: str,
        content: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Update a Notion page.

        Args:
            page_id: Page ID to update
            content: New content (replaces existing blocks)
            properties: Properties to update

        Returns:
            Updated page data
        """
        page_id = self._extract_id(page_id)

        # Update properties if provided
        if properties:
            self.client.pages.update(page_id=page_id, properties=properties)

        # Update content if provided
        if content is not None:
            # Delete existing blocks
            existing_blocks = self._get_all_blocks(page_id)
            for block in existing_blocks:
                try:
                    self.client.blocks.delete(block_id=block["id"])
                except APIResponseError:
                    pass  # Some blocks can't be deleted

            # Add new blocks
            new_blocks = self._text_to_blocks(content)
            if new_blocks:
                self.client.blocks.children.append(
                    block_id=page_id,
                    children=new_blocks,
                )

        return {"id": page_id, "updated": True}

    def append_to_page(self, page_id: str, content: str) -> Dict[str, Any]:
        """
        Append content to an existing page.

        Args:
            page_id: Page ID
            content: Content to append

        Returns:
            Result with appended block IDs
        """
        page_id = self._extract_id(page_id)
        blocks = self._text_to_blocks(content)

        if blocks:
            result = self.client.blocks.children.append(
                block_id=page_id,
                children=blocks,
            )
            return {
                "id": page_id,
                "appended_blocks": len(result.get("results", [])),
            }

        return {"id": page_id, "appended_blocks": 0}

    # =========================================================================
    # Database Operations
    # =========================================================================

    def get_database(self, database_id: str) -> Dict[str, Any]:
        """
        Get database metadata and schema.

        Args:
            database_id: Database ID or URL

        Returns:
            Database info including properties schema
        """
        database_id = self._extract_id(database_id)
        db = self.client.databases.retrieve(database_id=database_id)

        return {
            "id": db["id"],
            "title": self._extract_plain_text(db.get("title", [])),
            "properties": db.get("properties", {}),
            "url": db.get("url", ""),
        }

    def query_database(
        self,
        database_id: str,
        filter: Optional[Dict[str, Any]] = None,
        sorts: Optional[List[Dict[str, Any]]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query a Notion database.

        Args:
            database_id: Database ID or URL
            filter: Notion filter object
            sorts: Sort configuration
            limit: Maximum results

        Returns:
            List of page results
        """
        database_id = self._extract_id(database_id)

        query_params = {"database_id": database_id, "page_size": min(limit, 100)}

        if filter:
            query_params["filter"] = filter
        if sorts:
            query_params["sorts"] = sorts

        results = []
        has_more = True
        start_cursor = None

        while has_more and len(results) < limit:
            if start_cursor:
                query_params["start_cursor"] = start_cursor

            response = self.client.databases.query(**query_params)
            results.extend(response.get("results", []))
            has_more = response.get("has_more", False)
            start_cursor = response.get("next_cursor")

        # Process results
        processed = []
        for page in results[:limit]:
            title = self._extract_title(page.get("properties", {}))
            processed.append(
                {
                    "id": page["id"],
                    "title": title,
                    "properties": page.get("properties", {}),
                    "url": page.get("url", ""),
                    "created_time": page.get("created_time", ""),
                    "last_edited_time": page.get("last_edited_time", ""),
                }
            )

        return processed

    # =========================================================================
    # Comments
    # =========================================================================

    def add_comment(self, page_id: str, comment: str) -> Dict[str, Any]:
        """
        Add a comment to a Notion page.

        Args:
            page_id: Page ID
            comment: Comment text

        Returns:
            Created comment data
        """
        page_id = self._extract_id(page_id)

        result = self.client.comments.create(
            parent={"page_id": page_id},
            rich_text=[{"type": "text", "text": {"content": comment}}],
        )

        return {
            "id": result["id"],
            "page_id": page_id,
            "comment": comment,
        }

    def get_comments(self, page_id: str) -> List[Dict[str, Any]]:
        """
        Get all comments on a page.

        Args:
            page_id: Page ID

        Returns:
            List of comments
        """
        page_id = self._extract_id(page_id)

        response = self.client.comments.list(block_id=page_id)
        comments = []

        for comment in response.get("results", []):
            text = self._extract_plain_text(comment.get("rich_text", []))
            comments.append(
                {
                    "id": comment["id"],
                    "text": text,
                    "created_time": comment.get("created_time", ""),
                }
            )

        return comments

    # =========================================================================
    # Pipeline Integration
    # =========================================================================

    def create_project_page(
        self,
        database_id: str,
        project_name: str,
        description: str = "",
        status: str = "Not Started",
    ) -> Dict[str, Any]:
        """
        Create a new project page with standard template.

        Args:
            database_id: Projects database ID
            project_name: Name of the project
            description: Project description
            status: Initial status

        Returns:
            Created page data
        """
        # Standard project template content
        template_content = f"""## Overview
{description or 'Project description goes here.'}

## Goals
- [ ] Define project goals

## Timeline
- **Created**: {datetime.utcnow().strftime('%Y-%m-%d')}
- **Target Completion**: TBD

## Notes
"""

        # Properties for database page
        properties = {
            "Name": {"title": [{"text": {"content": project_name}}]},
        }

        # Add status if the database has a Status property
        # (will fail silently if property doesn't exist)
        try:
            db = self.get_database(database_id)
            if "Status" in db.get("properties", {}):
                properties["Status"] = {"select": {"name": status}}
        except Exception:
            pass

        return self.create_page(
            parent_id=database_id,
            title=project_name,
            content=template_content,
            properties=properties,
            is_database=True,
        )

    def log_pipeline_run(
        self,
        database_id: str,
        pipeline_name: str,
        input_text: str,
        output_text: str,
        success: bool,
        latency_ms: int,
        model_info: str = "",
    ) -> Dict[str, Any]:
        """
        Log a pipeline run to a Notion database.

        Args:
            database_id: Pipeline logs database ID
            pipeline_name: Name of the pipeline
            input_text: Input that was processed
            output_text: Output from pipeline
            success: Whether run succeeded
            latency_ms: Execution time in ms
            model_info: Models used (optional)

        Returns:
            Created log entry
        """
        timestamp = datetime.utcnow().isoformat()
        title = f"{pipeline_name} - {timestamp[:19]}"

        # Content for the log page
        content = f"""## Pipeline Run Details

**Pipeline**: {pipeline_name}
**Timestamp**: {timestamp}
**Status**: {'Success' if success else 'Failed'}
**Latency**: {latency_ms}ms
**Models**: {model_info or 'N/A'}

---

## Input
{input_text[:2000]}{'...' if len(input_text) > 2000 else ''}

---

## Output
{output_text[:5000]}{'...' if len(output_text) > 5000 else ''}
"""

        properties = {
            "Name": {"title": [{"text": {"content": title}}]},
        }

        return self.create_page(
            parent_id=database_id,
            title=title,
            content=content,
            properties=properties,
            is_database=True,
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _extract_id(self, id_or_url: str) -> str:
        """Extract Notion ID from URL or return ID as-is."""
        # Handle URLs
        if "notion.so" in id_or_url or "notion.site" in id_or_url:
            # Extract ID from URL (last 32 chars before any query params)
            match = re.search(r"([a-f0-9]{32})", id_or_url.replace("-", ""))
            if match:
                raw_id = match.group(1)
                # Format as UUID
                return f"{raw_id[:8]}-{raw_id[8:12]}-{raw_id[12:16]}-{raw_id[16:20]}-{raw_id[20:]}"

        # Remove dashes if present and reformat
        clean_id = id_or_url.replace("-", "")
        if len(clean_id) == 32:
            return f"{clean_id[:8]}-{clean_id[8:12]}-{clean_id[12:16]}-{clean_id[16:20]}-{clean_id[20:]}"

        return id_or_url

    def _get_all_blocks(self, block_id: str) -> List[Dict[str, Any]]:
        """Get all child blocks of a page/block."""
        blocks = []
        has_more = True
        start_cursor = None

        while has_more:
            params = {"block_id": block_id}
            if start_cursor:
                params["start_cursor"] = start_cursor

            response = self.client.blocks.children.list(**params)
            blocks.extend(response.get("results", []))
            has_more = response.get("has_more", False)
            start_cursor = response.get("next_cursor")

        return blocks

    def _blocks_to_text(self, blocks: List[Dict[str, Any]]) -> str:
        """Convert Notion blocks to plain text."""
        lines = []

        for block in blocks:
            block_type = block.get("type", "")
            block_data = block.get(block_type, {})

            if block_type == "paragraph":
                text = self._extract_plain_text(block_data.get("rich_text", []))
                lines.append(text)

            elif block_type in ("heading_1", "heading_2", "heading_3"):
                text = self._extract_plain_text(block_data.get("rich_text", []))
                level = int(block_type[-1])
                lines.append(f"{'#' * level} {text}")

            elif block_type == "bulleted_list_item":
                text = self._extract_plain_text(block_data.get("rich_text", []))
                lines.append(f"- {text}")

            elif block_type == "numbered_list_item":
                text = self._extract_plain_text(block_data.get("rich_text", []))
                lines.append(f"1. {text}")

            elif block_type == "to_do":
                text = self._extract_plain_text(block_data.get("rich_text", []))
                checked = block_data.get("checked", False)
                checkbox = "[x]" if checked else "[ ]"
                lines.append(f"- {checkbox} {text}")

            elif block_type == "code":
                text = self._extract_plain_text(block_data.get("rich_text", []))
                language = block_data.get("language", "")
                lines.append(f"```{language}\n{text}\n```")

            elif block_type == "quote":
                text = self._extract_plain_text(block_data.get("rich_text", []))
                lines.append(f"> {text}")

            elif block_type == "divider":
                lines.append("---")

            elif block_type == "callout":
                text = self._extract_plain_text(block_data.get("rich_text", []))
                lines.append(f"> **Note:** {text}")

        return "\n\n".join(lines)

    def _text_to_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Convert plain text/markdown to Notion blocks."""
        blocks = []
        lines = text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].rstrip()

            if not line:
                i += 1
                continue

            # Headings
            if line.startswith("### "):
                blocks.append(self._create_heading_block(line[4:], 3))
            elif line.startswith("## "):
                blocks.append(self._create_heading_block(line[3:], 2))
            elif line.startswith("# "):
                blocks.append(self._create_heading_block(line[2:], 1))

            # Bullet points
            elif line.startswith("- [ ] "):
                blocks.append(self._create_todo_block(line[6:], False))
            elif line.startswith("- [x] "):
                blocks.append(self._create_todo_block(line[6:], True))
            elif line.startswith("- ") or line.startswith("* "):
                blocks.append(self._create_bullet_block(line[2:]))

            # Numbered lists
            elif re.match(r"^\d+\. ", line):
                text = re.sub(r"^\d+\. ", "", line)
                blocks.append(self._create_numbered_block(text))

            # Code blocks
            elif line.startswith("```"):
                language = line[3:].strip()
                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                blocks.append(self._create_code_block("\n".join(code_lines), language))

            # Quotes
            elif line.startswith("> "):
                blocks.append(self._create_quote_block(line[2:]))

            # Dividers
            elif line in ("---", "***", "___"):
                blocks.append({"type": "divider", "divider": {}})

            # Regular paragraphs
            else:
                blocks.append(self._create_paragraph_block(line))

            i += 1

        return blocks

    def _create_paragraph_block(self, text: str) -> Dict[str, Any]:
        """Create a paragraph block."""
        return {
            "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]},
        }

    def _create_heading_block(self, text: str, level: int) -> Dict[str, Any]:
        """Create a heading block."""
        heading_type = f"heading_{level}"
        return {
            "type": heading_type,
            heading_type: {"rich_text": [{"type": "text", "text": {"content": text}}]},
        }

    def _create_bullet_block(self, text: str) -> Dict[str, Any]:
        """Create a bulleted list item block."""
        return {
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            },
        }

    def _create_numbered_block(self, text: str) -> Dict[str, Any]:
        """Create a numbered list item block."""
        return {
            "type": "numbered_list_item",
            "numbered_list_item": {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            },
        }

    def _create_todo_block(self, text: str, checked: bool) -> Dict[str, Any]:
        """Create a to-do block."""
        return {
            "type": "to_do",
            "to_do": {
                "rich_text": [{"type": "text", "text": {"content": text}}],
                "checked": checked,
            },
        }

    def _create_code_block(self, code: str, language: str = "") -> Dict[str, Any]:
        """Create a code block."""
        return {
            "type": "code",
            "code": {
                "rich_text": [{"type": "text", "text": {"content": code}}],
                "language": language or "plain text",
            },
        }

    def _create_quote_block(self, text: str) -> Dict[str, Any]:
        """Create a quote block."""
        return {
            "type": "quote",
            "quote": {"rich_text": [{"type": "text", "text": {"content": text}}]},
        }

    def _extract_plain_text(self, rich_text: List[Dict[str, Any]]) -> str:
        """Extract plain text from Notion rich text array."""
        return "".join(item.get("plain_text", "") for item in rich_text)

    def _extract_title(self, properties: Dict[str, Any]) -> str:
        """Extract title from page properties."""
        # Try common title property names
        for key in ("Name", "Title", "title", "name"):
            if key in properties:
                prop = properties[key]
                if prop.get("type") == "title":
                    return self._extract_plain_text(prop.get("title", []))

        # Check all properties for title type
        for prop in properties.values():
            if prop.get("type") == "title":
                return self._extract_plain_text(prop.get("title", []))

        return "Untitled"


# Singleton instance
_notion_client: Optional[NotionClient] = None


def get_notion_client() -> NotionClient:
    """Get or create NotionClient singleton."""
    global _notion_client
    if _notion_client is None:
        _notion_client = NotionClient()
    return _notion_client
