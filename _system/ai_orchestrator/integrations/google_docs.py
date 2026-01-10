"""
Google Docs integration for AI Orchestrator.
Provides read/write access to Google Docs.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Scopes required for Google Docs access
SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive.file",
]

# Default paths
DEFAULT_CREDENTIALS_PATH = (
    Path(__file__).parent.parent.parent / "google_credentials.json"
)
DEFAULT_TOKEN_PATH = Path(__file__).parent.parent.parent / "google_token.json"


class GoogleDocsClient:
    """
    Client for Google Docs API operations.

    Features:
    - Read document content
    - Create new documents
    - Update existing documents
    - Append content to documents
    """

    def __init__(
        self,
        credentials_path: Optional[Path] = None,
        token_path: Optional[Path] = None,
    ):
        """
        Initialize Google Docs client.

        Args:
            credentials_path: Path to OAuth credentials JSON file
            token_path: Path to store/load OAuth token
        """
        self.credentials_path = credentials_path or DEFAULT_CREDENTIALS_PATH
        self.token_path = token_path or DEFAULT_TOKEN_PATH

        self.creds = self._get_credentials()
        self.docs_service = build("docs", "v1", credentials=self.creds)
        self.drive_service = build("drive", "v3", credentials=self.creds)

    def _get_credentials(self) -> Credentials:
        """Get or refresh OAuth credentials."""
        creds = None

        # Load existing token
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)

        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self.credentials_path.exists():
                    raise FileNotFoundError(
                        f"Google credentials file not found: {self.credentials_path}"
                    )

                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save token for next run
            with open(self.token_path, "w") as token:
                token.write(creds.to_json())

        return creds

    # =========================================================================
    # Document Operations
    # =========================================================================

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Get a Google Doc by ID.

        Args:
            doc_id: Document ID or URL

        Returns:
            Document data including content as plain text
        """
        doc_id = self._extract_doc_id(doc_id)

        try:
            doc = self.docs_service.documents().get(documentId=doc_id).execute()

            # Extract text content
            content = self._extract_text(doc)

            return {
                "id": doc["documentId"],
                "title": doc.get("title", "Untitled"),
                "content": content,
                "url": f"https://docs.google.com/document/d/{doc['documentId']}/edit",
            }

        except HttpError as e:
            raise RuntimeError(f"Failed to get document: {e}")

    def get_document_content(self, doc_id: str) -> str:
        """
        Get just the text content of a document.

        Args:
            doc_id: Document ID or URL

        Returns:
            Plain text content
        """
        doc = self.get_document(doc_id)
        return doc["content"]

    def create_document(
        self,
        title: str,
        content: str = "",
        folder_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new Google Doc.

        Args:
            title: Document title
            content: Initial content
            folder_id: Google Drive folder ID (optional)

        Returns:
            Created document data
        """
        try:
            # Create empty document
            doc = self.docs_service.documents().create(body={"title": title}).execute()
            doc_id = doc["documentId"]

            # Add content if provided
            if content:
                self._insert_text(doc_id, content)

            # Move to folder if specified
            if folder_id:
                self._move_to_folder(doc_id, folder_id)

            return {
                "id": doc_id,
                "title": title,
                "url": f"https://docs.google.com/document/d/{doc_id}/edit",
            }

        except HttpError as e:
            raise RuntimeError(f"Failed to create document: {e}")

    def update_document(self, doc_id: str, content: str) -> Dict[str, Any]:
        """
        Replace document content entirely.

        Args:
            doc_id: Document ID or URL
            content: New content

        Returns:
            Updated document data
        """
        doc_id = self._extract_doc_id(doc_id)

        try:
            # Get current document to find content length
            doc = self.docs_service.documents().get(documentId=doc_id).execute()

            # Find the end index of current content
            body_content = doc.get("body", {}).get("content", [])
            end_index = 1
            for element in body_content:
                if "endIndex" in element:
                    end_index = max(end_index, element["endIndex"])

            # Build requests to clear and insert
            requests = []

            # Delete existing content (if any)
            if end_index > 2:
                requests.append(
                    {
                        "deleteContentRange": {
                            "range": {
                                "startIndex": 1,
                                "endIndex": end_index - 1,
                            }
                        }
                    }
                )

            # Insert new content
            if content:
                requests.append(
                    {"insertText": {"location": {"index": 1}, "text": content}}
                )

            # Execute batch update
            if requests:
                self.docs_service.documents().batchUpdate(
                    documentId=doc_id, body={"requests": requests}
                ).execute()

            return {
                "id": doc_id,
                "url": f"https://docs.google.com/document/d/{doc_id}/edit",
                "updated": True,
            }

        except HttpError as e:
            raise RuntimeError(f"Failed to update document: {e}")

    def append_to_document(self, doc_id: str, content: str) -> Dict[str, Any]:
        """
        Append content to the end of a document.

        Args:
            doc_id: Document ID or URL
            content: Content to append

        Returns:
            Updated document data
        """
        doc_id = self._extract_doc_id(doc_id)

        try:
            # Get current document to find end index
            doc = self.docs_service.documents().get(documentId=doc_id).execute()

            body_content = doc.get("body", {}).get("content", [])
            end_index = 1
            for element in body_content:
                if "endIndex" in element:
                    end_index = max(end_index, element["endIndex"])

            # Insert at end (before the final newline)
            insert_index = max(1, end_index - 1)

            # Add newlines before new content for separation
            text_to_insert = f"\n\n{content}"

            requests = [
                {
                    "insertText": {
                        "location": {"index": insert_index},
                        "text": text_to_insert,
                    }
                }
            ]

            self.docs_service.documents().batchUpdate(
                documentId=doc_id, body={"requests": requests}
            ).execute()

            return {
                "id": doc_id,
                "url": f"https://docs.google.com/document/d/{doc_id}/edit",
                "appended": True,
            }

        except HttpError as e:
            raise RuntimeError(f"Failed to append to document: {e}")

    def list_documents(
        self, folder_id: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        List Google Docs (optionally in a specific folder).

        Args:
            folder_id: Optional folder ID to list from
            limit: Maximum results

        Returns:
            List of document metadata
        """
        try:
            query = "mimeType='application/vnd.google-apps.document'"
            if folder_id:
                query += f" and '{folder_id}' in parents"

            results = (
                self.drive_service.files()
                .list(
                    q=query,
                    pageSize=limit,
                    fields="files(id, name, modifiedTime, webViewLink)",
                )
                .execute()
            )

            documents = []
            for file in results.get("files", []):
                documents.append(
                    {
                        "id": file["id"],
                        "title": file["name"],
                        "modified_time": file.get("modifiedTime", ""),
                        "url": file.get("webViewLink", ""),
                    }
                )

            return documents

        except HttpError as e:
            raise RuntimeError(f"Failed to list documents: {e}")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _extract_doc_id(self, doc_id_or_url: str) -> str:
        """Extract document ID from URL or return ID as-is."""
        # Handle Google Docs URLs
        patterns = [
            r"/document/d/([a-zA-Z0-9-_]+)",
            r"docs\.google\.com/.*?([a-zA-Z0-9-_]{20,})",
        ]

        for pattern in patterns:
            match = re.search(pattern, doc_id_or_url)
            if match:
                return match.group(1)

        return doc_id_or_url

    def _extract_text(self, doc: Dict[str, Any]) -> str:
        """Extract plain text from a Google Doc."""
        text_parts = []

        body = doc.get("body", {})
        content = body.get("content", [])

        for element in content:
            if "paragraph" in element:
                paragraph = element["paragraph"]
                for elem in paragraph.get("elements", []):
                    if "textRun" in elem:
                        text_parts.append(elem["textRun"].get("content", ""))

        return "".join(text_parts)

    def _insert_text(self, doc_id: str, text: str):
        """Insert text at the beginning of a document."""
        requests = [{"insertText": {"location": {"index": 1}, "text": text}}]

        self.docs_service.documents().batchUpdate(
            documentId=doc_id, body={"requests": requests}
        ).execute()

    def _move_to_folder(self, file_id: str, folder_id: str):
        """Move a file to a specific folder."""
        # Get current parents
        file = (
            self.drive_service.files().get(fileId=file_id, fields="parents").execute()
        )
        previous_parents = ",".join(file.get("parents", []))

        # Move to new folder
        self.drive_service.files().update(
            fileId=file_id,
            addParents=folder_id,
            removeParents=previous_parents,
            fields="id, parents",
        ).execute()


# Singleton instance
_google_docs_client: Optional[GoogleDocsClient] = None


def get_google_docs_client() -> GoogleDocsClient:
    """Get or create GoogleDocsClient singleton."""
    global _google_docs_client
    if _google_docs_client is None:
        _google_docs_client = GoogleDocsClient()
    return _google_docs_client


def is_google_docs_configured() -> bool:
    """Check if Google Docs credentials are available."""
    return DEFAULT_CREDENTIALS_PATH.exists()
