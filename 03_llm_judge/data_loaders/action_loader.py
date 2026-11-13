"""
Action document loader.

Reads local Conservation Evidence action text files from the directory
structure used by the student's codebase:
    <base_dir>/action_{id}_clean.txt

The base directory is configured at construction time; typically:
    action_data/background_key_messages/bg_km_all/

If a file is not found locally, the loader optionally falls back to fetching
the page from conservationevidence.com (requires network access).
"""

import logging
import os
import re
import time
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# Simple in-process cache so the same action isn't loaded multiple times
_action_cache: dict[str, Optional[str]] = {}


class ActionLoader:
    """
    Load Conservation Evidence action text by ID.

    Parameters
    ----------
    local_base_dir : str | None
        Path to the directory containing action_{id}_clean.txt files.
        If None, local loading is disabled.
    fetch_from_web : bool
        If True and local file is not found, fetch from CE website as fallback.
        Defaults to False.
    web_delay : float
        Seconds to wait between web requests (polite crawling).
    """

    def __init__(
        self,
        local_base_dir: Optional[str] = None,
        fetch_from_web: bool = False,
        web_delay: float = 1.0,
    ):
        self.local_base_dir = local_base_dir
        self.fetch_from_web = fetch_from_web
        self.web_delay = web_delay
        self._last_web_fetch: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_action_text(self, action_id: str) -> Optional[str]:
        """
        Return the text content for a given action ID, or None if unavailable.

        Tries (in order):
          1. In-process cache
          2. Local file  (if local_base_dir is set)
          3. CE website  (if fetch_from_web is True)
        """
        action_id = str(action_id).strip()
        if action_id in _action_cache:
            return _action_cache[action_id]

        text = None

        if self.local_base_dir:
            text = self._load_local(action_id)

        if text is None and self.fetch_from_web:
            text = self._fetch_web(action_id)

        if text is None:
            logger.warning(
                "Action text not available for action ID %s "
                "(local_base_dir=%s, fetch_from_web=%s)",
                action_id,
                self.local_base_dir,
                self.fetch_from_web,
            )

        _action_cache[action_id] = text
        return text

    def get_action_texts_combined(
        self,
        action_ids: list[str],
        separator: str = "\n\n---\n\n",
    ) -> Optional[str]:
        """
        Load multiple action documents and join them into a single string.

        Skips any IDs for which no text is available.
        Returns None if no documents could be loaded at all.
        """
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_ids = []
        for aid in action_ids:
            aid = str(aid).strip()
            if aid and aid not in seen:
                seen.add(aid)
                unique_ids.append(aid)

        texts = []
        for aid in unique_ids:
            text = self.get_action_text(aid)
            if text:
                texts.append(f"[Action ID: {aid}]\n{text}")

        if not texts:
            return None
        return separator.join(texts)

    def clear_cache(self):
        """Clear the in-process cache (useful between large runs)."""
        _action_cache.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_local(self, action_id: str) -> Optional[str]:
        filename = f"action_{action_id}_clean.txt"
        filepath = os.path.join(self.local_base_dir, filename)
        if not os.path.exists(filepath):
            logger.debug("Local file not found: %s", filepath)
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            logger.debug("Loaded action %s from local file.", action_id)
            return text
        except Exception as e:
            logger.error("Error reading local action file %s: %s", filepath, e)
            return None

    def _fetch_web(self, action_id: str) -> Optional[str]:
        """
        Fetch action page from conservationevidence.com.
        Extracts the key-messages / synopsis text from the HTML.
        """
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error(
                "requests and beautifulsoup4 are required for web fetching. "
                "Install them with: pip install requests beautifulsoup4"
            )
            return None

        # Polite rate limiting
        elapsed = time.time() - self._last_web_fetch
        if elapsed < self.web_delay:
            time.sleep(self.web_delay - elapsed)

        url = f"https://www.conservationevidence.com/actions/{action_id}"
        try:
            resp = requests.get(url, timeout=15)
            self._last_web_fetch = time.time()

            if resp.status_code != 200:
                logger.warning("HTTP %d for action %s", resp.status_code, action_id)
                return None

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract the main evidence text – try several selectors
            text_parts = []

            # Action title
            title_tag = soup.find("h1") or soup.find("h2")
            if title_tag:
                text_parts.append(f"Title: {title_tag.get_text(strip=True)}")

            # Key messages section
            for selector in [
                "div.key-messages",
                "section.key-messages",
                "div#key-messages",
                "div.synopsis",
            ]:
                section = soup.select_one(selector)
                if section:
                    text_parts.append(section.get_text(separator="\n", strip=True))
                    break

            # Evidence content / studies
            for selector in [
                "div.evidence-content",
                "div.action-text",
                "div#action-text",
                "article",
                "main",
            ]:
                section = soup.select_one(selector)
                if section:
                    text_parts.append(section.get_text(separator="\n", strip=True))
                    break

            if not text_parts:
                logger.warning("Could not extract text from CE page for action %s", action_id)
                return None

            full_text = "\n\n".join(text_parts)
            # Clean up excessive whitespace
            full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()
            logger.info("Fetched action %s from CE website.", action_id)
            return full_text

        except Exception as e:
            logger.error("Error fetching action %s from web: %s", action_id, e)
            return None
