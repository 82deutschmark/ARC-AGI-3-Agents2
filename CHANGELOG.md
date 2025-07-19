# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- **Windows Unicode Encoding Issue** (2025-01-18)
  - Fixed UnicodeEncodeError when logging AI assistant responses containing Unicode characters (like en-dash ‐)
  - Added UTF-8 encoding configuration for logging handlers on Windows systems
  - Improved cross-platform compatibility with fallback encoding handling
  - **Technical Details**: The issue occurred when AI responses contained Unicode characters that Windows' default cp1252 codec couldn't handle. The fix configures both stdout and file logging handlers to use UTF-8 encoding with error replacement.
  - **Modified Files**: `main.py` (logging configuration)
  - **Resolved Error**: `UnicodeEncodeError: 'charmap' codec can't encode character '\u2010'`

---
*Changes documented by Claude Sonnet 4* 