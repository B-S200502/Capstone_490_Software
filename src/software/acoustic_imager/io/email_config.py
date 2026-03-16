"""
Email/SMTP config for sharing captures. Stored in a dedicated JSON file (not gallery cache).
"""

from __future__ import annotations

import json
import logging
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

# Default max total attachment size (bytes). Many providers limit to 25 MB.
SHARE_ATTACHMENT_LIMIT_BYTES = 25 * 1024 * 1024

logger = logging.getLogger(__name__)

EMAIL_CONFIG_FILENAME = "email_config.json"

# Provider presets: (smtp_host, port, use_tls)
SMTP_PRESETS = {
    "gmail": ("smtp.gmail.com", 587, True),
    "outlook": ("smtp.office365.com", 587, True),
    "yahoo": ("smtp.mail.yahoo.com", 587, True),
}


def get_config_path(output_dir: Path) -> Path:
    """Path to email_config.json (same parent as output_dir so it persists)."""
    return output_dir / EMAIL_CONFIG_FILENAME


def _default_provider_data(provider: str) -> dict[str, Any]:
    """Default form data for a provider."""
    base = {"email": "", "password": "", "default_to": ""}
    if provider == "other":
        base["smtp_host"] = ""
        base["smtp_port"] = 587
        base["use_tls"] = True
    return base


def load_config(output_dir: Path) -> dict[str, Any]:
    """Load full email config from JSON. Prefer load_provider_config for form data."""
    path = get_config_path(output_dir)
    if not path.exists():
        return {"providers": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"providers": {}}
    if not isinstance(data, dict):
        return {"providers": {}}
    # Migrate old flat format to per-provider
    if "providers" not in data and "email" in data:
        flat = data
        data = {"providers": {"gmail": {"email": (flat.get("email") or "").strip().lower(),
                                       "password": (flat.get("password") or "").strip(),
                                       "default_to": (flat.get("default_to") or "").strip().lower(),
                                       "smtp_host": (flat.get("smtp_host") or "").strip(),
                                       "smtp_port": int(flat.get("smtp_port") or 587),
                                       "use_tls": bool(flat.get("use_tls", True))}}}
        for p in ("outlook", "yahoo", "other"):
            data["providers"][p] = _default_provider_data(p)
    if "providers" not in data:
        data["providers"] = {}
    return data


def get_email_verified(output_dir: Path) -> bool:
    """True if a test email has been sent successfully (email configured and working)."""
    data = load_config(output_dir)
    return bool(data.get("email_verified"))


def set_email_verified(output_dir: Path, value: bool) -> None:
    """Set the email_verified flag (e.g. after successful test send)."""
    data = load_config(output_dir)
    data["email_verified"] = bool(value)
    save_config(output_dir, data)


def load_provider_config(output_dir: Path, provider: str) -> dict[str, Any]:
    """Load form data for one provider. Returns email, password, default_to; for 'other' also smtp_host, smtp_port, use_tls."""
    data = load_config(output_dir)
    providers = data.get("providers") or {}
    out = _default_provider_data(provider)
    if provider not in providers:
        return out
    p = providers[provider]
    for key in ("email", "password", "default_to"):
        if key in p and isinstance(p[key], str):
            out[key] = p[key]
    if provider == "other":
        if "smtp_host" in p and isinstance(p["smtp_host"], str):
            out["smtp_host"] = p["smtp_host"]
        if "smtp_port" in p and isinstance(p["smtp_port"], (int, float)):
            out["smtp_port"] = int(p["smtp_port"])
        if "use_tls" in p and isinstance(p["use_tls"], bool):
            out["use_tls"] = p["use_tls"]
    return out


def save_provider_config(output_dir: Path, provider: str, form_data: dict[str, Any]) -> None:
    """Save form data for one provider into the config file."""
    data = load_config(output_dir)
    if "providers" not in data:
        data["providers"] = {}
    data["providers"][provider] = {
        "email": (form_data.get("email") or "").strip().lower(),
        "password": (form_data.get("password") or "").strip(),
        "default_to": (form_data.get("default_to") or "").strip().lower(),
    }
    if provider == "other":
        data["providers"][provider]["smtp_host"] = (form_data.get("smtp_host") or "").strip()
        data["providers"][provider]["smtp_port"] = int(form_data.get("smtp_port") or 587)
        data["providers"][provider]["use_tls"] = bool(form_data.get("use_tls", True))
    save_config(output_dir, data)


def save_config(output_dir: Path, config: dict[str, Any]) -> None:
    """Save email config to JSON. Prefer chmod 600 on the file for password safety."""
    path = get_config_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def get_smtp_params(provider: str, config: dict[str, Any]) -> tuple[str, int, bool]:
    """Return (host, port, use_tls) for the given provider. config may be full or one provider's data."""
    if provider in SMTP_PRESETS:
        return SMTP_PRESETS[provider]
    if "providers" in config and provider in config["providers"]:
        p = config["providers"][provider]
    else:
        p = config
    host = (p.get("smtp_host") or "").strip() or "smtp.gmail.com"
    port = int(p.get("smtp_port") or 587)
    use_tls = bool(p.get("use_tls", True))
    return host, port, use_tls


def send_test_email(provider: str, form_data: dict[str, Any]) -> tuple[bool, str]:
    """
    Send a test email using the given provider and form data.
    Returns (success, message). Message is short for UI (e.g. "Sent!" or error summary).
    """
    email = (form_data.get("email") or "").strip()
    password = (form_data.get("password") or "").strip()
    if not email or not password:
        logger.warning("Send test email: email or password missing")
        return False, "Email and password required"
    host, port, use_tls = get_smtp_params(provider, form_data)
    if provider == "other" and not (form_data.get("smtp_host") or "").strip():
        logger.warning("Send test email: SMTP host missing for provider 'other'")
        return False, "SMTP host required"
    logger.info("Send test email: connecting to %s:%s (TLS=%s) as %s", host, port, use_tls, email)
    try:
        if use_tls:
            server = smtplib.SMTP(host, port, timeout=15)
            server.starttls()
        else:
            server = smtplib.SMTP(host, port, timeout=15)
        server.login(email, password)
        msg = MIMEText("Test email from Acoustic Imager. Your email settings are working.")
        msg["Subject"] = "Acoustic Imager – test"
        msg["From"] = email
        msg["To"] = email
        server.sendmail(email, [email], msg.as_string())
        server.quit()
        logger.info("Send test email: sent successfully to %s", email)
        return True, "Sent!"
    except smtplib.SMTPAuthenticationError as e:
        logger.error("Send test email: SMTP authentication failed: %s", e, exc_info=True)
        # Gmail (and some others) require an App Password, not the normal account password
        resp = (e.args[1] if len(e.args) > 1 else b"") or b""
        if isinstance(resp, bytes):
            resp = resp.decode("utf-8", errors="replace").lower()
        else:
            resp = str(resp).lower()
        if "application-specific password" in resp or "app password" in resp or "invalidsecondfactor" in resp:
            return False, "Use App Password (Google)"
        return False, "Login failed"
    except smtplib.SMTPException as e:
        logger.error("Send test email: SMTP error: %s", e, exc_info=True)
        return False, str(e)[:40] or "SMTP error"
    except OSError as e:
        logger.error("Send test email: connection error: %s", e, exc_info=True)
        return False, str(e)[:40] or "Connection error"


def _get_first_configured_provider(output_dir: Path) -> Tuple[Optional[str], Optional[dict]]:
    """Return (provider, provider_config) for the first provider with email+password, else (None, None)."""
    data = load_config(output_dir)
    providers = data.get("providers") or {}
    for name in ("gmail", "outlook", "yahoo", "other"):
        if name not in providers:
            continue
        p = providers[name]
        if not (p.get("email") or "").strip() or not (p.get("password") or "").strip():
            continue
        if name == "other" and not (p.get("smtp_host") or "").strip():
            continue
        return name, p
    return None, None


def get_share_recipient(output_dir: Path) -> str:
    """Return the default recipient (default_to or from address) for share, or empty string."""
    _, cfg = _get_first_configured_provider(output_dir)
    if not cfg:
        return ""
    to = (cfg.get("default_to") or "").strip()
    if to:
        return to
    return (cfg.get("email") or "").strip()


def send_share_email(
    output_dir: Path,
    file_paths: List[Path],
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[bool, str, dict]:
    """
    Send an email with the given files as attachments using the first configured provider.
    progress_callback(progress: 0.0-1.0, phase: str) is called during preparation and send.
    Returns (success, message_for_ui, details). details has to_email, n_images, n_videos on success.
    """
    def _progress(p: float, phase: str) -> None:
        if progress_callback:
            progress_callback(p, phase)

    provider, cfg = _get_first_configured_provider(output_dir)
    if not provider or not cfg:
        return False, "Email not configured", {}
    email = (cfg.get("email") or "").strip()
    password = (cfg.get("password") or "").strip()
    to_email = (cfg.get("default_to") or "").strip() or email
    host, port, use_tls = get_smtp_params(provider, cfg)
    n_images = sum(1 for p in file_paths if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    n_videos = sum(1 for p in file_paths if p.suffix.lower() in (".mp4", ".avi", ".webm", ".mov"))
    try:
        _progress(0.0, "preparing")
        msg = MIMEMultipart()
        msg["Subject"] = "Acoustic Imager – shared capture(s)"
        msg["From"] = email
        msg["To"] = to_email
        msg.attach(MIMEText(f"Shared {len(file_paths)} file(s) from Acoustic Imager.", "plain"))
        valid_paths = [p for p in file_paths if p.exists()]
        n = max(1, len(valid_paths))
        for i, path in enumerate(valid_paths):
            with open(path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment", filename=path.name)
            msg.attach(part)
            _progress((i + 1) / n * 0.7, "preparing")  # 0–70% for attachment prep
        _progress(0.75, "sending")
        if use_tls:
            server = smtplib.SMTP(host, port, timeout=30)
            server.starttls()
        else:
            server = smtplib.SMTP(host, port, timeout=30)
        server.login(email, password)
        _progress(0.9, "sending")
        server.sendmail(email, [to_email], msg.as_string())
        server.quit()
        _progress(1.0, "sending")
        logger.info("Share email sent to %s: %s files", to_email, len(file_paths))
        return True, "Sent!", {"to_email": to_email, "n_images": n_images, "n_videos": n_videos}
    except smtplib.SMTPAuthenticationError as e:
        logger.error("Share email: SMTP auth failed: %s", e, exc_info=True)
        return False, "Login failed", {}
    except smtplib.SMTPException as e:
        logger.error("Share email: SMTP error: %s", e, exc_info=True)
        return False, str(e)[:50] or "SMTP error", {}
    except OSError as e:
        logger.error("Share email: connection error: %s", e, exc_info=True)
        return False, str(e)[:50] or "Connection error", {}
