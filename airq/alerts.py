"""Early-warning alerts for forecast Poor / Very Poor / Severe air.

The message is always built. It is emailed only when SMTP settings exist in the
environment; nothing is sent otherwise:

    AQI_SMTP_HOST, AQI_SMTP_PORT (default 587), AQI_SMTP_USER, AQI_SMTP_PASSWORD, AQI_ALERT_TO
"""
import os
import smtplib
from email.message import EmailMessage


def build_message(result: dict) -> str:
    return (
        f"Air quality alert for {result['city']} on {result['target_date']}\n\n"
        f"Forecast AQI: {result['predicted_aqi']:.0f} ({result['predicted_category']})\n"
        f"Today's AQI: {result['today_aqi']:.0f} ({result['today_category']})\n\n"
        f"Health advice: {result['advice']}\n"
        f"(Model: {result['model']})"
    )


def email_configured() -> bool:
    return all(os.environ.get(k) for k in ("AQI_SMTP_HOST", "AQI_SMTP_USER", "AQI_SMTP_PASSWORD", "AQI_ALERT_TO"))


def send_email(result: dict) -> bool:
    """Send the alert by email. Returns False if SMTP is not configured."""
    if not email_configured():
        return False
    msg = EmailMessage()
    msg["Subject"] = f"AQI alert: {result['city']} forecast {result['predicted_category']}"
    msg["From"] = os.environ["AQI_SMTP_USER"]
    msg["To"] = os.environ["AQI_ALERT_TO"]
    msg.set_content(build_message(result))
    with smtplib.SMTP(os.environ["AQI_SMTP_HOST"], int(os.environ.get("AQI_SMTP_PORT", 587))) as s:
        s.starttls()
        s.login(os.environ["AQI_SMTP_USER"], os.environ["AQI_SMTP_PASSWORD"])
        s.send_message(msg)
    return True
