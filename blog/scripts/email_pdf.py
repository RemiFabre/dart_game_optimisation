"""Email a file attachment via the locally configured msmtp.

Usage:
    python blog/scripts/email_pdf.py <to-address> <subject> <body-file-or-"-"> <attachment> [more attachments...]

If the body argument is "-", reads body from stdin. Otherwise the argument is
a path to a UTF-8 file whose contents become the email body.

msmtp is invoked as a subprocess; auth/SMTP details come from ~/.msmtprc.
"""

from __future__ import annotations

import mimetypes
import subprocess
import sys
from email.message import EmailMessage
from pathlib import Path


def build_message(to_addr: str, subject: str, body: str, attachments: list[Path]) -> EmailMessage:
    msg = EmailMessage()
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)
    for path in attachments:
        ctype, encoding = mimetypes.guess_type(str(path))
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        with open(path, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype=maintype,
                subtype=subtype,
                filename=path.name,
            )
    return msg


def send(msg: EmailMessage, to_addr: str) -> None:
    proc = subprocess.run(
        ["msmtp", "-a", "default", "--", to_addr],
        input=msg.as_bytes(),
        capture_output=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", errors="replace"))
        sys.exit(proc.returncode)


def main() -> None:
    if len(sys.argv) < 5:
        sys.stderr.write(__doc__)
        sys.exit(2)
    to_addr = sys.argv[1]
    subject = sys.argv[2]
    body_arg = sys.argv[3]
    attachments = [Path(p) for p in sys.argv[4:]]
    for p in attachments:
        if not p.exists():
            sys.stderr.write(f"attachment not found: {p}\n")
            sys.exit(2)

    body = sys.stdin.read() if body_arg == "-" else Path(body_arg).read_text(encoding="utf-8")

    msg = build_message(to_addr, subject, body, attachments)
    send(msg, to_addr)
    print(f"sent: {subject} -> {to_addr} ({len(attachments)} attachment(s))")


if __name__ == "__main__":
    main()
