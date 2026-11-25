import hmac
from typing import Optional
from fastapi import Header, HTTPException, status
from app.core.config import settings


def verify_internal_secret(
    x_internal_secret: Optional[str] = Header(None, alias="X-Internal-Secret"),
) -> bool:
    """
    Xác thực request nội bộ từ Spring Boot Backend bằng header X-Internal-Secret.
    Ở môi trường development có thể bỏ qua nếu chưa truyền key, nhưng production bắt buộc khớp.
    """
    expected = (settings.INTERNAL_SECRET_KEY or "").strip()
    if settings.ENVIRONMENT.lower() != "production" and not x_internal_secret:
        return True

    if not x_internal_secret or not hmac.compare_digest(x_internal_secret.strip(), expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Internal-Secret header",
        )
    return True
