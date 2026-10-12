"""工单接口：列表、详情、状态流转、导出。"""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app import db as dbmod
from app.config import get_settings
from app.tickets.service import EDITABLE_FIELDS, STATUS_ORDER, TicketService

router = APIRouter(prefix="/api", tags=["tickets"])


def _with_service(handler):
    """每个请求一个连接 + 一个服务实例。"""

    def wrapper(*args, **kwargs):
        settings = get_settings()
        settings.ensure_dirs()
        conn = dbmod.connect(settings=settings)
        try:
            return handler(TicketService(conn), *args, **kwargs)
        finally:
            conn.close()

    return wrapper


class StatusUpdate(BaseModel):
    to_status: str = Field(..., description="目标状态：待处理 / 处理中 / 已解决 / 已关闭")
    note: str = Field("", description="变更备注")
    operator: str = Field("客服", description="操作人")


class FieldUpdate(BaseModel):
    fields: dict[str, str] = Field(
        ...,
        description="要修改的字段（customer_request / product_model / fault_phenomenon / "
        "error_info / tried_actions / urgency / suggestion）",
    )
    note: str = Field("", description="备注；留空则自动记录字段变化")
    operator: str = Field("客服", description="操作人")


@router.get("/tickets")
def list_tickets(
    status: str | None = Query(None, description="按状态筛选"),
    q: str | None = Query(None, description="关键字（工单号/诉求/型号/现象/报错/建议）"),
    conv_id: str | None = Query(None, description="按会话筛选"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    @_with_service
    def handler(service: TicketService):
        payload = service.list(status=status, q=q, conv_id=conv_id, limit=limit, offset=offset)
        payload["status_options"] = list(STATUS_ORDER)
        return payload

    return handler()


@router.get("/tickets/export.csv")
def export_tickets(
    status: str | None = Query(None),
    q: str | None = Query(None),
):
    """导出 CSV（UTF-8 BOM，Excel 可直接打开）。声明在 /tickets/{id} 之前避免路径冲突。"""

    @_with_service
    def handler(service: TicketService):
        return service.export_csv(status=status, q=q)

    content = handler()
    ascii_name = "tickets.csv"
    utf8_name = quote("工单导出.csv")
    return Response(
        content=content,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{utf8_name}'
            )
        },
    )


@router.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: int):
    @_with_service
    def handler(service: TicketService):
        ticket = service.get(ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail="工单不存在")
        ticket["events"] = service.events(ticket_id)
        return ticket

    return handler()


@router.patch("/tickets/{ticket_id}")
def update_ticket(ticket_id: int, payload: StatusUpdate):
    """状态流转。非法流转返回 400，并在响应里说明允许的目标状态。"""

    @_with_service
    def handler(service: TicketService):
        try:
            ticket = service.update_status(
                ticket_id,
                payload.to_status,
                note=payload.note,
                operator=payload.operator,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        ticket["events"] = service.events(ticket_id)
        return ticket

    return handler()


@router.patch("/tickets/{ticket_id}/fields")
def update_ticket_fields(ticket_id: int, payload: FieldUpdate):
    """人工修正工单字段（AI 抽取难免有偏差，人工可覆盖，改动记入时间线）。"""

    @_with_service
    def handler(service: TicketService):
        try:
            ticket = service.update_fields(
                ticket_id,
                payload.fields,
                note=payload.note,
                operator=payload.operator,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        ticket["events"] = service.events(ticket_id)
        ticket["editable_fields"] = list(EDITABLE_FIELDS)
        return ticket

    return handler()
