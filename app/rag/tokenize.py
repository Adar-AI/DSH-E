"""中文友好的轻量分词，供 SQLite FTS5 使用。

为什么需要它：FTS5 自带的 `unicode61` 分词器会把一整串连续汉字当成**一个** token，
「更换硒鼓的步骤」整句变成一个词，用户搜「硒鼓」永远匹配不到。这里把中文按
**二元（bigram）** 切分：'打印机报错' -> '打印 印机 机报 报错'，检索时同样切分，
即可获得可用的中文召回率，且不引入第三方分词依赖。

英文、数字、型号（E05、L3250）原样保留并转小写，保证报错码能精确命中。
"""

from __future__ import annotations

import re

# CJK 统一表意文字（含扩展 A 与兼容区）
_CJK_CLASS = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_CJK_RE = re.compile(f"[{_CJK_CLASS}]")
_TOKEN_RE = re.compile(f"[{_CJK_CLASS}]+|[A-Za-z0-9_]+")


def is_cjk(char: str) -> bool:
    """判断单个字符是否属于 CJK 表意文字。"""
    return bool(_CJK_RE.fullmatch(char))


def segment(text: str) -> str:
    """把文本切成检索用 token，空格连接。

    >>> segment('打印机报错E05')
    '打印 印机 机报 报错 e05'
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text or ""):
        piece = match.group(0)
        if _CJK_RE.match(piece):
            if len(piece) == 1:
                tokens.append(piece)
            else:
                tokens.extend(piece[i : i + 2] for i in range(len(piece) - 1))
        else:
            tokens.append(piece.lower())
    return " ".join(tokens)


def tokens_of(text: str) -> list[str]:
    """返回去重后的 token 列表（保持出现顺序）。"""
    seen: dict[str, None] = {}
    for token in segment(text).split():
        seen.setdefault(token, None)
    return list(seen)


def build_match_query(text: str) -> str:
    """把用户问题转成 FTS5 MATCH 表达式（token 之间 OR）。

    所有 token 都用双引号包裹，避免用户输入里的 `*`、`:`、`NEAR` 等被当成
    FTS5 语法。空问题返回空串，调用方据此跳过关键词检索。
    """
    tokens = tokens_of(text)
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in tokens)
