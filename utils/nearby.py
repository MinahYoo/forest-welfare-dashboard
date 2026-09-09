"""Distance-sorted nearby forest-welfare facilities from verified public coordinates.

facilities.json / sigungu_centroids.json are the same artifacts the 숲BTI web demo
uses: 전국휴양림표준데이터·치유의숲 현황(WGS84) and 시군구 행정경계 중심좌표.
No new model or score — only a great-circle distance from the chosen 시군구 centre.
"""
from __future__ import annotations
import json
import math
import re

from .preprocessing import ROOT

_FALLBACK = [36.5, 127.8]


def _load():
    fac = json.loads((ROOT / 'data/facilities.json').read_text(encoding='utf-8'))
    cent = json.loads((ROOT / 'data/sigungu_centroids.json').read_text(encoding='utf-8'))
    return fac, cent


def _haversine(a, b):
    radius = 6371.0
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    dlat = math.radians(b[0] - a[0])
    dlon = math.radians(b[1] - a[1])
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(h))


def origin_point(sido, sigungu_name, cent):
    """Return ([lat, lon], label) for the chosen region centre."""
    base = cent['기초']
    sido_c = cent['시도']
    if sigungu_name:
        key = f'{sido} {sigungu_name}'
        if key in base:
            return base[key], key
        merged = re.sub(r'시\S*구$', '시', sigungu_name)  # 수원시영통구 -> 수원시
        if f'{sido} {merged}' in base:
            return base[f'{sido} {merged}'], f'{sido} {merged}'
        if sido == '세종' and '세종특별자치시' in base:
            return base['세종특별자치시'], '세종특별자치시'
    return sido_c.get(sido, _FALLBACK), sido


def nearby_facilities(sido, sigungu_name=None, preferred_types=(), limit=6):
    """Nearest facilities to the region centre, recommended types shown first."""
    fac, cent = _load()
    origin, label = origin_point(sido, sigungu_name, cent)
    preferred = {t.replace(' ', '') for t in preferred_types}
    ranked = sorted(
        ({**f, '거리_km': _haversine(origin, [f['lat'], f['lon']])} for f in fac),
        key=lambda x: x['거리_km'],
    )[:25]
    ranked.sort(key=lambda x: (0 if x['type'].replace(' ', '') in preferred else 1, x['거리_km']))
    return ranked[:limit], label
