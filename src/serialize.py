import json
PRIORITY_KEYS = ['бренд', 'бренд в одежде и обуви', 'торговая марка', 'тип', 'модель', 'артикул', 'партномер (артикул производителя)', 'артикул производителя', 'oem-номер', 'oe-код', 'цвет', 'цвет товара', 'название цвета', 'размер', 'российский размер', 'объем, мл', 'объем', 'вес товара, г', 'состав', 'материал', 'количество, штук', 'единиц в одном товаре']

def make_text(name, category, attrs_json, max_attr_chars=350):
    try:
        d = json.loads(attrs_json)
        d = {str(k).strip().lower(): str(v).strip() for (k, v) in d.items()}
    except Exception:
        d = {}
    (parts, used) = ([], set())
    for k in PRIORITY_KEYS:
        if k in d and d[k] and (d[k].lower() not in ('нет', '-')):
            parts.append(f'{k}={d[k][:60]}')
            used.add(k)
    for (k, v) in d.items():
        if k not in used and v:
            parts.append(f'{k}={v[:60]}')
    attr = '; '.join(parts)[:max_attr_chars]
    return f'{category} | {name} | {attr}'
