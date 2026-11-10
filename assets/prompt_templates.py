"""Curated, local-only Chinese prompt templates for the H3 Studio UI.

No model requests, remote APIs, databases, or user data are used here.
Templates are suggestions; they do not change any inference parameters.
"""
from __future__ import annotations

TEMPLATES = (
    {
        "id": "film-dolly",
        "title": "电影感推进镜头",
        "category": "电影镜头",
        "summary": "浅景深、柔和侧光、缓慢推进",
        "sample_subject": "一位站在雨后街道上的旅行者",
        "tags": ("电影感", "运镜", "写实"),
        "prompt": "{主体}。镜头从中景开始平稳缓慢向前推进，背景有自然的空间层次，柔和侧光勾勒轮廓，光影随镜头移动产生细微变化。主体动作自然连续，电影摄影质感，浅景深，细节清晰，画面稳定，无文字或水印。",
    },
    {
        "id": "film-orbit",
        "title": "环绕运镜",
        "category": "电影镜头",
        "summary": "环绕主体、保持焦点、动态视差",
        "sample_subject": "一辆停在海边公路上的复古摩托车",
        "tags": ("环绕", "镜头运动", "空间感"),
        "prompt": "以{主体}为画面焦点，摄影机沿平滑弧线缓慢环绕，始终保持主体清晰，前景与背景产生自然视差。环境光变化细腻，物体比例稳定，运动轨迹连贯，细节丰富，无字幕和水印。",
    },
    {
        "id": "film-closeup",
        "title": "细节特写",
        "category": "电影镜头",
        "summary": "微距特写、纹理与焦点转换",
        "sample_subject": "一片沾着晨露的绿色叶片",
        "tags": ("微距", "特写", "质感"),
        "prompt": "{主体}的特写镜头。摄影机非常缓慢地横向移动并进行柔和的焦点转换，突出表面纹理、细小反光与真实材质，背景自然虚化，光线温柔而有方向性，画面宁静精致，运动连续。",
    },
    {
        "id": "city-neon",
        "title": "雨夜霓虹城市",
        "category": "城市科幻",
        "summary": "赛博都市、雨水反射、街景推镜",
        "sample_subject": "一名撑着透明雨伞的行人",
        "tags": ("霓虹", "雨夜", "城市"),
        "prompt": "未来都市的雨夜，{主体}沿湿润的街道缓慢前行。霓虹灯在路面与建筑玻璃上形成流动倒影，远处车灯穿过薄雾，镜头平稳跟随主体向前移动。冷暖光交织，空气透视真实，电影级构图，画面干净，没有屏幕文字。",
    },
    {
        "id": "city-sunrise",
        "title": "城市清晨航拍",
        "category": "城市科幻",
        "summary": "黄金时刻、城市纵深、缓慢俯瞰",
        "sample_subject": "一座沿河展开的现代城市",
        "tags": ("航拍", "清晨", "建筑"),
        "prompt": "{主体}在日出后的金色晨光中苏醒。镜头从高处缓慢前移并轻微降低高度，展示道路、建筑与河流的空间关系，薄雾在楼宇间散开，细节丰富，色彩自然，镜头运动稳定平顺。",
    },
    {
        "id": "nature-forest",
        "title": "森林晨雾",
        "category": "自然风光",
        "summary": "漫射阳光、草木细节、轻柔跟拍",
        "sample_subject": "一只在森林里行走的梅花鹿",
        "tags": ("森林", "雾气", "野生动物"),
        "prompt": "清晨薄雾中的原始森林，{主体}缓慢穿过带着露水的蕨类植物，树冠间洒下柔和的光束，镜头以低机位平稳跟拍。树叶随微风轻轻摆动，远近景层次分明，真实自然纪录片质感。",
    },
    {
        "id": "nature-coast",
        "title": "海岸日落",
        "category": "自然风光",
        "summary": "暖色夕照、海浪节奏、广角景别",
        "sample_subject": "一座静静矗立在悬崖上的灯塔",
        "tags": ("海岸", "日落", "风景"),
        "prompt": "黄昏海岸，{主体}沐浴在温暖落日中，海浪有节奏地拍打礁石，海风吹动周围植物。摄影机以宽广构图缓慢横移，天空呈现丰富而自然的暖色渐变，画面宁静清晰，环境运动真实。",
    },
    {
        "id": "anime-adventure",
        "title": "动画冒险场景",
        "category": "动画创意",
        "summary": "轻快冒险、层次丰富的动画镜头",
        "sample_subject": "背着小书包的少年探险家",
        "tags": ("动画", "奇幻", "冒险"),
        "prompt": "精致二维动画风格，{主体}在充满奇妙植物和漂浮光点的小路上向前奔跑，镜头从侧前方轻快地跟随，背景具有细腻的分层运动，明亮配色与柔和手绘质感，角色动作连贯自然，欢乐而富有探索感。",
    },
    {
        "id": "anime-cozy",
        "title": "温暖治愈日常",
        "category": "动画创意",
        "summary": "温馨室内、柔和手绘、生活细节",
        "sample_subject": "一只趴在窗台晒太阳的橘猫",
        "tags": ("治愈", "动画", "日常"),
        "prompt": "温柔的手绘动画风格，{主体}待在阳光透过窗帘的安静房间里，微风拂动薄纱和室内绿植，镜头从中景缓慢推进，温暖低饱和色彩，细微表情和动作自然流畅，画面舒适轻松。",
    },
    {
        "id": "product-hero",
        "title": "产品视觉展示",
        "category": "产品展示",
        "summary": "高光轮廓、慢速转台、精致材质",
        "sample_subject": "一款简约设计的智能手表",
        "tags": ("产品", "商业", "材质"),
        "prompt": "专业商业产品摄影，{主体}放置在简洁的深色展示台上，摄影机缓慢环绕并略微推近，柔和的轮廓光勾勒外形，材质细节和反射真实，背景干净，构图居中，产品比例始终稳定，无品牌文字或水印。",
    },
    {
        "id": "product-lifestyle",
        "title": "生活方式广告",
        "category": "产品展示",
        "summary": "生活场景、柔光、自然使用动作",
        "sample_subject": "一杯刚冲泡好的手工咖啡",
        "tags": ("生活方式", "商业", "暖光"),
        "prompt": "明亮舒适的生活方式广告画面，{主体}置于自然光充足的木质桌面，窗边光线柔和，背景细节克制。镜头从略高角度缓慢推近，记录细腻的光影变化和物体材质，动作连续、色彩真实，画面简约高级。",
    },
    {
        "id": "documentary-human",
        "title": "人物纪录片",
        "category": "人物纪实",
        "summary": "自然表情、手持质感、环境叙事",
        "sample_subject": "一位正在制作陶器的手艺人",
        "tags": ("人物", "纪录片", "真实"),
        "prompt": "人物纪录片风格，{主体}在熟悉的工作环境中自然地进行手上的动作。镜头保持稳定但带有轻微自然手持质感，从半身景别缓缓靠近，捕捉真实的神情、工具与环境细节，柔和环境光，肤色自然，动作连贯。",
    },
)

CATEGORIES = ("全部", "电影镜头", "城市科幻", "自然风光", "动画创意", "产品展示", "人物纪实")
_BY_ID = {item["id"]: item for item in TEMPLATES}


def search_templates(category: str = "全部", keyword: str = "") -> list[dict]:
    query = (keyword or "").strip().casefold()
    result = []
    for item in TEMPLATES:
        if category not in ("全部", "", None) and item["category"] != category:
            continue
        haystack = " ".join(
            [item["title"], item["category"], item["summary"], *item["tags"], item["prompt"]]
        ).casefold()
        if query and query not in haystack:
            continue
        result.append(item)
    return result


def get_template(template_id: str | None) -> dict | None:
    return _BY_ID.get(template_id or "")


def template_choices(category: str = "全部", keyword: str = "") -> list[tuple[str, str]]:
    return [
        (f'{item["title"]}  ·  {item["summary"]}', item["id"])
        for item in search_templates(category, keyword)
    ]


def render_template(template_id: str | None, subject: str = "") -> str:
    item = get_template(template_id)
    if item is None:
        return ""
    value = (subject or "").strip() or item["sample_subject"]
    return item["prompt"].replace("{主体}", value)


def template_details(template_id: str | None) -> str:
    item = get_template(template_id)
    if item is None:
        return "暂无匹配模板，请更换分类或关键词。"
    return f'**{item["title"]}** · {item["category"]}  |  适用：{item["summary"]}  |  关键词：{" / ".join(item["tags"])}'
