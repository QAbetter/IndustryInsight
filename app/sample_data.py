"""生成离线演示语料：一份 PDF 研报、一份 Word 研报、一份 Markdown 政策/风险简报。

所有数字均为演示用合成数据，仅用于验证 RAG 与多 Agent 流水线，不构成真实研究结论。
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("insight.sample")

_SENTENCES = {
    "market": "中汽协数据显示，2024 年中国新能源汽车销量达到 1,286.6 万辆，同比增长 35.5%，"
    "渗透率提升至 40.9%。2020-2024 年销量复合增长率约为 42%，行业增长动力由政策补贴转向产品供给与消费需求。"
    "2024 年新能源汽车出口 128.4 万辆，同比增长 6.7%，增速较上一年明显回落。",
    "battery": "动力电池装机量达到 548.4 GWh，同比增长 41.0%。宁德时代全球动力电池市占率约 37.9%，"
    "比亚迪弗迪电池约 16.4%，LG 新能源约 10.8%，中创新航约 6.9%，亿纬锂能约 4.5%。",
    "charging": "截至 2024 年底，全国充电基础设施累计数量达到 1,281.8 万台，同比增长 49.1%；"
    "其中公共充电桩保有量 357.9 万台，同比增长 31.0%，私人充电桩约 923.9 万台。",
    "profit": "价格战向整车与电池环节传导。样本整车企业平均毛利率约 15.6%，较 2023 年下降约 1.2 个百分点；"
    "宁德时代毛利率约 24.4%，仍显著高于整车环节。2024 年汽车行业整体利润率约 4.3%，处于历史低位。",
    "policy": "按照现行政策，2024-2025 年新能源汽车免征车辆购置税，每辆免税额不超过 3 万元；"
    "2026-2027 年减半征收，每辆减税额不超过 1.5 万元。以旧换新补贴与双积分政策继续支撑需求，"
    "但补贴退坡后部分需求可能被提前透支。",
    "risk": "主要风险包括价格战压缩利润、补贴退坡、欧盟关税与本地化要求、原材料价格波动，"
    "以及充电基础设施区域分布不均。欧盟对中国电动车加征最高约 35.3% 的关税，出口企业面临合规与成本压力。",
    "outlook": "机构预计 2025 年中国新能源汽车销量达到 1,550-1,600 万辆，同比增速回落至 20%-25%，"
    "渗透率继续提升。出口与海外本地化建厂将成为第二增长曲线。",
}


def _register_pdf_font() -> str:
    from reportlab.pdfbase import pdfmetrics

    fonts_dir = Path("C:/Windows/Fonts")
    for filename in ("msyh.ttc", "simsun.ttc"):
        path = fonts_dir / filename
        if not path.exists():
            continue
        try:
            from reportlab.pdfbase.ttfonts import TTFont

            pdfmetrics.registerFont(TTFont("CN", str(path), subfontIndex=0))
            return "CN"
        except Exception as exc:  # noqa: BLE001
            logger.debug("注册字体 %s 失败：%s", path, exc)
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    return "STSong-Light"


def _build_pdf(path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    font = _register_pdf_font()
    title_style = ParagraphStyle(
        "TitleCN", fontName=font, fontSize=20, leading=28, spaceAfter=10, textColor=colors.HexColor("#0f1f3d")
    )
    h1 = ParagraphStyle(
        "H1CN", fontName=font, fontSize=14, leading=20, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#1f5eff")
    )
    body = ParagraphStyle(
        "BodyCN", fontName=font, fontSize=10.5, leading=19, firstLineIndent=21, alignment=TA_JUSTIFY
    )
    small = ParagraphStyle("SmallCN", fontName=font, fontSize=9, leading=14, textColor=colors.HexColor("#5f6b7a"))

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title="2024年中国新能源汽车行业研究报告（演示数据）",
        author="IndustryInsight Demo Corpus",
    )
    story = [
        Paragraph("2024 年中国新能源汽车行业研究报告", title_style),
        Paragraph("演示语料 · 合成数据 · 仅用于工程验证", small),
        Spacer(1, 6 * mm),
        Paragraph("一、核心结论", h1),
        Paragraph(_SENTENCES["market"], body),
        Paragraph(_SENTENCES["battery"], body),
        Paragraph(_SENTENCES["charging"], body),
        Spacer(1, 4 * mm),
        Paragraph("二、关键指标", h1),
    ]
    rows = [
        ["指标", "2024 年数值", "同比/口径"],
        ["新能源汽车销量", "1,286.6 万辆", "+35.5%"],
        ["渗透率", "40.9%", "批发口径"],
        ["出口量", "128.4 万辆", "+6.7%"],
        ["动力电池装机量", "548.4 GWh", "+41.0%"],
        ["公共充电桩", "357.9 万台", "+31.0%"],
    ]
    table_data = [[Paragraph(cell, small) for cell in row] for row in rows]
    table = Table(table_data, colWidths=[55 * mm, 55 * mm, 50 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3ff")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7deea")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(table)
    story.append(PageBreak())

    story += [
        Paragraph("三、产业链与盈利", h1),
        Paragraph(_SENTENCES["profit"], body),
        Paragraph(
            "产业链上游为锂、镍等资源与正负极、隔膜、电解液材料；中游为动力电池、电机、电控与热管理；"
            "下游为整车制造、充电基础设施、后市场服务与电池回收。2024 年碳酸锂价格较 2023 年高点明显回落，"
            "缓解了电池成本压力，但也带来材料企业库存减值。",
            body,
        ),
        Paragraph("四、竞争格局概览", h1),
        Paragraph(
            "整车端呈现“一超多强”格局。比亚迪凭借垂直整合与规模效应保持领先，2024 年新能源汽车市场份额约 33.2%；"
            "特斯拉中国份额约 6.0%；吉利、长安、奇瑞通过多品牌与混动路线追赶；理想、蔚来、小鹏在 20 万元以上市场"
            "以智能化与服务差异化竞争。",
            body,
        ),
        Paragraph("五、政策与风险", h1),
        Paragraph(_SENTENCES["policy"], body),
        Paragraph(_SENTENCES["risk"], body),
        Paragraph("六、2025 年展望", h1),
        Paragraph(_SENTENCES["outlook"], body),
        Spacer(1, 6 * mm),
        Paragraph(
            "免责声明：本文档为演示用合成语料，数字经过简化处理，不构成任何投资建议。",
            small,
        ),
    ]
    doc.build(story)


def _build_docx(path: Path) -> None:
    from docx import Document
    from docx.oxml.ns import qn
    from docx.shared import Pt

    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Microsoft YaHei"
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

    document.add_heading("2024 年中国新能源汽车竞争格局与动力电池研究（演示数据）", level=1)
    document.add_paragraph(
        "本报告为 IndustryInsight 离线演示语料，数据为合成数据，仅用于验证 PDF/Word 解析与向量检索效果。"
    )
    document.add_heading("一、整车竞争格局", level=2)
    document.add_paragraph(
        "2024 年中国新能源汽车市场呈现“一超多强”格局。比亚迪以 33.2% 的份额保持第一，"
        "特斯拉中国份额约 6.0%，吉利汽车约 7.8%，长安汽车约 6.5%，奇瑞汽车约 5.9%。"
        "理想、蔚来、小鹏份额分别约 4.8%、3.8% 和 3.5%，在 20 万元以上市场形成差异化竞争。"
    )
    document.add_paragraph(
        "价格战使整车企业利润率承压。样本整车企业平均毛利率约 15.6%，较 2023 年下降约 1.2 个百分点，"
        "行业整体利润率约 4.3%。头部企业通过规模效应、供应链降本与海外市场维持盈利韧性。"
    )
    table = document.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    headers = table.rows[0].cells
    headers[0].text = "企业"
    headers[1].text = "2024 年份额"
    headers[2].text = "备注"
    rows = [
        ("比亚迪", "33.2%", "规模领先，垂直整合"),
        ("特斯拉中国", "6.0%", "高端纯电，面临本土竞争"),
        ("吉利汽车", "7.8%", "多品牌与混动路线"),
        ("长安汽车", "6.5%", "深蓝/阿维塔双线"),
        ("奇瑞汽车", "5.9%", "出口占比高"),
        ("理想汽车", "4.8%", "增程+家庭定位"),
        ("蔚来", "3.8%", "换电与服务"),
        ("小鹏", "3.5%", "智能化标签"),
    ]
    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            cells[index].text = value

    document.add_heading("二、动力电池竞争格局", level=2)
    document.add_paragraph(
        "2024 年全球动力电池装机量约 894.4 GWh，同比增长 27.2%；中国动力电池装机量 548.4 GWh，同比增长 41.0%。"
        "宁德时代全球市占率约 37.9%，比亚迪弗迪电池约 16.4%，LG 新能源约 10.8%，中创新航约 6.9%，"
        "亿纬锂能约 4.5%。头部企业通过长协、合资建厂与海外产能锁定客户。"
    )
    battery_table = document.add_table(rows=1, cols=2)
    battery_table.style = "Table Grid"
    battery_table.rows[0].cells[0].text = "电池企业"
    battery_table.rows[0].cells[1].text = "全球市占率"
    for name, share in [
        ("宁德时代", "37.9%"),
        ("比亚迪弗迪电池", "16.4%"),
        ("LG 新能源", "10.8%"),
        ("中创新航", "6.9%"),
        ("亿纬锂能", "4.5%"),
    ]:
        cells = battery_table.add_row().cells
        cells[0].text = name
        cells[1].text = share

    document.add_heading("三、产业链与盈利", level=2)
    document.add_paragraph(
        "上游碳酸锂价格回落缓解电池成本压力，但材料企业面临库存减值；中游电池环节受益于装机增长，"
        "宁德时代毛利率约 24.4%，显著高于整车环节；下游整车与充电服务仍在补短板。"
        "2024 年全国充电基础设施累计 1,281.8 万台，其中公共充电桩 357.9 万台。"
    )
    document.save(str(path))


def _build_markdown(path: Path) -> None:
    content = f"""# 新能源汽车政策、风险与 2025 年展望（演示数据）

> 演示语料，合成数据，仅用于工程验证。

## 一、政策环境

{_SENTENCES['policy']}

## 二、需求与出口

{_SENTENCES['market']}

{_SENTENCES['outlook']}

## 三、风险清单

{_SENTENCES['risk']}

## 四、充电基础设施

{_SENTENCES['charging']}

## 五、关键数据表

| 指标 | 2024 年 | 同比 |
| --- | --- | --- |
| 新能源汽车销量 | 1,286.6 万辆 | +35.5% |
| 渗透率 | 40.9% | 批发口径 |
| 出口量 | 128.4 万辆 | +6.7% |
| 动力电池装机量 | 548.4 GWh | +41.0% |
| 公共充电桩 | 357.9 万台 | +31.0% |
| 充电基础设施总量 | 1,281.8 万台 | +49.1% |
| 汽车行业利润率 | 4.3% | 历史低位 |

## 六、口径说明

本语料中的销量采用批发口径，渗透率为新能源汽车销量占汽车总销量比例；出口数据单独统计，不与国内零售直接相加。
"""
    path.write_text(content, encoding="utf-8")


def ensure_demo_corpus(root: Path | str) -> list[Path]:
    root = Path(root)
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    targets = {
        raw / "nev_2024_industry_report.pdf": _build_pdf,
        raw / "nev_2024_competition.docx": _build_docx,
        raw / "nev_policy_risk.md": _build_markdown,
    }
    created: list[Path] = []
    for path, builder in targets.items():
        if path.exists():
            continue
        builder(path)
        created.append(path)
        logger.info("已生成演示语料：%s", path)
    return created
