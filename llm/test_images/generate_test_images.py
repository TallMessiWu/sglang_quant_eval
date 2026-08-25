#!/usr/bin/env python3
"""生成 curl.sh 测试用的两张占位图片（合成图形，非真实照片）。"""

from pathlib import Path

from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).parent


def make_outdoor_courtyard():
    img = Image.new("RGB", (640, 480), color=(135, 206, 235))  # 天空蓝
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 320, 640, 480], fill=(120, 180, 90))  # 草地
    draw.rectangle([180, 200, 460, 380], fill=(200, 170, 140))  # 建筑墙面
    draw.polygon([(160, 200), (320, 100), (480, 200)], fill=(150, 60, 50))  # 屋顶
    draw.rectangle([290, 280, 350, 380], fill=(90, 60, 40))  # 门
    draw.ellipse([500, 60, 560, 120], fill=(255, 220, 80))  # 太阳
    draw.ellipse([60, 340, 140, 400], fill=(70, 50, 40))  # 树干阴影占位
    draw.ellipse([40, 260, 160, 360], fill=(60, 140, 60))  # 树冠
    img.save(OUT_DIR / "outdoor-courtyard.png")


def make_indoor_kitchen():
    img = Image.new("RGB", (640, 480), color=(245, 235, 220))  # 墙面
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 340, 640, 480], fill=(160, 120, 90))  # 地板
    draw.rectangle([40, 200, 300, 340], fill=(210, 210, 215))  # 橱柜台面
    draw.rectangle([40, 200, 300, 220], fill=(180, 180, 185))  # 台面边缘
    draw.rectangle([350, 160, 470, 340], fill=(90, 90, 95))  # 冰箱
    draw.rectangle([360, 220, 460, 224], fill=(60, 60, 65))  # 冰箱把手线
    draw.rectangle([500, 260, 600, 340], fill=(70, 70, 75))  # 灶台
    for x in range(520, 590, 25):
        draw.ellipse([x, 275, x + 15, 290], fill=(30, 30, 30))  # 灶眼
    img.save(OUT_DIR / "indoor-kitchen.png")


if __name__ == "__main__":
    make_outdoor_courtyard()
    make_indoor_kitchen()
    print("已生成 outdoor-courtyard.png, indoor-kitchen.png")
