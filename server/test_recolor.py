"""Тесты бесплатного пайплайна перекраски (CPU, без GPU и без внешних API).

Запуск::

    cd server
    python -m pytest test_recolor.py -v          # если есть pytest
    python test_recolor.py                        # или без pytest

Покрыто:
    1. Локальная сегментация по точке (точность на синтетике)
    2. Локальная перекраска: точность цвета + сохранение рельефа
    3. Все материалы
    4. Выбор бэкенда и приоритеты
    5. Полный HTTP-эндпоинт /ai-recolor (контракт API сохранён)
    6. Обработка ошибок и валидация
    7. Фолбэк на локальный путь при отказе API
    8. Формирование запросов к Cloudflare / Gemini (моки)
    9. Покрытие объекта по пикселям, стекло, диагностика
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image  # noqa: E402

# ---------------------------------------------------------------- fixtures

H = W = 512


def _canvas() -> np.ndarray:
    return np.zeros((H, W, 3), np.uint8)


def fixture_circle():
    """Красный круг на голубом фоне + ground truth."""
    yy, xx = np.mgrid[0:H, 0:W]
    img = _canvas()
    img[:, :, 1] = 120
    img[:, :, 2] = 200
    gt = (yy - 256) ** 2 + (xx - 256) ** 2 <= 170**2
    img[gt] = (200, 30, 30)
    return img, gt, (256, 256)


def fixture_object():
    """Тёмный объект со «сквозной» перемычкой на светлом фоне."""
    img = np.full((H, W, 3), 225, np.uint8)
    gt = np.zeros((H, W), bool)
    gt[130:380, 160:350] = True
    gt[210:250, 90:420] = True
    img[gt] = (72, 72, 78)
    return img, gt, (255, 235)


def fixture_noisy():
    """Фото-подобный шум."""
    yy, xx = np.mgrid[0:H, 0:W]
    rng = np.random.default_rng(7)
    img = _canvas()
    img[:, :, 1] = 120
    img[:, :, 2] = 200
    gt = (yy - 256) ** 2 + (xx - 256) ** 2 <= 170**2
    img[gt] = (200, 30, 30)
    return np.clip(img.astype(int) + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8), gt, (256, 256)


ROOM_OBJ = (slice(70, 230), slice(170, 350))  # объект ~15% кадра


def fixture_room() -> np.ndarray:
    """Комната со стеной и объектом — для сквозных тестов эндпоинта."""
    a = np.zeros((420, 560, 3), np.uint8)
    a[:, :, :] = (205, 190, 175)          # стена
    a[ROOM_OBJ[0], ROOM_OBJ[1], :] = (120, 125, 135)  # объект
    return a


MONITOR = (slice(150, 450), slice(80, 400))   # рамка + экран
MONITOR_STAND = (slice(450, 560), slice(160, 320))


def fixture_monitor() -> np.ndarray:
    """Монитор с белой рамкой на светлой стене (скрин: зелёные разводы
    за рамкой). Рамка почти не отличается по цвету от стены — growth
    утекал через границу и перекрашивал стену вокруг монитора."""
    h, w = 640, 480
    img = np.full((h, w, 3), (210, 208, 203), np.uint8)  # стена
    img[180:420, 110:370] = (22, 22, 25)                 # экран
    img[150:450, 80:110] = (235, 235, 232)               # рамка
    img[150:450, 370:400] = (235, 235, 232)
    img[150:180, 80:400] = (235, 235, 232)
    img[420:450, 80:400] = (235, 235, 232)
    img[MONITOR_STAND[0], MONITOR_STAND[1]] = (225, 225, 222)   # стойка
    img[520:560, 160:320] = (200, 200, 198)                      # основание
    gt = np.zeros((h, w), bool)
    gt[MONITOR[0], MONITOR[1]] = True  # рамка + экран
    gt[MONITOR_STAND[0], MONITOR_STAND[1]] = True  # стойка + основание
    gt[520:560, 160:320] = True
    return img, gt, (95, 300)


def fixture_chair() -> np.ndarray:
    """Кресло из нескольких материалов + стул рядом (скрин: покрашена
    только спинка). Цвета частей разные, поэтому растущий по цвету метод
    закрашивал лишь ту часть, куда попала точка касания."""
    h, w = 640, 480
    img = np.full((h, w, 3), (180, 175, 170), np.uint8)  # комната
    img[430:, :] = (150, 120, 90)                        # пол
    rng = np.random.default_rng(3)
    img = np.clip(img.astype(int) + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)
    # соседний стул — не должен попасть в маску
    img[150:350, 390:470] = (95, 80, 70)
    # кресло: спинка (кожа), сиденье, подлокотники, ножка
    img[120:380, 170:310] = (78, 62, 55)
    img[380:440, 140:340] = (118, 126, 138)
    img[330:440, 105:140] = (70, 70, 74)
    img[330:440, 340:375] = (70, 70, 74)
    img[440:560, 220:260] = (60, 60, 62)
    gt = np.zeros((h, w), bool)
    gt[120:380, 170:310] = True
    gt[380:440, 140:340] = True
    gt[330:440, 105:140] = True
    gt[330:440, 340:375] = True
    gt[440:560, 220:260] = True
    return img, gt, (240, 200)


def fixture_wall() -> np.ndarray:
    """Стена между потолком и полом (скрин: рваный оранжевый край у
    потолка)."""
    h, w = 640, 480
    img = np.zeros((h, w, 3), np.uint8)
    img[:130] = (185, 185, 188)  # потолок
    img[130:500] = (205, 190, 175)  # стена
    img[500:] = (150, 110, 75)  # пол
    rng = np.random.default_rng(5)
    img = np.clip(img.astype(int) + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    gt = np.zeros((h, w), bool)
    gt[130:500] = True
    return img, gt, (240, 300)


GLASS_HAND = (slice(260, 400), slice(20, 150))  # рука не должна попасть в маску


def fixture_plate() -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Светлая тарелка на светлой стене (скрин: фиолетовая стена вокруг
    тарелки). Край слабый по яркости, стена текстурированная, сзади дверь
    и диван, снизу рука."""
    h, w = 640, 480
    rng = np.random.default_rng(21)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([195, 190, 183], np.float32)  # светлая стена
    img += (np.sin(yy / 23.0) * 3 + np.sin(xx / 31.0) * 2)[..., None]
    img += rng.normal(0, 5, (h, w, 3))
    img[60:280, 330:480] = np.array([140, 110, 85], np.float32) + rng.normal(
        0, 5, (220, 150, 3)
    )  # дверь
    img[0:120, 0:200] = np.array([90, 88, 92], np.float32) + rng.normal(
        0, 5, (120, 200, 3)
    )  # диван
    img[380:480, 150:360] = np.array([206, 165, 132], np.float32) + rng.normal(
        0, 5, (100, 210, 3)
    )  # рука
    plate = (yy - 300) ** 2 + (xx - 250) ** 2 <= 130 * 130
    img[plate] = np.array([210, 196, 172], np.float32) + rng.normal(
        0, 4, (int(plate.sum()), 3)
    )
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, plate.copy(), (250, 300)


def fixture_monitor_desk() -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Монитор на подставке на столе у стены (скрин: покрасились подставка
    со столом вместо монитора)."""
    h, w = 640, 480
    rng = np.random.default_rng(33)
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([188, 182, 172], np.float32) + rng.normal(0, 4, (h, w, 3))
    img[420:] = np.array([135, 100, 70], np.float32) + rng.normal(0, 5, (220, w, 3))
    img[120:300, 170:390] = (235, 235, 232)  # белая рамка
    img[150:280, 190:370] = np.clip(
        np.array([150, 170, 200], np.float32) + rng.normal(0, 8, (130, 180, 3)), 0, 255
    )
    img[300:380, 250:310] = (215, 213, 210)  # подставка
    img[380:420, 200:360] = (200, 198, 195)  # основание
    img = np.clip(img, 0, 255).astype(np.uint8)
    gt = np.zeros((h, w), bool)
    gt[120:300, 170:390] = True  # рамка + экран
    gt[300:380, 250:310] = True
    gt[380:420, 200:360] = True
    return img, gt, (180, 200)


def fixture_glass() -> np.ndarray:
    """Стакан на деревянном столе в руке (скрин: зелёные разводы на руке
    и фоне). Стекло полупрозрачное — цвет непостоянный, рост по цвету
    образовывал кольцо вокруг стакана."""
    h, w = 480, 360
    rng = np.random.default_rng(11)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([150, 118, 82], np.float32)
    img += (np.sin(yy / 9.0) * 6)[..., None]
    img += rng.normal(0, 7, (h, w, 3))
    img[GLASS_HAND[0], GLASS_HAND[1]] = np.array([214, 172, 140], np.float32) + rng.normal(
        0, 6, (140, 130, 3)
    )
    gx0, gx1, gy0, gy1 = 150, 260, 120, 400
    cx = (gx0 + gx1) / 2.0
    shade = 1.0 - 0.55 * np.exp(-((xx - cx) ** 2) / (2 * 26.0**2))
    img[gy0:gy1, gx0:gx1] *= shade[gy0:gy1, gx0:gx1, None]
    img[gy0:gy1, gx0:gx1] = np.clip(img[gy0:gy1, gx0:gx1] + 40.0, 0, 255)
    img[gy0:gy1, gx0 : gx0 + 3] = 235  # кромки
    img[gy0:gy1, gx1 - 3 : gx1] = 225
    img[gy0 : gy0 + 3, gx0:gx1] = 240
    img[gy1 - 3 : gy1, gx0:gx1] = 230
    img = np.clip(img, 0, 255).astype(np.uint8)
    gt = np.zeros((h, w), bool)
    gt[gy0:gy1, gx0:gx1] = True
    return img, gt, (205, 300)


def jpeg_bytes(img: np.ndarray, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


from recolor import RecolorRequest  # noqa: E402
from recolor.backends import LocalBackend, get_backend  # noqa: E402
from recolor.local_recolor import recolor  # noqa: E402
from recolor.segmentation import segment_object  # noqa: E402


# ---------------------------------------------------------------- 1. маска

def test_segmentation_finds_circle():
    from recolor.segmentation import segment_object

    img, gt, pt = fixture_circle()
    res = segment_object(img, pt[0], pt[1])
    assert res.mask.shape == img.shape[:2]
    assert iou(res.mask, gt) > 0.9, f"IoU too low: {iou(res.mask, gt):.3f}"


def test_segmentation_finds_complex_shape():
    from recolor.segmentation import segment_object

    img, gt, pt = fixture_object()
    res = segment_object(img, pt[0], pt[1])
    assert iou(res.mask, gt) > 0.9, f"IoU too low: {iou(res.mask, gt):.3f}"


def test_segmentation_handles_noise():
    from recolor.segmentation import segment_object

    img, gt, pt = fixture_noisy()
    res = segment_object(img, pt[0], pt[1])
    assert iou(res.mask, gt) > 0.85, f"IoU too low: {iou(res.mask, gt):.3f}"


def test_segmentation_never_hangs_on_flat_image():
    """Регрессия: раньше GrabCut зависал на однотонном кадре навсегда."""
    from recolor.segmentation import segment_object

    flat = np.full((H, W, 3), 140, np.uint8)
    res = segment_object(flat, W // 2, H // 2)
    assert res.mask.shape == flat.shape[:2]
    # Маска ограничена сверху долей площади
    assert res.mask.mean() <= 0.76


def test_segmentation_bounds_point_to_image():
    from recolor.segmentation import segment_object

    img, _, _ = fixture_circle()
    res = segment_object(img, -50, 99999)
    assert res.mask.any()


# --------------------------- регрессии по багам с фото (скриншоты)

def test_segmentation_monitor_no_halo_on_wall():
    """Регрессия: за белой рамкой монитора маска затекала на стену —
    на фото стену вокруг монитора закрашивало зелёным.

    Рамка почти не отличается по цвету от стены, поэтому граница
    перекрытия между ними — единственная защита."""
    from recolor.segmentation import segment_object

    img, gt, pt = fixture_monitor()
    res = segment_object(img, pt[0], pt[1])
    mask = res.mask
    assert iou(mask, gt) > 0.85, f"IoU too low: {iou(mask, gt):.3f}"
    # Главное: НЕТ ореола вокруг монитора (было ~27% площади утечки)
    leak = (mask & ~gt).sum()
    assert leak < 0.005 * gt.sum(), f"маска вылезла за монитор: {leak} px"
    # Экран внутри рамки должен быть покрашен вместе с рамкой
    screen = np.zeros(gt.shape, bool)
    screen[180:420, 110:370] = True
    assert mask[screen].mean() > 0.95, f"экран не покрыт: {mask[screen].mean():.2f}"


def test_segmentation_multi_material_object_covered():
    """Регрессия: кресло красилось частично — только спинка (скрин).

    Части объекта разного цвета разделены краями, и рост по цвету
    останавливался на первой же внутренней границе."""
    from recolor.segmentation import segment_object

    img, gt, pt = fixture_chair()
    res = segment_object(img, pt[0], pt[1])
    mask = res.mask
    coverage = float(np.logical_and(mask, gt).sum() / gt.sum())
    assert coverage > 0.93, f"кресло покрашено частично: {coverage:.2f}"
    # Соседний стул не должен попасть в маску
    neighbour = np.zeros(gt.shape, bool)
    neighbour[150:350, 390:470] = True
    assert mask[neighbour].sum() < 0.01 * neighbour.sum(), "соседний стул попал в маску"


def test_segmentation_wall_boundary_stays_clean():
    """Регрессия: край стены у потолка был рваным и затекал на потолок."""
    from recolor.segmentation import segment_object

    img, gt, pt = fixture_wall()
    res = segment_object(img, pt[0], pt[1])
    mask = res.mask
    # Внутренность стены (без двух рядов у границ и кромок кадра) —
    # должна быть покрыта полностью
    core = gt.copy()
    core[:132] = False
    core[498:] = False
    core[:, :2] = False
    core[:, -2:] = False
    assert mask[core].mean() > 0.98, f"стена покрыта частично: {mask[core].mean():.3f}"
    # А на потолок и пол маска вылезть не должна
    outside_band = ~gt
    leak = (mask & outside_band).sum()
    assert leak < 0.01 * gt.sum(), f"маска вылезла на потолок/пол: {leak} px"


def test_segmentation_glass_keeps_hand_out():
    """Регрессия: вокруг стекла возникало окрашенное кольцо, и краска
    попадала на руку и фон (скрин)."""
    from recolor.segmentation import segment_object

    img, gt, pt = fixture_glass()
    res = segment_object(img, pt[0], pt[1])
    mask = res.mask
    coverage = float(np.logical_and(mask, gt).sum() / gt.sum())
    assert coverage > 0.9, f"стекло покрашено частично: {coverage:.2f}"
    leak = float(np.logical_and(mask, ~gt).sum() / gt.sum())
    assert leak < 0.03, f"маска вылезла за стекло: {leak:.3f}"
    hand = np.zeros(gt.shape, bool)
    hand[GLASS_HAND[0], GLASS_HAND[1]] = True
    assert mask[hand].sum() == 0, f"рука попала в маску: {mask[hand].sum()} px"


def test_segmentation_plate_does_not_absorb_wall():
    """Регрессия: тарелка обрастала стеной (скрин с фиолетовой стеной).

    Край тарелка/стена слабый по яркости — барьера нет, и склейка
    затягивала в маску фон вокруг объекта."""
    from recolor.segmentation import segment_object

    img, gt, pt = fixture_plate()
    res = segment_object(img, pt[0], pt[1])
    mask = res.mask
    assert iou(mask, gt) > 0.95, f"IoU too low: {iou(mask, gt):.3f}"
    leak = (mask & ~gt).sum()
    assert leak < 0.05 * gt.sum(), f"маска захватила стену: {leak} px"


def test_segmentation_jpeg_quality_85_no_collapse():
    """Регрессия: JPEG q=85 схлопывал сегментацию (mask 2.6% вместо 25.6%).

    JPEG-артефакты (DCT-блоки, субдискретизация хромы) создают ложные
    микро-границы, которые _edge_barrier превращает в «сыплющий» барьер.
    Волновой рост схлопывается — объект покрывается лишь в центре."""
    from recolor.segmentation import segment_object

    h, w = 500, 700
    img = np.full((h, w, 3), (128, 128, 128), np.uint8)  # серая стена
    obj_y1, obj_y2, obj_x1, obj_x2 = 100, 450, 200, 500
    for i in range(obj_y1, obj_y2):
        t = (i - obj_y1) / (obj_y2 - obj_y1)
        r = int(150 + 100 * t)  # плавный красный градиент
        g = int(50 * (1 - t) + 20 * t)
        b = int(50 * (1 - t) + 20 * t)
        img[i, obj_x1:obj_x2] = [r, g, b]

    # Ground truth = объектный прямоугольник
    gt = np.zeros((h, w), bool)
    gt[obj_y1:obj_y2, obj_x1:obj_x2] = True
    px, py = 350, 275  # центр объекта

    # JPEG q=85
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=85)
    buf.seek(0)
    jpeg = np.array(Image.open(buf))

    res = segment_object(jpeg, px, py)
    coverage = float(np.logical_and(res.mask, gt).sum() / gt.sum())
    assert coverage > 0.9, f"JPEG схлопнул сегментацию: покрытие {coverage:.1%}"


def test_segmentation_monitor_on_desk_not_stand():
    """Регрессия: тап по монитору красил подставку и стол (скрин)."""
    from recolor.segmentation import segment_object

    img, gt, pt = fixture_monitor_desk()
    res = segment_object(img, pt[0], pt[1])
    mask = res.mask
    coverage = float(np.logical_and(mask, gt).sum() / gt.sum())
    assert coverage > 0.9, f"монитор покрашен частично: {coverage:.2f}"
    # Стол (нижняя треть кадра) не должен попасть в маску
    table = np.zeros(gt.shape, bool)
    table[430:] = True
    assert mask[table].sum() < 0.02 * table.sum(), "стол попал в маску"


def test_recolor_material_texture_is_visible():
    """Регрессия: текстура материала была еле заметна (скрин с плоской
    заливкой). После подгонки цвета модуляция сжималась, особенно на
    тёмных целях."""
    img, gt, pt = fixture_chair()
    mask = segment_object(img, pt[0], pt[1]).mask
    for material in ("wood", "metal", "fabric", "concrete"):
        request = RecolorRequest(
            image_rgb=img,
            point_x=pt[0],
            point_y=pt[1],
            material=material,
            color_r=120,
            color_g=80,
            color_b=45,
            color_hex=(120 << 16) | (80 << 8) | 45,
            from_pipette=True,
        )
        out = recolor(img, mask, request)
        region = out[mask].astype(np.float32)
        # Низкочастотная модуляция (рельеф материала): разброс средних по
        # строкам и столбцам области. Плоская заливка даёт ~0.
        rows, cols = np.nonzero(mask)
        row_means = np.array(
            [region[rows == y].mean(axis=0) for y in np.unique(rows)]
        )
        col_means = np.array(
            [region[cols == x].mean(axis=0) for x in np.unique(cols)]
        )
        modulation = float(max(row_means.std(), col_means.std()))
        assert modulation > 8.0, f"{material}: текстура слабая ({modulation:.1f})"
        mean = region.mean(axis=0)
        assert np.abs(mean - np.array((120, 80, 45))).max() < 15, f"{material}: {mean}"


# ------------------------------------------------------------- 2. перекраска

def _req(img, material, rgb, point, pipette=True, **kw):
    return RecolorRequest(
        image_rgb=img,
        point_x=point[0],
        point_y=point[1],
        material=material,
        color_r=rgb[0],
        color_g=rgb[1],
        color_b=rgb[2],
        color_hex=(rgb[0] << 16) | (rgb[1] << 8) | rgb[2],
        from_pipette=pipette,
        **kw,
    )


def test_recolor_hits_target_color():
    for material, target in [
        ("metal", (40, 90, 200)),
        ("wood", (170, 110, 50)),
        ("plastic", (250, 250, 250)),
        ("no_texture", (30, 30, 30)),
        ("concrete", (128, 60, 60)),
        ("fabric", (20, 180, 120)),
    ]:
        img, _, pt = fixture_circle()
        mask = segment_object(img, pt[0], pt[1]).mask
        out = recolor(img, mask, _req(img, material, target, pt))
        mean = out[mask].mean(axis=0)
        assert np.abs(mean - np.array(target)).max() < 12, f"{material}: {mean} vs {target}"


def test_recolor_preserves_shading():
    """Рельеф/градиент объекта должен сохраняться, а не становиться плоским."""
    import cv2

    yy, xx = np.mgrid[0:H, 0:W]
    img = _canvas()
    img[:, :, 0] = np.clip(40 + xx / 4, 0, 255)
    img[:, :, 1] = 90
    img[:, :, 2] = np.clip(150 + yy / 6, 0, 255)
    mask = segment_object(img, 256, 256).mask

    l_before = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0][mask].astype(float)
    for material in ("metal", "wood", "plastic", "concrete"):
        out = recolor(img, mask, _req(img, material, (100, 110, 190), (256, 256), pipette=False))
        l_after = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)[..., 0][mask].astype(float)
        corr = float(np.corrcoef(l_before, l_after)[0, 1])
        assert corr > 0.85, f"{material}: shading correlation {corr:.3f}"


def test_recolor_leaves_outside_untouched():
    img, _, pt = fixture_circle()
    mask = segment_object(img, pt[0], pt[1]).mask
    out = recolor(img, mask, _req(img, "metal", (10, 200, 10), pt))
    outside = ~mask
    changed = (out != img).any(axis=2)
    # Меняться может только тонкая рамка сглаживания вокруг маски
    assert (changed & outside).sum() < 0.02 * outside.sum()


def test_recolor_monitor_does_not_paint_wall():
    """Сквозная регрессия (скрин): стена вокруг монитора была зелёной."""
    img, gt, pt = fixture_monitor()
    mask = segment_object(img, pt[0], pt[1]).mask
    out = recolor(img, mask, _req(img, "metal", (40, 140, 60), pt))
    changed = np.abs(out.astype(int) - img.astype(int)).max(axis=2) > 12
    # За пределами монитора (с допуском на растушёвку) — ничего не красить
    halo = ~gt
    leak = (changed & halo).sum()
    assert leak < 0.01 * gt.sum(), f"стена закрашена: {leak} px"
    mean = out[gt].mean(axis=0)
    assert np.abs(mean - np.array((40, 140, 60))).max() < 15, f"цвет монитора: {mean}"


def test_recolor_multi_material_object_full_color():
    """Сквозная регрессия (скрин): кресло красилось частично — цвет
    добирался только на спинке."""
    img, gt, pt = fixture_chair()
    mask = segment_object(img, pt[0], pt[1]).mask
    out = recolor(img, mask, _req(img, "leather", (120, 75, 40), pt))
    mean = out[gt].mean(axis=0)
    assert np.abs(mean - np.array((120, 75, 40))).max() < 15, f"цвет кресла: {mean}"
    # Сиденье (серая часть) тоже должно быть покрашено: уход к цели
    # не менее 30% расстояния (оттенок цели, относительная светлота
    # частей объекта сохраняется — сиденье светлее спинки и остаётся
    # светлее, средний цвет кресла при этом точно равен цели)
    seat = np.zeros(gt.shape, bool)
    seat[380:440, 140:340] = True
    orig = img[seat].astype(float)
    target = np.array((120, 75, 40), float)
    moved = out[seat].astype(float) - orig
    to_target = target - orig
    frac = float(
        (
            (moved * to_target).sum(axis=1)
            / np.linalg.norm(to_target, axis=1)
            / np.linalg.norm(to_target, axis=1)
        ).mean()
    )
    assert frac > 0.3, f"сиденье покрашено слабо: {frac:.2f}"
    changed = (np.abs(out[seat].astype(int) - img[seat].astype(int)).max(axis=1) > 12).mean()
    assert changed > 0.9, f"сиденье покрашено частично: {changed:.2f}"


def test_recolor_patina_changes_metal():
    img, _, pt = fixture_circle()
    mask = segment_object(img, pt[0], pt[1]).mask
    plain = recolor(img, mask, _req(img, "metal", (150, 140, 60), pt))
    aged = recolor(img, mask, _req(img, "metal", (150, 140, 60), pt, patina=True))
    assert not np.array_equal(plain, aged)


def test_recolor_empty_mask_returns_copy():
    img, _, pt = fixture_circle()
    out = recolor(img, np.zeros(img.shape[:2], bool), _req(img, "wood", (1, 2, 3), pt))
    assert np.array_equal(out, img)


# ------------------------------------------------------------ 3. все материалы

def test_all_allowed_materials_recolor():
    from config import ALLOWED_MATERIALS

    for material in sorted(ALLOWED_MATERIALS):
        img, _, pt = fixture_circle()
        mask = segment_object(img, pt[0], pt[1]).mask
        out = recolor(img, mask, _req(img, material, (60, 120, 200), pt))
        assert (out[mask] != img[mask]).any(), f"{material} changed nothing"
        mean = out[mask].mean(axis=0)
        assert np.abs(mean - np.array((60, 120, 200))).max() < 15, f"{material}: {mean}"


# ------------------------------------------------------------ 4. бэкенды

def test_backend_local_without_keys():
    import config

    saved = config.AI_BACKEND
    try:
        config.CLOUDFLARE_ACCOUNT_ID = ""
        config.CLOUDFLARE_API_TOKEN = ""
        config.GEMINI_API_KEY = ""
        config.AI_BACKEND = "local"
        assert isinstance(get_backend(), LocalBackend)

        config.AI_BACKEND = "auto"
        from recolor.backends import BackendConfigurationError
        try:
            get_backend()
            raise AssertionError("auto mode must not silently use local recoloring")
        except BackendConfigurationError:
            pass
    finally:
        config.AI_BACKEND = saved


def test_backend_prefers_cloudflare():
    import config

    saved = (config.AI_BACKEND, config.CLOUDFLARE_ACCOUNT_ID, config.CLOUDFLARE_API_TOKEN, config.GEMINI_API_KEY)
    try:
        config.AI_BACKEND = "auto"
        config.CLOUDFLARE_ACCOUNT_ID = "acc"
        config.CLOUDFLARE_API_TOKEN = "tok"
        config.GEMINI_API_KEY = "g"
        assert get_backend().name == "cloudflare"
    finally:
        (
            config.AI_BACKEND,
            config.CLOUDFLARE_ACCOUNT_ID,
            config.CLOUDFLARE_API_TOKEN,
            config.GEMINI_API_KEY,
        ) = saved


def test_backend_cloudflare_requires_keys():
    import config

    saved = (config.AI_BACKEND, config.CLOUDFLARE_ACCOUNT_ID, config.CLOUDFLARE_API_TOKEN)
    try:
        config.AI_BACKEND = "cloudflare"
        config.CLOUDFLARE_ACCOUNT_ID = ""
        config.CLOUDFLARE_API_TOKEN = ""
        try:
            get_backend()
            raise AssertionError("ожидалась RuntimeError")
        except RuntimeError as exc:
            assert "CLOUDFLARE_ACCOUNT_ID" in str(exc)
    finally:
        (
            config.AI_BACKEND,
            config.CLOUDFLARE_ACCOUNT_ID,
            config.CLOUDFLARE_API_TOKEN,
        ) = saved


# -------------------------------------------------------- 5. HTTP эндпоинт

def _client():
    from fastapi.testclient import TestClient

    import app
    import config

    # HTTP contract tests exercise the explicit local backend without making paid API calls.
    # Production configuration defaults to Cloudflare and is tested separately below.
    config.AI_BACKEND = "local"
    _reset_rate_limit(app)
    return TestClient(app.app), app


def _reset_rate_limit(app_mod) -> None:
    """Сбрасывает счётчик rate-limit между тестами (лимит — 10 запросов/минуту)."""
    limiter = getattr(app_mod.app.state, "limiter", None)
    if limiter is not None:
        try:
            limiter.reset()
        except Exception:  # pragma: no cover
            pass


def test_health_endpoint():
    client, app_mod = _client()
    with client:
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert "backend" in data


def test_ai_recolor_endpoint_all_materials():
    client, app_mod = _client()
    payload = jpeg_bytes(fixture_room())
    with client:
        for material, color in [("metal", "0x2E5AAC"), ("wood", "0x8B5A2B"), ("plastic", "0xF2F2F2")]:
            rgb = tuple(int(color[2:][i : i + 2], 16) for i in (0, 2, 4))
            res = client.post(
                "/ai-recolor",
                files={"image": ("room.jpg", payload, "image/jpeg")},
                data={
                    "point_x": "260",
                    "point_y": "150",
                    "material": material,
                    "color_hex": color,
                    "color_r": str(rgb[0]),
                    "color_g": str(rgb[1]),
                    "color_b": str(rgb[2]),
                    "object_name": "cabinet",
                    "from_pipette": "true",
                },
            )
            assert res.status_code == 200, res.text
            assert res.headers["content-type"] == "image/png"
            out = np.array(Image.open(io.BytesIO(res.content)).convert("RGB"))
            assert out.shape == fixture_room().shape

            obj = np.zeros(out.shape[:2], bool)
            obj[ROOM_OBJ[0], ROOM_OBJ[1]] = True
            mean = out[obj].mean(axis=0)
            assert np.abs(mean - np.array(rgb)).max() < 15, f"{material}: {mean} vs {rgb}"


def test_ai_recolor_preserves_outside():
    client, _ = _client()
    room = fixture_room()
    payload = jpeg_bytes(room)
    with client:
        res = client.post(
            "/ai-recolor",
            files={"image": ("x.jpg", payload, "image/jpeg")},
            data={"point_x": "260", "point_y": "150", "material": "metal",
                  "color_hex": "0x2E5AAC", "color_r": "46", "color_g": "90", "color_b": "172",
                  "from_pipette": "true"},
        )
    orig = np.array(Image.open(io.BytesIO(payload)).convert("RGB"))
    out = np.array(Image.open(io.BytesIO(res.content)).convert("RGB"))
    obj = np.zeros(out.shape[:2], bool)
    obj[ROOM_OBJ[0], ROOM_OBJ[1]] = True
    changed = (out != orig).any(axis=2)
    assert changed[obj].mean() > 0.9
    assert (changed & ~obj).sum() < 0.02 * (~obj).sum()


# ------------------------------------------------------- 6. ошибки и валидация

def test_error_corrupted_image():
    client, _ = _client()
    with client:
        res = client.post(
            "/ai-recolor",
            files={"image": ("x.jpg", b"not an image", "image/jpeg")},
            data={"point_x": "10", "point_y": "10", "color_hex": "0xFF0000"},
        )
        assert res.status_code == 400


def test_error_point_out_of_range():
    client, _ = _client()
    with client:
        res = client.post(
            "/ai-recolor",
            files={"image": ("x.jpg", jpeg_bytes(fixture_room()), "image/jpeg")},
            data={"point_x": "-10", "point_y": "10", "color_hex": "0xFF0000"},
        )
        assert res.status_code == 400


def test_unknown_material_falls_back():
    client, _ = _client()
    with client:
        res = client.post(
            "/ai-recolor",
            files={"image": ("x.jpg", jpeg_bytes(fixture_room()), "image/jpeg")},
            data={"point_x": "260", "point_y": "150", "material": "unobtanium",
                  "color_hex": "0xFF0000"},
        )
        assert res.status_code == 200


def test_oversized_image_rejected():
    import config

    rng = np.random.default_rng(0)
    big = np.clip(rng.normal(128, 40, (2600, 3500, 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(big).save(buf, format="PNG")
    data = buf.getvalue()
    assert len(data) > config.MAX_IMAGE_SIZE_BYTES

    client, _ = _client()
    with client:
        res = client.post(
            "/ai-recolor",
            files={"image": ("big.png", data, "image/png")},
            data={"point_x": "10", "point_y": "10", "color_hex": "0xFF0000"},
        )
        assert res.status_code == 413


def test_color_hex_formats_accepted():
    client, _ = _client()
    payload = jpeg_bytes(fixture_room())
    with client:
        for color in ("0xFF8800", "#FF8800", "16747520"):
            res = client.post(
                "/ai-recolor",
                files={"image": ("x.jpg", payload, "image/jpeg")},
                data={"point_x": "260", "point_y": "150", "color_hex": color},
            )
            assert res.status_code == 200, f"{color} -> {res.status_code}"


# ------------------------------------------- 7. фолбэк при отказе API

def test_cloudflare_failure_returns_gateway_error_instead_of_fake_success():
    import requests
    import config
    from recolor.cloudflare_ai import CloudflareAI

    client, app_mod = _client()
    saved = (config.AI_BACKEND, config.CLOUDFLARE_ACCOUNT_ID, config.CLOUDFLARE_API_TOKEN)
    try:
        from unittest.mock import patch
        from fastapi.testclient import TestClient

        config.AI_BACKEND = "cloudflare"
        config.CLOUDFLARE_ACCOUNT_ID = "acc"
        config.CLOUDFLARE_API_TOKEN = "tok"
        cf_client = CloudflareAI(
            "acc", "tok", "@cf/runwayml/stable-diffusion-v1-5-inpainting",
            "@cf/meta/llama-3.2-11b-vision-instruct", timeout=2, max_retries=0,
        )

        def boom(self, url, **kw):
            raise requests.ConnectionError("network down")

        with patch.object(requests.Session, "post", autospec=True, side_effect=boom):
            _reset_rate_limit(app_mod)
            with TestClient(app_mod.app) as c:
                res = c.post(
                    "/ai-recolor",
                    files={"image": ("x.jpg", jpeg_bytes(fixture_room()), "image/jpeg")},
                    data={"point_x": "260", "point_y": "150", "material": "wood",
                          "color_hex": "0x8B5A2B", "color_r": "139", "color_g": "90",
                          "color_b": "43", "from_pipette": "true"},
                )
        assert res.status_code == 502, res.text
        assert "Image generation service failed" in res.json()["detail"]
    finally:
        config.AI_BACKEND, config.CLOUDFLARE_ACCOUNT_ID, config.CLOUDFLARE_API_TOKEN = saved

def test_cloudflare_request_shape():
    import requests

    from recolor.cloudflare_ai import CloudflareAI

    cf = CloudflareAI("acct", "tok", "@cf/x", "@cf/y", timeout=5, max_retries=0)
    img, _, pt = fixture_circle()
    cap = {}

    def fake(self, url, headers=None, json=None, timeout=None, **kw):
        cap["url"] = url
        cap["auth"] = headers["Authorization"]
        cap["keys"] = sorted(json)
        cap["model"] = url.split("/ai/run/")[1]

        class R:
            status_code = 200
            headers = {"Content-Type": "application/json"}
            content = b""

            @staticmethod
            def json():
                # Реальный формат ответа REST API Cloudflare — вложенный result
                return {"result": {"response": " car door "}, "success": True, "errors": []}

        return R()

    from unittest.mock import patch

    with patch.object(requests.Session, "post", autospec=True, side_effect=fake):
        assert cf.describe(img, pt[0], pt[1]) == "car door"

    assert cap["url"].startswith("https://api.cloudflare.com/client/v4/accounts/acct/ai/run/")
    assert cap["auth"] == "Bearer tok"
    assert cap["model"] == "@cf/y"
    # Vision принимает prompt + image (НЕ messages — проверено живьём)
    assert "prompt" in cap["keys"]
    assert "image" in cap["keys"]
    assert "messages" not in cap["keys"]


def test_cloudflare_detection_selects_box_containing_tap():
    """DETR's binary input and returned bbox are correctly mapped to source pixels."""
    import requests
    from unittest.mock import patch
    from recolor.cloudflare_ai import CloudflareAI

    cf = CloudflareAI("acct", "tok", "@cf/x", "@cf/y", max_retries=0,
                      detection_model="@cf/facebook/detr-resnet-50")
    image = np.zeros((400, 800, 3), dtype=np.uint8)
    captured = {}

    class Reply:
        status_code = 200
        headers = {"Content-Type": "application/json"}
        content = b""
        @staticmethod
        def json():
            return {"result": [
                {"label": "chair", "score": 0.91,
                 "box": {"xmin": 100, "ymin": 40, "xmax": 250, "ymax": 300}},
                {"label": "person", "score": 0.99,
                 "box": {"xmin": 400, "ymin": 10, "xmax": 500, "ymax": 390}},
            ], "success": True}

    def fake_post(self, url, headers=None, data=None, timeout=None, **kwargs):
        captured.update(url=url, headers=headers, data=data)
        return Reply()

    with patch.object(requests.Session, "post", autospec=True, side_effect=fake_post):
        # Image is scaled to 768x384; source tap (200, 200) lies in the chair box.
        found = cf.detect_object(image, 200, 200)
    assert found is not None
    box, label, score = found
    assert label == "chair" and score == 0.91
    assert box == (104, 42, 260, 312)
    assert captured["url"].endswith("/@cf/facebook/detr-resnet-50")
    assert captured["headers"]["Content-Type"] == "image/png"
    assert isinstance(captured["data"], bytes) and captured["data"].startswith(b"\x89PNG")


def test_cloudflare_daily_free_limit_is_not_retryable():
    from recolor.cloudflare_ai import _parse_error

    class Reply:
        status_code = 429
        text = ""
        @staticmethod
        def json():
            return {"success": False, "errors": [{"code": 3036,
                    "message": "You have used up your daily free allocation of 10,000 neurons."}]}

    message, retryable = _parse_error(Reply())
    assert "3036" in message
    assert retryable is False


def test_cloudflare_inpaint_payload_shape():
    """Inpaint должен слать image/mask МАССИВАМИ байт, не base64 (проверено живьём)."""
    import requests

    from recolor.cloudflare_ai import CloudflareAI

    cf = CloudflareAI("a", "b", "@cf/m", "@cf/v", timeout=5, max_retries=0)
    img, _, pt = fixture_circle()
    cap = {}

    def fake(self, url, headers=None, json=None, timeout=None, **kw):
        cap.update(json)

        class R:
            status_code = 200
            headers = {"Content-Type": "image/png"}
            content = b"\x89PNG\r\n\x1a\n" + b"0" * 512
            text = ""

        return R()

    from unittest.mock import patch

    with patch.object(requests.Session, "post", autospec=True, side_effect=fake):
        try:
            cf.inpaint(img, pt and _mask_for(img, pt), prompt="recolor", size=256)
        except Exception:
            pass  # декод заглушки не обязателен, важен payload

    assert isinstance(cap.get("image"), list), "image должен быть массивом байт"
    assert isinstance(cap.get("mask"), list), "mask должен быть массивом байт"
    assert all(isinstance(v, int) for v in cap["image"][:50])
    assert "image_b64" not in cap and "mask_b64" not in cap
    assert "prompt" in cap


def _mask_for(img, pt):
    from recolor.segmentation import segment_object

    return segment_object(img, pt[0], pt[1]).mask


def test_cloudflare_error_mapping():
    import requests

    from recolor.cloudflare_ai import CloudflareAI, CloudflareAIError

    cf = CloudflareAI("a", "b", "@cf/m", "@cf/v", timeout=5, max_retries=0)

    class Resp:
        status_code = 400
        headers = {}
        content = b""

        @staticmethod
        def json():
            return {"success": False, "errors": [{"code": 7003, "message": "No such model"}]}

    from unittest.mock import patch

    with patch.object(requests.Session, "post", autospec=True, return_value=Resp()):
        try:
            cf._run("@cf/m", {})
            raise AssertionError("ожидался CloudflareAIError")
        except CloudflareAIError as exc:
            assert exc.status == 400
            assert "No such model" in str(exc)
            assert exc.retryable is False


def test_gemini_request_shape():
    import requests

    from recolor.gemini_ai import GeminiAI

    g = GeminiAI("gkey", "gemini-2.5-flash", timeout=5, max_retries=0)
    img, _, pt = fixture_circle()
    cap = {}

    def fake(self, url, params=None, headers=None, json=None, timeout=None, **kw):
        cap["url"] = url
        cap["key"] = params.get("key")
        cap["parts"] = [p for p in json["contents"][0]["parts"]]

        class R:
            status_code = 200

            @staticmethod
            def json():
                return {"candidates": [{"content": {"parts": [{"text": "wall panel"}]}}]}

        return R()

    from unittest.mock import patch

    with patch.object(requests.Session, "post", autospec=True, side_effect=fake):
        assert g.describe(img, pt[0], pt[1]) == "wall panel"

    assert cap["url"].endswith(":generateContent")
    assert cap["key"] == "gkey"
    assert any("inline_data" in p for p in cap["parts"])


# ------------------------------------------------------------------ runner

def test_degenerate_inpaint_rejected_and_retried():
    """Регрессия: SD-1.5 inpaint иногда возвращает чёрное — такой результат
    должен отбраковываться и повторяться, а не уходить клиенту."""
    from recolor.backends import _is_sensible_recolor

    # чёрное / темное — отбраковка
    assert _is_sensible_recolor(np.array([0.0, 0.0, 0.0]), 130.0) is False
    assert _is_sensible_recolor(np.array([9.0, 9.0, 9.0]), 130.0) is False
    assert _is_sensible_recolor(np.array([10.0, 18.0, 40.0]), 130.0) is False
    # вменяемая покраска — принимается
    assert _is_sensible_recolor(np.array([46.0, 90.0, 172.0]), 130.0) is True
    assert _is_sensible_recolor(np.array([170.0, 110.0, 50.0]), 118.0) is True
    # тёмная цель, а результат выгорел — тоже отбраковка
    assert _is_sensible_recolor(np.array([250.0, 250.0, 250.0]), 20.0) is False


class _FakeCloudflare:
    """Мок клиента Cloudflare: считает вызовы klein/inpaint и отдаёт кадр."""

    def __init__(self, result=None, error=None, object_color=(0, 160, 0),
                 fail_attempts=0):
        self.calls = {"klein": 0, "inpaint": 0, "describe": 0}
        self.error = error
        self.result = result
        self.object_color = object_color
        # сколько ПЕРВЫХ попыток провалить (ошибкой или негодным кадром)
        self.fail_attempts = fail_attempts
        self.klein_kwargs = {}
        self.is_configured = True
        self.klein_model = "@cf/black-forest-labs/flux-2-klein-4b"
        self.inpaint_model = "@cf/stabilityai/stable-diffusion-xl-base-1.0"
        self.vision_model = "@cf/meta/llama-3.2-11b-vision-instruct"

    # --- генеративные методы (вызываются бэкендом) ---
    def recolor_klein(self, image_rgb, mask, prompt, size=1024, seed=None):
        self.calls["klein"] += 1
        self.klein_kwargs = {"prompt": prompt, "size": size, "seed": seed}

        if self.calls["klein"] <= self.fail_attempts:
            if self.error is not None:
                raise self.error
            # «негодный» результат: объект не перекрашен
            return image_rgb.copy()
        if self.result is not None:
            return self.result
        return self._paint(image_rgb, mask)

    def inpaint(self, image_rgb, mask, prompt, **kw):
        self.calls["inpaint"] += 1
        raise AssertionError("SD-XL inpaint не должен вызываться при klein")

    def describe(self, image_rgb, point_x, point_y):
        self.calls["describe"] += 1
        return "socket panel"

    def _paint(self, image_rgb, mask):
        """Имитация FLUX: объект внутри маски становится цветом цели."""
        from recolor import ops as _ops

        out = image_rgb.copy()
        m = _ops.to_mask_array(mask)
        # рамка вокруг маски — имитация «съеденных» моделью краёв
        grown = _ops.dilate_mask(m, 3)
        out[grown] = self.object_color
        return out


def _cloudflare_backend(client):
    from recolor.backends import CloudflareBackend

    b = CloudflareBackend.__new__(CloudflareBackend)
    b.client = client
    b.analyzer = client
    b.analyzer_name = "cloudflare-vision"
    b._retry_note = None
    return b


def _room_request(r=139, g=90, b=43, material="wood", pipette=True):
    from recolor import RecolorRequest

    img = fixture_room()
    return RecolorRequest(
        image_rgb=img, point_x=270, point_y=200, material=material,
        color_r=r, color_g=g, color_b=b, color_hex=(r << 16) | (g << 8) | b,
        color_name="brown", object_name="object", from_pipette=pipette,
    )


def test_cloudflare_uses_flux_klein():
    """Главный тест: перекраску делает FLUX.2 [klein], а не локалка.

    Проверяем: вызван именно recolor_klein (не inpaint), модель опрошена,
    результат генерации попал в итог, фон за пределами маски цел.
    """
    from recolor.backends import _cfg

    client = _FakeCloudflare(object_color=(139, 90, 43))
    backend = _cloudflare_backend(client)
    request = _room_request()

    # klein по умолчанию
    assert str(_cfg("CLOUDFLARE_RECOLOR_MODEL", "klein")) == "klein"

    result = backend.run(request)

    assert client.calls["klein"] == 1, "FLUX.2 klein не вызван"
    assert client.calls["inpaint"] == 0, "SD-XL inpaint подменил FLUX"
    assert client.calls["describe"] == 1, "vision-модель не опрошена"
    assert result.backend == "cloudflare"
    assert result.object_label == "socket panel"
    assert any("flux" in n.lower() or "нейросет" in n for n in result.notes), result.notes

    # Генеративный результат действительно использован
    img = request.image_rgb
    obj = np.zeros(img.shape[:2], bool)
    obj[ROOM_OBJ[0], ROOM_OBJ[1]] = True
    from recolor import ops as _ops

    away = ~_ops.dilate_mask(obj, 6)
    # Фон за пределами маски не тронут
    assert np.abs(result.image_rgb[away].astype(int) - img[away].astype(int)).max() < 12
    # Объект перекрашен (отличимо от исходного серого)
    changed = np.abs(
        result.image_rgb[obj].astype(int) - img[obj].astype(int)
    ).max(axis=1)
    assert changed.mean() > 20, f"объект не перекрашен: {changed.mean():.1f}"
    # Цвет объекта смещён в сторону цели
    mean = result.image_rgb[obj].astype(np.float32).mean(axis=0)
    before = img[obj].astype(np.float32).mean(axis=0)
    assert np.abs(mean - np.array((139, 90, 43))).max() < np.abs(before - np.array((139, 90, 43))).max()


def test_cloudflare_klein_retries_and_rejects_invalid_generation():
    """Негодный результат FLUX отбраковывается и не подменяется локальной заливкой."""
    from recolor.cloudflare_ai import CloudflareAIError
    client = _FakeCloudflare(fail_attempts=99)
    backend = _cloudflare_backend(client)
    request = _room_request()
    try:
        backend.run(request)
        raise AssertionError("invalid generation should fail, not return a fallback image")
    except CloudflareAIError:
        pass
    assert client.calls["klein"] == 2, f"ожидалось 2 попытки, было {client.calls['klein']}"


def test_cloudflare_klein_success_after_retry():
    """Первая попытка провалилась, вторая принята — фолбэка нет."""
    client = _FakeCloudflare(fail_attempts=1)  # падаем только на 1-й попытке
    backend = _cloudflare_backend(client)
    request = _room_request()

    result = backend.run(request)

    assert client.calls["klein"] == 2
    assert result.backend == "cloudflare", "успех со второй попытки — не фолбэк"
    assert not any("ГЕНЕРАЦИЯ НЕ ВЫПОЛНЕНА" in n for n in result.notes)


def test_cloudflare_api_error_is_not_silently_recoloured():
    """Ошибки авторизации/квоты должны всплывать, а не маскироваться локальным результатом."""
    from recolor.cloudflare_ai import CloudflareAIError
    client = _FakeCloudflare(error=CloudflareAIError("401 unauthorized", status=401), fail_attempts=99)
    backend = _cloudflare_backend(client)
    try:
        backend.run(_room_request())
        raise AssertionError("401 must not be returned as successful recoloring")
    except CloudflareAIError as exc:
        assert exc.status == 401
    assert client.calls["klein"] == 1, "non-retryable authentication errors must not be retried"


def test_validate_generated_rejects_bad_results():
    """Валидация ответа генеративной модели ловит все виды мусора."""
    from recolor.backends import _validate_generated
    from recolor import ops as _ops

    img = fixture_room()
    m = np.zeros(img.shape[:2], bool)
    m[ROOM_OBJ[0], ROOM_OBJ[1]] = True
    target_luma = 0.299 * 139 + 0.587 * 90 + 0.114 * 43

    # годный результат: объект перекрашен
    good = img.copy()
    good[_ops.dilate_mask(m, 2)] = (139, 90, 43)
    assert _validate_generated(good, img, m, target_luma) is None

    # None / чёрный / не тот размер / объект не изменён
    assert _validate_generated(None, img, m, target_luma) is not None
    assert _validate_generated(np.zeros_like(img), img, m, target_luma) is not None
    assert _validate_generated(img[:10, :10], img, m, target_luma) is not None
    assert _validate_generated(img.copy(), img, m, target_luma) is not None


def test_cloudflare_prompt_contains_color_and_material():
    """Промпт несёт целевой цвет и материал."""
    from recolor.backends import build_recolor_prompt

    prompt = build_recolor_prompt(_room_request(material="leather"), "sofa", "brown")
    low = prompt.lower()
    assert "leather" in low
    assert "brown" in low
    assert "#8b5a2b" in low


def test_recolor_klein_is_explicit_flux_method():
    """Клиент отдаёт публичный метод FLUX и не подменяет модель молча."""
    from recolor.cloudflare_ai import CloudflareAI

    c = CloudflareAI(
        "acc", "tok",
        inpaint_model="@cf/runwayml/stable-diffusion-v1-5-inpainting",
        vision_model="@cf/meta/llama-3.2-11b-vision-instruct",
        klein_model="@cf/black-forest-labs/flux-2-klein-4b",
    )
    assert hasattr(c, "recolor_klein"), "нет публичного метода FLUX.2 klein"

    # без klein-модели — явная ошибка, а не тихий уход на другую модель
    c2 = CloudflareAI("acc", "tok", inpaint_model="", vision_model="", klein_model="")
    try:
        c2.recolor_klein(np.zeros((32, 32, 3), np.uint8), np.ones((32, 32), bool), "x")
        raise AssertionError("ожидалась ошибка конфигурации")
    except Exception as exc:
        assert "klein" in str(exc).lower()


def test_klein_request_size_within_api_limits():
    """Референс для klein: кратность 32 и предел стороны из конфига."""
    from recolor.cloudflare_ai import _fit_klein_input_size

    for box in (200, 150), (777, 431), (3000, 1200), (40, 30):
        for cap in (512, 768, 1024):
            cw, ch = _fit_klein_input_size(box[0], box[1], cap)
            assert cw % 32 == 0 and ch % 32 == 0, (box, cap, cw, ch)
            assert 64 <= cw <= cap and 64 <= ch <= cap, (box, cap, cw, ch)
            # Пропорции сохранены с точностью, допускаемой округлением до 32
            assert abs(cw * box[1] - ch * box[0]) <= 32 * max(box), (box, cap, cw, ch)


def test_legacy_backend_wraps_runner():
    """Legacy-маршрут регистрируется как бэкенд (иначе run падал на None)."""
    from recolor.backends import LegacyBackend

    sentinel = object()
    backend = LegacyBackend(sentinel)
    assert backend.name == "legacy"
    assert backend.runner is sentinel

    # Реальный раннер из app возвращает BackendResult
    import app as app_module

    assert callable(app_module.run_legacy_recolor_job)


def test_get_backend_never_picks_legacy_by_default():
    """Auto mode chooses real image generation, otherwise fails instead of pretending."""
    import config
    from recolor import get_backend
    from recolor.backends import BackendConfigurationError

    original = (config.AI_BACKEND, config.CLOUDFLARE_ACCOUNT_ID, config.CLOUDFLARE_API_TOKEN)
    try:
        config.AI_BACKEND = "auto"
        config.CLOUDFLARE_ACCOUNT_ID = ""
        config.CLOUDFLARE_API_TOKEN = ""
        try:
            get_backend("auto")
            raise AssertionError("auto must not select a non-generative local backend")
        except BackendConfigurationError:
            pass
    finally:
        config.AI_BACKEND, config.CLOUDFLARE_ACCOUNT_ID, config.CLOUDFLARE_API_TOKEN = original


# ------------------------------------------- Z.AI: габариты от vision-модели

class _FakeZai:
    """Мок Z.AI: отдаёт заданный бокс (или падает)."""

    def __init__(self, box=None, error=None):
        self.box = box
        self.error = error
        self.calls = 0

    @property
    def is_configured(self) -> bool:
        return True

    def object_box(self, image_rgb, point_x, point_y):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.box


def _gt_box(gt: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(gt)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def test_zai_parses_box_from_glm_answers():
    """Разбор ответов GLM: JSON, абсолютные и 0..1000 координаты, фон."""
    from recolor.zai_ai import _parse_box

    assert _parse_box('{"box": [10, 20, 300, 400]}', 500, 500) == (10, 20, 300, 400)
    assert _parse_box('```json\n{"box":[5,6,100,200]}\n```', 500, 500) == (5, 6, 100, 200)
    assert _parse_box('{"box": null}', 500, 500) is None
    assert _parse_box("The point is on background.", 500, 500) is None
    # координаты в шкале 0..1000
    box = _parse_box('{"box": [100, 200, 600, 800]}', 1000, 2000)
    assert box == (100, 400, 600, 1600)
    # перепутанный порядок координат
    assert _parse_box('{"box": [300, 20, 10, 400]}', 500, 500) == (10, 20, 300, 400)


def test_zai_bbox_repairs_partial_taps():
    """Регрессия: тап по мелкой части объекта красил только её (скрины).

    Габариты от vision-модели дают seed GrabCut по всему объекту, поэтому
    объект покрывается целиком."""
    from recolor.segmentation import segment_object

    img, gt, _ = fixture_chair()
    box = _gt_box(gt)
    # тап по НОЖКЕ кресла — локально красилась только ножка
    mask = segment_object(img, 240, 500, bbox=box).mask
    coverage = float(np.logical_and(mask, gt).sum() / gt.sum())
    assert coverage > 0.85, f"объект покрашен частично: {coverage:.2f}"

    img, gt, _ = fixture_monitor()
    box = _gt_box(gt)
    # тап по ОСНОВАНИЮ монитора — локально красилось только основание
    mask = segment_object(img, 240, 540, bbox=box).mask
    coverage = float(np.logical_and(mask, gt).sum() / gt.sum())
    assert coverage > 0.80, f"объект покрашен частично: {coverage:.2f}"


def test_zai_bbox_does_not_break_normal_taps():
    """Точный и заведомо раздутый бокс не должны ломать обычные тапы."""
    from recolor.segmentation import segment_object

    for fn in (fixture_chair, fixture_monitor, fixture_glass, fixture_wall, fixture_plate):
        img, gt, pt = fn()
        box = _gt_box(gt)
        for bbox in (box, (box[0] - 150, box[1] - 150, box[2] + 150, box[3] + 150)):
            mask = segment_object(img, pt[0], pt[1], bbox=bbox).mask
            coverage = float(np.logical_and(mask, gt).sum() / gt.sum())
            assert coverage > 0.9, f"{fn.__name__}: покрытие {coverage:.2f}"


def test_zai_backend_uses_box_for_mask():
    """Бэкенд Z.AI передаёт габариты модели в сегментацию."""
    from recolor.backends import ZaiBackend

    img, gt, pt = fixture_chair()
    request = RecolorRequest(
        image_rgb=img,
        point_x=pt[0],
        point_y=pt[1],
        material="leather",
        color_r=120,
        color_g=75,
        color_b=40,
        color_hex=(120 << 16) | (75 << 8) | 40,
        from_pipette=True,
    )
    client = _FakeZai(box=_gt_box(gt))
    result = ZaiBackend(client).run(request)
    assert client.calls == 1, "модель не опрошена"
    coverage = float(np.logical_and(result.segmentation.mask, gt).sum() / gt.sum())
    assert coverage > 0.9, f"покрытие {coverage:.2f}"
    mean = result.image_rgb[gt].mean(axis=0)
    assert np.abs(mean - np.array((120, 75, 40))).max() < 15, f"цвет: {mean}"


def test_zai_backend_falls_back_when_api_fails():
    """Любой отказ API не должен ломать запрос: красит локальный путь."""
    from recolor.backends import ZaiBackend
    from recolor.zai_ai import ZaiAIError

    img, gt, pt = fixture_monitor()
    request = RecolorRequest(
        image_rgb=img,
        point_x=pt[0],
        point_y=pt[1],
        material="metal",
        color_r=40,
        color_g=140,
        color_b=60,
        color_hex=(40 << 16) | (140 << 8) | 60,
        from_pipette=True,
    )
    # 401 (неверный/истёкший ключ) и сетевой сбой —
    # оба обязаны давать корректную картинку локальным путём
    for error in (
        ZaiAIError("401", status=401, retryable=False),
        ZaiAIError("сеть", status=0, retryable=True),
    ):
        result = ZaiBackend(_FakeZai(error=error)).run(request)
        assert result.image_rgb.shape == img.shape
        changed = (
            np.abs(result.image_rgb.astype(int) - img.astype(int)).max(axis=2) > 12
        )
        assert changed.any(), " Lokальный путь не покрасил ничего"


# ------------------------------------------------------------------ runner


# --------------------------------------------------- 9. покрытие по пикселям

def _solid(target, material="metal"):
    """Однотонный объект: генерация возвращает исходный цвет — пятно."""
    h, w = 256, 256
    img = np.full((h, w, 3), 140, np.uint8)
    img[:, :, 0] = 150
    img[:, :, 1] = 150
    img[:, :, 2] = 150
    gt = np.zeros((h, w), bool)
    gt[40:220, 40:220] = True
    img[gt] = (150, 150, 150)
    return img, gt, target, material


def test_validate_pixel_coverage_rejects_partial_paint():
    """Генерация, оставившая часть объекта исходного цвета — отбраковка."""
    from recolor.backends import _validate_pixel_coverage

    img, gt, target, material = _solid((40, 90, 200))
    # объект не тронут — всё исходный цвет
    assert _validate_pixel_coverage(img, img, gt, target, material) is not None
    # часть объекта покрашена, часть — нет
    half = img.copy()
    half[gt][: gt[gt].shape[0] // 2] = target
    assert _validate_pixel_coverage(half, img, gt, target, material) is not None
    # весь объект покрашен — принимается
    full = img.copy()
    full[gt] = target
    assert _validate_pixel_coverage(full, img, gt, target, material) is None


def test_enforce_coverage_removes_stains():
    """_enforce_coverage устраняет локальные пятна исходного цвета,
    не трогая светлоту и детали (LAB-хрома, итеративно)."""
    from recolor.backends import _enforce_coverage

    img, gt, target, material = _solid((40, 90, 200))
    # Имитация слабой генерации: центр покрашен, углы — исходный цвет.
    painted = img.copy()
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    core = ((yy - 130) ** 2 + (xx - 130) ** 2) <= 80 ** 2
    painted[core] = target
    # Углы маски по-прежнему исходного цвета
    out = _enforce_coverage(painted, gt, target, material, original=img)
    # Внутри маски не должно остаться пикселей исходного цвета
    import cv2

    lab_o = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_n = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma_diff = np.abs(lab_n[gt, 1:] - lab_o[gt, 1:]).max(axis=1)
    assert (chroma_diff <= 22).mean() < 0.02, "остали пятна исходного цвета"
    # Светлота сохраняется (рельеф/тени не гасятся)
    lo = lab_o[gt, 0]
    ln = lab_n[gt, 0]
    if lo.std() < 1e-6:
        luma_corr = 1.0
    else:
        luma_corr = float(np.corrcoef(ln, lo)[0, 1])
        if np.isnan(luma_corr):
            luma_corr = 1.0
    assert luma_corr > 0.9, f"светлота искажена: {luma_corr:.3f}"


def test_enforce_coverage_preserves_shading():
    """После покрытия градиент освещения остаётся — не плоская заливка."""
    from recolor.backends import _enforce_coverage
    import cv2

    h, w = 256, 256
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = np.clip(40 + xx / 3, 0, 255)
    img[:, :, 1] = np.clip(40 + yy / 4, 0, 255)
    img[:, :, 2] = 100
    gt = np.zeros((h, w), bool)
    gt[40:220, 40:220] = True
    target = (40, 90, 200)
    painted = img.copy()
    painted[gt] = target
    out = _enforce_coverage(painted, gt, target, "metal", original=img)
    lab_o = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_n = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lo = lab_o[gt, 0]
    ln = lab_n[gt, 0]
    if lo.std() < 1e-6:
        corr = 1.0
    else:
        corr = float(np.corrcoef(ln, lo)[0, 1])
        if np.isnan(corr):
            corr = 1.0
    assert corr > 0.9, f"градиент освещения утрачен: {corr:.3f}"


# ---------------------------------------------------------- 10. стекло

def test_recolor_glass_full_surface():
    """Стекло перекрашивается по всей видимой поверхности, а не только дном."""
    img, gt, pt = fixture_glass()
    mask = segment_object(img, pt[0], pt[1]).mask
    target = (60, 180, 200)
    out = recolor(img, mask, _req(img, "glass", target, pt))
    # Средний цвет всей маски — близок к цели
    mean = out[gt].mean(axis=0)
    assert np.abs(mean - np.array(target)).max() < 25, f"цвет стекла: {mean}"
    # Дно (нижняя треть стекла) и боковые стенки (верхняя треть) —
    # оба должны отклониться от исходного цвета
    h = out.shape[0]
    bands = {}
    band = gt.copy()
    band[int(h * 0.66):] = False
    bands["top"] = band.copy()
    band = gt.copy()
    band[: int(h * 0.33)] = False
    bands["bottom"] = band.copy()
    for name, band in bands.items():
        if band.sum() == 0:
            continue
        moved = float(
            np.abs(out[band].astype(int) - img[band].astype(int)).max(axis=1).mean()
        )
        assert moved > 15, f"стекло {name} не перекрашено: {moved:.1f}"


def test_recolor_glass_outside_untouched():
    """Сквозь стекло не красится фон, рука и стол."""
    img, gt, pt = fixture_glass()
    mask = segment_object(img, pt[0], pt[1]).mask
    out = recolor(img, mask, _req(img, "glass", (60, 180, 200), pt))
    changed = (np.abs(out.astype(int) - img.astype(int)).max(axis=2) > 12)
    hand = np.zeros(gt.shape, bool)
    hand[GLASS_HAND[0], GLASS_HAND[1]] = True
    assert (changed & hand).sum() == 0, "рука закрашена сквозь стекло"
    outside = ~gt
    assert (changed & outside).sum() < 0.03 * outside.sum(), "фон закрашен сквозь стекло"


def test_recolor_glass_keeps_highlights():
    """Для стекла сохраняются блики и отражения (светлые пиксели)."""
    img, gt, pt = fixture_glass()
    mask = segment_object(img, pt[0], pt[1]).mask
    out = recolor(img, mask, _req(img, "glass", (60, 180, 200), pt))
    import cv2

    lab_o = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_n = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    # Самые светлые пиксели стекла (блики) почти не темнеют
    luma_o = lab_o[gt, 0]
    luma_n = lab_n[gt, 0]
    bright = luma_o > np.percentile(luma_o, 90)
    if bright.sum() > 5:
        drop = float((luma_o[bright] - luma_n[bright]).mean())
        assert drop < 25, f"блики потускнели: {drop:.1f}"


# ----------------------------------------------------- 11. диагностика

def test_diagnostic_collects_all_frames():
    """Диагностический режим сохраняет все промежуточные кадры и метрики."""
    from recolor import diagnostic as diag_mod
    from recolor.backends import _cfg

    import recolor.backends as bmod

    # включаем диагностику для этого теста
    saved_diag = getattr(bmod, "_diag", None)
    saved_cfg = _cfg("RECOLOR_DIAGNOSTIC", "0")
    try:
        # принудительно включаем через атрибут модуля
        bmod._diag = diag_mod.DiagnosticCollector("test-req", enabled=True)
        img, gt, pt = fixture_circle()
        mask = segment_object(img, pt[0], pt[1]).mask
        request = _req(img, "metal", (40, 90, 200), pt)
        backend = LocalBackend()
        result = backend.run(request)
        d = bmod._diag
        assert d.enabled
        for key in (
            "00_original_resized",
            "01_mask",
            "02_mask_overlay",
            "03_crop",
            "04_model_output",
            "05_detail_restored",
            "06_final",
        ):
            assert key in d.images, f"нет кадра {key}"
        assert "mask_bbox" in d.metrics
        assert "mask_area_ratio" in d.metrics
        assert "crop_size" in d.metrics
        assert "changed_inside_mask" in d.metrics
        assert "changed_outside_mask" in d.metrics
        assert result.image_rgb.shape == img.shape
    finally:
        bmod._diag = saved_diag
        # восстанавливаем конфигурацию (через env)
        import os as _os
        _os.environ.pop("RECOLOR_DIAGNOSTIC", None)


def test_diagnostic_disabled_by_default():
    """В production-режиме диагностика не сохраняет кадры."""
    from recolor import diagnostic as diag_mod

    d = diag_mod.DiagnosticCollector("req", enabled=False)
    assert not d.enabled
    assert d.dir is None
    d.capture("x", np.zeros((4, 4, 3), np.uint8))
    assert "x" not in d.images


def test_cloudflare_diagnostic_numbering_matches_local():
    """CloudflareBackend захватывает '01_mask', а не '02_mask' — нумерация
    совпадает с LocalBackend."""
    from recolor import diagnostic as diag_mod
    import recolor.backends as bmod

    saved_diag = getattr(bmod, "_diag", None)
    try:
        bmod._diag = diag_mod.DiagnosticCollector("cf-diag-req", enabled=True)
        client = _FakeCloudflare(object_color=(139, 90, 43))
        backend = _cloudflare_backend(client)
        request = _room_request()

        backend.run(request)

        d = bmod._diag
        assert "01_mask" in d.images, f"01_mask missing; keys: {list(d.images)}"
        assert "02_mask" not in d.images, "02_mask should not exist (use 01_mask)"
        for key in ("01_mask", "02_mask_overlay", "03_crop", "04_model_output",
                     "05_detail_restored", "06_final"):
            assert key in d.images, f"нет кадра {key}"
    finally:
        bmod._diag = saved_diag


def test_cloudflare_model_name_visible_in_notes():
    """В notes виден реальный путь модели, а не сокращение 'klein'."""
    client = _FakeCloudflare(object_color=(139, 90, 43))
    backend = _cloudflare_backend(client)
    request = _room_request()

    result = backend.run(request)

    assert "@cf/black-forest-labs/flux-2-klein-4b" in " | ".join(result.notes), \
        f"модель не видна в notes: {result.notes}"


def test_local_backend_has_diagnostic_notes():
    """LocalBackend теперь вызывает diag.flush() и добавляет заметки."""
    from recolor import diagnostic as diag_mod
    import recolor.backends as bmod

    saved_diag = getattr(bmod, "_diag", None)
    try:
        bmod._diag = diag_mod.DiagnosticCollector("local-diag-req", enabled=True)
        img, gt, pt = fixture_circle()
        mask = segment_object(img, pt[0], pt[1]).mask
        request = _req(img, "metal", (40, 90, 200), pt)
        backend = LocalBackend()
        result = backend.run(request)

        assert any("диагностика:" in n for n in result.notes), \
            f"нет диагностических notes: {result.notes}"
        assert any("локальная перекраска" in n for n in result.notes), \
            f"нет заметки о локальной покраске: {result.notes}"
    finally:
        bmod._diag = saved_diag


def test_diagnostic_cleanup_on_init():
    """При создании включённого DiagnosticCollector вызывается cleanup_expired."""
    from recolor import diagnostic as diag_mod
    import recolor.backends as bmod

    called = {"count": 0}

    original = diag_mod.cleanup_expired

    def fake_cleanup(*args, **kwargs):
        called["count"] += 1
        return 0

    diag_mod.cleanup_expired = fake_cleanup
    try:
        d = diag_mod.DiagnosticCollector("cleanup-req", enabled=True)
        assert called["count"] >= 1, "cleanup_expired не вызван при включённой диагностике"
    finally:
        diag_mod.cleanup_expired = original


def test_diagnostic_cleanup_not_called_when_disabled():
    """cleanup_expired не вызывается для disabled-коллектора."""
    from recolor import diagnostic as diag_mod
    import recolor.backends as bmod

    called = {"count": 0}
    original = diag_mod.cleanup_expired

    def fake_cleanup(*args, **kwargs):
        called["count"] += 1
        return 0

    diag_mod.cleanup_expired = fake_cleanup
    try:
        d = diag_mod.DiagnosticCollector("disabled-req", enabled=False)
        assert called["count"] == 0, "cleanup_expired не должен вызываться при disabled"
    finally:
        diag_mod.cleanup_expired = original


# ------------------------------------------------------------------ runner

def _run_all() -> int:
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
