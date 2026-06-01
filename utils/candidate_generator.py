import math
from typing import Dict, List


def generate_grid_candidates(
    bg_width: int,
    bg_height: int,
    fg_width: int,
    fg_height: int,
    candidate_count: int = 9,
    margin_ratio: float = 0.08,
) -> List[Dict]:
    """
    使用网格采样生成候选位置。

    x, y 表示前景左上角坐标。
    """
    candidate_count = int(candidate_count)
    candidate_count = max(1, candidate_count)

    cols = math.ceil(math.sqrt(candidate_count))
    rows = math.ceil(candidate_count / cols)

    margin_x = int(bg_width * margin_ratio)
    margin_y = int(bg_height * margin_ratio)

    min_x = margin_x
    max_x = max(margin_x, bg_width - fg_width - margin_x)

    min_y = margin_y
    max_y = max(margin_y, bg_height - fg_height - margin_y)

    candidates = []
    cid = 1

    for r in range(rows):
        for c in range(cols):
            if cid > candidate_count:
                break

            if cols == 1:
                x = (min_x + max_x) // 2
            else:
                x = int(min_x + c * (max_x - min_x) / (cols - 1))

            if rows == 1:
                y = (min_y + max_y) // 2
            else:
                y = int(min_y + r * (max_y - min_y) / (rows - 1))

            candidates.append(
                {
                    "id": cid,
                    "x": x,
                    "y": y,
                }
            )

            cid += 1

    return candidates