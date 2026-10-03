import pandas as pd

from keiba_place_lab.scope_expansion import straight_course_mask


def test_straight_course_mask_handles_english_and_japanese():
    frame = pd.DataFrame(
        {
            "turn_direction": ["left", "right", "straight", "直線", None],
        }
    )
    mask = straight_course_mask(frame)
    assert mask.tolist() == [False, False, True, True, False]
