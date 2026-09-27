from typing import List, Tuple

class ToneSandhiEngine:
    """
    Applies Mandarin Tone Sandhi rules to determine contextual tone targets.
    Rules:
    1. 3 + 3 -> 2 + 3 (Two 3rd tones: first becomes 2nd tone).
    2. '一' (yī):
       - Before Tone 4 -> Tone 2 (yí)
       - Before Tone 1, 2, 3 -> Tone 4 (yì)
    3. '不' (bù):
       - Before Tone 4 -> Tone 2 (bú)
    4. Neutral tone (Tone 0) reduction rules.
    """

    @staticmethod
    def apply_sandhi_rules(pinyin_list: List[str], lexical_tones: List[int]) -> List[Tuple[int, bool, str]]:
        """
        Returns list of (contextual_target_tone, is_modified, rule_applied)
        """
        n = len(lexical_tones)
        contextual_targets = list(lexical_tones)
        metadata = [("", False) for _ in range(n)]

        results = []

        # Process 3-3 Sandhi
        i = 0
        while i < n:
            target_tone = lexical_tones[i]
            is_sandhi = False
            rule_name = ""

            py = pinyin_list[i].lower() if i < len(pinyin_list) else ""

            # Check 3-3 sandhi
            if target_tone == 3 and i + 1 < n and lexical_tones[i+1] == 3:
                target_tone = 2
                is_sandhi = True
                rule_name = "3-3_tone_sandhi"

            # Check '一' (yi) sandhi
            elif "yi" in py or py == "yī" or py == "yí" or py == "yǐ" or py == "yì":
                if i + 1 < n:
                    next_tone = lexical_tones[i+1]
                    if next_tone == 4:
                        target_tone = 2
                        is_sandhi = True
                        rule_name = "yi_before_tone4"
                    elif next_tone in (1, 2, 3):
                        target_tone = 4
                        is_sandhi = True
                        rule_name = "yi_before_tone123"

            # Check '不' (bu) sandhi
            elif "bu" in py or py == "bù" or py == "bú":
                if i + 1 < n and lexical_tones[i+1] == 4:
                    target_tone = 2
                    is_sandhi = True
                    rule_name = "bu_before_tone4"

            results.append((target_tone, is_sandhi, rule_name))
            i += 1

        return results
