from __future__ import annotations

# FrozenLake map layouts used for RM context derivation (NL -> RM guardrails).
#
# The layouts follow the same emoji convention used by `parse_map_emoji`:
# - '⛔' are holes (ignored by the RM event vocabulary)
# - letters/digits are goal symbols (become allowed RM events)
# - any other symbol (e.g., 🟩) is floor

config = {
    "maps": {
        "map1": {
            # Three goals (A, B, C) and some holes.
            "layout": """
              B 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
             🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
             🟩 🟩 🟩 ⛔ ⛔ 🟩 🟩 🟩 🟩 🟩
             🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
             🟩 🟩 🟩 🟩 A  🟩 🟩 🟩 🟩 🟩
             🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
             🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
             ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ 🟩 ⛔ ⛔
             🟩 🟩 🟩 🟩  C 🟩 🟩 🟩 🟩 🟩
             🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
            """,
        }
    }
}
