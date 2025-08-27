# app.py — ARAM 대시보드 (레포 파일명 자동 매핑 + 빈칸/빈행 처리)
# ----------------------------------------------------------------
# 레포에 다음 파일이 있어야 합니다:
#  - aram_matches_superlight2.csv     (참가자/매치 원본)
#  - spell_icons.csv                  (id, name, icon_url)
#  - rune_icons.csv                   (id, name, icon_url)  # 8000~8400 포함(영감=8300)
#  - champion_icons.csv               (id or key, name, icon_url)
# ----------------------------------------------------------------

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="ARAM PS Dashboard", layout="wide")

# ===== 파일 경로 (레포 루트 기준) =====
ROOT = Path(__file__).parent
PLAYERS_CSV = ROOT / "aram_matches_superlight2.csv"
SPELL_CSV   = ROOT / "spell_icons.csv"
RUNE_CSV    = ROOT / "rune_icons.csv"
CHAMP_CSV   = ROOT / "champion_icons.csv"

# ===== 유틸 =====
def pick_first(df: pd.DataFrame, candidates):
    """DataFrame에 존재하는 첫 후보 컬럼명을 반환. 없으면 None"""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def must_pick(df: pd.DataFrame, candidates, human_name: str):
    c = pick_first(df, candidates)
    if c is None:
        raise ValueError(f"[필수 컬럼 누락] {human_name} 에 해당하는 컬럼이 없습니다. 후보: {candidates} / 실제: {list(df.columns)}")
    return c

def safe_int(x, default=-1):
    try:
        if isinstance(x, (bool, np.bool_)):
            return int(x)
        return int(float(x))
    except Exception:
        return default

def fill_blanks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].fillna(-1)
        else:
            out[c] = out[c].fillna("—")
    return out

def drop_totally_empty_rows(df: pd.DataFrame, key_cols) -> pd.DataFrame:
    # 핵심 컬럼이 모두 비었으면 제거 (빈행 제거)
    mask_all_empty = df[key_cols].isna().all(axis=1)
    return df.loc[~mask_all_empty].copy()

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {path.name}")
    return pd.read_csv(path)

# ===== 데이터 로드 =====
players_raw = load_csv(PLAYERS_CSV)
spells = load_csv(SPELL_CSV)
runes  = load_csv(RUNE_CSV)
champs = load_csv(CHAMP_CSV)

# 스펠/룬/챔프 id, name, icon_url 컬럼 추출(이름이 달라도 대응)
spell_id_col  = must_pick(spells, ["id", "key", "spell_id"], "스펠 ID")
spell_namecol = must_pick(spells, ["name", "kr_name", "en_name"], "스펠 이름")
spell_iconcol = must_pick(spells, ["icon_url", "icon", "img"], "스펠 아이콘")

rune_id_col   = must_pick(runes,  ["id", "key", "rune_id"], "룬트리 ID")
rune_namecol  = must_pick(runes,  ["name", "kr_name", "en_name"], "룬트리 이름")
rune_iconcol  = must_pick(runes,  ["icon_url", "icon", "img"], "룬트리 아이콘")

champ_id_col  = pick_first(champs, ["id","key","championId"])
champ_namecol = pick_first(champs, ["name","kr_name","en_name"])
champ_iconcol = pick_first(champs, ["icon_url","icon","img"])

# 맵 생성
spell_map_name = dict(zip(spells[spell_id_col].map(safe_int), spells[spell_namecol]))
spell_map_icon = dict(zip(spells[spell_id_col].map(safe_int), spells[spell_iconcol]))

rune_map_name  = dict(zip(runes[rune_id_col].map(safe_int),  runes[rune_namecol]))
rune_map_icon  = dict(zip(runes[rune_id_col].map(safe_int),  runes[rune_iconcol]))

champ_map_name = dict(zip(champs[champ_id_col].map(safe_int), champs[champ_namecol])) if champ_id_col and champ_namecol else {}
champ_map_icon = dict(zip(champs[champ_id_col].map(safe_int), champs[champ_iconcol])) if champ_id_col and champ_iconcol else {}

# players_raw에서 필요한 컬럼 자동 탐색
spell1_col = must_pick(players_raw, ["spell1Id","spell1","spell_1","summoner1Id"], "스펠1 ID")
spell2_col = must_pick(players_raw, ["spell2Id","spell2","spell_2","summoner2Id"], "스펠2 ID")
pstyle_col = must_pick(players_raw, ["perkPrimaryStyle","primaryStyle","primary_perk_style","primaryStyleId"], "핵심룬 트리 ID")
sstyle_col = must_pick(players_raw, ["perkSubStyle","subStyle","sub_perk_style","subStyleId"], "보조룬 트리 ID")
win_col    = must_pick(players_raw, ["win","isWin","result","victory"], "승리 여부(0/1)")

champ_col_opt = pick_first(players_raw, ["championId","champion","champ_id","key"])

players = players_raw.copy()
for c in [spell1_col, spell2_col, pstyle_col, sstyle_col, win_col, champ_col_opt]:
    if c:
        players[c] = players[c].map(safe_int)

# 매핑 적용
players["스펠1"] = players[spell1_col].map(spell_map_icon).fillna("")
players["스펠2"] = players[spell2_col].map(spell_map_icon).fillna("")
players["스펠1 이름"] = players[spell1_col].map(spell_map_name).fillna("미확인")
players["스펠2 이름"] = players[spell2_col].map(spell_map_name).fillna("미확인")

players["핵심룬"] = players[pstyle_col].map(rune_map_icon).fillna("")
players["보조트리"] = players[sstyle_col].map(rune_map_icon).fillna("")
players["핵심룬 이름"] = players[pstyle_col].map(rune_map_name).fillna("미확인")
players["보조트리 이름"] = players[sstyle_col].map(rune_map_name).fillna("미확인")

if champ_col_opt:
    players["챔피언"] = players[champ_col_opt].map(champ_map_name).fillna("—")
    players["챔피언 아이콘"] = players[champ_col_opt].map(champ_map_icon).fillna("")
else:
    players["챔피언"] = "—"
    players["챔피언 아이콘"] = ""

# 완전 빈행 제거 (핵심표시 필드 전부 결측인 줄)
players = drop_totally_empty_rows(players, ["스펠1 이름","스펠2 이름","핵심룬 이름","보조트리 이름"])

# ===== 스펠 조합 집계 =====
players["_s1"] = players[spell1_col].map(safe_int)
players["_s2"] = players[spell2_col].map(safe_int)
players[["_min","_max"]] = np.sort(players[["_s1","_s2"]].values, axis=1)

grp_spell = players.groupby(["_min","_max"], dropna=False).agg(
    게임수 = (win_col, "count"),
    승수   = (win_col, "sum"),
).reset_index()

grp_spell["스펠1"] = grp_spell["_min"].map(spell_map_icon).fillna("")
grp_spell["스펠2"] = grp_spell["_max"].map(spell_map_icon).fillna("")
grp_spell["스펠1 이름"] = grp_spell["_min"].map(spell_map_name).fillna("미확인")
grp_spell["스펠2 이름"] = grp_spell["_max"].map(spell_map_name).fillna("미확인")
grp_spell["승률(%)"] = (grp_spell["승수"] / grp_spell["게임수"] * 100).round(2)

spell_table = grp_spell[["스펠1","스펠1 이름","스펠2","스펠2 이름","게임수","승수","승률(%)"]]
spell_table = fill_blanks(spell_table)
spell_table = spell_table[spell_table["게임수"].astype(float) > 0].sort_values(
    ["게임수","승률(%)"], ascending=[False, False]
).head(10)

# ===== 룬 트리 조합 집계 =====
grp_rune = players.groupby([pstyle_col, sstyle_col], dropna=False).agg(
    게임수 = (win_col, "count"),
    승수   = (win_col, "sum"),
).reset_index()

grp_rune["핵심룬"] = grp_rune[pstyle_col].map(rune_map_icon).fillna("")
grp_rune["보조트리"] = grp_rune[sstyle_col].map(rune_map_icon).fillna("")
grp_rune["핵심룬 이름"] = grp_rune[pstyle_col].map(rune_map_name).fillna("미확인")
grp_rune["보조트리 이름"] = grp_rune[sstyle_col].map(rune_map_name).fillna("미확인")
grp_rune["승률(%)"] = (grp_rune["승수"] / grp_rune["게임수"] * 100).round(2)

rune_table = grp_rune[["핵심룬","핵심룬 이름","보조트리","보조트리 이름","게임수","승수","승률(%)"]]
rune_table = fill_blanks(rune_table)
rune_table = rune_table[rune_table["게임수"].astype(float) > 0].sort_values(
    ["게임수","승률(%)"], ascending=[False, False]
).head(10)

# ===== UI =====
st.markdown("## Recommended Spell Combos")
st.dataframe(spell_table, use_container_width=True, hide_index=True)

st.markdown("## Recommended Rune Combos")
st.dataframe(rune_table, use_container_width=True, hide_index=True)

with st.expander("Raw rows (sample)"):
    view_cols = ["챔피언","스펠1 이름","스펠2 이름","핵심룬 이름","보조트리 이름", win_col]
    view_cols = [c for c in view_cols if c in players.columns]
    st.dataframe(fill_blanks(players[view_cols].head(300)), use_container_width=True, hide_index=True)
