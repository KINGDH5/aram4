# app.py — 빈행/빈칸 제거판 (Recommended Spell / Rune Combos)
# ----------------------------------------------------------------
# 필요 파일:
# 1) 참가자 데이터 CSV: PLAYERS_CSV
#    - 필수 컬럼 예시: ['champion','spell1Id','spell2Id','perkPrimaryStyle','perkSubStyle','win']
# 2) 스펠 매핑 CSV: SPELL_CSV (spell_icons.csv)
#    - 필수 컬럼 예시: ['id','name','icon_url']   # id = 정수(스펠ID)
#
# 출력:
# - 빈칸은 '—' 로 대체
# - 완전 빈 행 제거
# - '보조트리(영감)' 누락 없이 텍스트 표시
# ----------------------------------------------------------------

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="ARAM PS Dashboard (No Blanks)", layout="wide")

# ===== 파일 경로 =====
PLAYERS_CSV = "aram_participants_with_icons_superlight.csv"   # 참가자 원본 (수정 가능)
SPELL_CSV   = "spell_icons.csv"                               # 스펠 매핑

# ===== 컬럼명 매핑(당신 CSV에 맞게 조정) =====
COLS = {
    "champ": "champion",
    "spell1": "spell1Id",
    "spell2": "spell2Id",
    "pstyle": "perkPrimaryStyle",   # 핵심룬 트리 ID
    "sstyle": "perkSubStyle",       # 보조룬 트리 ID
    "win": "win"
}

# ===== 룬 트리 한글 매핑(완전판, 특히 영감=8300 포함) =====
RUNE_TREE_NAME = {
    8000: "정밀",
    8100: "지배",
    8200: "마법",
    8300: "영감",
    8400: "결의"
}
RUNE_TREE_ICON = {
    8000: "https://ddragon.leagueoflegends.com/cdn/img/perk-images/Styles/Precision/Precision.png",
    8100: "https://ddragon.leagueoflegends.com/cdn/img/perk-images/Styles/Domination/Domination.png",
    8200: "https://ddragon.leagueoflegends.com/cdn/img/perk-images/Styles/Sorcery/Sorcery.png",
    8300: "https://ddragon.leagueoflegends.com/cdn/img/perk-images/Styles/Inspiration/Inspiration.png",
    8400: "https://ddragon.leagueoflegends.com/cdn/img/perk-images/Styles/Resolve/Resolve.png",
}

# ===== 유틸: 안전 채움/제거 =====
def fill_blanks(df: pd.DataFrame) -> pd.DataFrame:
    """표시용 문자열 결측은 '—' 로, 숫자형은 -1로 채움"""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].fillna(-1)
        else:
            out[c] = out[c].fillna("—")
    return out

def drop_totally_empty_rows(df: pd.DataFrame, key_cols) -> pd.DataFrame:
    """핵심 컬럼이 전부 결측/공백인 행 제거 -> '빈행' 제거"""
    mask_all_empty = df[key_cols].isna().all(axis=1) | (df[key_cols].astype(str).replace("nan", "").str.strip().eq("").all(axis=1))
    return df.loc[~mask_all_empty].copy()

def safe_int(x, default=-1):
    try:
        # True/False처럼 bool 들어오면 int로
        if isinstance(x, (bool, np.bool_)):
            return int(x)
        return int(float(x))
    except Exception:
        return default

# ===== 데이터 로드 =====
@st.cache_data(show_spinner=False)
def load_players(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 타입 정리
    for c in [COLS["spell1"], COLS["spell2"], COLS["pstyle"], COLS["sstyle"]]:
        if c in df.columns:
            df[c] = df[c].apply(safe_int)
    if COLS["win"] in df.columns:
        df[COLS["win"]] = df[COLS["win"]].apply(safe_int)
    return df

@st.cache_data(show_spinner=False)
def load_spells(path: str) -> pd.DataFrame:
    # spell_icons.csv : id, name, icon_url (id가 정수형)
    s = pd.read_csv(path)
    # 혹시 id가 문자열이면 정수화
    if "id" in s.columns:
        s["id"] = s["id"].apply(safe_int)
    return s[["id", "name", "icon_url"]].drop_duplicates("id")

players = load_players(PLAYERS_CSV)
spells  = load_spells(SPELL_CSV)

# ===== 스펠 매핑 (ID -> 이름/아이콘) =====
spell_map_name = dict(zip(spells["id"], spells["name"]))
spell_map_icon = dict(zip(spells["id"], spells["icon_url"]))

players["스펠1"] = players[COLS["spell1"]].map(spell_map_icon)
players["스펠1 이름"] = players[COLS["spell1"]].map(spell_map_name)
players["스펠2"] = players[COLS["spell2"]].map(spell_map_icon)
players["스펠2 이름"] = players[COLS["spell2"]].map(spell_map_name)

# 스펠 매핑 실패(아이콘/이름 None) → 표시용 대체
players["스펠1"] = players["스펠1"].fillna("")
players["스펠2"] = players["스펠2"].fillna("")
players["스펠1 이름"] = players["스펠1 이름"].fillna("미확인")
players["스펠2 이름"] = players["스펠2 이름"].fillna("미확인")

# ===== 룬 트리 매핑 (ID -> 한글명/아이콘) =====
players["핵심룬"] = players[COLS["pstyle"]].map(RUNE_TREE_ICON)
players["핵심룬 이름"] = players[COLS["pstyle"]].map(RUNE_TREE_NAME)
players["보조트리"] = players[COLS["sstyle"]].map(RUNE_TREE_ICON)
players["보조트리 이름"] = players[COLS["sstyle"]].map(RUNE_TREE_NAME)

# 보조룬 '영감(8300)' 등 누락 방지: 위 딕셔너리에 이미 포함되어 있음
players["핵심룬"] = players["핵심룬"].fillna("")
players["보조트리"] = players["보조트리"].fillna("")
players["핵심룬 이름"] = players["핵심룬 이름"].fillna("미확인")
players["보조트리 이름"] = players["보조트리 이름"].fillna("미확인")

# ===== 완전 빈 행 제거(스펠/룬 식별 불가한 완전 공백 줄) =====
KEYS_FOR_EMPTY = ["스펠1 이름", "스펠2 이름", "핵심룬 이름", "보조트리 이름"]
players = drop_totally_empty_rows(players, KEYS_FOR_EMPTY)

# ===== 추천 스펠 조합 표 생성 =====
spell_cols = ["스펠1 이름", "스펠2 이름"]
# 정렬된(작은ID→큰ID) 조합으로 동일쌍 집계
players["_s1"] = players[COLS["spell1"]].apply(safe_int)
players["_s2"] = players[COLS["spell2"]].apply(safe_int)
players[["_min","_max"]] = np.sort(players[["_s1","_s2"]].values, axis=1)

grp_spell = players.groupby(["_min","_max"], dropna=False).agg(
    게임수 = (COLS["win"], "count"),
    승수 = (COLS["win"], "sum"),
).reset_index()

# 이름/아이콘 붙이기
grp_spell["스펠1"] = grp_spell["_min"].map(spell_map_icon).fillna("")
grp_spell["스펠2"] = grp_spell["_max"].map(spell_map_icon).fillna("")
grp_spell["스펠1 이름"] = grp_spell["_min"].map(spell_map_name).fillna("미확인")
grp_spell["스펠2 이름"] = grp_spell["_max"].map(spell_map_name).fillna("미확인")
grp_spell["승률(%)"] = (grp_spell["승수"] / grp_spell["게임수"] * 100).round(2)

# 표시용 컬럼 선택 & 빈칸 채움
spell_table = grp_spell[["스펠1","스펠1 이름","스펠2","스펠2 이름","게임수","승수","승률(%)"]]
spell_table = fill_blanks(spell_table)
# 완전 빈 조합(게임수 0) 제거 안전장치
spell_table = spell_table[spell_table["게임수"].astype(float) > 0].copy()
spell_table = spell_table.sort_values(["게임수","승률(%)"], ascending=[False, False]).head(10)

# ===== 추천 룬 트리 조합 표 생성 =====
grp_rune = players.groupby([COLS["pstyle"], COLS["sstyle"]], dropna=False).agg(
    게임수 = (COLS["win"], "count"),
    승수 = (COLS["win"], "sum"),
).reset_index()

grp_rune["핵심룬"] = grp_rune[COLS["pstyle"]].map(RUNE_TREE_ICON).fillna("")
grp_rune["보조트리"] = grp_rune[COLS["sstyle"]].map(RUNE_TREE_ICON).fillna("")
grp_rune["핵심룬 이름"] = grp_rune[COLS["pstyle"]].map(RUNE_TREE_NAME).fillna("미확인")
grp_rune["보조트리 이름"] = grp_rune[COLS["sstyle"]].map(RUNE_TREE_NAME).fillna("미확인")
grp_rune["승률(%)"] = (grp_rune["승수"] / grp_rune["게임수"] * 100).round(2)

rune_table = grp_rune[["핵심룬","핵심룬 이름","보조트리","보조트리 이름","게임수","승수","승률(%)"]]
rune_table = fill_blanks(rune_table)
rune_table = rune_table[rune_table["게임수"].astype(float) > 0].copy()
rune_table = rune_table.sort_values(["게임수","승률(%)"], ascending=[False, False]).head(10)

# ===== Streamlit 표시 =====
st.markdown("## Recommended Spell Combos")
st.dataframe(
    spell_table,
    use_container_width=True,
    hide_index=True
)

st.markdown("## Recommended Rune Combos")
st.dataframe(
    rune_table,
    use_container_width=True,
    hide_index=True
)

# ===== Raw rows(선택 챔피언) 옵션 예시 =====
with st.expander("Raw rows (selected champion)"):
    champs = ["(전체)"] + sorted(players[COLS["champ"]].dropna().astype(str).unique().tolist())
    pick = st.selectbox("Champion", champs, index=0)
    if pick != "(전체)":
        raw = players[players[COLS["champ"]].astype(str) == pick].copy()
    else:
        raw = players.copy()
    view_cols = [COLS["champ"], "스펠1 이름","스펠2 이름","핵심룬 이름","보조트리 이름", COLS["win"]]
    view_cols = [c for c in view_cols if c in raw.columns]
    st.dataframe(fill_blanks(raw[view_cols]), use_container_width=True, hide_index=True)
