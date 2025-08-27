# app.py — ARAM 스펠/룬 추천표 (업로드 CSV 컬럼에 정확히 맞춘 버전)
# - 매핑 딕셔너리 불필요: spell_icons.csv / rune_icons.csv 자체가 한글/아이콘 포함
# - 빈칸 -> 안전 대체, 완전 빈행 제거
# - win 은 aram_matches_superlight2.csv 에서 조인

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="ARAM PS Dashboard", layout="wide")

ROOT = Path(__file__).parent
PLAYERS_CSV = ROOT / "aram_matches_superlight2.csv"
SPELL_CSV   = ROOT / "spell_icons.csv"
RUNE_CSV    = ROOT / "rune_icons.csv"

# ---------- 공통 유틸 ----------
def fill_blanks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].fillna(-1)
        else:
            out[c] = out[c].fillna("—")      # 표시용 대체
    return out

def drop_totally_empty_rows(df: pd.DataFrame, key_cols) -> pd.DataFrame:
    mask_all_empty = df[key_cols].isna().all(axis=1) | df[key_cols].apply(lambda s: s.astype(str).str.strip().eq("").all())
    return df.loc[~mask_all_empty].copy()

@st.cache_data(show_spinner=False)
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

# ---------- 데이터 로드 ----------
players = read_csv(PLAYERS_CSV)  # win 보유
spells  = read_csv(SPELL_CSV)    # spell1_name_fix, spell1_icon, spell2_name_fix, spell2_icon
runes   = read_csv(RUNE_CSV)     # rune_core, rune_core_icon, rune_sub, rune_sub_icon

# 키 정리 (조인용)
KEYS = ["matchId", "summonerName"]
for df in (players, spells, runes):
    for k in KEYS:
        if k in df.columns:
            df[k] = df[k].astype(str)

# ---------- 승패 조인 ----------
players_small = players[KEYS + ["win"]].copy()
spells_m = spells.merge(players_small, on=KEYS, how="left")
runes_m  = runes.merge(players_small, on=KEYS, how="left")

# ---------- 스펠 표 생성 ----------
# 안전 채움
for c in ["spell1_name_fix", "spell2_name_fix", "spell1_icon", "spell2_icon"]:
    if c in spells_m.columns:
        spells_m[c] = spells_m[c].fillna("")

# 완전 빈행(두 스펠 이름이 모두 없음/공백) 제거
spells_m = spells_m[~(spells_m["spell1_name_fix"].astype(str).str.strip().eq("") &
                      spells_m["spell2_name_fix"].astype(str).str.strip().eq(""))].copy()

# 조합 정규화: 이름 기준 알파벳/한글 순서로 정렬(같은 조합을 하나로)
spells_m["_a"] = spells_m["spell1_name_fix"].astype(str)
spells_m["_b"] = spells_m["spell2_name_fix"].astype(str)
spells_m[["s_min","s_max"]] = np.sort(spells_m[["_a","_b"]].values, axis=1)

# 아이콘도 같은 순서로 맞춤
def pick_icon_row(row):
    if row["_a"] <= row["_b"]:
        return pd.Series({"i1": row["spell1_icon"], "i2": row["spell2_icon"]})
    else:
        return pd.Series({"i1": row["spell2_icon"], "i2": row["spell1_icon"]})
spells_m[["i1","i2"]] = spells_m.apply(pick_icon_row, axis=1)

grp_spell = spells_m.groupby(["s_min","s_max","i1","i2"], dropna=False).agg(
    게임수=("win","count"),
    승수=("win","sum"),
).reset_index()
grp_spell["승률(%)"] = (grp_spell["승수"]/grp_spell["게임수"]*100).round(2)

spell_table = grp_spell.rename(columns={
    "s_min":"스펠1 이름","s_max":"스펠2 이름","i1":"스펠1","i2":"스펠2"
})[["스펠1","스펠1 이름","스펠2","스펠2 이름","게임수","승수","승률(%)"]]

spell_table = fill_blanks(spell_table)
spell_table = spell_table[spell_table["게임수"].astype(float) > 0]
spell_table = spell_table.sort_values(["게임수","승률(%)"], ascending=[False, False]).head(10)

# ---------- 룬 표 생성 ----------
for c in ["rune_core", "rune_core_icon", "rune_sub", "rune_sub_icon"]:
    if c in runes_m.columns:
        runes_m[c] = runes_m[c].fillna("")

# 완전 빈행(핵심/보조 둘 다 이름 없음) 제거
runes_m = runes_m[~(runes_m["rune_core"].astype(str).str.strip().eq("") &
                    runes_m["rune_sub"].astype(str).str.strip().eq(""))].copy()

grp_rune = runes_m.groupby(["rune_core","rune_core_icon","rune_sub","rune_sub_icon"], dropna=False).agg(
    게임수=("win","count"),
    승수=("win","sum"),
).reset_index()
grp_rune["승률(%)"] = (grp_rune["승수"]/grp_rune["게임수"]*100).round(2)

rune_table = grp_rune.rename(columns={
    "rune_core":"핵심룬 이름",
    "rune_core_icon":"핵심룬",
    "rune_sub":"보조트리 이름",
    "rune_sub_icon":"보조트리",
})[["핵심룬","핵심룬 이름","보조트리","보조트리 이름","게임수","승수","승률(%)"]]

rune_table = fill_blanks(rune_table)
rune_table = rune_table[rune_table["게임수"].astype(float) > 0]
rune_table = rune_table.sort_values(["게임수","승률(%)"], ascending=[False, False]).head(10)

# ---------- UI ----------
st.markdown("## Recommended Spell Combos")
st.dataframe(spell_table, use_container_width=True, hide_index=True)

st.markdown("## Recommended Rune Combos")
st.dataframe(rune_table, use_container_width=True, hide_index=True)

with st.expander("Raw rows (sample)"):
    cols = ["matchId","summonerName","spell1_name_fix","spell2_name_fix","rune_core","rune_sub","win"]
    cols = [c for c in cols if c in spells_m.columns or c in runes_m.columns or c in players.columns]
    # 샘플 표시는 중복 방지용으로 spells_m에서 300행만
    raw = spells_m.merge(runes_m[KEYS+["rune_core","rune_sub"]], on=KEYS, how="left")
    st.dataframe(fill_blanks(raw[cols].head(300)), use_container_width=True, hide_index=True)
