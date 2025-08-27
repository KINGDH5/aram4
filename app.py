# app.py — ZeroDivisionError 방지판 (네 CSV 컬럼에 맞춤)
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="ARAM PS Dashboard", layout="wide")

ROOT = Path(__file__).parent
PLAYERS_CSV = ROOT / "aram_matches_superlight2.csv"
SPELL_CSV   = ROOT / "spell_icons.csv"
RUNE_CSV    = ROOT / "rune_icons.csv"

def fill_blanks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].fillna(-1)
        else:
            out[c] = out[c].fillna("—")
    return out

@st.cache_data(show_spinner=False)
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

# ---------- 데이터 로드 ----------
players = read_csv(PLAYERS_CSV)  # win 포함
spells  = read_csv(SPELL_CSV)    # spell1_name_fix, spell1_icon, spell2_name_fix, spell2_icon
runes   = read_csv(RUNE_CSV)     # rune_core, rune_core_icon, rune_sub, rune_sub_icon

# 키 통일
KEYS = ["matchId", "summonerName"]
for df in (players, spells, runes):
    for k in KEYS:
        if k in df.columns:
            df[k] = df[k].astype(str)

# win을 0/1 정수로 보정(누락=0)
if "win" not in players.columns:
    st.error("players 파일에 'win' 컬럼이 없습니다.")
    st.stop()
players["win"] = players["win"].fillna(0).astype(int).clip(0, 1)

players_small = players[KEYS + ["win"]].copy()

# ---------- 스펠 표 ----------
spells_m = spells.merge(players_small, on=KEYS, how="left")
for c in ["spell1_name_fix", "spell2_name_fix", "spell1_icon", "spell2_icon"]:
    spells_m[c] = spells_m.get(c, "").fillna("")

# 이름 둘 다 비면 제거(빈행 제거)
spells_m = spells_m[~(spells_m["spell1_name_fix"].astype(str).str.strip().eq("") &
                      spells_m["spell2_name_fix"].astype(str).str.strip().eq(""))].copy()

# 조합 정규화(이름 기준)
spells_m["_a"] = spells_m["spell1_name_fix"].astype(str)
spells_m["_b"] = spells_m["spell2_name_fix"].astype(str)
spells_m[["s_min","s_max"]] = np.sort(spells_m[["_a","_b"]].values, axis=1)

# 아이콘도 정렬 방향에 맞춤
def pick_icon_row(row):
    if row["_a"] <= row["_b"]:
        return pd.Series({"i1": row["spell1_icon"], "i2": row["spell2_icon"]})
    else:
        return pd.Series({"i1": row["spell2_icon"], "i2": row["spell1_icon"]})
spells_m[["i1","i2"]] = spells_m.apply(pick_icon_row, axis=1)

# 게임수=size, 승수=win 합
grp_s = spells_m.groupby(["s_min","s_max","i1","i2"], dropna=False)
spell_agg = grp_s.size().to_frame("게임수").reset_index()
wins = grp_s["win"].sum().reset_index(name="승수")
grp_spell = spell_agg.merge(wins, on=["s_min","s_max","i1","i2"], how="left")
grp_spell["승수"] = grp_spell["승수"].fillna(0).astype(int)

# 승률(분모>0에서만)
grp_spell["승률(%)"] = np.where(
    grp_spell["게임수"] > 0,
    (grp_spell["승수"] / grp_spell["게임수"] * 100).round(2),
    0.0
)

spell_table = grp_spell.rename(columns={
    "s_min":"스펠1 이름","s_max":"스펠2 이름","i1":"스펠1","i2":"스펠2"
})[["스펠1","스펠1 이름","스펠2","스펠2 이름","게임수","승수","승률(%)"]]
spell_table = fill_blanks(spell_table).sort_values(
    ["게임수","승률(%)"], ascending=[False, False]
).head(10)

# ---------- 룬 표 ----------
runes_m = runes.merge(players_small, on=KEYS, how="left")
for c in ["rune_core", "rune_core_icon", "rune_sub", "rune_sub_icon"]:
    runes_m[c] = runes_m.get(c, "").fillna("")

# 핵심/보조 둘 다 공백이면 제거
runes_m = runes_m[~(runes_m["rune_core"].astype(str).str.strip().eq("") &
                    runes_m["rune_sub"].astype(str).str.strip().eq(""))].copy()

grp_r = runes_m.groupby(["rune_core","rune_core_icon","rune_sub","rune_sub_icon"], dropna=False)
rune_agg = grp_r.size().to_frame("게임수").reset_index()
wins_r = grp_r["win"].sum().reset_index(name="승수")
grp_rune = rune_agg.merge(wins_r, on=["rune_core","rune_core_icon","rune_sub","rune_sub_icon"], how="left")
grp_rune["승수"] = grp_rune["승수"].fillna(0).astype(int)

grp_rune["승률(%)"] = np.where(
    grp_rune["게임수"] > 0,
    (grp_rune["승수"] / grp_rune["게임수"] * 100).round(2),
    0.0
)

rune_table = grp_rune.rename(columns={
    "rune_core":"핵심룬 이름",
    "rune_core_icon":"핵심룬",
    "rune_sub":"보조트리 이름",
    "rune_sub_icon":"보조트리",
})[["핵심룬","핵심룬 이름","보조트리","보조트리 이름","게임수","승수","승률(%)"]]
rune_table = fill_blanks(rune_table).sort_values(
    ["게임수","승률(%)"], ascending=[False, False]
).head(10)

# ---------- UI ----------
st.markdown("## Recommended Spell Combos")
st.dataframe(spell_table, use_container_width=True, hide_index=True)

st.markdown("## Recommended Rune Combos")
st.dataframe(rune_table, use_container_width=True, hide_index=True)

with st.expander("Raw rows (sample)"):
    cols = ["matchId","summonerName","spell1_name_fix","spell2_name_fix","rune_core","rune_sub","win"]
    raw = spells_m.merge(runes_m[KEYS+["rune_core","rune_sub"]], on=KEYS, how="left")
    have = [c for c in cols if c in raw.columns]
    st.dataframe(fill_blanks(raw[have].head(300)), use_container_width=True, hide_index=True)
