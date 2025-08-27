# app.py — ARAM 챔피언 대시보드 (슈퍼라이트2 대응/아이콘 렌더 고정)
import os, re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ARAM PS Dashboard", layout="wide")

# ===== 파일 경로 =====
PLAYERS_CSV   = "aram_matches_superlight2.csv"  # 참가자 행 (슈퍼라이트2)
CHAMP_CSV     = "champion_icons.csv"            # champion, champion_icon (또는 icon/icon_url)
RUNE_CSV      = "rune_icons.csv"                # (선택) 이름→아이콘 매핑이 있을 때만 사용
SPELL_CSV     = "spell_icons.csv"               # (선택) 스펠명→아이콘 매핑

DD_VERSION = "15.17.1"  # Data Dragon 폴백 버전

# ===== 유틸 =====
def _exists(path: str) -> bool:
    ok = os.path.exists(path)
    if not ok:
        st.warning(f"파일 없음: `{path}`")
    return ok

def _norm(x: str) -> str:
    return re.sub(r"\s+", "", str(x)).strip().lower()

def _looks_like_url(s: str) -> bool:
    return isinstance(s, str) and s.startswith(("http://", "https://"))

# ===== 로더 =====
@st.cache_data
def load_players(path: str) -> pd.DataFrame:
    if not _exists(path):
        st.stop()
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949")

    # 승패 정리
    if "win_clean" not in df.columns:
        if "win" in df.columns:
            df["win_clean"] = df["win"].astype(str).str.lower().isin(["true","1","t","yes"]).astype(int)
        else:
            df["win_clean"] = 0

    # 널/공백 정리
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].fillna("").astype(str).str.strip()
    return df

@st.cache_data
def load_champion_icons(path: str) -> dict:
    if not _exists(path):
        return {}
    df = pd.read_csv(path)
    name_col = next((c for c in ["champion","Champion","championName"] if c in df.columns), None)
    icon_col = next((c for c in ["champion_icon","icon","icon_url"] if c in df.columns), None)
    if not name_col or not icon_col:
        return {}
    df[name_col] = df[name_col].astype(str).str.strip()
    return dict(zip(df[name_col], df[icon_col]))

@st.cache_data
def load_rune_icons(path: str) -> dict:
    if not _exists(path):
        return {"core": {}, "sub": {}}
    df = pd.read_csv(path)
    core_map, sub_map = {}, {}
    if "rune_core" in df.columns:
        ic = "rune_core_icon" if "rune_core_icon" in df.columns else None
        if ic: core_map = dict(zip(df["rune_core"].astype(str), df[ic].astype(str)))
    if "rune_sub" in df.columns:
        ic = "rune_sub_icon" if "rune_sub_icon" in df.columns else None
        if ic: sub_map = dict(zip(df["rune_sub"].astype(str), df[ic].astype(str)))
    return {"core": core_map, "sub": sub_map}

@st.cache_data
def load_spell_icons(path: str) -> dict:
    if not _exists(path):
        return {}
    df = pd.read_csv(path)
    cand_name = [c for c in df.columns if _norm(c) in {"spell","spellname","name","스펠","스펠명"}]
    cand_icon = [c for c in df.columns if "icon" in c.lower()]
    m = {}
    if cand_name and cand_icon:
        name_col, icon_col = cand_name[0], cand_icon[0]
        for n, i in zip(df[name_col].astype(str), df[icon_col].astype(str)):
            m[_norm(n)] = i
            m[str(n).strip()] = i
    elif df.shape[1] >= 2:
        for n, i in zip(df.iloc[:,0].astype(str), df.iloc[:,1].astype(str)):
            m[_norm(n)] = i
            m[str(n).strip()] = i
    return m

# ===== 데이터 =====
df        = load_players(PLAYERS_CSV)
champ_map = load_champion_icons(CHAMP_CSV)
rune_maps = load_rune_icons(RUNE_CSV)
spell_map = load_spell_icons(SPELL_CSV)

# ===== 사이드바 =====
st.sidebar.title("ARAM PS Controls")
champs = sorted(df["champion"].dropna().unique().tolist()) if "champion" in df.columns else []
selected = st.sidebar.selectbox("Champion", champs, index=0 if champs else None)

# ===== 상단 요약 =====
dsel = df[df["champion"] == selected].copy() if len(champs) else df.head(0).copy()
games = len(dsel)
match_cnt_all = df["matchId"].nunique() if "matchId" in df.columns else len(df)
match_cnt_sel = dsel["matchId"].nunique() if "matchId" in dsel.columns else games
winrate = round(dsel["win_clean"].mean()*100, 2) if games else 0.0
pickrate = round((match_cnt_sel / match_cnt_all * 100), 2) if match_cnt_all else 0.0

c0, ctitle = st.columns([1, 5])
with c0:
    cicon = champ_map.get(selected, "")
    if cicon:
        st.image(cicon, width=64)
with ctitle:
    st.title(f"{selected}")

c1, c2, c3 = st.columns(3)
c1.metric("Games", f"{games}")
c2.metric("Win Rate", f"{winrate}%")
c3.metric("Pick Rate", f"{pickrate}%")

# ===== 아이템 추천 (슈퍼라이트2 내의 아이콘 사용, 외부 요약 의존 제거) =====
st.subheader("Recommended Items")
item_name_cols = [c for c in dsel.columns if re.fullmatch(r"item[0-6]_name", c)]
item_icon_cols = [c for c in dsel.columns if re.fullmatch(r"item[0-6]_icon", c)]
if games and item_name_cols and item_icon_cols:
    stacks = []
    for i in range(7):
        ncol = f"item{i}_name"
        icol = f"item{i}_icon"
        if ncol in dsel.columns and icol in dsel.columns:
            part = dsel[[ncol, icol, "win_clean"]].rename(columns={ncol:"item", icol:"icon_url"})
            stacks.append(part)
    union = pd.concat(stacks, ignore_index=True)
    union = union[union["item"].astype(str).str.strip() != ""]
    top_items = (
        union.groupby(["item","icon_url"])
        .agg(total_picks=("item","count"), wins=("win_clean","sum"))
        .reset_index()
    )
    # 같은 아이템명이 여러 아이콘으로 나뉘면 가장 많이 쓰인 아이콘을 대표로
    top_items = (
        top_items.sort_values(["item","total_picks"], ascending=[True,False])
        .drop_duplicates(subset=["item"])
    )
    top_items["win_rate"] = (top_items["wins"]/top_items["total_picks"]*100).round(2)
    top_items = top_items.sort_values(["total_picks","win_rate"], ascending=[False, False]).head(20)

    st.dataframe(
        top_items[["icon_url","item","total_picks","wins","win_rate"]],
        use_container_width=True,
        column_config={
            "icon_url": st.column_config.ImageColumn("아이콘", width="small"),
            "item": "아이템", "total_picks": "픽수", "wins": "승수", "win_rate": "승률(%)"
        }
    )
else:
    st.info("아이템 이름/아이콘 컬럼이 없어 아이템 집계를 만들 수 없습니다. (item*_name, item*_icon 필요)")

# ===== 스펠 추천 (순서무시 통합 + 이름/아이콘 보정) =====
st.subheader("Recommended Spell Combos")

# 표준 한글명 <-> DDragon 키
KOR_TO_DDRAGON = {
    "점멸":"SummonerFlash",
    "표식":"SummonerSnowball",
    "유체화":"SummonerHaste",
    "회복":"SummonerHeal",
    "점화":"SummonerDot",
    "정화":"SummonerBoost",
    "탈진":"SummonerExhaust",
    "방어막":"SummonerBarrier",
    "총명":"SummonerMana",
    "순간이동":"SummonerTeleport",
}
DDRAGON_TO_KOR = {v:k for k,v in KOR_TO_DDRAGON.items()}
ID_TO_DDRAGON = {  # 숫자ID -> 키
    "4":"SummonerFlash", "32":"SummonerSnowball", "6":"SummonerHaste", "7":"SummonerHeal",
    "14":"SummonerDot", "1":"SummonerBoost", "3":"SummonerExhaust", "21":"SummonerBarrier",
    "13":"SummonerMana", "12":"SummonerTeleport"
}

def spell_icon_from_key(key: str) -> str:
    if not key: return ""
    return f"https://ddragon.leagueoflegends.com/cdn/{DD_VERSION}/img/spell/{key}.png"

def pick_spell_source(df_):
    """
    우선순위:
    1) spell1_name_fix/spell2_name_fix (이름)
    2) spell1/spell2 가 URL이면 -> spell1Id/spell2Id 로 보정
    3) spell1/spell2 (이름)
    """
    if {"spell1_name_fix","spell2_name_fix"}.issubset(df_.columns):
        return ("name", "spell1_name_fix", "spell2_name_fix")
    if {"spell1","spell2"}.issubset(df_.columns):
        s1, s2 = df_["spell1"].astype(str).head(1).tolist()[0] if len(df_) else "", df_["spell2"].astype(str).head(1).tolist()[0] if len(df_) else ""
        if _looks_like_url(s1) or _looks_like_url(s2):
            if {"spell1Id","spell2Id"}.issubset(df_.columns):
                return ("id", "spell1Id", "spell2Id")
        return ("name", "spell1", "spell2")
    return (None, None, None)

mode, col1, col2 = pick_spell_source(dsel)
if games and mode:
    tmp = dsel[[col1, col2, "win_clean"]].copy()
    if mode == "id":
        # 숫자 → 키 → 한글명
        tmp["k1"] = tmp[col1].astype(str).map(ID_TO_DDRAGON).fillna("")
        tmp["k2"] = tmp[col2].astype(str).map(ID_TO_DDRAGON).fillna("")
        tmp["n1"] = tmp["k1"].map(DDRAGON_TO_KOR).fillna(tmp["k1"])
        tmp["n2"] = tmp["k2"].map(DDRAGON_TO_KOR).fillna(tmp["k2"])
    else:
        # 이름 → 키 시도 (spell_icons.csv가 있으면 먼저 사용)
        def to_key(name: str) -> str:
            n = str(name).strip()
            # spell_icons.csv를 아이콘 URL로 쓰는 경우도 있어서, 이름이면 매핑, 아니면 그대로
            for try_n in (n, _norm(n)):
                if try_n in spell_map:
                    # 아이콘 파일명에서 키를 추출할 수 없는 경우가 많음 → 한글명으로 역매핑
                    pass
            # 한글이 표준이면 직접 매핑
            return KOR_TO_DDRAGON.get(n, "")
        tmp["k1"] = tmp[col1].apply(lambda x: to_key(x))
        tmp["k2"] = tmp[col2].apply(lambda x: to_key(x))
        # 표시는 원래 이름
        tmp["n1"] = tmp[col1].astype(str)
        tmp["n2"] = tmp[col2].astype(str)

    # 순서 무시 통합
    tmp["pair_key"] = tmp.apply(lambda r: "|".join(sorted([r["k1"], r["k2"]])), axis=1)
    grp = (
        tmp.groupby("pair_key")
        .agg(games=("win_clean","count"), wins=("win_clean","sum"),
             k1=("k1","first"), k2=("k2","first"),
             n1=("n1","first"), n2=("n2","first"))
        .reset_index(drop=True)
    )
    grp["win_rate"] = (grp["wins"]/grp["games"]*100).round(2)

    # 아이콘 (키 기반)
    grp[["disp_k1","disp_k2"]] = grp.apply(lambda r: pd.Series(sorted([r["k1"], r["k2"]])), axis=1)
    grp["spell1_icon"] = grp["disp_k1"].apply(spell_icon_from_key)
    grp["spell2_icon"] = grp["disp_k2"].apply(spell_icon_from_key)

    # 이름도 키 기준으로 정렬된 순서에 맞춰 표시 (한글 없으면 키 그대로)
    grp["spell1_name"] = grp["disp_k1"].map(DDRAGON_TO_KOR).fillna(grp["n1"])
    grp["spell2_name"] = grp["disp_k2"].map(DDRAGON_TO_KOR).fillna(grp["n2"])

    grp = grp.sort_values(["games","win_rate"], ascending=[False, False]).head(10)

    st.dataframe(
        grp[["spell1_icon","spell1_name","spell2_icon","spell2_name","games","wins","win_rate"]],
        use_container_width=True,
        column_config={
            "spell1_icon": st.column_config.ImageColumn("스펠1", width="small"),
            "spell2_icon": st.column_config.ImageColumn("스펠2", width="small"),
            "spell1_name": "스펠1 이름", "spell2_name": "스펠2 이름",
            "games":"게임수","wins":"승수","win_rate":"승률(%)"
        }
    )
else:
    st.info("스펠 컬럼을 찾지 못했습니다. (spell1Id or spell1_name_fix / spell1 필요)")

# ===== 룬 추천 (슈퍼라이트2가 URL이면 그걸 바로 아이콘으로 사용) =====
st.subheader("Recommended Rune Combos")

def is_url_series(s: pd.Series) -> bool:
    return s.astype(str).str.startswith(("http://","https://")).all()

if games and {"rune_core","rune_sub"}.issubset(dsel.columns):
    ru = (
        dsel.groupby(["rune_core","rune_sub"])
        .agg(games=("win_clean","count"), wins=("win_clean","sum"))
        .reset_index()
    )
    ru["win_rate"] = (ru["wins"]/ru["games"]*100).round(2)

    # 아이콘 결정: 값 자체가 URL이면 그대로 사용, 아니면 외부 매핑 사용
    if is_url_series(ru["rune_core"]):
        ru["rune_core_icon"] = ru["rune_core"]
        ru["rune_core_name"] = ""
    else:
        core_map = rune_maps.get("core", {})
        ru["rune_core_icon"] = ru["rune_core"].map(core_map).fillna("")
        ru["rune_core_name"] = ru["rune_core"]

    if is_url_series(ru["rune_sub"]):
        ru["rune_sub_icon"] = ru["rune_sub"]
        ru["rune_sub_name"] = ""
    else:
        sub_map = rune_maps.get("sub", {})
        ru["rune_sub_icon"]  = ru["rune_sub"].map(sub_map).fillna("")
        ru["rune_sub_name"]  = ru["rune_sub"]

    ru = ru.sort_values(["games","win_rate"], ascending=[False,False]).head(10)

    st.dataframe(
        ru[["rune_core_icon","rune_core_name","rune_sub_icon","rune_sub_name","games","wins","win_rate"]],
        use_container_width=True,
        column_config={
            "rune_core_icon": st.column_config.ImageColumn("핵심룬", width="small"),
            "rune_sub_icon":  st.column_config.ImageColumn("보조트리", width="small"),
            "rune_core_name":"핵심룬 이름","rune_sub_name":"보조트리 이름",
            "games":"게임수","wins":"승수","win_rate":"승률(%)"
        }
    )
else:
    st.info("룬 컬럼(rune_core, rune_sub)이 없습니다.")

# ===== 원본(선택 챔피언) =====
with st.expander("Raw rows (selected champion)"):
    st.dataframe(dsel, use_container_width=True)
